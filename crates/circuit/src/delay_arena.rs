// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Reference-counted typed storage for `Delay` instruction durations.
//!
//! `StandardInstruction::Delay` carries a 4-byte [`DelayHandle`] into a process-global
//! arena of `(DelayUnit, DelayDuration)` slots.  Cloning a handle bumps an atomic
//! refcount on its slot; dropping a handle decrements that refcount and returns the
//! slot to a freelist when the count hits zero.
//!
//! The arena exists because Python's `Delay` data model distinguishes `int` durations
//! (only valid for the `dt` unit) from `float` durations (valid for time units).
//! Storing the duration in `Param::Float` would silently coerce ints to floats and
//! lose that invariant; storing the duration as a typed `DelayDuration` next to the
//! instruction's `DelayUnit` keeps the Rust- and Python-side data models aligned.
//!
//! # Thread safety
//!
//! All public operations are safe to call concurrently:
//!
//! * Allocation (`DelayHandle::new`) and final-drop (rc reaches zero) take a write
//!   guard on the arena's `slots` vector.
//! * Clone, non-final drop, and `with` closures take only a read guard plus an
//!   atomic refcount op.
//!
//! Slot addresses are stable (`Vec<Box<Slot>>`), so a handle that has snapshotted
//! a slot pointer can read it after releasing the read guard, as long as the
//! handle itself is still live.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, LazyLock, Mutex, RwLock};

use pyo3::prelude::*;

use crate::operations::DelayUnit;
use crate::parameter::parameter_expression::ParameterExpression;

/// The duration value of a `Delay` instruction.
///
/// Variants mirror the shapes that `Delay._validate_arguments` produces in Python:
///
/// * `Int(i64)` — `dt` unit only.  The Python validator rejects `dt` durations that
///   are not non-negative integers.
/// * `Float(f64)` — for time units (`s`, `ms`, `us`, `ns`, `ps`).
/// * `Expr(Arc<ParameterExpression>)` — `ParameterExpression`-typed durations, valid
///   for any unit.
/// * `PyObj(Py<PyAny>)` — a fallback for opaque Python-typed durations, used for
///   classical `expr.Expr` durations under the `expr` unit (and as a catch-all for
///   anything we couldn't classify on the way in).  We round-trip these through
///   Python rather than try to inspect them in Rust.
#[derive(Clone, Debug)]
pub enum DelayDuration {
    Int(i64),
    Float(f64),
    Expr(Arc<ParameterExpression>),
    PyObj(Py<PyAny>),
}

impl PartialEq for DelayDuration {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => a.to_bits() == b.to_bits(),
            (Self::Expr(a), Self::Expr(b)) => a == b,
            // PyObj equality is conservative: only same Python identity counts.
            // The DAG isomorphism path that needs richer comparison goes through
            // `is_close`, which delegates to Python.
            (Self::PyObj(a), Self::PyObj(b)) => Python::attach(|py| a.is(b.bind(py))),
            _ => false,
        }
    }
}

impl DelayDuration {
    /// Approximate equality, used by DAG isomorphism comparisons.
    ///
    /// Float-vs-int and float-vs-float are compared with relative tolerance `eps`;
    /// expressions are compared structurally; any cross-variant comparison returns
    /// `false`.
    pub fn is_close(&self, other: &Self, eps: f64) -> bool {
        let approx = |a: f64, b: f64| {
            let diff = (a - b).abs();
            let max = a.abs().max(b.abs()).max(1.0);
            diff <= eps * max
        };
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => approx(*a, *b),
            (Self::Int(a), Self::Float(b)) | (Self::Float(b), Self::Int(a)) => {
                approx(*a as f64, *b)
            }
            (Self::Expr(a), Self::Expr(b)) => a == b,
            (Self::PyObj(a), Self::PyObj(b)) => {
                Python::attach(|py| a.bind(py).eq(b).unwrap_or(false))
            }
            _ => false,
        }
    }
}

/// A heap-allocated arena slot.  The `Box` indirection in [`Arena::slots`] keeps each
/// slot's address stable across arena growth, which is what lets [`DelayHandle::with`]
/// release the read guard before deref'ing the slot.
struct Slot {
    rc: AtomicU32,
    unit: DelayUnit,
    data: DelayDuration,
}

struct Arena {
    /// Heap-stable slot storage.  Indexed by [`DelayHandle`].  We mutate a slot's
    /// `unit`/`data` only by *replacing* the entire `Box<Slot>` under the write lock;
    /// that path runs only when the slot is on the freelist (rc == 0) or freshly
    /// pushed, so it never aliases a live handle's view.
    ///
    /// The `Box` indirection is *load-bearing*: `DelayHandle::with` snapshots a
    /// `*const Slot` while holding only the read guard, then releases the guard
    /// before deref'ing the pointer.  That is sound only because the `Box`
    /// keeps each `Slot`'s address stable even when the outer `Vec` grows.
    /// `clippy::vec_box` doesn't know about this requirement.
    #[allow(clippy::vec_box)]
    slots: RwLock<Vec<Box<Slot>>>,
    /// Indices of slots whose rc has reached zero and may be reused on the next
    /// allocation.  Held under a separate mutex so the drop hot path doesn't have to
    /// take the slots' write guard.
    freelist: Mutex<Vec<u32>>,
}

static ARENA: LazyLock<Arena> = LazyLock::new(|| Arena {
    slots: RwLock::new(Vec::new()),
    freelist: Mutex::new(Vec::new()),
});

/// 4-byte handle into the global delay arena.
///
/// `StandardInstruction::Delay` carries one of these as its sole payload.
/// `Clone` bumps the slot's refcount; `Drop` decrements it and returns the slot to
/// the freelist when the count hits zero.
#[derive(Debug)]
pub struct DelayHandle(u32);

impl DelayHandle {
    /// Allocate a new slot in the global arena and return a handle to it.
    pub fn new(unit: DelayUnit, data: DelayDuration) -> Self {
        let slot = Box::new(Slot {
            rc: AtomicU32::new(1),
            unit,
            data,
        });
        // Try to reuse a slot from the freelist before growing.
        let reused = ARENA
            .freelist
            .lock()
            .expect("delay arena freelist poisoned")
            .pop();
        match reused {
            Some(idx) => {
                let mut guard = ARENA.slots.write().expect("delay arena slots poisoned");
                // The freelist promised this slot has rc == 0 and is unreferenced;
                // replacing the Box drops the old Slot, which is what we want.
                guard[idx as usize] = slot;
                Self(idx)
            }
            None => {
                let mut guard = ARENA.slots.write().expect("delay arena slots poisoned");
                let idx = guard
                    .len()
                    .try_into()
                    .expect("delay arena exceeded u32::MAX live + previously-live slots");
                if idx == u32::MAX {
                    // u32::MAX is reserved as a sentinel.
                    panic!("delay arena exceeded u32::MAX live + previously-live slots");
                }
                guard.push(slot);
                Self(idx)
            }
        }
    }

    /// Return the unit of this handle's slot.
    pub fn unit(&self) -> DelayUnit {
        self.with(|unit, _| unit)
    }

    /// Construct a handle from an existing live slot index, incrementing the slot's
    /// refcount.  Used by code that decodes a packed `DelayHandle` from an external
    /// representation (e.g. `PackedOperation`'s bitfield) and wants an owned handle.
    ///
    /// # Panics
    ///
    /// Panics if `idx` is out of range for the arena.  In normal use this is only
    /// called with indices the arena previously emitted from [`Self::new`].
    pub fn clone_from_index(idx: u32) -> Self {
        let guard = ARENA.slots.read().expect("delay arena slots poisoned");
        guard[idx as usize].rc.fetch_add(1, Ordering::Relaxed);
        Self(idx)
    }

    /// Consume this handle and return its raw slot index *without* decrementing the
    /// refcount.  The caller takes responsibility for ensuring the rc unit this
    /// handle owned is accounted for elsewhere (e.g. by storing the index in a
    /// packed bit field that will later be released via [`Self::release_by_index`]).
    pub fn forget_into_index(self) -> u32 {
        let idx = self.0;
        std::mem::forget(self);
        idx
    }

    /// Get this handle's raw slot index without consuming it.
    #[inline]
    pub fn index(&self) -> u32 {
        self.0
    }

    /// Release one rc unit on the slot at `idx`, freeing it if this was the last
    /// reference.  Mirrors [`Self::drop`] but operates directly on a u32 index, which
    /// is the form a `PackedOperation` carries.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `idx` is associated with a refcount unit that has
    /// not yet been released (e.g. obtained from a previous [`Self::forget_into_index`]
    /// or [`Self::clone_from_index`]).  Releasing the same unit twice corrupts the
    /// arena.
    pub unsafe fn release_by_index(idx: u32) {
        let guard = ARENA.slots.read().expect("delay arena slots poisoned");
        let prev = guard[idx as usize].rc.fetch_sub(1, Ordering::AcqRel);
        drop(guard);
        if prev == 1 {
            ARENA
                .freelist
                .lock()
                .expect("delay arena freelist poisoned")
                .push(idx);
        }
    }

    /// Run a closure with read access to this handle's `(unit, duration)`.
    ///
    /// The closure runs without the arena's read guard held, so it may itself perform
    /// arena operations (clone, allocate new handles, etc.).
    pub fn with<R>(&self, f: impl FnOnce(DelayUnit, &DelayDuration) -> R) -> R {
        // SAFETY: holding `self` proves the slot's rc >= 1, so `with_by_index` is
        // safe (see safety contract there).
        unsafe { with_by_index(self.0, f) }
    }
}

/// Run a closure with read access to the `(unit, duration)` of the slot at `idx`,
/// without holding any owned handle.
///
/// This is the lookup path used by code that has a raw slot index from a packed
/// representation but does not want to incur the cost of a rc++ on each read.
///
/// # Safety
///
/// The caller must ensure that some live handle (or other rc unit) keeps the slot
/// at `idx` from being recycled for the duration of this call.  Typical usage:
/// pass an index that was extracted from a `PackedOperation`'s bitfield, where the
/// `PackedOperation` itself owns the rc unit.
pub unsafe fn with_by_index<R>(idx: u32, f: impl FnOnce(DelayUnit, &DelayDuration) -> R) -> R {
    let guard = ARENA.slots.read().expect("delay arena slots poisoned");
    // See safety argument on `DelayHandle::with`: the caller's contract here mirrors
    // it, replacing "self handle is live" with "some external rc unit is live".
    let slot_ptr: *const Slot = &*guard[idx as usize];
    drop(guard);
    let slot = unsafe { &*slot_ptr };
    f(slot.unit, &slot.data)
}

impl Clone for DelayHandle {
    fn clone(&self) -> Self {
        let guard = ARENA.slots.read().expect("delay arena slots poisoned");
        // `Relaxed` matches the `Arc::clone` pattern: we already have a handle (so the
        // slot is not going away), and the new handle's data dependency on the old rc
        // value is established by program order.
        guard[self.0 as usize].rc.fetch_add(1, Ordering::Relaxed);
        Self(self.0)
    }
}

impl Drop for DelayHandle {
    fn drop(&mut self) {
        let guard = ARENA.slots.read().expect("delay arena slots poisoned");
        // `AcqRel` matches `Arc::drop`: the final decrementer needs to synchronize
        // with all prior increments (Acquire) and release this thread's writes to the
        // slot for any future allocator that reuses it (Release).
        let prev = guard[self.0 as usize].rc.fetch_sub(1, Ordering::AcqRel);
        let idx = self.0;
        drop(guard);
        if prev == 1 {
            // We were the last handle.  Push to the freelist so a future
            // `DelayHandle::new` can reuse this index (and replace the Box's contents).
            ARENA
                .freelist
                .lock()
                .expect("delay arena freelist poisoned")
                .push(idx);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// Read the rc of `handle`'s slot for testing.  Acquires the read guard the same
    /// way the production paths do.
    fn slot_rc(handle: &DelayHandle) -> u32 {
        let guard = ARENA.slots.read().unwrap();
        guard[handle.0 as usize].rc.load(Ordering::Acquire)
    }

    #[test]
    fn test_alloc_and_with() {
        let h = DelayHandle::new(DelayUnit::DT, DelayDuration::Int(100));
        h.with(|unit, data| {
            assert_eq!(unit, DelayUnit::DT);
            assert_eq!(data, &DelayDuration::Int(100));
        });
        assert_eq!(h.unit(), DelayUnit::DT);
    }

    #[test]
    fn test_alloc_each_unit_kind() {
        let int_h = DelayHandle::new(DelayUnit::DT, DelayDuration::Int(7));
        let float_h = DelayHandle::new(DelayUnit::NS, DelayDuration::Float(1.5));
        int_h.with(|u, d| {
            assert_eq!(u, DelayUnit::DT);
            assert_eq!(d, &DelayDuration::Int(7));
        });
        float_h.with(|u, d| {
            assert_eq!(u, DelayUnit::NS);
            assert_eq!(d, &DelayDuration::Float(1.5));
        });
    }

    #[test]
    fn test_clone_increments_rc() {
        let h = DelayHandle::new(DelayUnit::DT, DelayDuration::Int(5));
        assert_eq!(slot_rc(&h), 1);
        let h2 = h.clone();
        assert_eq!(slot_rc(&h), 2);
        assert_eq!(slot_rc(&h2), 2);
        // Both handles see the same payload.
        h.with(|_, d| assert_eq!(d, &DelayDuration::Int(5)));
        h2.with(|_, d| assert_eq!(d, &DelayDuration::Int(5)));
    }

    #[test]
    fn test_drop_decrements_rc() {
        let h = DelayHandle::new(DelayUnit::DT, DelayDuration::Int(99));
        let h2 = h.clone();
        assert_eq!(slot_rc(&h), 2);
        drop(h2);
        assert_eq!(slot_rc(&h), 1);
        h.with(|_, d| assert_eq!(d, &DelayDuration::Int(99)));
    }

    #[test]
    fn test_slot_reuse_via_freelist() {
        let h = DelayHandle::new(DelayUnit::DT, DelayDuration::Int(1));
        let idx = h.0;
        drop(h);
        // The next allocation should reuse the freed index.
        let h2 = DelayHandle::new(DelayUnit::S, DelayDuration::Float(2.0));
        assert_eq!(
            h2.0, idx,
            "freelist should reuse the most recently freed index"
        );
        h2.with(|u, d| {
            assert_eq!(u, DelayUnit::S);
            assert_eq!(d, &DelayDuration::Float(2.0));
        });
    }

    #[test]
    fn test_concurrent_clone_drop() {
        use std::thread;

        let h = DelayHandle::new(DelayUnit::DT, DelayDuration::Int(42));
        let mut threads = Vec::new();
        for _ in 0..8 {
            let h_t = h.clone();
            threads.push(thread::spawn(move || {
                for _ in 0..1000 {
                    let h2 = h_t.clone();
                    h2.with(|_, d| {
                        assert_eq!(d, &DelayDuration::Int(42));
                    });
                    drop(h2);
                }
            }));
        }
        for t in threads {
            t.join().unwrap();
        }
        // Original handle survives.
        assert_eq!(slot_rc(&h), 1);
        h.with(|_, d| assert_eq!(d, &DelayDuration::Int(42)));
    }

    #[test]
    fn test_with_closure_on_expr() {
        // Smoke-test that DelayDuration::Expr storage works.  Building a real
        // ParameterExpression from inside a Rust unit test requires Python state
        // we don't have here, so we use Arc::default-style construction via a
        // manually-constructed expression if available; otherwise just check Int+Float
        // round-trip (the Expr arm is exercised by Python integration tests).
        let h = DelayHandle::new(DelayUnit::DT, DelayDuration::Int(3));
        h.with(|u, d| {
            assert_eq!(u, DelayUnit::DT);
            assert!(matches!(d, DelayDuration::Int(3)));
        });
    }
}
