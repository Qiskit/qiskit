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

use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError};
use std::{mem, ops, ptr};

// The `expect(dead_code)` used on the variants here is because we don't _expect_ to actually need
// to read from them.  They are simply RAII handles that we leak to C; all we need from them are
// their `Drop` implementations.

/// Type erasure of a read or a write lock; C doesn't need to care.
enum CGuardType<T: 'static> {
    #[expect(dead_code)]
    Read(RwLockReadGuard<'static, T>),
    #[expect(dead_code)]
    Write(RwLockWriteGuard<'static, T>),
}

/// An RAII handle representing an owned lock.
///
/// The purpose of this is purely to leak it to C.  The `guard` _must_ be derived from the same
/// `RwLock` stored in the associated `Arc`; this self-referencing is what allows us to erase the
/// standard lifetime of a `RwLockReadGuard` (or so) into `'static`; the true lifetime of the
/// `guard` is now bounded by the lifetime of the `Arc` we hold.
///
/// We never expect to actually use the code within this structure from Rust; there is no advantage
/// to it over the standard library guard types.  The purpose of it is simply to tie the `guard` and
/// `arc`'s lifetimes together while leaked, and so that the `Drop` implementation cleans up both at
/// the same time.
pub struct CGuard<T: 'static> {
    // SAFETY: the safety of this struct depends on `guard` being dropped before `arc`, because the
    // `guard` is only valid while the `arc` pointer is guaranteeing that the `RwLock` is alive.
    // Rust drop order in structs is in declaration order.
    /// The internal guard object, which can be either a read or a write; we don't need to expose
    /// the types to C because we leak out the reference by other means.
    #[expect(dead_code)]
    guard: CGuardType<T>,
    #[expect(dead_code)]
    arc: Arc<RwLock<T>>,
}
impl<T: 'static> CGuard<T> {
    /// Leak a `ReadGuard` or `WriteGuard` of the correct type into a raw pointer.
    ///
    /// You must call `Self::release` exactly once at some later point in time to release the lock.
    pub fn leak<G>(guard: G) -> *mut Self
    where
        G: Into<Self>,
    {
        Box::into_raw(Box::new(guard.into()))
    }

    /// Release the held lock.
    ///
    /// # Safety
    ///
    /// `ptr` must be exactly equal to a return value from `Self::leak`, and must not previously
    /// have been released.
    pub unsafe fn release(ptr: *mut Self) {
        // SAFETY: per documentation, `ptr` is a type-erased pointer produced by `Self::leak`.
        _ = unsafe { Box::from_raw(ptr.cast::<Self>()) };
    }
}
impl<T: 'static> From<ReadGuard<T>> for CGuard<T> {
    fn from(val: ReadGuard<T>) -> Self {
        Self {
            guard: CGuardType::Read(val.guard),
            arc: val.arc,
        }
    }
}
impl<T: 'static> From<WriteGuard<T>> for CGuard<T> {
    fn from(val: WriteGuard<T>) -> Self {
        Self {
            guard: CGuardType::Write(val.guard),
            arc: val.arc,
        }
    }
}

/// A lifetime-erased read guard held against a given `RwLock`.
pub struct ReadGuard<T: 'static> {
    // SAFETY: the safety of this struct depends on `guard` being dropped before `arc`, because the
    // `guard` is only valid while the `arc` pointer is guaranteeing that the `RwLock` is alive.
    // Rust drop order in structs is in declaration order.
    //
    // We don't actually ever expect to drop one of these structs, however, only to destructure it
    // into a `CGuard`.
    /// A guard taken out against the lock stored in the `arc`.
    guard: RwLockReadGuard<'static, T>,
    /// A keep-alive pointer to the lock, so it can't be released while we exist.
    arc: Arc<RwLock<T>>,
}

impl<T: 'static> ReadGuard<T> {
    /// Create the static guard from an arbitrary lifetime-bound guard and an `Arc` that protects
    /// the lock object from getting dropped.
    ///
    /// # Safety
    ///
    /// The `guard` must be derived from the same `RwLock` that is in the `arc`.
    unsafe fn from_guard<'a>(guard: RwLockReadGuard<'a, T>, arc: Arc<RwLock<T>>) -> Self {
        // SAFETY: per documentation, `arc` protects the same `RwLock` as the guard contains (via
        // reference) from going out of scope, so the guard is valid for as long as we hold `arc`.
        // The transmute types are trivially the same, since only the lifetime changes.
        let guard =
            unsafe { mem::transmute::<RwLockReadGuard<'a, T>, RwLockReadGuard<'static, T>>(guard) };
        Self { guard, arc }
    }

    /// Acquire a read guard on the underlying lock.
    ///
    /// This method blocks until the guard can be obtained.
    ///
    /// # Panics
    ///
    /// If the lock is poisoned.
    pub fn blocking(lock: &Arc<RwLock<T>>) -> Self {
        let guard = lock
            .read()
            .expect("lock poisoning is unhandleable in the C API");
        // SAFETY: `guard` is derived from `lock` right above.
        unsafe { Self::from_guard(guard, lock.clone()) }
    }

    /// Attempt to acquire a read guard on the underlying lock.
    ///
    /// This method returns immediately, returning `None` if the lock cannot be obtained without
    /// blocking for it.
    ///
    /// # Panics
    ///
    /// If the lock is poisoned.
    pub fn nonblocking(lock: &Arc<RwLock<T>>) -> Option<Self> {
        match lock.try_read() {
            Ok(guard) => {
                // SAFETY: `guard` is derived from `lock` right above.
                Some(unsafe { Self::from_guard(guard, lock.clone()) })
            }
            Err(TryLockError::WouldBlock) => None,
            Err(TryLockError::Poisoned(_)) => {
                panic!("lock poisoning is unhandleable in the C API")
            }
        }
    }

    /// Write this guard into a C-owned location, returning a reference to Rust space.
    ///
    /// As far as Rust is concerned, the data is valid for the rest of the lifetime of the program.
    /// Actually, the lifetime is bound to that of the `CGuard`, but that's no longer owned by us.
    /// The caller must ensure that the `CGuard` is only released after all references derived from
    /// it have been released.
    ///
    /// # Safety
    ///
    /// `guard` must be aligned and valid for one write of the correct type.  `guard` does not need
    /// to point to initialized data.
    pub unsafe fn leak_to_c(self, guard: *mut *mut CGuard<T>) -> &'static T {
        let out = ptr::from_ref::<T>(&self);
        // SAFETY: per documentation, `guard` is valid for one write.
        unsafe { guard.write(CGuard::leak(self)) };
        // SAFETY: the pointer is valid as a reference because we only just derived it above.  The
        // lifetime is now leaked to be owned by the lifetime of the `CGuard`, which per
        // documentation the caller will ensure is valid.
        unsafe { &*out }
    }
}

impl<T: 'static> ops::Deref for ReadGuard<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

/// A lifetime-erased write guard held against a given `RwLock`.
pub struct WriteGuard<T: 'static> {
    // SAFETY: the safety of this struct depends on `guard` being dropped before `arc`, because the
    // `guard` is only valid while the `arc` pointer is guaranteeing that the `RwLock` is alive.
    // Rust drop order in structs is in declaration order.
    //
    // We don't actually ever expect to drop one of these structs, however, only to destructure it
    // into a `CGuard`.
    /// A guard taken out against the lock stored in the `arc`.
    guard: RwLockWriteGuard<'static, T>,
    /// A keep-alive pointer to the lock, so it can't be released while we exist.
    arc: Arc<RwLock<T>>,
}

impl<T: 'static> WriteGuard<T> {
    /// Create the static guard from an arbitrary lifetime-bound guard and an `Arc` that protects
    /// the lock object from getting dropped.
    ///
    /// # Safety
    ///
    /// The `guard` must be derived from the same `RwLock` that is in the `arc`.
    unsafe fn from_guard<'a>(guard: RwLockWriteGuard<'a, T>, arc: Arc<RwLock<T>>) -> Self {
        // SAFETY: per documentation, `arc` protects the same `RwLock` as the guard contains (via
        // reference) from going out of scope, so the guard is valid for as long as we hold `arc`.
        // The transmute types are trivially the same, since only the lifetime changes.
        let guard = unsafe {
            mem::transmute::<RwLockWriteGuard<'a, T>, RwLockWriteGuard<'static, T>>(guard)
        };
        Self { guard, arc }
    }

    /// Write this guard into a C-owned location, returning a reference to Rust space.
    ///
    /// As far as Rust is concerned, the data is valid for the rest of the lifetime of the program.
    /// Actually, the lifetime is bound to that of the `CGuard`, but that's no longer owned by us.
    /// The caller must ensure that the `CGuard` is only released after all references derived from
    /// it have been released.
    ///
    /// # Safety
    ///
    /// `guard` must be aligned and valid for one write of the correct type.  `guard` does not need
    /// to point to initialized data.
    pub unsafe fn leak_to_c(mut self, guard: *mut *mut CGuard<T>) -> &'static mut T {
        let out = ptr::from_mut::<T>(&mut self);
        // SAFETY: per documentation, `guard` is valid for one write.
        unsafe { guard.write(CGuard::leak(self)) };
        // SAFETY: the pointer is valid as a reference because we only just derived it above.  The
        // lifetime is now leaked to be owned by the lifetime of the `CGuard`, which per
        // documentation the caller will ensure is valid.
        unsafe { &mut *out }
    }

    /// Acquire a write guard on the underlying lock.
    ///
    /// This method blocks until the guard can be obtained.
    ///
    /// # Panics
    ///
    /// If the lock is poisoned.
    pub fn blocking(lock: &Arc<RwLock<T>>) -> Self {
        let guard = lock
            .write()
            .expect("lock poisoning is unhandleable in the C API");
        // SAFETY: `guard` is derived from `lock` right above.
        unsafe { Self::from_guard(guard, lock.clone()) }
    }

    /// Attempt to acquire a write guard on the underlying lock.
    ///
    /// This method returns immediately, returning `None` if the lock cannot be obtained without
    /// blocking for it.
    ///
    /// # Panics
    ///
    /// If the lock is poisoned.
    pub fn nonblocking(lock: &Arc<RwLock<T>>) -> Option<Self> {
        match lock.try_write() {
            Ok(guard) => {
                // SAFETY: `guard` is derived from `lock` right above.
                Some(unsafe { Self::from_guard(guard, lock.clone()) })
            }
            Err(TryLockError::WouldBlock) => None,
            Err(TryLockError::Poisoned(_)) => {
                panic!("lock poisoning is unhandleable in the C API")
            }
        }
    }
}

impl<T: 'static> ops::Deref for WriteGuard<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}
impl<T: 'static> ops::DerefMut for WriteGuard<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
    }
}
