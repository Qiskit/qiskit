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

use std::ffi::{CStr, c_char, c_void};
use std::sync::Arc;
use std::{mem, ptr};

use super::{CompilationError, IrHandle, VtableEntry};
use crate::dyn_types::*;
use crate::pointers::{
    ExposesOwnedPointers, arc_clone_from_raw, const_ptr_as_ref, expose_by_arc, expose_by_box,
};
use qiskit_passmanager::{IR, PassContext, PassError, Predicate, UntilStable};
use qiskit_util::dyn_types::*;

/// @ingroup pass-manager
/// The available methods that can be implemented by a custom predicate.
///
/// These are the valid values for `QkVtableEntry::slot` and the associated function-pointer types
/// required for the `table` argument of `qk_predicate_vtable_new`.
///
/// # Safety
///
/// All functions, including the destructor, must be callable from any thread.
#[derive(Clone, Copy, derive_more::TryFrom, Debug)]
#[try_from(repr)]
#[repr(u32)]
pub enum PredicateSlot {
    /// Evaluate the predicate. *Required*.
    ///
    /// Signature:
    /// ```c
    /// bool evaluate(void *this, const void *ir, const void *context,
    ///               QkCompilationError **error);
    /// ```
    ///
    /// # Implementation
    ///
    /// `this` will be equal to a value passed to `qk_predicate_new` with this vtable.  `ir`
    /// points to *borrowed* data of the type specified by the vtable's `ir` field.  `context` is
    /// reserved and cannot be inspected; there are no functions that read it.
    ///
    /// If the function wants to indicate an error state, it must write a valid owned
    /// `QkCompilationError *` object into the `error` pointer, in which case the returned
    /// value is ignored.
    Evaluate = 0,
    /// A destructor for the `this` argument of a predicate, at the time that the predicate is
    /// destructed. *Optional*.
    ///
    /// Signature:
    /// ```c
    /// void delete(void *this);
    /// ```
    Delete = 1,
}

/// @ingroup pass-manager
/// A "type object" representing the behavior of a custom predicate.
///
/// These objects represent only the behavior; they must be combined with a data pointer of the
/// correct type to produce a complete predicate.
///
/// These are constructed using `qk_predicate_vtable_new`, and are given to
/// `qk_predicate_new`.
///
/// See `QkPredicateSlot` for an enumeration of the possible behavior and its semantics.
pub struct PredicateVtable {
    /// A human-readable name for the predicate.
    name: String,
    /// A handle to the complete IR methods of the type the predicate reads.
    ir: Arc<dyn DynTraitExposer<dyn IR>>,
    /// Evaluate the predicate.  See [`PredicateSlot::Evaluate`].
    evaluate: unsafe extern "C" fn(
        *mut c_void,
        *const c_void,
        *const PassContext,
        *mut *mut CompilationError,
    ) -> bool,
    /// The destructor of the [`CPredicate::this`] pointer.  See [`PredicateSlot::Delete`].
    delete: Option<unsafe extern "C" fn(*mut c_void) -> c_void>,
}
// SAFETY: `PredicateVtable` is always exposed and freed via an `Arc`.
const _: () = unsafe { expose_by_arc!(PredicateVtable) };
/// The primary constructor of `PredicateVtable` from Rust.
impl TryFrom<PredicateVtablePartial> for PredicateVtable {
    /// The first slot encountered that's required but absent.
    type Error = PredicateSlot;

    fn try_from(partial: PredicateVtablePartial) -> Result<Self, Self::Error> {
        Ok(Self {
            name: partial.name,
            ir: partial.ir,
            evaluate: partial.evaluate.ok_or(PredicateSlot::Evaluate)?,
            delete: partial.delete,
        })
    }
}

/// @ingroup pass-manager
/// Create a new "type object" for a custom predicate.
///
/// Once you have the output of this function, you create instances of the predicate by combining it
/// with a data pointer using `qk_predicate_new`.
///
/// @param name A borrowed nul-terminated human-readable name for the predicate. Used for debugging.
/// @param ir A borrowed handle to the IR "type object" the predicate reads.  This must match the
///     type of the body the predicate is used with, even if the predicate inspects only the
///     pass context.
/// @param table A borrowed table of `QkVtableEntry` objects terminated by an entry with `.slot =
///     -1`.  See `QkPredicateSlot` for the allowed slot values and associated function-pointer
///     types.  Unknown values of `slot` are ignored.
/// @return An owned type object that can be lent to `qk_predicate_new` to create instances of
///     the predicate.
///
/// # Safety
///
/// Behavior is undefined if any of the following are violated:
///
/// * `name` is a nul-terminated string.
/// * `ir` points to a valid `QkIrHandle` object.
/// * `table` points to contiguous, valid instances of `VtableEntry` that are all consistent entries
///   as defined by `QkPredicateSlot`, with no duplicate `slot` entries and terminated by `.slot
///   = -1`.
/// * `table` contains slot items for all required slots.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_predicate_vtable_new(
    name: *const c_char,
    ir: *const IrHandle,
    mut table: *const VtableEntry,
) -> *mut PredicateVtable {
    // SAFETY: per documentation `name` is a pointer to nul-terminated `char`s.
    let name = unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned();
    // SAFETY: per documentation, `ir` points to a valid `IrHandle` object.
    let ir = unsafe { const_ptr_as_ref(ir) };
    let mut partial = PredicateVtablePartial::new(name, Arc::clone(&ir.0));
    loop {
        // SAFETY: per documentation, `table` is valid for reads until we see the sentinel all-ones
        // pattern in a `slot`.
        let entry = unsafe { table.read() };
        table = if entry.slot == u32::MAX {
            break;
        } else {
            table.wrapping_add(1)
        };
        let Ok(slot) = PredicateSlot::try_from(entry.slot) else {
            // We assume this is a slot from a later version of Qiskit.
            continue;
        };
        // SAFETY: per documentation, `entry.ptr` is of the expected function-pointer type and valid
        // to call, because `entry.slot` was not all-ones.
        if unsafe { partial.set(slot, entry.ptr) } {
            // This a documented UB case.
            return ptr::dangling_mut();
        }
    }
    // SAFETY: per documentation, all required methods were set.
    (unsafe { PredicateVtable::try_from(partial).unwrap_unchecked() }).into_leaked()
}

/// @ingroup pass-manager
/// Free a single reference to a `QkPredicateVtable`.
///
/// You can call this once you have finished constructing instances of criteria that need this
/// vtable; each constructed predicate owns its own reference to the table.
///
/// Does nothing if `vtable` is `NULL`.
///
/// @param vtable The owned reference to release.
///
/// # Safety
///
/// Behavior is undefined if `vtable` is not null or a valid owned reference to an
/// `PredicateVtable`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_predicate_vtable_free(vtable: *mut PredicateVtable) {
    _ = (!vtable.is_null()).then(|| unsafe { PredicateVtable::steal(vtable) });
}

/// Internal Rust builder object for constructing `PredicateVtable` via its [`TryFrom`]
/// implementation.
struct PredicateVtablePartial {
    name: String,
    ir: Arc<dyn DynTraitExposer<dyn IR>>,
    evaluate: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *const c_void,
            *const PassContext,
            *mut *mut CompilationError,
        ) -> bool,
    >,
    delete: Option<unsafe extern "C" fn(*mut c_void) -> c_void>,
}
impl PredicateVtablePartial {
    /// Initialize the builder.
    ///
    /// After this, call [`set`](Self::set) repeatedly to fill in the slots.
    fn new(name: String, ir: Arc<dyn DynTraitExposer<dyn IR>>) -> Self {
        Self {
            name,
            ir,
            evaluate: None,
            delete: None,
        }
    }

    /// Set the `slot` to the corresponding function `ptr`.
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid function pointer of the type expected by the corresponding method in
    /// [`PredicateVtable`].
    unsafe fn set(&mut self, slot: PredicateSlot, ptr: *mut c_void) -> bool {
        // This lint suppression is because there is no actual safety provided by writing out the
        // entire type again; the safety is only guaranteed as a long-range interaction of a C
        // caller matching the correct type signature in the documentation of `PredicateSlot`.
        #[allow(clippy::missing_transmute_annotations)]
        match slot {
            PredicateSlot::Evaluate => {
                // SAFETY: per documentation, caller ensures pointer type validity.
                let ptr = unsafe { mem::transmute::<*mut c_void, _>(ptr) };
                self.evaluate.replace(ptr).is_some()
            }
            PredicateSlot::Delete => {
                // SAFETY: per documentation, caller ensures pointer type validity.
                let ptr = unsafe { mem::transmute::<*mut c_void, _>(ptr) };
                self.delete.replace(ptr).is_some()
            }
        }
    }
}

/// @ingroup pass-manager
/// A predicate usable by a pass-manager task.
///
/// This holds either a predicate whose behavior was defined through the C API using
/// `qk_predicate_vtable_new` and `qk_predicate_new`, or one of Qiskit's own predicates retrieved
/// with `qk_predicate_builtin`.
///
/// These are given to `qk_passmanager_push_while`.
pub struct CPredicate(pub(super) Box<dyn Predicate>);
// SAFETY: `CPredicate` is only created and freed via `Box`.
const _: () = unsafe { expose_by_box!(CPredicate) };

/// A predicate whose [`Predicate`] implementation was defined dynamically from C.
///
/// # Safety
///
/// The `this` data pointer must be safe to send and share between threads.  All methods in the
/// [`PredicateVtable`] must be safely callable from any thread.
struct VtablePredicate {
    /// The data pointer provided by C, and used as the `this` parameter in all
    /// [`PredicateVtable`] methods.
    this: *mut c_void,
    vtable: Arc<PredicateVtable>,
}
impl VtablePredicate {
    /// Create a new instance of the predicate.
    ///
    /// # Safety
    ///
    /// 1. `this` must point to data of the type that is expected by all "`this`" arguments in
    ///    functions in the `vtable`.
    /// 2. the data pointed to by `this` must be safe to share and send between threads.
    unsafe fn new(this: *mut c_void, vtable: Arc<PredicateVtable>) -> Self {
        Self { this, vtable }
    }
}
// SAFETY: per struct documentation, the `this` pointer must be safe to send between threads.
unsafe impl Send for VtablePredicate {}
// SAFETY: per struct documentation, the `this` pointer must be safe to share between threads.
unsafe impl Sync for VtablePredicate {}
impl Predicate for VtablePredicate {
    fn ir_id(&self) -> DynTypeId<'_> {
        self.vtable.ir.object_dyn_type_id()
    }
    fn name(&self) -> &str {
        &self.vtable.name
    }
    fn evaluate(&self, ir: &dyn IR, context: &PassContext) -> Result<bool, PassError> {
        let mut error = None::<ptr::NonNull<CompilationError>>;
        let error_ptr = (&raw mut error).cast::<*mut CompilationError>();
        let ir = self.vtable.ir.borrow(ir).cast::<c_void>();
        // SAFETY: per the documentation of `PredicateSlot`, `evaluate` is callable from any
        // thread with a borrowed `ir` of the vtable's type, and writes `error` only with a valid
        // owned pointer.
        let exit =
            unsafe { (self.vtable.evaluate)(self.this, ir, ptr::from_ref(context), error_ptr) };
        match error {
            // A written error takes precedence over the returned value.
            Some(error) => {
                // SAFETY: per documentation, a written `error` is a valid owned pointer.
                let error = unsafe { Box::from_raw(error.as_ptr()) };
                Err(PassError::Runtime(error.0))
            }
            None => Ok(exit),
        }
    }
}
impl Drop for VtablePredicate {
    fn drop(&mut self) {
        if let Some(delete) = self.vtable.delete {
            // SAFETY: per documentation of `PredicateVtable`, if the `delete` method is set, it
            // is valid to be passed `self.this` from any thread.
            unsafe { delete(self.this) };
        }
    }
}

/// @ingroup pass-manager
/// Enumeration of Qiskit's own predicates that are usable from the C API.
///
/// These are the valid inputs to `qk_predicate_builtin`.
#[derive(Clone, Copy, derive_more::TryFrom, Debug)]
#[try_from(repr)]
#[repr(u32)]
pub enum PredicateBuiltin {
    /// Stop as soon as the loop body reports that it left the IR alone.
    UntilStable = 0,
}

/// @ingroup pass-manager
/// Create a new instance of a predicate whose behavior was previously defined.
///
/// The resulting predicate object is typically then given to `qk_passmanager_push_while`.
///
/// @param data The owned data pointer for this `vtable`.
/// @param vtable A borrowed "type object" created by `qk_predicate_vtable_new`.
/// @return An owned predicate object.
///
/// # Safety
///
/// Behavior is undefined if any of the following are violated:
///
/// * `data` points to data of the correct type for the methods in `vtable`.
/// * `vtable` points to a valid instance of `QkPredicateVtable`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_predicate_new(
    data: *mut c_void,
    vtable: *const PredicateVtable,
) -> *mut CPredicate {
    // SAFETY: per documentation, `vtable` is the still-valid result of
    // `qk_predicate_vtable_new`, which returns the result of `Arc::into_raw`.
    let vtable = unsafe { arc_clone_from_raw(vtable) };
    // SAFETY: per documentation, `data` points to data of the type expected by `vtable` methods.
    let predicate = unsafe { VtablePredicate::new(data, vtable) };
    CPredicate(Box::new(predicate)).into_leaked()
}

/// @ingroup pass-manager
/// Get one of Qiskit's own predicates.
///
/// A built-in predicate reads no IR of its own, so it is compatible with any loop body.
///
/// @param predicate An identifier for the desired predicate.  See `QkPredicateBuiltin` for the
///     allowed values.
/// @param ir A borrowed handle to the IR "type object" of the loop body the predicate will be used
///     with.
/// @return An owned predicate object, or `NULL` if `predicate` is not a valid
///     `QkPredicateBuiltin`.
///
/// # Safety
///
/// Behavior is undefined if `ir` does not point to a valid `QkIrHandle` object.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_predicate_builtin(
    predicate: u32,
    ir: *const IrHandle,
) -> *mut CPredicate {
    // SAFETY: per documentation, `ir` points to a valid `IrHandle` object.
    let ir = unsafe { const_ptr_as_ref(ir) };
    match PredicateBuiltin::try_from(predicate) {
        Ok(PredicateBuiltin::UntilStable) => CPredicate(Box::new(ContextPredicate {
            name: "qiskit.until_stable",
            ir: Arc::clone(&ir.0),
            evaluate: UntilStable::is_stable,
        }))
        .into_leaked(),
        Err(_) => ptr::null_mut(),
    }
}

/// A [`Predicate`] that reads only the [`PassContext`], taking its IR type from the loop body it was
/// built against.
///
/// Qiskit's own IR-agnostic predicates are generic over the IR type, which a C caller cannot supply,
/// so this pairs the condition with an IR handle chosen at run time.
struct ContextPredicate {
    name: &'static str,
    ir: Arc<dyn DynTraitExposer<dyn IR>>,
    evaluate: fn(&PassContext) -> bool,
}
impl Predicate for ContextPredicate {
    fn ir_id(&self) -> DynTypeId<'_> {
        self.ir.object_dyn_type_id()
    }
    fn name(&self) -> &str {
        self.name
    }
    fn evaluate(&self, _ir: &dyn IR, context: &PassContext) -> Result<bool, PassError> {
        Ok((self.evaluate)(context))
    }
}

/// @ingroup pass-manager
/// Free a custom predicate instance.
///
/// You typically do not need to call this, because the pass-manager building functions steal
/// ownership of `QkPredicate` arguments.
///
/// @param predicate An owned pointer to the predicate to free.
///
/// # Safety
///
/// Behavior is undefined if `predicate` is not either null or a valid owned pointer to a
/// `QkPredicate`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_predicate_free(predicate: *mut CPredicate) {
    _ = (!predicate.is_null()).then(|| unsafe { CPredicate::steal(predicate) });
}
