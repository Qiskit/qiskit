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

use qiskit_passmanager::{
    DynTypeId, Pass, PassContext, PassError, PassManager, PassManagerContext,
};
use std::{
    any::Any,
    ffi::{CStr, c_char, c_void},
    mem, ptr,
    sync::Arc,
};

use crate::{
    ExitCode,
    pointers::{arc_clone_from_raw, const_ptr_as_ref, mut_ptr_as_ref},
};

type CFuncPtr = unsafe extern "C" fn() -> c_void;

// TODO: docs
/// @ingroup QkPassManager
/// An entry in a vtable for defining objects with custom behavior.
#[repr(C)]
pub struct VTableEntry {
    /// The "slot" of the function, or the sentinel `(uint32_t)-1` to mark the final array entry.
    ///
    /// This is typically set to some `enum` value, where the particular `enum` varies depending on
    /// which vtable you are defining.
    pub slot: u32,
    /// Any additional "flags" for the particular table entry.  These will typically be or'd (`|`)
    /// together, and the valid set of flags will be documented by the table user.
    pub flags: u32,
    /// A function pointer implementing the correct signature for the combination of the `slot` and
    /// `flags`.
    ///
    /// This can be `NULL` only in the case of `slots` being the sentinel `-1`.
    pub ptr: Option<CFuncPtr>,
}

// TODO: docs
pub struct Error(anyhow::Error);

/// @ingroup QkError
/// Create a new error message.
///
/// @param msg A borrowed pointer to a nul-terminated string.
///
/// # Safety
///
/// `msg` must point to nul-terminated bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_error_new(msg: *const c_char) -> ptr::NonNull<Error> {
    // SAFETY: per documentation, `msg` points to nul-terminated bytes.
    let msg = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
    let ptr = Box::into_raw(Box::new(Error(anyhow::Error::msg(msg))));
    // SAFETY: `Box` is always non-null.
    unsafe { ptr::NonNull::new_unchecked(ptr) }
}

// TODO: docs
#[derive(Clone, Copy, derive_more::TryFrom, Debug)]
#[try_from(repr)]
#[repr(u32)]
pub enum PassSlot {
    /// Run the pass on an IR of the correct type. *Required*.
    ///
    /// Signature:
    /// ```c
    /// void *run(void *this, void *ir, QkPassContext *context, QkError **error);
    /// ```
    ///
    /// # Implementation
    ///
    /// This implements the pass behavior.  `this` will be equal to a value passed to `qk_pass_new`
    /// with this vtable.  `ir` will point to data of the type specified by the vtable's `ir_in`
    /// field.
    // TODO: ownership of `ir`?
    Run = 0,
    /// A destructor for the `this` argument of a pass, at the time that the pass is destructed.
    /// *Optional*.
    ///
    /// Signature:
    /// ```c
    /// void delete(void *this);
    /// ```
    ///
    /// # Safety
    ///
    /// This must be safe to call from any thread.
    Delete = 1,
}

// TODO: docs
pub struct PassVTable {
    name: String,
    ir_in: Arc<IRVTable>,
    ir_out: Arc<IRVTable>,
    run: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        *mut PassContext,
        *mut *mut Error,
    ) -> *mut c_void,
    delete: Option<unsafe extern "C" fn(*mut c_void) -> c_void>,
}
impl TryFrom<PassVTablePartial> for PassVTable {
    /// The first slot encountered that's required but absent.
    type Error = PassSlot;

    fn try_from(partial: PassVTablePartial) -> Result<Self, Self::Error> {
        Ok(Self {
            name: partial.name,
            ir_in: partial.ir_in,
            ir_out: partial.ir_out,
            run: partial.run.ok_or(PassSlot::Run)?,
            delete: partial.delete,
        })
    }
}

// TODO: docs
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_pass_vtable_new(
    name: *const c_char,
    ir_in: *const IRVTable,
    ir_out: *const IRVTable,
    mut table: *const VTableEntry,
) -> *const PassVTable {
    // SAFETY: per documentation `name` is a pointer to nul-terminated `char`s.
    let name = unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned();
    // SAFETY: per documentation, `ir_in` is the valid output of an `IRVTable` constructor.
    let ir_in = unsafe { arc_clone_from_raw(ir_in) };
    // SAFETY: per documentation, `ir_out` is the valid output of an `IRVTable` constructor.
    let ir_out = unsafe { arc_clone_from_raw(ir_out) };
    let mut partial = PassVTablePartial::new(name, ir_in, ir_out);
    loop {
        // SAFETY: per documentation, `table` is valid for reads until we see the sentinel all-ones
        // pattern in a `slot`.
        let entry = unsafe { table.read() };
        table = if entry.slot == u32::MAX {
            break;
        } else {
            table.wrapping_add(1)
        };
        let Ok(slot) = PassSlot::try_from(entry.slot) else {
            // We assume this is a slot from a later version of Qiskit.
            // TODO: add an envvar / global to turn on debug information in these cases?
            continue;
        };
        // SAFETY: per documentation, `entry.ptr` is not null because `entry.slot` was not all ones.
        let ptr = unsafe { entry.ptr.unwrap_unchecked() };
        // SAFETY: per documentation, `ptr` is of the expected function-pointer type.
        unsafe { partial.set(slot, ptr) };
    }
    let vtable = PassVTable::try_from(partial).unwrap();
    Arc::into_raw(Arc::new(vtable))
}

// TODO: we can very likely do a bit of macro trickery to simplify the creation of vtable objects,
// taking care that the filled public one needs to be visible to `cbindgen`.  For initial
// implementation, hard-coding is good enough.
#[derive(Debug)]
struct PassVTablePartial {
    name: String,
    ir_in: Arc<IRVTable>,
    ir_out: Arc<IRVTable>,
    run: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *mut PassContext,
            *mut *mut Error,
        ) -> *mut c_void,
    >,
    delete: Option<unsafe extern "C" fn(*mut c_void) -> c_void>,
}
impl PassVTablePartial {
    fn new(name: String, ir_in: Arc<IRVTable>, ir_out: Arc<IRVTable>) -> Self {
        Self {
            name,
            ir_in,
            ir_out,
            run: None,
            delete: None,
        }
    }

    /// Set the `slot` to the corresponding function `ptr`.
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid function pointer of the type expected by the corresponding method in
    /// [`PassVTable`].
    unsafe fn set(&mut self, slot: PassSlot, ptr: CFuncPtr) -> bool {
        match slot {
            PassSlot::Run => {
                // SAFETY: per documentation, caller ensures pointer type validity.
                let ptr = unsafe { mem::transmute::<CFuncPtr, _>(ptr) };
                self.run.replace(ptr).is_some()
            }
            PassSlot::Delete => {
                // SAFETY: per documentation, caller ensures pointer type validity.
                let ptr = unsafe { mem::transmute::<CFuncPtr, _>(ptr) };
                self.delete.replace(ptr).is_some()
            }
        }
    }
}

// TODO: docs
pub struct CPass {
    /// The pass object, aka `self`.
    this: *mut c_void,
    vtable: Arc<PassVTable>,
}
impl CPass {
    /// Create a new instance of the pass.
    ///
    /// # Safety
    ///
    /// 1. `this` must point ot data of the type that is expected by all "`this`" arguments in
    ///    functions in the `vtable`.
    /// 2. the data pointed to by `this` must be safe to share and send between threads.
    unsafe fn new(this: *mut c_void, vtable: Arc<PassVTable>) -> Self {
        Self { this, vtable }
    }
}
impl Pass for CPass {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn ir_id_in(&self) -> DynTypeId<'_> {
        self.vtable.ir_in.dyn_type_id()
    }
    fn ir_id_out(&self) -> DynTypeId<'_> {
        self.vtable.ir_out.dyn_type_id()
    }
    fn name(&self) -> &str {
        &self.vtable.name
    }
    fn run(&self, ir: Box<dyn Any>, context: &mut PassContext) -> Result<Box<dyn Any>, PassError> {
        let mut error = None::<ptr::NonNull<Error>>;
        let error_ptr = (&raw mut error).cast::<*mut Error>();
        let ir_out = unsafe { (self.vtable.run)(self.this, ir, context, error_ptr) };
        match error {
            Some(error) => {
                let error = unsafe { Box::from_raw(error.as_ptr()) };
                Err(PassError::Runtime(error.0))
            }
            None => Ok(ir_out),
        }
    }
}
impl Drop for CPass {
    fn drop(&mut self) {
        if let Some(delete) = self.vtable.delete {
            // SAFETY: per documentation of `PassVTable`, if the `delete` method is set, it is valid
            // to be passed `self.this` from any thread.
            unsafe { delete(self.this) };
        }
    }
}
// SAFETY: per struct documentation, the `this` pointer must be safe to send between threads.
unsafe impl Send for CPass {}
// SAFETY: per struct documentation, the `this` pointer must be safe to share between threads.
unsafe impl Sync for CPass {}

// TODO: docs
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_pass_new(this: *mut c_void, vtable: *const PassVTable) -> *mut CPass {
    // SAFETY: per documentation, `vtable` is the still-valid result of `qk_pass_vtable_new`, which
    // returns the result of `Arc::into_raw`.
    let vtable = unsafe { arc_clone_from_raw(vtable) };
    // SAFETY: per documentation, `this` points to data of the type expected by `vtable` methods.
    let pass = unsafe { CPass::new(this, vtable) };
    Box::into_raw(Box::new(pass))
}

/// @ingroup QkPassManager
/// Free the pass. Note that :c:func:`qk_passmanager_push_pass` consumes passes and they must not
/// be freed manually anymore.
///
/// @param pass A pointer to the pass to free.
///
/// # Safety
///
/// Behavior is undefined if ``pass`` is not either null or a valid pointer to a ``QkPass``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_pass_free(pass: Option<ptr::NonNull<CPass>>) {
    if let Some(ptr) = pass {
        // SAFETY: per documentation, `pass` points to the result of `qk_pass_new`, which returns a
        // leaked boxed `CPass`.
        _ = unsafe { Box::from_raw(ptr.as_ptr()) };
    }
}

/// @ingroup QkPassManager
/// Create an empty pass manager.
///
/// This object must be freed by the user.
#[unsafe(no_mangle)]
pub extern "C" fn qk_passmanager_new() -> *mut PassManager {
    Box::into_raw(Box::new(PassManager::new()))
}

/// @ingroup QkPassManager
/// Free the pass manager.
///
/// @param pm A pointer to the pass manager to free.
///
/// # Safety
///
/// Behavior is undefined if ``pm`` is not either null or a valid pointer to a ``QkPassManager``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_passmanager_free(pm: *mut PassManager) {
    if !pm.is_null() {
        if !pm.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(pm);
        }
    }
}

/// @ingroup QkPassManager
/// Push a pass onto the pass manager.
///
/// @param pm The pass manager.
/// @param pass The pass to push. This consumes the `QkPass` object, which must not be freed after.
///
/// @return ``QkExitCode_Success`` upon succesful push, else an exit code explaining the failure.
///
/// # Safety
///
/// Behavior is undefined in ``pm`` is not a non-null, valid pointer to a ``QkPassManager`` or
/// ``pass`` is not a non-null, valid pointer to a ``QkPass``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_passmanager_push_pass(
    pm: *mut PassManager,
    pass: *mut PassFromC,
) -> ExitCode {
    // SAFETY: per documentation the pointer is non-null and valid
    let pm = unsafe { mut_ptr_as_ref(pm) };
    // SAFETY: per documentation the pointer is non-null and valid
    let pass: Box<PassFromC> = unsafe { Box::from_raw(pass) };

    if let Err(e) = pm.try_push_pass(pass) {
        e.into()
    } else {
        ExitCode::Success
    }
}

/// @ingroup QkPassManager
/// Get a value from the local pass context.
///
/// @param context A pointer to the pass context to read from.
/// @param key A char pointer to the key string.
/// @param value A pointer to a `void *` to write the value into.
///
/// @return A `QkExitCode_CastingError` if the key exists but the value could not be cast
///     to `void *`. Else `QkExitCode_Success`.
///
/// # Safety
///
/// Behavior is undefined if
///
/// * `context` is not a aligned, non-null pointer to a `QkPassContext`, or
/// * `key` is not a pointer to a valid, nul-terminated character array, or
/// * `value` is not safely writeable with a `void *`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_pass_context_get(
    context: *const PassContext,
    key: *const c_char,
    value: *mut *mut c_void,
) -> ExitCode {
    // SAFETY: Per documentation, `key` is a valid, nul-terminated char pointer
    let key = unsafe { CStr::from_ptr(key) }
        .to_str()
        .expect("Invalid UTF-8 character")
        .to_string();

    // SAFETY: Per documentation, `context` is a valid, non-null pointer to `PassContext`
    let context = unsafe { const_ptr_as_ref(context) };
    if let Some(as_any) = context.get(&key) {
        if let Some(as_void) = as_any.downcast_ref::<*mut c_void>().copied() {
            // SAFETY: Per documentation, `value` is safe to write
            unsafe { *value = as_void };
            ExitCode::Success
        } else {
            ExitCode::CastingError
        }
    } else {
        ExitCode::Success
    }
}

/// @ingroup QkPassManager
/// Set a value in the local pass context.
///
/// @param context A pointer to the pass context to read from.
/// @param key A char pointer to the key string.
/// @param value A pointer to write into the pass context.
///
/// # Safety
///
/// Behavior is undefined if
///
/// * `context` is not a aligned, non-null pointer to a `QkPassContext`
/// * `key` is not a pointer to a valid, nul-terminated character array
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_pass_context_set(
    context: *mut PassContext,
    key: *const c_char,
    value: *const c_void,
) {
    // SAFETY: Per documentation, `key` is a valid, nul-terminated char pointer
    let key = unsafe { CStr::from_ptr(key) }
        .to_str()
        .expect("Invalid UTF-8 character")
        .to_string();

    // SAFETY: Per documentation, `context` is a valid, non-null pointer to `PassContext`
    let context = unsafe { mut_ptr_as_ref(context) };
    // TODO The alternative is to have some Value::CPtr(ptr) here, since storing a
    // Box<*mut c_void> seems strange
    context.set(key, Box::new(value))
}

#[repr(C)]
pub struct PassManagerResult {
    ir: *mut c_void,
    context: *mut PassManagerContext,
}

/// @ingroup QkPassManager
/// Run the pass manager.
///
/// @param pm A pointer to the pass manager to run.
/// @param ir A `void *` to the IR to run.
/// @param result A pointer to a `QkPassManagerResult` object to write the results into.
///
/// @return An exit code describing the error if the compilation failed and the result pointers
/// are set to `NULL`.
///
/// # Safety
///
/// Behavior is undefined if
///
/// * `pm` not a valid, non-null pointer to a `QkPassManager`, or
/// * `callback` is not either null or a valid pointer to a `QkCallback`, or
/// * `result` is not a valid, non-null pointer to a `QkPassManagerResult`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_passmanager_run(
    pm: *mut PassManager,
    ir: *mut c_void,
    result: *mut PassManagerResult,
    // callback:
) -> ExitCode {
    // SAFETY: Per documentation, `pm` is non-null and valid
    let pm = unsafe { mut_ptr_as_ref(pm) };
    // SAFETY: Per documentation, `result` is non-null and valid
    let result = unsafe { mut_ptr_as_ref(result) };

    match pm.run(ir) {
        Ok((ir_out, context)) => {
            result.ir = ir_out;
            result.context = Box::into_raw(Box::new(context));
            ExitCode::Success
        }
        Err(e) => {
            result.ir = null_mut();
            result.context = null_mut();
            ExitCode::from(e)
        }
    }
}

/// @ingroup QkPassManager
/// Free the pass manager context.
///
/// @param context A pointer to the pass manager context to free.
///
/// # Safety
///
/// Behavior is undefined if ``context`` is not either null or a valid pointer to a
/// ``QkPassManagerContext``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_passmanager_context_free(context: *mut PassManagerContext) {
    if !context.is_null() {
        if !context.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(context);
        }
    }
}
