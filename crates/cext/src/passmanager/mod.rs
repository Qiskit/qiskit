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

use qiskit_passmanager::{AnyPass, Callback, Pass, PassContext, PassManager, PassManagerContext};
use std::{
    any::Any,
    ffi::{CStr, c_char, c_void},
    ptr::null_mut,
};

use crate::{
    ExitCode,
    pointers::{const_ptr_as_ref, mut_ptr_as_ref},
};

/// @ingroup QkPassManager
/// A slot in a vtable, with an `index` and a `ptr` to the function pointer.
///
/// This is a generic slot that can be used for different vtables.
#[repr(C)]
pub struct VTableSlot {
    index: usize,
    ptr: *mut c_void,
}

/// The pass' run method with signature
/// void* run(void *self, void *ir, QkPassContext *context);
type RunFunctionPtr = extern "C" fn(*mut c_void, *mut c_void, *mut PassContext) -> *mut c_void;

/// The pass vtable.
struct PassVTable {
    run_ptr: RunFunctionPtr,
}

impl PassVTable {
    /// # Safety
    ///
    /// Behavior is undefined if `slots` is not a pointer to `num_slots` consecutive, valid,
    /// non-null [VTableSlot]s, or if any of the slots does not point to a function with
    /// the signature documented in [PassFromC].
    unsafe fn from_slots(slots: *const VTableSlot, num_slots: usize) -> Result<Self, &'static str> {
        // SAFETY: Per documentation, the pointer points to `num_slots` valid slots.
        let slots = unsafe { ::std::slice::from_raw_parts(slots, num_slots) };

        let run_ptr = if let Some(run_slot) = slots.first() {
            // SAFETY: Per documentation, the slot has the right type.
            unsafe { ::std::mem::transmute::<*mut c_void, RunFunctionPtr>(run_slot.ptr) }
        } else {
            return Err("Expected slot 0 to contain the `run_ptr`, but slot is empty.");
        };

        Ok(Self { run_ptr })
    }
}

/// A struct representing a Qiskit pass in C.
///
/// The pass is exposed as opaque pointer to keep the ability of adding more fields
/// without breaking the ABI.
///
/// | Slot   | Signature                                                   | Required |
/// |--------|-------------------------------------------------------------|----------|
/// | 0      | `void (*run)(void *self, void *ir, QkPassContext *context)` |    Yes   |
pub struct PassFromC {
    /// The pass object, aka `self`.
    self_ptr: *mut c_void,
    /// The pass' vtable.
    vtable: PassVTable,
}

impl Pass for PassFromC {
    type InputIR = *mut c_void;
    type OutputIR = *mut c_void;

    fn run(&self, ir: Self::InputIR, context: &mut PassContext) -> anyhow::Result<Self::OutputIR> {
        Ok((self.vtable.run_ptr)(self.self_ptr, ir, context))
    }
}

/// @ingroup QkPassManager
/// Create a new pass.
///
/// @param self_ptr A `void *` to a config struct, or `self`. Can be `NULL`.
/// @param slots A pointer to the vtable slots specifying the pass methods.
/// @param num_slots The number of slots.
///
/// # Safety
///
/// Behavior is undefined if `slots` is not a pointer to `num_slots` consecutive, valid,
/// non-null [VTableSlot]s, or if any of the slots does not point to a function with
/// the signature documented in [PassFromC].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_pass_new(
    self_ptr: *mut c_void,
    slots: *const VTableSlot,
    num_slots: usize,
) -> *mut PassFromC {
    let Ok(vtable) = (unsafe { PassVTable::from_slots(slots, num_slots) }) else {
        return null_mut();
    };

    Box::into_raw(Box::new(PassFromC { self_ptr, vtable }))
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
pub unsafe extern "C" fn qk_pass_free(pass: *mut PassFromC) {
    if !pass.is_null() {
        if !pass.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(pass);
        }
    }
}

/// The pass vtable.
struct CallbackVTable {
    trigger: extern "C" fn(*mut c_void, u8) -> bool,
    ir_and_context: Option<extern "C" fn(*mut c_void, *mut c_void, *const PassContext)>,
    with_pass:
        Option<extern "C" fn(*mut c_void, *const PassFromC, *mut c_void, *const PassContext)>,
}

impl CallbackVTable {
    /// # Safety
    ///
    /// Behavior is undefined if `slots` is not a pointer to `num_slots` consecutive, valid,
    /// non-null [VTableSlot]s, or if any of the slots does not point to a function with
    /// the signature documented in [CCallback].
    unsafe fn from_slots(slots: *const VTableSlot, num_slots: usize) -> Result<Self, &'static str> {
        // SAFETY: Per documentation, the pointer points to `num_slots` valid slots.
        let slots = unsafe { ::std::slice::from_raw_parts(slots, num_slots) };

        let trigger = if let Some(slot) = slots.first() {
            // SAFETY: Per documentation, the slot has the right type.
            unsafe {
                ::std::mem::transmute::<*mut c_void, extern "C" fn(*mut c_void, u8) -> bool>(
                    slot.ptr,
                )
            }
        } else {
            return Err("Expected slot 0 to contain the `trigger`, but slot is empty.");
        };

        let ir_and_context = slots
            .get(1)
            .map(|slot| {
                // SAFETY: Per documentation, the slot has the right type.
                Some(unsafe {
                    ::std::mem::transmute::<
                        *mut c_void,
                        extern "C" fn(*mut c_void, *mut c_void, *const PassContext),
                    >(slot.ptr)
                })
            })
            .unwrap_or(None);

        let with_pass = slots
            .get(2)
            .map(|slot| {
                // SAFETY: Per documentation, the slot has the right type.
                Some(unsafe {
                    ::std::mem::transmute::<
                        *mut c_void,
                        extern "C" fn(
                            *mut c_void,
                            *const PassFromC,
                            *mut c_void,
                            *const PassContext,
                        ),
                    >(slot.ptr)
                })
            })
            .unwrap_or(None);

        Ok(Self {
            trigger,
            ir_and_context,
            with_pass,
        })
    }
}

/// A struct representing a callback for Qiskit's pass manager.
///
/// The callback is exposed as opaque pointer to keep the ability of adding more fields
/// without breaking the ABI.
///
/// | Slot | Signature                                                                    | Required |
/// |------|------------------------------------------------------------------------------|----------|
/// | 0    | `bool trigger(void *self, uint8_t hookpoint)`                                |    Yes   |
/// | 1    | `void ir_and_context(void *self, void *ir, QkPassContext *context)`          |    No    |
/// | 2    | `void with_pass(void *self, QkPass *pass, void *ir, QkPassContext *context)` |    No    |
///
/// The hookpoints are
///
/// | Value | Hookpoint  |
/// |-------|------------|
/// | 0     | Post pass  |
/// | 1     | Post task  |
/// | 2     | Post stage |
pub struct CCallback {
    /// The pass object, aka `self`.
    self_ptr: *mut c_void,
    /// The pass' vtable.
    vtable: CallbackVTable,
}

impl Callback for CCallback {
    fn trigger(&self, hookpoint: &qiskit_passmanager::CallbackType) -> bool {
        (self.vtable.trigger)(self.self_ptr, *hookpoint as u8)
    }

    fn ir_and_context(&self, ir: &dyn Any, context: &PassContext) {
        if let Some(callback) = self.vtable.ir_and_context {
            (callback)(
                self.self_ptr,
                *ir.downcast_ref::<*mut c_void>()
                    .expect("Failed casting IR to void*."),
                context as *const PassContext,
            )
        }
    }

    fn with_pass(&self, pass: &dyn AnyPass, ir: &dyn Any, context: &PassContext) {
        if let Some(callback) = self.vtable.with_pass {
            let Some(c_pass) = pass.as_any().downcast_ref::<PassFromC>() else {
                panic!("Failed casting pass.");
            };

            (callback)(
                self.self_ptr,
                c_pass as *const PassFromC,
                ir.downcast_ref::<*mut c_void>()
                    .copied()
                    .expect("Failed casting IR to void*."),
                context as *const PassContext,
            )
        }
    }
}

/// @ingroup QkPassManager
/// Create a new callback.
///
/// @param self_ptr A `void *` to a config struct, or `self`. Can be `NULL`.
/// @param slots A pointer to the vtable slots specifying the callback methods.
/// @param num_slots The number of slots.
///
/// # Safety
///
/// Behavior is undefined if `slots` is not a pointer to `num_slots` consecutive, valid,
/// non-null [VTableSlot]s, or if any of the slots does not point to a function with
/// the signature documented in [QkCallback].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_callback_new(
    self_ptr: *mut c_void,
    slots: *const VTableSlot,
    num_slots: usize,
) -> *mut CCallback {
    let Ok(vtable) = (unsafe { CallbackVTable::from_slots(slots, num_slots) }) else {
        return null_mut();
    };

    Box::into_raw(Box::new(CCallback { self_ptr, vtable }))
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
    callback: *const CCallback,
    result: *mut PassManagerResult,
    // callback:
) -> ExitCode {
    // SAFETY: Per documentation, `pm` is non-null and valid
    let pm = unsafe { mut_ptr_as_ref(pm) };
    let callback = match callback.is_null() {
        true => None,
        // SAFETY: The pointer is non-null, and per documentation it is then safe to read
        false => Some(unsafe { const_ptr_as_ref(callback) } as &dyn Callback),
    };
    // SAFETY: Per documentation, `result` is non-null and valid
    let result = unsafe { mut_ptr_as_ref(result) };

    match pm.run(ir, callback) {
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
