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

use qiskit_passmanager::{AnyPass, PassContext, PassManager, PassManagerContext};
use std::{
    any::{Any, TypeId},
    ffi::c_void,
    ptr::null_mut,
};

use crate::{
    ExitCode,
    pointers::{const_ptr_as_ref, mut_ptr_as_ref},
};

/// The pass' run method with signature
/// void* run(void *self, void *ir, QkPassContext *context);
type RunFunctionPtr = extern "C" fn(*mut c_void, *mut c_void, *mut PassContext) -> *mut c_void;

/// A struct representing a Qiskit pass in C.
///
/// The pass is exposed as opaque pointer to keep the ability of adding more fields
/// without breaking the ABI.
pub struct PassFromC {
    /// The pass' run function is taking the IR as void pointer and returns a void
    /// pointer, since we cannot do general type checking C-side
    run_ptr: Option<RunFunctionPtr>,
    /// The pass object, aka `self`.
    self_ptr: Option<*mut c_void>,
}

impl AnyPass for &PassFromC {
    fn input_type_id(&self) -> TypeId {
        TypeId::of::<*mut c_void>()
    }

    fn output_type_id(&self) -> TypeId {
        TypeId::of::<*mut c_void>()
    }

    fn run(&self, ir: Box<dyn Any>, context: &mut PassContext) -> anyhow::Result<Box<dyn Any>> {
        let ir = ir.downcast::<*mut c_void>().expect("Anything is c_void");
        let self_ptr = self
            .self_ptr
            .expect("C pass was required to be complete: `self` is missing.");
        let run_ptr = self
            .run_ptr
            .expect("C pass was required to be complete: `run` is missing.");
        let out = run_ptr(self_ptr, *ir, context);
        Ok(Box::new(out))
    }
}

/// @ingroup QkPassManager
/// Create a new pass.
///
/// This pass *must* be populated by calling ``qk_pass_set_run`` to register a function pointer
/// for pass execution.
#[unsafe(no_mangle)]
pub extern "C" fn qk_pass_new() -> *mut PassFromC {
    Box::into_raw(Box::new(PassFromC {
        self_ptr: None,
        run_ptr: None,
    }))
}

/// @ingroup QkPassManager
/// Free the pass.
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

/// @ingroup QkPassManager
/// Set the function pointer for pass execution.
///
/// This must be called at least once for a new ``QkPass`` to be valid. Not setting the
/// function pointer and executing a pass manager with such a pass is undefined behavior.
///
/// # Safety
///
/// Behavior is undefined if ``pass`` is not a non-null, valid pointer to a ``QkPass``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_pass_set_run(pass: *mut PassFromC, run_ptr: RunFunctionPtr) {
    // SAFETY: Per documentation, `pass` is valid and non-null
    let pass = unsafe { mut_ptr_as_ref(pass) };
    pass.run_ptr = Some(run_ptr);
}

/// @ingroup QkPassManager
/// Set the function pointer for self.
///
/// This must be called at least once for a new ``QkPass`` to be valid. Not setting the
/// function pointer and executing a pass manager with such a pass is undefined behavior.
///
/// # Safety
///
/// Behavior is undefined if ``pass`` is not a non-null, valid pointer to a ``QkPass``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_pass_set_self(pass: *mut PassFromC, self_ptr: *mut c_void) {
    // SAFETY: Per documentation, `pass` is valid and non-null
    let pass = unsafe { mut_ptr_as_ref(pass) };
    pass.self_ptr = Some(self_ptr);
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
/// @param pass The pass to push.
///
/// @return ``QkExitCode_Success`` upon succesful push, else an exit code explaining the failure.
///
/// # Safety
///
/// Behavior is undefined in ``pm`` is not a non-null, valid pointer to a ``QkPassManager`` or
/// ``pass`` is not a non-null, valid pointer to a ``QkPass``.
#[unsafe(no_mangle)]
pub extern "C" fn qk_passmanager_push_pass(
    pm: *mut PassManager,
    pass: *const PassFromC,
) -> ExitCode {
    // SAFETY: per documentation the pointer is non-null and valid
    let pm = unsafe { mut_ptr_as_ref(pm) };
    // SAFETY: per documentation the pointer is non-null and valid
    let pass = unsafe { const_ptr_as_ref(pass) };

    if let Err(e) = pm.try_push_pass(Box::new(pass)) {
        e.into()
    } else {
        ExitCode::Success
    }
}

#[repr(C)]
pub struct PassManagerResult {
    ir: *mut c_void,
    context: *mut PassManagerContext,
}

#[unsafe(no_mangle)]
pub extern "C" fn qk_passmanager_run(
    pm: *mut PassManager,
    ir: *mut c_void,
    result: *mut PassManagerResult,
    // callback:
) -> ExitCode {
    // SAFETY: Per documentation the pointer is non-null and valid
    let pm = unsafe { mut_ptr_as_ref(pm) };
    let result = unsafe { mut_ptr_as_ref(result) };

    match pm.run(ir, None) {
        Ok((ir_out, context)) => {
            result.ir = ir_out;
            result.context = Box::into_raw(Box::new(context));
            ExitCode::Success
        }
        Err(e) => {
            result.ir = null_mut();
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
pub extern "C" fn qk_passmanager_context_free(context: *mut PassManagerContext) {
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
