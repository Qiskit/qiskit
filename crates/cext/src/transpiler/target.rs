// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::pointers::const_ptr_as_ref;
use qiskit_accelerate::target_transpiler::{InstructionProperties, Target};

/// @ingroup QkTarget
/// Construct a new Target with the given number of qubits
///
/// @param num_qubits The number of qubits the Target will support
///
/// @return A pointer to the new Target
///
/// # Example
///     
///     QkTarget *target = qk_target_new(5);
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_target_new(num_qubits: u32) -> *mut Target {
    let target = Target::new(
        None,
        Some(num_qubits as usize),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    Box::into_raw(Box::new(target))
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_num_qubits(target: *const Target) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    target.num_qubits.unwrap_or_default() as u32
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_free(target: *mut Target) {
    if !target.is_null() {
        if !target.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(target);
        }
    }
}

pub extern "C" fn qk_instruction_properties_new(
    duration: f64,
    error: f64,
) -> *const InstructionProperties {
    Box::into_raw(Box::new(InstructionProperties::new(
        Some(duration),
        Some(error),
    )))
}

// #[no_mangle]
// #[cfg(feature = "cbinding")]
// pub unsafe extern "C" fn qk_target_add_instruction(target: *mut Target, operation: )
