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

use crate::pointers::{const_ptr_as_ref, mut_ptr_as_ref};
use qiskit_accelerate::{
    nlayout::PhysicalQubit,
    target_transpiler::{InstructionProperties, Qargs, Target},
};
use qiskit_circuit::operations::{Operation, Param, StandardGate};

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

/// Represents the mapping between qargs and `InstructionProperties`
pub struct PropertyMap(Vec<(Qargs, Option<InstructionProperties>)>);

pub extern "C" fn qk_propety_map_new() -> *const PropertyMap {
    Box::into_raw(Box::new(PropertyMap(vec![])))
}

pub unsafe extern "C" fn qk_property_map_add(
    property_map: *mut PropertyMap,
    qargs: *const u32,
    num_qubits: u32,
    instruction_properties: *const InstructionProperties,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let prop_map = unsafe { mut_ptr_as_ref(property_map) };
    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
    let qubits: Qargs = unsafe {
        (0..num_qubits)
            .map(|idx| PhysicalQubit(*qargs.wrapping_add(idx as usize)))
            .collect()
    };
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let instruction_properties = unsafe { const_ptr_as_ref(instruction_properties) };
    prop_map.0.push((qubits, Some(*instruction_properties)));
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_add_instruction(
    target: *mut Target,
    operation: StandardGate,
    params: *const f64,
    property_map: *const PropertyMap,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { mut_ptr_as_ref(target) };

    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let property_map = unsafe { const_ptr_as_ref(property_map) };
    // SAFETY: Per the documentation the params pointers are arrays of num_params() elements.
    let parsed_params: &[Param] = unsafe {
        match operation.num_params() {
            0 => &[],
            1 => &[(*params.wrapping_add(0)).into()],
            2 => &[
                (*params.wrapping_add(0)).into(),
                (*params.wrapping_add(1)).into(),
            ],
            3 => &[
                (*params.wrapping_add(0)).into(),
                (*params.wrapping_add(1)).into(),
                (*params.wrapping_add(2)).into(),
            ],
            4 => &[
                (*params.wrapping_add(0)).into(),
                (*params.wrapping_add(1)).into(),
                (*params.wrapping_add(2)).into(),
                (*params.wrapping_add(3)).into(),
            ],
            // There are no standard gates that take > 4 params
            _ => unreachable!(),
        }
    };
    let props_map = if property_map.0.is_empty() {
        None
    } else {
        Some(
            property_map
                .0
                .iter()
                .map(|(q, i)| (q.clone(), *i))
                .collect(),
        )
    };

    target
        .add_instruction(operation.into(), parsed_params, None, props_map)
        .unwrap();
}
