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

use crate::exit_codes::ExitCode;
use crate::pointers::{const_ptr_as_ref, mut_ptr_as_ref};
use indexmap::IndexMap;
use qiskit_accelerate::{
    nlayout::PhysicalQubit,
    target_transpiler::{InstructionProperties, Qargs, Target},
};
use qiskit_circuit::operations::{Operation, Param, StandardGate};
use std::ffi::{c_char, CStr, CString};
use std::mem::forget;
use std::ptr::null;

/// @ingroup QkTarget
/// Construct a new Target with the given number of qubits.
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

#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_instruction_properties_new(
    duration: f64,
    error: f64,
) -> *mut InstructionProperties {
    Box::into_raw(Box::new(InstructionProperties::new(
        Some(duration),
        Some(error),
    )))
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_instruction_properties_get_duration(
    instruction_properties: *const InstructionProperties,
) -> f64 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let instruction_properties = unsafe { const_ptr_as_ref(instruction_properties) };

    let Some(duration) = instruction_properties.duration else {
        // TODO: Double check if this is a valid option
        return 0.0;
    };
    duration
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_instruction_properties_get_error(
    instruction_properties: *const InstructionProperties,
) -> f64 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let instruction_properties = unsafe { const_ptr_as_ref(instruction_properties) };

    let Some(error) = instruction_properties.error else {
        // TODO: Double check if this is a valid option
        return 0.0;
    };
    error
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_instruction_properties_free(
    instruction_properties: *mut InstructionProperties,
) {
    if !instruction_properties.is_null() {
        if !instruction_properties.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(instruction_properties);
        }
    }
}

/// Represents the mapping between qargs and `InstructionProperties`
pub struct PropertyMap(IndexMap<Qargs, Option<InstructionProperties>, ahash::RandomState>);

#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_property_map_new() -> *mut PropertyMap {
    Box::into_raw(Box::new(PropertyMap(Default::default())))
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_property_map_length(property_map: *const PropertyMap) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let prop_map = unsafe { const_ptr_as_ref(property_map) };
    prop_map.0.len()
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_property_map_free(property_map: *mut PropertyMap) {
    if !property_map.is_null() {
        if !property_map.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(property_map);
        }
    }
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_property_map_contains_qargs(
    property_map: *mut PropertyMap,
    qargs: *const u32,
    num_qubits: u32,
) -> bool {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let prop_map = unsafe { mut_ptr_as_ref(property_map) };

    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
    let qargs = unsafe { parse_qargs(qargs, num_qubits) };

    prop_map.0.contains_key(&qargs)
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_property_map_get(
    property_map: *mut PropertyMap,
    qargs: *const u32,
    num_qubits: u32,
) -> *const InstructionProperties {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let prop_map = unsafe { mut_ptr_as_ref(property_map) };

    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
    let qargs = unsafe { parse_qargs(qargs, num_qubits) };

    if let Some(Some(prop)) = prop_map.0.get(&qargs) {
        Box::into_raw(Box::new(*prop))
    } else {
        null()
    }
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_property_map_add(
    property_map: *mut PropertyMap,
    qargs: *const u32,
    num_qubits: u32,
    instruction_properties: *const InstructionProperties,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let prop_map = unsafe { mut_ptr_as_ref(property_map) };
    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
    let qubits: Qargs = unsafe { parse_qargs(qargs, num_qubits) };
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let instruction_properties = unsafe { const_ptr_as_ref(instruction_properties) };
    prop_map.0.insert(qubits, Some(*instruction_properties));
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_add_instruction(
    target: *mut Target,
    operation: StandardGate,
    params: *const f64,
    property_map: *const PropertyMap,
) -> ExitCode {
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
        Some(property_map.0.clone())
    };

    if let Ok(_) = target.add_instruction(operation.into(), parsed_params, None, props_map) {
        ExitCode::Success
    } else {
        ExitCode::CInputError
    }
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_update_instruction_prop(
    target: *mut Target,
    name: *const c_char,
    qargs: *const u32,
    num_qubits: u32,
    instruction_properties: *const InstructionProperties,
) -> ExitCode {
    // SAFETY: TBD
    let name: Box<CStr> = unsafe { Box::from(CStr::from_ptr(name)) };

    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { mut_ptr_as_ref(target) };
    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
    let qargs: Qargs = unsafe { parse_qargs(qargs, num_qubits) };
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let properties = unsafe { const_ptr_as_ref(instruction_properties) };

    if let Ok(_) = target.update_instruction_properties(
        name.to_str().expect("Error while extracting string"),
        &qargs,
        Some(*properties),
    ) {
        ExitCode::Success
    } else {
        ExitCode::CInputError
    }
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_get_prop_map(
    target: *const Target,
    name: *const c_char,
) -> *const PropertyMap {
    // SAFETY: TBD
    let name: Box<CStr> = unsafe { Box::from(CStr::from_ptr(name)) };

    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    if let Some(props) = target.get(name.to_str().expect("Error extracting str")) {
        Box::into_raw(Box::new(PropertyMap(props.clone())))
    } else {
        null()
    }
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_get_inst_prop(
    target: *const Target,
    name: *const c_char,
    qargs: *const u32,
    num_qubits: u32,
) -> *const InstructionProperties {
    // SAFETY: TBD
    let name: Box<CStr> = unsafe { Box::from(CStr::from_ptr(name)) };

    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
    let qargs: Qargs = unsafe { parse_qargs(qargs, num_qubits) };

    if let Some(Some(Some(props))) = target
        .get(name.to_str().expect("Error extracting str"))
        .map(|map| map.get(&qargs))
    {
        Box::into_raw(Box::new(*props))
    } else {
        null()
    }
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_operation_names(target: *const Target) -> *mut *mut c_char {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    let mut names: Vec<*mut c_char> = target
        .operation_names()
        .map(|name| CString::new(name).unwrap().into_raw())
        .collect();
    let pointer = names.as_mut_ptr();

    // Prevent vec from being destroyed
    forget(names);
    pointer
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_phyisical_qubits(target: *const Target) -> *mut usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    let mut physical_qubits: Vec<usize> = target.physical_qubits().collect();

    physical_qubits.as_mut_ptr()
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_non_global_operation_names(
    target: *mut Target,
    strict_direction: bool,
) -> *mut *mut c_char {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { mut_ptr_as_ref(target) };

    let mut operation_names: Vec<*mut c_char> = target
        .get_non_global_operation_names(strict_direction)
        .unwrap()
        .iter()
        .map(|items| CString::new(items.as_str()).unwrap().into_raw())
        .collect();
    let ptr = operation_names.as_mut_ptr();
    // Prevent original vec from being forgotten
    forget(operation_names);
    ptr
}

// TODO: Figure out how to properly represent qargs
// #[repr(C)]
// pub enum QkQargs {
//     Global,
//     Concrete(*mut u32, usize),
// }
//
// impl QkQargs {
//     // pub extern "C" fn new()
//
//     pub unsafe fn to_qargs(&self) -> Qargs {
//         match self {
//             QkQargs::Global => Qargs::Global,
//             QkQargs::Concrete(qargs, size) => {
//                 // SAFETY: The qargs pointer is guaranteed to have a size N offset.
//                 unsafe {
//                     (0..*size)
//                         .map(|idx| PhysicalQubit(*qargs.wrapping_add(idx)))
//                         .collect()
//                 }
//             }
//         }
//     }
// }

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_operation_names_for_qargs(
    target: *const Target,
    qargs: *const u32,
    num_qubits: u32,
) -> *mut *const c_char {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
    let qargs: Qargs = unsafe { parse_qargs(qargs, num_qubits) };

    let mut result: Vec<*const c_char> = if let Ok(names) = target.operation_names_for_qargs(&qargs)
    {
        // Temporary measure to ensure consistent results for the instruction names
        let mut temp_vec = Vec::from_iter(names);
        temp_vec.sort_unstable();
        temp_vec
            .into_iter()
            .map(|name| CString::new(name).unwrap().into_raw().cast_const())
            .collect()
    } else {
        vec![]
    };

    let ptr = result.as_mut_ptr();

    // Prevent origin from being destroyed
    forget(result);
    ptr
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_qargs_for_operation_names(
    target: *const Target,
    name: *const c_char,
) -> *const *const u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    // SAFETY: TBD
    let name = unsafe { CStr::from_ptr(name) };

    let result: Vec<*const u32> =
        if let Ok(Some(qargs)) = target.qargs_for_operation_name(name.to_str().unwrap()) {
            qargs
                .map(|qargs| {
                    if let Qargs::Concrete(qargs) = qargs {
                        let value = qargs.iter().map(|bit| bit.0).collect::<Vec<_>>();
                        let value_ptr = value.as_ptr();
                        forget(value);
                        value_ptr
                    } else {
                        null()
                    }
                })
                .collect()
        } else {
            vec![]
        };

    let ptr = if result.is_empty() {
        null()
    } else {
        result.as_ptr()
    };

    // Prevent original from being destroyed
    forget(result);
    ptr
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_qargs(target: *const Target) -> *const *const u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    let result: Vec<*const u32> = if let Some(qargs) = target.qargs() {
        qargs
            .map(|qargs| {
                if let Qargs::Concrete(qargs) = qargs {
                    let value = qargs.iter().map(|bit| bit.0).collect::<Vec<_>>();
                    let value_ptr = value.as_ptr();
                    forget(value);
                    value_ptr
                } else {
                    null()
                }
            })
            .collect()
    } else {
        vec![]
    };
    let ptr = if result.is_empty() {
        null()
    } else {
        result.as_ptr()
    };

    // Prevent original from being destroyed
    forget(result);
    ptr
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_instruction_supported(
    target: *const Target,
    name: *const c_char,
    qargs: *const u32,
    num_qubits: u32,
) -> bool {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    // SAFETY: TBD
    let operation_name = unsafe { CStr::from_ptr(name) };

    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
    let qargs: Qargs = unsafe { parse_qargs(qargs, num_qubits) };

    target.instruction_supported(operation_name.to_str().unwrap(), &qargs)
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_contains_instr(
    target: *const Target,
    name: *const c_char,
) -> bool {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    // SAFETY: TBD
    let operation_name = unsafe { CStr::from_ptr(name) };

    target.contains_key(operation_name.to_str().unwrap())
}

#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_length(target: *const Target) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    target.len()
}

// Helpers

/// Parses qargs based on a pointer and its size.
unsafe fn parse_qargs(qargs: *const u32, num_qubits: u32) -> Qargs {
    if qargs.is_null() {
        Qargs::Global
    } else {
        // SAFETY: Per the documentation the qubits pointer is non-null and points to an array of num_qubits elements
        unsafe {
            (0..num_qubits)
                .map(|idx| PhysicalQubit(*qargs.wrapping_add(idx as usize)))
                .collect()
        }
    }
}
