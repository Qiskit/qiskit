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
use std::ptr::null_mut;

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

/// @ingroup QkTarget
/// Returns the number of qubits of this Target.
///
/// @param target A pointer to the Target.
///
/// @return The number of qubits this target can use.
///
/// # Example
///     
///     QkTarget *target = qk_target_new(5);
///     uint32_t num_qubits = qk_target_num_qubits(target);
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_num_qubits(target: *const Target) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    target.num_qubits.unwrap_or_default() as u32
}

/// @ingroup QkTarget
/// Free the Target.
///
/// @param target A pointer to the Target to free.
///
/// # Example
///     
///     QkTarget *target = qk_target_new(5);
///     qk_target_free(target);
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
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

/// @ingroup QkInstructionProps
/// Construct a new InstructionProperties object with the defined properties.
///
/// @param duration The duration of the instruction.
/// @param error The error rate of the instruction.
///
/// @return A pointer to the new instance of InstructionProperties.
///
/// # Example
///
///     QkInstructionProps *inst_props = qk_instruction_properties_new(1.098e-9, 2.000109e-10);
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

/// @ingroup QkInstructionProps
/// Gets the duration of the instruction.
///
/// @param instruction_properties The pointer to the instruction property.
///
/// @return The duration of the instruction.
///
/// # Example
///
///     QkInstructionProps *inst_props = qk_instruction_properties_new(1.098e-9, 2.000109e-10);
///     double duration = qk_instruction_properties_get_duration(inst_props);
///
/// # Safety
///
/// Behavior is undefined if ``instruction_properties`` is not a valid, non-null pointer to
/// a ``QkInstructionProps``.
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

/// @ingroup QkInstructionProps
/// Gets the error rate of the instruction.
///
/// @param instruction_properties The pointer to the instruction property.
///
/// @return The error rate of the instruction.
///
/// # Example
///
///     QkInstructionProps *inst_props = qk_instruction_properties_new(1.098e-9, 2.000109e-10);
///     double error = qk_instruction_properties_get_errorn(inst_props);
///
/// # Safety
///
/// Behavior is undefined if ``instruction_properties`` is not a valid, non-null pointer to
/// a ``QkInstructionProps``.
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

/// @ingroup QkInstructionProps
/// Free the InstructionProperties object.
///
/// @param target A pointer to the InstructionProperties object to free.
///
/// # Example
///     
///     QkInstructionProps *inst_props = qk_instruction_properties_new(1.098e-9, 2.000109e-10);
///     qk_instruction_properties_free(inst_props);
///
/// # Safety
///
/// Behavior is undefined if ``instruction_properties`` is not a valid, non-null pointer to
/// a ``QkInstructionProps`` object.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_instruction_properties_free(
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

/// @ingroup QkPropsMap
/// Creates an object that will serve as a mapping between an instruction's
/// qargs and instruction properties.
///
/// @return The Property Mapping structure.
///
/// # Example
///
///     QkPropsMap *props_map = qk_property_map_new();
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_property_map_new() -> *mut PropertyMap {
    Box::into_raw(Box::new(PropertyMap(Default::default())))
}

/// @ingroup QkPropsMap
/// Retrieves the length of the current property map.
///
/// @param property_map The pointer to the mapping object.
///
/// @return The length of the PropertyMap.
///
/// # Example
///
///     QkPropsMap *props_map = qk_property_map_new();
///     size_t props_size = qk_property_map_length(props_map);
///
/// # Safety
///
/// The behavior is undefined if ``property_map`` is not a valid,
/// non-null pointer to a ``QkPropsMap`` object.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_property_map_length(property_map: *const PropertyMap) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let prop_map = unsafe { const_ptr_as_ref(property_map) };
    prop_map.0.len()
}

/// @ingroup QkPropsMap
/// Frees the property map.
///
/// @param property_map The pointer to the mapping object to be freed.
///
/// # Example
///
///     QkPropsMap *props_map = qk_property_map_new();
///     qk_property_map_free(props_map);
///
/// # Safety
///
/// The behavior is undefined if ``property_map`` is not a valid,
/// non-null pointer to a ``QkPropsMap`` object.
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

/// @ingroup QkPropsMap
/// Checks if some qargs exist within the property map.
///
/// @param property_map The pointer to the mapping object.
/// @param qargs A pointer to the array of ``uint32_t`` qubit indices to add the
///     gate on, can be a null pointer to check for global properties.
/// @param num_qubits The length of the qargs array.
///
/// @return Whether the qargs are present or not.
///
/// # Example
///
///     QkPropsMap *props_map = qk_property_map_new();
///     uint32_t qargs[2] = {0, 1};
///     qk_property_map_add(props_map, qargs, 2, NULL);
///     qk_property_map_contains_qargs(props_map, qargs, 2);
///
/// # Safety
///
/// The behavior is undefined if ``property_map`` is not a valid, non-null pointer
/// to a ``QkPropsMap`` object.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_property_map_contains_qargs(
    property_map: *mut PropertyMap,
    qargs: *mut u32,
    num_qubits: u32,
) -> bool {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let prop_map = unsafe { mut_ptr_as_ref(property_map) };

    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
    let qargs = unsafe { parse_qargs(qargs, num_qubits) };

    prop_map.0.contains_key(&qargs)
}

/// @ingroup QkPropsMap
/// Retrieves an InstructionProperty based on its assigned qargs.
///
/// @param property_map The pointer to the mapping object.
/// @param qargs A pointer to the array of ``uint32_t`` qubit indices to add the
///     gate on, can be a null pointer to check for global properties.
/// @param num_qubits The length of the qargs array.
///
/// @return The properties associated with the qargs.
///
/// # Example
///
///     QkPropsMap *props_map = qk_property_map_new();
///     uint32_t qargs[2] = {0, 1};
///     qk_property_map_add(props_map, qargs, 2, qk_instruction_properties_new(0.0, 0.1));
///     qk_property_map_contains_qargs(props_map, qargs, 2);
///
/// # Safety
///
/// The behavior is undefined if ``property_map`` is not a valid, non-null pointer
/// to a ``QkPropsMap`` object.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_property_map_get(
    property_map: *mut PropertyMap,
    qargs: *mut u32,
    num_qubits: u32,
) -> *mut InstructionProperties {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let prop_map = unsafe { mut_ptr_as_ref(property_map) };

    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
    let qargs = unsafe { parse_qargs(qargs, num_qubits) };

    if let Some(Some(prop)) = prop_map.0.get(&qargs) {
        Box::into_raw(Box::new(*prop))
    } else {
        null_mut()
    }
}

/// @ingroup QkPropsMap
/// Adds an InstructionProperty based on its assigned qargs.
///
/// @param property_map The pointer to the mapping object.
/// @param qargs A pointer to the array of ``uint32_t`` qubit indices to add the
///     gate on, can be a null pointer to check for global properties.
/// @param num_qubits The length of the qargs array.
/// @param instruction_properties The instruction properties to be added.
///
/// # Example
///
///     QkPropsMap *props_map = qk_property_map_new();
///     uint32_t qargs[2] = {0, 1};
///     qk_property_map_add(props_map, qargs, 2, qk_instruction_properties_new(0.0, 0.1));
///
/// # Safety
///
/// The behavior is undefined if ``property_map`` is not a valid, non-null pointer
/// to a ``QkPropsMap`` object.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_property_map_add(
    property_map: *mut PropertyMap,
    qargs: *mut u32,
    num_qubits: u32,
    instruction_properties: *const InstructionProperties,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let prop_map = unsafe { mut_ptr_as_ref(property_map) };
    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
    let qubits: Qargs = unsafe { parse_qargs(qargs, num_qubits) };
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let instruction_properties = unsafe {
        if instruction_properties.is_null() {
            None
        } else {
            Some(const_ptr_as_ref(instruction_properties))
        }
    };
    prop_map.0.insert(qubits, instruction_properties.copied());
}

/// @ingroup QkTarget
/// Adds a StandardGate to the Target.
///
/// @param target A pointer to the Target.
/// @param operation The StandardGate to be added to the Target.
/// @param params The pointer to the array of ``double`` values to use as
/// parameters to the StandardGate. This can be a null pointer if there
/// are no parameters to be added.
/// @param property_map The mapping of qargs and InstructionProperties to
/// be associated with this instruction.
///
/// # Example
///     
///     QkTarget *target = qk_target_new(5);
///     QkPropsMap *props_map = qk_property_map_new();
///     uint32_t qargs[2] = {0, 1};
///     double params[1] = {3.1415};
///     qk_property_map_add(props_map, qargs, 2, qk_instruction_properties_new(0.0, 0.1));
///     qk_target_add_instruction(target, QkGate_CRX, params, props_map);
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
///
/// The ``params`` type is expected to be a pointer to an array of ``double`` where the length
/// matches the the expectations of the standard gate. If the array is insufficently long the
/// behavior of this function is undefined as this will read outside the bounds of the array.
/// It can be a null pointer if there are no params for a given gate. You can check
/// `qk_gate_num_params` to determine how many qubits are required for a given gate.
///
/// Behavior is undefined if ``property_map`` is not a valid, non-null pointer to a ``QkPropsMap``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_add_instruction(
    target: *mut Target,
    operation: StandardGate,
    params: *mut f64,
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

    if target
        .add_instruction(operation.into(), parsed_params, None, props_map)
        .is_ok()
    {
        ExitCode::Success
    } else {
        ExitCode::CInputError
    }
}

/// @ingroup QkTarget
/// Modifies the properties of a gate in the Target.
///
/// @param target A pointer to the Target.
/// @param name The name of the gate to modify.
/// @param qargs The pointer to the array of ``uint32_t`` values to use as
/// qargs. Can be ``NULL`` if global.
/// @param num_qubits The number of qubits in the array.
/// @param instruction_properties The instruction properties objects to replace by.
///
/// # Example
///     
///     QkTarget *target = qk_target_new(5);
///     QkPropsMap *props_map = qk_property_map_new();
///     uint32_t qargs[2] = {0, 1};
///     double params[1] = {3.1415};
///     qk_property_map_add(props_map, qargs, 2, qk_instruction_properties_new(0.0, 0.1));
///     qk_target_add_instruction(target, QkGate_CRX, params, props_map);
///
///     qk_target_update_instruction_properties(target, "cx", qargs, 2, qk_instruction_properties_new(0.0012, 1.1))
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
///
/// The ``name`` type expected to be a pointer to a null-terminated valid string. If the
/// pointer is not valid or not null ending, the behavior will be undefined.
///
/// The ``qargs` type is expected to be a pointer to an array of ``u32int_t`` where the length
/// matches is specified by ``num_qubits`` and has to match the expectation of the gate. If the
/// array is insufficently long the behavior of this function is undefined as this will read
/// outside the bounds of the array. It can be a null pointer if there are no qubits for
/// a given gate. You can check `qk_gate_num_qubits` to determine how many qubits are required
/// for a given gate.
///
/// Behavior is undefined if ``instruction_properties`` is not a valid, non-null pointer
/// to a ``QkInstructionProps``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_update_instruction_prop(
    target: *mut Target,
    name: *mut c_char,
    qargs: *mut u32,
    num_qubits: u32,
    instruction_properties: *const InstructionProperties,
) -> ExitCode {
    // SAFETY: Per documentation, name points to a null-terminated string of characters.
    let name: Box<CStr> = unsafe { Box::from(CStr::from_ptr(name)) };

    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { mut_ptr_as_ref(target) };
    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
    let qargs: Qargs = unsafe { parse_qargs(qargs, num_qubits) };
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let properties = unsafe { const_ptr_as_ref(instruction_properties) };

    if target
        .update_instruction_properties(
            name.to_str().expect("Error while extracting string"),
            &qargs,
            Some(*properties),
        )
        .is_ok()
    {
        ExitCode::Success
    } else {
        ExitCode::CInputError
    }
}

/// @ingroup QkTarget
/// Retrieves the mapping of qargs and properties from the Target.
///
/// @param target A pointer to the Target.
/// @param name The name of the gate to modify.
///
/// @return The property map associated with the gate name.
///
/// # Example
///     
///     QkTarget *target = qk_target_new(5);
///     QkPropsMap *props_map = qk_property_map_new();
///     uint32_t qargs[2] = {0, 1};
///     double params[1] = {3.1415};
///     qk_property_map_add(props_map, qargs, 2, qk_instruction_properties_new(0.0, 0.1));
///     qk_target_add_instruction(target, QkGate_CRX, params, props_map);
///
///     qk_target_get_prop_map(target, "cx")
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
///
/// The ``name`` type expected to be a pointer to a null-terminated valid string. If the
/// pointer is not valid or not null ending, the behavior will be undefined.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_get_prop_map(
    target: *const Target,
    name: *mut c_char,
) -> *const PropertyMap {
    // SAFETY: Per documentation, name points to a null-terminated string of characters.
    let name: Box<CStr> = unsafe { Box::from(CStr::from_ptr(name)) };

    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    if let Some(props) = target.get(name.to_str().expect("Error extracting str")) {
        Box::into_raw(Box::new(PropertyMap(props.clone())))
    } else {
        null_mut()
    }
}

/// @ingroup QkTarget
/// Retrieves the instruction properties associated with some qargs for an
/// instruction in the Target.
///
/// @param target A pointer to the Target.
/// @param name The name of the gate to modify.
/// @param qargs The pointer to the array of ``uint32_t`` values to use as
/// qargs. Can be ``NULL`` if global.
/// @param num_qubits The number of qubits in the array.
///
/// @return The InstructionProperties instance associated with the name and qargs.
///
/// # Example
///     
///     QkTarget *target = qk_target_new(5);
///     QkPropsMap *props_map = qk_property_map_new();
///     uint32_t qargs[2] = {0, 1};
///     double params[1] = {3.1415};
///     qk_property_map_add(props_map, qargs, 2, qk_instruction_properties_new(0.0, 0.1));
///     qk_target_add_instruction(target, QkGate_CRX, params, props_map);
///
///     qk_target_get_inst_prop(target, "cx", qargs, 2)
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
///
/// The ``name`` type expected to be a pointer to a null-terminated valid string. If the
/// pointer is not valid or not null ending, the behavior will be undefined.
///
/// The ``qargs` type is expected to be a pointer to an array of ``u32int_t`` where the length
/// matches is specified by ``num_qubits`` and has to match the expectation of the gate. If the
/// array is insufficently long the behavior of this function is undefined as this will read
/// outside the bounds of the array. It can be a null pointer if there are no qubits for
/// a given gate. You can check `qk_gate_num_qubits` to determine how many qubits are required
/// for a given gate.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_get_inst_prop(
    target: *const Target,
    name: *mut c_char,
    qargs: *mut u32,
    num_qubits: u32,
) -> *mut InstructionProperties {
    // SAFETY: Per documentation, name points to a null-terminated string of characters.
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
        null_mut()
    }
}

/// @ingroup QkTarget
/// Retrieves the names of all the operations in this Target.
///
/// @param target A pointer to the Target.
///
/// @return A list of the operation names.
///
/// # Example
///     
///     QkTarget *target = qk_target_new(5);
///     QkPropsMap *props_map = qk_property_map_new();
///     qk_property_map_add(props_map, NULL, 0, qk_instruction_properties_new(0.0, 0.1));
///     qk_target_add_instruction(target, QkGate_CRX, *[3.14], props_map);
///     qk_target_add_instruction(target, QkGate_H, NULL, qk_property_map_new());
///
///     qk_target_operation_names(target)
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
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

/// @ingroup QkTarget
/// Retrieves the physical qubits in this Target.
///
/// @param target A pointer to the Target.
///
/// @return An array with all of the physical qubits in the Target.
///
/// # Example
///     
///     QkTarget *target = qk_target_new(5);
///     qk_target_phyisical_qubits(target)
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_phyisical_qubits(target: *const Target) -> *mut usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    let mut physical_qubits: Vec<usize> = target.physical_qubits().collect();

    physical_qubits.as_mut_ptr()
}

/// @ingroup QkTarget
/// Retrieves the names of all non-global operations in this Target.
///
/// @param target A pointer to the Target.
/// @param strict_direction Checks whether the direction of the instruction's
/// qargs should be considered when classifying it.
///
/// @return A list of the names for the global operations in the Target.
///
/// # Example
///     
///     QkTarget *target = qk_target_new(5);
///     QkPropsMap *props_map = qk_property_map_new();
///     qk_property_map_add(props_map, NULL, 0, qk_instruction_properties_new(0.0, 0.1));
///     qk_target_add_instruction(target, QkGate_CRX, *[3.14], props_map);
///     qk_target_add_instruction(target, QkGate_H, NULL, qk_property_map_new());
///
///     qk_target_non_global_operation_names(target, true)
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
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

/// @ingroup QkTarget
/// Retrieves the names of all the operations in this Target which have
/// defined properties for the provided qargs.
///
/// @param target A pointer to the Target.
/// @param qargs The pointer to the array of ``uint32_t`` values to use as
/// qargs. Can be ``NULL`` if global.
/// @param num_qubits The number of qubits in the array.
///
/// @return The list of all the operation names with those qargs associated.
///
/// # Example
///
///     QkTarget *target = qk_target_new(5);
///     QkPropsMap *props_map = qk_property_map_new();
///     qk_property_map_add(props_map, NULL, 0, qk_instruction_properties_new(0.0, 0.1));
///     qk_target_add_instruction(target, QkGate_CRX, *[3.14], props_map);
///     qk_target_add_instruction(target, QkGate_H, NULL, qk_property_map_new());
///
///     qk_target_operation_names_for_qargs(target, *[0, 1], 2)
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
///
/// The ``qargs` type is expected to be a pointer to an array of ``u32int_t`` where the length
/// matches is specified by ``num_qubits`` and has to match the expectation of the gate. If the
/// array is insufficently long the behavior of this function is undefined as this will read
/// outside the bounds of the array. It can be a null pointer if there are no qubits for
/// a given gate. You can check `qk_gate_num_qubits` to determine how many qubits are required
/// for a given gate.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_operation_names_for_qargs(
    target: *const Target,
    qargs: *mut u32,
    num_qubits: u32,
) -> *mut *mut c_char {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
    let qargs: Qargs = unsafe { parse_qargs(qargs, num_qubits) };

    let mut result: Vec<*mut c_char> = if let Ok(names) = target.operation_names_for_qargs(&qargs) {
        // Temporary measure to ensure consistent results for the instruction names
        let mut temp_vec = Vec::from_iter(names);
        temp_vec.sort_unstable();
        temp_vec
            .into_iter()
            .map(|name| CString::new(name).unwrap().into_raw())
            .collect()
    } else {
        vec![]
    };

    let ptr = result.as_mut_ptr();

    // Prevent origin from being destroyed
    forget(result);
    ptr
}

/// @ingroup QkTarget
/// Retrieves the specified qargs for the provided operation name.
///
/// @param target A pointer to the Target.
/// @param name The name of the gate to modify.
///
/// @return A list of the qargs associated with this operation.
///
/// # Example
///     
///     QkTarget *target = qk_target_new(5);
///     QkPropsMap *props_map = qk_property_map_new();
///     qk_property_map_add(props_map, NULL, 0, qk_instruction_properties_new(0.0, 0.1));
///     qk_target_add_instruction(target, QkGate_CRX, *[3.14], props_map);
///     qk_target_add_instruction(target, QkGate_H, NULL, qk_property_map_new());
///
///     qk_target_qargs_for_operation_names(target, "x")
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
///
/// The ``name`` type expected to be a pointer to a null-terminated valid string. If the
/// pointer is not valid or not null ending, the behavior will be undefined.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_qargs_for_operation_names(
    target: *const Target,
    name: *mut c_char,
) -> *mut *mut u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.

    let target = unsafe { const_ptr_as_ref(target) };
    // SAFETY: Per documentation, name points to a null-terminated string of characters.
    let name = unsafe { CStr::from_ptr(name) };

    let mut result: Vec<*mut u32> =
        if let Ok(Some(qargs)) = target.qargs_for_operation_name(name.to_str().unwrap()) {
            qargs
                .map(|qargs| {
                    if let Qargs::Concrete(qargs) = qargs {
                        let mut value = qargs.iter().map(|bit| bit.0).collect::<Vec<_>>();
                        let value_ptr = value.as_mut_ptr();
                        forget(value);
                        value_ptr
                    } else {
                        null_mut()
                    }
                })
                .collect()
        } else {
            vec![]
        };

    let ptr = if result.is_empty() {
        null_mut()
    } else {
        result.as_mut_ptr()
    };

    // Prevent original from being destroyed
    forget(result);
    ptr
}

/// @ingroup QkTarget
/// Retrieves all of the qargs specified in the Target.
///
/// @param target A pointer to the Target.
///
/// @return all of the specified qargs in the Target.
///
/// # Example
///
///     QkTarget *target = qk_target_new(5);
///     QkPropsMap *props_map = qk_property_map_new();
///     qk_property_map_add(props_map, NULL, 0, qk_instruction_properties_new(0.0, 0.1));
///     qk_target_add_instruction(target, QkGate_CRX, *[3.14], props_map);
///     qk_target_add_instruction(target, QkGate_H, NULL, qk_property_map_new());
///
///     qk_target_qargs(target)
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_qargs(target: *const Target) -> *mut *mut u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    let mut result: Vec<*mut u32> = if let Some(qargs) = target.qargs() {
        qargs
            .map(|qargs| {
                if let Qargs::Concrete(qargs) = qargs {
                    let mut value = qargs.iter().map(|bit| bit.0).collect::<Vec<_>>();
                    let value_ptr = value.as_mut_ptr();
                    forget(value);
                    value_ptr
                } else {
                    null_mut()
                }
            })
            .collect()
    } else {
        vec![]
    };
    let ptr = if result.is_empty() {
        null_mut()
    } else {
        result.as_mut_ptr()
    };

    // Prevent original from being destroyed
    forget(result);
    ptr
}

/// @ingroup QkTarget
/// Checks if the provided instruction and its qargs are supported by this
/// Target.
///
/// @param target A pointer to the Target.
/// @parsm name The name of the instruction to check.
/// @param qargs The pointer to the array of ``uint32_t`` values to use as
/// qargs. Can be ``NULL`` if global.
/// @param num_qubits The number of qubits in the array.
///
/// @return Whether the instruction is supported or not.
///
/// # Example
///
///     QkTarget *target = qk_target_new(5);
///     QkPropsMap *props_map = qk_property_map_new();
///     qk_property_map_add(props_map, NULL, 0, qk_instruction_properties_new(0.0, 0.1));
///     qk_target_add_instruction(target, QkGate_CRX, *[3.14], props_map);
///     qk_target_add_instruction(target, QkGate_H, NULL, qk_property_map_new());
///
///     qk_target_instruction_supported(target, "cx*, [0, 1], 2)
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
///
/// The ``name`` type expected to be a pointer to a null-terminated valid string. If the
/// pointer is not valid or not null ending, the behavior will be undefined.
///
/// The ``qargs` type is expected to be a pointer to an array of ``u32int_t`` where the length
/// matches is specified by ``num_qubits`` and has to match the expectation of the gate. If the
/// array is insufficently long the behavior of this function is undefined as this will read
/// outside the bounds of the array. It can be a null pointer if there are no qubits for
/// a given gate. You can check `qk_gate_num_qubits` to determine how many qubits are required
/// for a given gate.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_instruction_supported(
    target: *const Target,
    name: *mut c_char,
    qargs: *mut u32,
    num_qubits: u32,
) -> bool {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    // SAFETY: Per documentation, name points to a null-terminated string of characters.
    let operation_name = unsafe { CStr::from_ptr(name) };

    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
    let qargs: Qargs = unsafe { parse_qargs(qargs, num_qubits) };

    target.instruction_supported(operation_name.to_str().unwrap(), &qargs)
}

/// @ingroup QkTarget
/// Check if the provided operation name exists within the Target.
///
/// @param target A pointer to the Target.
/// @param name The name of the gate to check.
///
/// @return Whether the gate is present in the Target or not.
///
/// # Example
///     
///     QkTarget *target = qk_target_new(5);
///     QkPropsMap *props_map = qk_property_map_new();
///     qk_property_map_add(props_map, NULL, 0, qk_instruction_properties_new(0.0, 0.1));
///     qk_target_add_instruction(target, QkGate_CRX, *[3.14], props_map);
///     qk_target_add_instruction(target, QkGate_H, NULL, qk_property_map_new());
///
///     qk_target_contains_instr(target, "x")
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
///
/// The ``name`` type expected to be a pointer to a null-terminated valid string. If the
/// pointer is not valid or not null ending, the behavior will be undefined.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_contains_instr(
    target: *const Target,
    name: *mut c_char,
) -> bool {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    // SAFETY: Per documentation, name points to a null-terminated string of characters.
    let operation_name = unsafe { CStr::from_ptr(name) };

    target.contains_key(operation_name.to_str().unwrap())
}

/// @ingroup QkTarget
/// Get the length of the Target.
///
/// @param target A pointer to the Target.
///
/// @return The length of the target.
///
///
/// # Example
///     
///     QkTarget *target = qk_target_new(5);
///     qk_target_add_instruction(target, QkGate_H, NULL, qk_property_map_new());
///
///     qk_target_length(target, "x")
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
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
