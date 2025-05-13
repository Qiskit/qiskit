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
use core::f64;
use indexmap::IndexMap;
use qiskit_circuit::operations::{Operation, Param, StandardGate};
use qiskit_circuit::PhysicalQubit;
use qiskit_transpiler::target::{InstructionProperties, Qargs, Target, TargetError};
use std::mem::forget;
use std::ptr::null_mut;

// Helper structures

/// Represent a list of instruction names coming from the target.
#[repr(C)]
pub struct QkTargetOpsList {
    list: *mut StandardGate,
    length: usize,
}

/// Represent a list of qargs coming from the target.
#[repr(C)]
pub struct QkTargetQargsList {
    list: *mut QkTargetQargs,
    length: usize,
}

/// Represent qargs coming from the target.
#[repr(C)]
pub struct QkTargetQargs {
    args: *mut u32,
    length: usize,
}

/// Represents the properties of an instruction in the Target
#[repr(C)]
pub struct QkInstructionProps {
    duration: f64,
    error: f64,
}

/// @ingroup QkTarget
/// Construct a new ``Target`` with the given number of qubits.
/// The number of qubits is bound to change if an instruction is added with properties
/// that apply to a collection of qargs in which any index is higher than the specified
/// number of qubits
///
/// @param num_qubits The number of qubits the ``Target`` will explicitly support.
///
/// @return A pointer to the new ``Target``
///
/// # Example
///     
///     QkTarget *target = qk_target_new(5);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_target_new(num_qubits: usize) -> *mut Target {
    let target = Target::new(
        None,
        Some(num_qubits),
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
/// Returns the number of qubits of this ``Target``.
///
/// @param target A pointer to the ``Target``.
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
pub unsafe extern "C" fn qk_target_num_qubits(target: *const Target) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    target.num_qubits.unwrap_or_default()
}

/// @ingroup QkTarget
/// Returns the dt value of this ``Target``.
///
/// @param target A pointer to the ``Target``.
///
/// @return The dt value of this ``Target``.
///
/// # Example
///     const dt = 10e-9;
///     QkTarget *target = qk_target_new(5);
///     qk_target_set_dt(target, 10e-9);
///     double dt = qk_target_dt(target);
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_dt(target: *const Target) -> f64 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    target.dt.unwrap_or_default()
}

/// @ingroup QkTarget
/// Returns the granularity value of this ``Target``.
///
/// @param target A pointer to the ``Target``.
///
/// @return The granularity value of this ``Target``.
///
/// # Example
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 1
///     uint32_t granularity = qk_target_granularity(target);
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_granularity(target: *const Target) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    target.granularity
}

/// @ingroup QkTarget
/// Returns the `min_length` value of this ``Target``.
///
/// @param target A pointer to the ``Target``.
///
/// @return The min_length value of this ``Target``.
///
/// # Example
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 1
///     size_t min_length = qk_target_min_length(target);
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_min_length(target: *const Target) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    target.min_length
}

/// @ingroup QkTarget
/// Returns the `pulse_alignment` value of this ``Target``.
///
/// @param target A pointer to the ``Target``.
///
/// @return The pulse_alignment value of this ``Target``.
///
/// # Example
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 1
///     uint32_t pulse_alignment = qk_target_pulse_alignment(target);
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_pulse_alignment(target: *const Target) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    target.pulse_alignment
}

/// @ingroup QkTarget
/// Returns the `acquire_alignment` value of this ``Target``.
///
/// @param target A pointer to the ``Target``.
///
/// @return The acquire_alignment value of this ``Target``.
///
/// # Example
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 0
///     uint32_t acquire_alignment = qk_target_pulse_alignment(target);
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_acquire_alignment(target: *const Target) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    target.acquire_alignment
}

/// @ingroup QkTarget
/// Sets the dt value of this ``Target``.
///
/// @param target A pointer to the ``Target``.
/// @param dt The dt value for the system time resolution of input.
///
/// # Example
///     QkTarget *target = qk_target_new(5);
///     double dt = qk_target_set_dt(target, 10e-9);
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_set_dt(target: *mut Target, dt: f64) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { mut_ptr_as_ref(target) };
    target.dt = Some(dt);
    ExitCode::Success
}

/// @ingroup QkTarget
/// Sets the granularity value of this ``Target``.
///
/// @param target A pointer to the ``Target``.
/// @param granularity The value for the minimum pulse gate resolution in
///     units of ``dt``.
///
/// # Example
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 1
///     qk_target_set_granularity(target, 2);
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_set_granularity(
    target: *mut Target,
    granularity: u32,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { mut_ptr_as_ref(target) };
    target.granularity = granularity;
    ExitCode::Success
}

/// @ingroup QkTarget
/// Sets the `min_length` value of this ``Target``.
///
/// @param target A pointer to the ``Target``.
/// @param min_length minimum pulse gate length value in units of ``dt``.
///
/// # Example
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 1
///     qk_target_set_min_length(target, 3);
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_set_min_length(
    target: *mut Target,
    min_length: usize,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { mut_ptr_as_ref(target) };
    target.min_length = min_length;
    ExitCode::Success
}

/// @ingroup QkTarget
/// Returns the `pulse_alignment` value of this ``Target``.
///
/// @param target A pointer to the ``Target``.
/// @param pulse_alignment value representing a time resolution of gate.
///
/// # Example
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 1
///     qk_target_set_pulse_alignment(target, 4);
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_set_pulse_alignment(
    target: *mut Target,
    pulse_alignment: u32,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { mut_ptr_as_ref(target) };
    target.pulse_alignment = pulse_alignment;
    ExitCode::Success
}

/// @ingroup QkTarget
/// Sets the `acquire_alignment` value of this ``Target``.
///
/// @param target A pointer to the ``Target``.
/// @param acquire_alignment value representing a time resolution of measure instruction
///     starting time.
///
/// # Example
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 0
///     qk_target_set_acquire_alignment(target, 5);
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_set_acquire_alignment(
    target: *mut Target,
    acquire_alignment: u32,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { mut_ptr_as_ref(target) };
    target.acquire_alignment = acquire_alignment;
    ExitCode::Success
}

/// @ingroup QkTarget
/// Free the ``Target``.
///
/// @param target A pointer to the ``Target`` to free.
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
///     qk_property_map_add(props_map, qargs, 2, 0.0, 0.1);
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
        Box::into_raw(Box::new(prop.clone()))
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
/// @param duration The instruction's duration in seconds on the specific set of
///     qubits.
/// @param error The instruction's average error rate on the specific set of qubits.
///
/// # Example
///
///     QkPropsMap *props_map = qk_property_map_new();
///     uint32_t qargs[2] = {0, 1};
///     qk_property_map_add(props_map, qargs, 2, 0.0, 0.1);
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
    duration: f64,
    error: f64,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let prop_map = unsafe { mut_ptr_as_ref(property_map) };
    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
    let qubits: Qargs = unsafe { parse_qargs(qargs, num_qubits) };
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let duration = if duration.is_nan() {
        None
    } else {
        Some(duration)
    };
    let error = if error.is_nan() { None } else { Some(error) };
    prop_map
        .0
        .insert(qubits, Some(InstructionProperties::new(duration, error)));
}

/// @ingroup QkTarget
/// Adds a StandardGate to the ``Target``.
///
/// @param target A pointer to the ``Target``.
/// @param instruction The StandardGate to be added to the ``Target``.
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
///     qk_property_map_add(props_map, qargs, 2, 0.0, 0.1);
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
    instruction: StandardGate,
    params: *mut f64,
    property_map: *const PropertyMap,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { mut_ptr_as_ref(target) };

    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let property_map = unsafe {
        if property_map.is_null() {
            None
        } else {
            Some(const_ptr_as_ref(property_map))
        }
    };
    // SAFETY: Per the documentation the params pointers are arrays of num_params() elements.
    let parsed_params: &[Param] = unsafe {
        match instruction.num_params() {
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
    let props_map = property_map.map(|props| props.0.clone());

    match target.add_instruction(instruction.into(), parsed_params, None, props_map) {
        Ok(_) => ExitCode::Success,
        Err(e) => target_error_to_error_code(e),
    }
}

/// @ingroup QkTarget
/// Modifies the properties of a gate in the ``Target``.
///
/// @param target A pointer to the ``Target``.
/// @param instruction The instruction to modify.
/// @param qargs The pointer to the array of ``uint32_t`` values to use as
/// qargs. Can be ``NULL`` if global.
/// @param num_qubits The number of qubits in the array.
/// @param duration The instruction's duration in seconds on the specific set of
///     qubits.
/// @param error The instruction's average error rate on the specific set of qubits.
/// # Example
///     
///     QkTarget *target = qk_target_new(5);
///     QkPropsMap *props_map = qk_property_map_new();
///     uint32_t qargs[2] = {0, 1};
///     double params[1] = {3.1415};
///     qk_property_map_add(props_map, qargs, 2, 0.0, 0.1);
///     qk_target_add_instruction(target, QkGate_CRX, params, props_map);
///
///     qk_target_update_instruction_properties(target, QkGate_CRX, qargs, 2, 0.0012, 1.1)
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
///
/// The ``qargs`` type is expected to be a pointer to an array of ``u32int_t`` where the length
/// matches is specified by ``num_qubits`` and has to match the expectation of the gate. If the
/// array is insufficently long the behavior of this function is undefined as this will read
/// outside the bounds of the array. It can be a null pointer if there are no qubits for
/// a given gate. You can check `qk_gate_num_qubits` to determine how many qubits are required
/// for a given gate.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_update_instruction_prop(
    target: *mut Target,
    instruction: StandardGate,
    qargs: *mut u32,
    num_qubits: u32,
    duration: f64,
    error: f64,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { mut_ptr_as_ref(target) };
    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
    let qargs: Qargs = unsafe { parse_qargs(qargs, num_qubits) };

    let duration = if duration.is_nan() {
        None
    } else {
        Some(duration)
    };
    let error = if error.is_nan() { None } else { Some(error) };

    match target.update_instruction_properties(
        instruction.name(),
        &qargs,
        Some(InstructionProperties::new(duration, error)),
    ) {
        Ok(_) => ExitCode::Success,
        Err(e) => target_error_to_error_code(e),
    }
}

/// @ingroup QkTarget
/// Retrieves the mapping of qargs and properties from the ``Target``.
///
/// @param target A pointer to the ``Target``.
/// @param instruction The instruction to query.
///
/// @return The property map associated with the instruction.
///
/// # Example
///     
///     QkTarget *target = qk_target_new(5);
///     QkPropsMap *props_map = qk_property_map_new();
///     uint32_t qargs[2] = {0, 1};
///     double params[1] = {3.1415};
///     qk_property_map_add(props_map, qargs, 2, 0.0, 0.1);
///     qk_target_add_instruction(target, QkGate_CRX, params, props_map);
///
///     qk_target_get_prop_map(target, QkGate_CRX)
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_get_prop_map(
    target: *const Target,
    instruction: StandardGate,
) -> *const PropertyMap {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    if let Some(props) = target.get(instruction.name()) {
        Box::into_raw(Box::new(PropertyMap(props.clone())))
    } else {
        null_mut()
    }
}

/// @ingroup QkTarget
/// Retrieves the instruction properties associated with some qargs for an
/// instruction in the ``Target``.
///
/// @param target A pointer to the ``Target``.
/// @param instruction The instruction to query.
/// @param qargs The pointer to the array of ``uint32_t`` values to use as
/// qargs. Can be ``NULL`` if global.
/// @param num_qubits The number of qubits in the array.
///
/// @return The InstructionProperties instance associated with the instruction and qargs.
///
/// # Example
///     
///     QkTarget *target = qk_target_new(5);
///     QkPropsMap *props_map = qk_property_map_new();
///     uint32_t qargs[2] = {0, 1};
///     double params[1] = {3.1415};
///     qk_property_map_add(props_map, qargs, 2, 0.0, 0.1);
///     qk_target_add_instruction(target, QkGate_CRX, params, props_map);
///
///     qk_target_get_inst_prop(target, QkGate_CRX, qargs, 2)
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
///
/// The ``qargs`` type is expected to be a pointer to an array of ``u32int_t`` where the length
/// matches is specified by ``num_qubits`` and has to match the expectation of the gate. If the
/// array is insufficently long the behavior of this function is undefined as this will read
/// outside the bounds of the array. It can be a null pointer if there are no qubits for
/// a given gate. You can check `qk_gate_num_qubits` to determine how many qubits are required
/// for a given gate.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_get_inst_prop(
    target: *const Target,
    instruction: StandardGate,
    qargs: *mut u32,
    num_qubits: u32,
) -> QkInstructionProps {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
    let qargs: Qargs = unsafe { parse_qargs(qargs, num_qubits) };

    if let Some(Some(Some(props))) = target.get(instruction.name()).map(|map| map.get(&qargs)) {
        instruction_props_to_qk_inst_props(props)
    } else {
        panic!("The instruction or qargs are not present in this Target")
    }
}

/// @ingroup QkTarget
/// Retrieves the list of all the operations in this ``Target``.
///
/// @param target A pointer to the ``Target``.
///
/// @return A list of the operations.
///
/// # Example
///     
///     QkTarget *target = qk_target_new(5);
///     QkPropsMap *props_map = qk_property_map_new();
///     qk_property_map_add(props_map, NULL, 0, 0.0, 0.1);
///     qk_target_add_instruction(target, QkGate_CRX, *[3.14], props_map);
///     qk_target_add_instruction(target, QkGate_H, NULL, NULL);
///
///     qk_target_operations(target)
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_operations(target: *const Target) -> QkTargetOpsList {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    let mut operations: Vec<StandardGate> = target
        .operations()
        .map(|instruction| instruction.operation.standard_gate())
        .collect();
    let pointer = operations.as_mut_ptr();
    let length = operations.len();

    // Prevent vec from being destroyed
    forget(operations);
    QkTargetOpsList {
        list: pointer,
        length,
    }
}

/// @ingroup QkTarget
/// Retrieves the physical qubits in this ``Target``.
///
/// @param target A pointer to the ``Target``.
///
/// @return An array of size ``num_qubits`` with all of the physical qubits in the ``Target``.
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
/// Retrieves the names of all non-global operations in this ``Target``.
///
/// @param target A pointer to the ``Target``.
/// @param strict_direction Checks whether the direction of the instruction's
/// qargs should be considered when classifying it.
///
/// @return A list of the names for the global operations in the ``Target``.
///
/// # Example
///     
///     QkTarget *target = qk_target_new(5);
///     QkPropsMap *props_map = qk_property_map_new();
///     qk_property_map_add(props_map, NULL, 0, 0.0, 0.1);
///     qk_target_add_instruction(target, QkGate_CRX, *[3.14], props_map);
///     qk_target_add_instruction(target, QkGate_H, NULL, NULL);
///
///     qk_target_non_global_operation_names(target, true)
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_non_global_operations(
    target: *mut Target,
    strict_direction: bool,
) -> QkTargetOpsList {
    // SAFETY: Per documentation, the pointer is non-null and aligned.

    use qiskit_transpiler::target::TargetOperation;
    let target = unsafe { mut_ptr_as_ref(target) };

    let operation_names = target
        .get_non_global_operation_names(strict_direction)
        .unwrap()
        .to_owned();
    let mut operations: Vec<StandardGate> = operation_names
        .into_iter()
        .filter_map(|op_name| match target.operation_from_name(&op_name) {
            Some(TargetOperation::Normal(normal)) => Some(normal.operation.standard_gate()),
            _ => None,
        })
        .collect();
    let length = operations.len();
    let ptr = operations.as_mut_ptr();
    // Prevent original vec from being forgotten
    forget(operations);
    QkTargetOpsList { list: ptr, length }
}

/// @ingroup QkTarget
/// Retrieves the names of all the operations in this ``Target`` which have
/// defined properties for the provided qargs.
///
/// @note The order of the gate names is not guaranteed to be the same
/// between runs of the program.
///
/// @param target A pointer to the ``Target``.
/// @param qargs The pointer to the array of ``uint32_t`` values to use as
/// qargs. Can be ``NULL`` if global.
/// @param num_qubits The number of qubits in the array.
///
/// @return The list of all the instruction names with those qargs associated.
///
/// # Example
///
///     QkTarget *target = qk_target_new(5);
///     QkPropsMap *props_map = qk_property_map_new();
///     qk_property_map_add(props_map, NULL, 0, 0.0, 0.1);
///     qk_target_add_instruction(target, QkGate_CRX, *[3.14], props_map);
///     qk_target_add_instruction(target, QkGate_H, NULL, NULL);
///     
///     uint32_t qargs[2] = { 0, 1 };
///     qk_target_operations_for_qargs(target, qargs, 2);
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
///
/// The ``qargs`` type is expected to be a pointer to an array of ``u32int_t`` where the length
/// matches is specified by ``num_qubits`` and has to match the expectation of the gate. If the
/// array is insufficently long the behavior of this function is undefined as this will read
/// outside the bounds of the array. It can be a null pointer if there are no qubits for
/// a given gate. You can check `qk_gate_num_qubits` to determine how many qubits are required
/// for a given gate.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_operations_for_qargs(
    target: *const Target,
    qargs: *mut u32,
    num_qubits: u32,
) -> QkTargetOpsList {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
    let qargs: Qargs = unsafe { parse_qargs(qargs, num_qubits) };

    let mut result: Vec<StandardGate> = if let Ok(names) = target.operations_for_qargs(&qargs) {
        // Temporary measure to ensure consistent results for the instruction names
        names
            .into_iter()
            .map(|operation| operation.operation.standard_gate())
            .collect()
    } else {
        vec![]
    };
    let length = result.len();
    let ptr = result.as_mut_ptr();

    // Prevent origin from being destroyed
    forget(result);
    QkTargetOpsList { list: ptr, length }
}

/// @ingroup QkTarget
/// Retrieves the specified qargs for the provided operatioh.
///
/// @param target A pointer to the ``Target``.
/// @param operation The instruction whose properties we want to modify.
///
/// @return A list of the qargs associated with this instruction.
///
/// # Example
///     
///     QkTarget *target = qk_target_new(5);
///     QkPropsMap *props_map = qk_property_map_new();
///     qk_property_map_add(props_map, NULL, 0, 0.0, 0.1);
///     qk_target_add_instruction(target, QkGate_CRX, *[3.14], props_map);
///     qk_target_add_instruction(target, QkGate_H, NULL, NULL);
///
///     qk_target_qargs_for_operation(target, QkGate_CX)
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_qargs_for_operation(
    target: *const Target,
    operation: StandardGate,
) -> QkTargetQargsList {
    // SAFETY: Per documentation, the pointer is non-null and aligned.

    let target = unsafe { const_ptr_as_ref(target) };

    let mut result: Vec<QkTargetQargs> =
        if let Ok(Some(qargs)) = target.qargs_for_operation_name(operation.name()) {
            qargs
                .map(|qargs| {
                    if let Qargs::Concrete(qargs) = qargs {
                        let mut value = qargs.iter().map(|bit| bit.0).collect::<Vec<_>>();
                        let value_ptr = value.as_mut_ptr();
                        let length = value.len();
                        forget(value);

                        QkTargetQargs {
                            args: value_ptr,
                            length,
                        }
                    } else {
                        QkTargetQargs {
                            args: null_mut(),
                            length: 0,
                        }
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
    let length = result.len();

    // Prevent original from being destroyed
    forget(result);
    QkTargetQargsList { list: ptr, length }
}

/// @ingroup QkTarget
/// Retrieves all of the qargs specified in the ``Target``.
///
/// @param target A pointer to the ``Target``.
///
/// @return all of the specified qargs in the ``Target``.
///
/// # Example
///
///     QkTarget *target = qk_target_new(5);
///     QkPropsMap *props_map = qk_property_map_new();
///     qk_property_map_add(props_map, NULL, 0, 0.0, 0.1);
///     qk_target_add_instruction(target, QkGate_CRX, *[3.14], props_map);
///     qk_target_add_instruction(target, QkGate_H, NULL, NULL);
///
///     qk_target_qargs(target)
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_qargs(target: *const Target) -> QkTargetQargsList {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    let mut result: Vec<QkTargetQargs> = if let Some(qargs) = target.qargs() {
        qargs
            .map(|qargs| {
                if let Qargs::Concrete(qargs) = qargs {
                    let mut value = qargs.iter().map(|bit| bit.0).collect::<Vec<_>>();
                    let value_ptr = value.as_mut_ptr();
                    let length = value.len();
                    forget(value);
                    QkTargetQargs {
                        args: value_ptr,
                        length,
                    }
                } else {
                    QkTargetQargs {
                        args: null_mut(),
                        length: 0,
                    }
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
    let length = result.len();

    // Prevent original from being destroyed
    forget(result);
    QkTargetQargsList { list: ptr, length }
}

/// @ingroup QkTarget
/// Checks if the provided instruction and its qargs are supported by this
/// ``Target``.
///
/// @param target A pointer to the ``Target``.
/// @param operation The instruction to check for.
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
///     qk_property_map_add(props_map, NULL, 0, 0.0, 0.1);
///     qk_target_add_instruction(target, QkGate_CRX, *[3.14], props_map);
///     qk_target_add_instruction(target, QkGate_H, NULL, NULL);
///
///     qk_target_instruction_supported(target, QkGate_CRX, [0, 1], 2)
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
///
/// The ``qargs`` type is expected to be a pointer to an array of ``u32int_t`` where the length
/// matches is specified by ``num_qubits`` and has to match the expectation of the gate. If the
/// array is insufficently long the behavior of this function is undefined as this will read
/// outside the bounds of the array. It can be a null pointer if there are no qubits for
/// a given gate. You can check `qk_gate_num_qubits` to determine how many qubits are required
/// for a given gate.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_instruction_supported(
    target: *const Target,
    operation: StandardGate,
    qargs: *mut u32,
    num_qubits: u32,
) -> bool {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
    let qargs: Qargs = unsafe { parse_qargs(qargs, num_qubits) };

    target.instruction_supported(operation.name(), &qargs)
}

/// @ingroup QkTarget
/// Check if the provided instruction exists within the ``Target``.
///
/// @param target A pointer to the ``Target``.
/// @param instruction The instruction to check for.
///
/// @return Whether the instruction is present in the ``Target`` or not.
///
/// # Example
///     
///     QkTarget *target = qk_target_new(5);
///     QkPropsMap *props_map = qk_property_map_new();
///     qk_property_map_add(props_map, NULL, 0, 0.0, 0.1);
///     qk_target_add_instruction(target, QkGate_CRX, *[3.14], props_map);
///     qk_target_add_instruction(target, QkGate_H, NULL, NULL);
///
///     qk_target_contains_instr(target, "x")
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_contains_instr(
    target: *const Target,
    instruction: StandardGate,
) -> bool {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    target.contains_key(instruction.name())
}

/// @ingroup QkTarget
/// Get the length of the ``Target``.
///
/// @param target A pointer to the ``Target``.
///
/// @return The length of the target.
///
///
/// # Example
///     
///     QkTarget *target = qk_target_new(5);
///     qk_target_add_instruction(target, QkGate_H, NULL, NULL);
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

fn target_error_to_error_code(e: TargetError) -> ExitCode {
    match e {
        TargetError::InvalidKey(_) => ExitCode::TargetInvalidInstKey,
        TargetError::AlreadyExists(_) => ExitCode::TargetInstAlreadyExists,
        TargetError::QargsMismatch {
            instruction: _,
            arguments: _,
        } => ExitCode::TargetQargMismatch,
        TargetError::InvalidQargsKey {
            instruction: _,
            arguments: _,
        } => ExitCode::TargetInvalidQargsKey,
        TargetError::QargsWithoutInstruction(_) => ExitCode::TargetQargsWithoutInstruction,
    }
}

// fn qk_inst_props_to_instruction_props(props: QkInstructionProps) -> InstructionProperties {
//     let duration = if props.duration.is_nan() {None}  else {Some(props.duration)};
//     let error = if props.error.is_nan() {None}  else {Some(props.error)};
//     InstructionProperties { duration, error }
// }

fn instruction_props_to_qk_inst_props(props: &InstructionProperties) -> QkInstructionProps {
    QkInstructionProps {
        duration: props.duration.unwrap_or(f64::NAN),
        error: props.error.unwrap_or(f64::NAN),
    }
}
