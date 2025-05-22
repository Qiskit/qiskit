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
use qiskit_transpiler::target::{InstructionProperties, Qargs, Target};

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
/// @return The dt value of this ``Target`` or `NaN` if not assigned.
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
    target.dt.unwrap_or(f64::NAN)
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
/// Creates a copy of the ``Target``.
///
/// @param target A pointer to the ``Target`` to copy.
///
/// @return A pointer to the new copy of the ``Target``.
///
/// # Example
///     
///     QkTarget *target = qk_target_new(5);
///     QkPropsMap *props_map = qk_property_map_new();
///     uint32_t qargs[2] = {0, 1};
///     qk_property_map_add(props_map, qargs, 2, 0.0, 0.1);
///     qk_target_add_instruction(target, QkGate_CX, props_map);
///
///     QkTarget *copied = qk_target_copy(target);
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_copy(target: *mut Target) -> *mut Target {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    Box::into_raw(target.clone().into())
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
/// Adds a non-parameteric StandardGate to the ``Target``.
///
/// This function will fail if a parametric gate is added.
///
/// @note Adding parametric gates with non-fixed parameters is currently
/// not supported. See ``qk_target_add_instruction_fixed_params()`` for
/// adding gates with fixed parameters.
///
/// @param target A pointer to the ``Target``.
/// @param instruction The StandardGate to be added to the ``Target``.
/// @param property_map The mapping of qargs and InstructionProperties to
/// be associated with this instruction.
///
/// # Example
///     
///     QkTarget *target = qk_target_new(5);
///     QkPropsMap *props_map = qk_property_map_new();
///     uint32_t qargs[2] = {0, 1};
///     qk_property_map_add(props_map, qargs, 2, 0.0, 0.1);
///     qk_target_add_instruction(target, QkGate_CX, props_map);
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
///
/// Behavior is undefined if ``property_map`` is not a valid, non-null pointer to a ``QkPropsMap``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_add_instruction(
    target: *mut Target,
    instruction: StandardGate,
    property_map: *const PropertyMap,
) -> ExitCode {
    // Fast-fail if the gate is parametric.
    if instruction.num_params() != 0 {
        return ExitCode::TargetNonFixedParametricGate;
    }

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
    let props_map = property_map.map(|props| props.0.clone());

    match target.add_instruction(instruction.into(), &[], None, props_map) {
        Ok(_) => ExitCode::Success,
        Err(e) => e.into(),
    }
}

/// @ingroup QkTarget
/// Adds a StandardGate to the ``Target`` with fixed parameters.
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
///     qk_target_add_instruction_fixed_params(target, QkGate_CRX, params, props_map);
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
pub unsafe extern "C" fn qk_target_add_instruction_fixed_params(
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
        Err(e) => e.into(),
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
///     qk_target_add_instruction_fixed_params(target, QkGate_CRX, params, props_map);
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
        Err(e) => e.into(),
    }
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
///     qk_target_add_instruction(target, QkGate_H, NULL);
///
///     qk_target_length(target)
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
