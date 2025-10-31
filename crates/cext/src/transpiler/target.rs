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

use std::f64;
use std::ffi::{CStr, CString, c_char};
use std::ptr::{null_mut, write};
use std::sync::Arc;

use crate::dag::COperationKind;
use crate::exit_codes::{CInputError, ExitCode};
use crate::pointers::{check_ptr, const_ptr_as_ref, mut_ptr_as_ref};
use indexmap::IndexMap;
use qiskit_circuit::PhysicalQubit;
use qiskit_circuit::operations::StandardInstruction;
use qiskit_circuit::operations::{Operation, Param, StandardGate};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
use qiskit_circuit::parameter::symbol_expr::Symbol;
use qiskit_transpiler::target::{InstructionProperties, Qargs, Target};
use smallvec::{SmallVec, smallvec};

/// @ingroup QkTarget
/// Construct a new ``QkTarget`` with the given number of qubits.
/// The number of qubits is bound to change if an instruction is added with properties
/// that apply to a collection of qargs in which any index is higher than the specified
/// number of qubits
///
/// @param num_qubits The number of qubits the ``QkTarget`` will explicitly support.
///
/// @return A pointer to the new ``QkTarget``
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
/// ```
///
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_target_new(num_qubits: u32) -> *mut Target {
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
/// Returns the number of qubits of this ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
///
/// @return The number of qubits this target can use.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     uint32_t num_qubits = qk_target_num_qubits(target);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_num_qubits(target: *const Target) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    target.num_qubits.unwrap_or_default()
}

/// @ingroup QkTarget
/// Returns the dt value of this ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
///
/// @return The dt value of this ``QkTarget`` or ``NAN`` if not assigned.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     qk_target_set_dt(target, 10e-9);
///     double dt = qk_target_dt(target);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_dt(target: *const Target) -> f64 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    target.dt.unwrap_or(f64::NAN)
}

/// @ingroup QkTarget
/// Returns the granularity value of this ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
///
/// @return The ``granularity`` value of this ``QkTarget``.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 1
///     uint32_t granularity = qk_target_granularity(target);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_granularity(target: *const Target) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    target.granularity
}

/// @ingroup QkTarget
/// Returns the ``min_length`` value of this ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
///
/// @return The ``min_length`` value of this ``QkTarget``.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 1
///     size_t min_length = qk_target_min_length(target);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_min_length(target: *const Target) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    target.min_length
}

/// @ingroup QkTarget
/// Returns the ``pulse_alignment`` value of this ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
///
/// @return The ``pulse_alignment`` value of this ``QkTarget``.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 1
///     uint32_t pulse_alignment = qk_target_pulse_alignment(target);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_pulse_alignment(target: *const Target) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    target.pulse_alignment
}

/// @ingroup QkTarget
/// Returns the ``acquire_alignment`` value of this ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
///
/// @return The ``acquire_alignment`` value of this ``QkTarget``.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 0
///     uint32_t acquire_alignment = qk_target_pulse_alignment(target);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_acquire_alignment(target: *const Target) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    target.acquire_alignment
}

/// @ingroup QkTarget
/// Sets the dt value of this ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
/// @param dt The ``dt`` value for the system time resolution of input.
///
/// @return ``QkExitCode`` specifying if the operation was successful.
///
/// # Example
///
/// ```c
///     QkTarget *target = qk_target_new(5);
///     double dt = qk_target_set_dt(target, 10e-9);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_set_dt(target: *mut Target, dt: f64) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { mut_ptr_as_ref(target) };
    target.dt = if dt.is_nan() { None } else { Some(dt) };
    ExitCode::Success
}

/// @ingroup QkTarget
/// Sets the ``granularity`` value of this ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
/// @param granularity The value for the minimum pulse gate resolution in
///     units of ``dt``.
///
/// @return ``QkExitCode`` specifying if the operation was successful.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 1
///     qk_target_set_granularity(target, 2);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
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
/// Sets the ``min_length`` value of this ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
/// @param min_length The minimum pulse gate length value in units of ``dt``.
///
/// @return ``QkExitCode`` specifying if the operation was successful.
///
/// # Example
///
/// ```c
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 1
///     qk_target_set_min_length(target, 3);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_set_min_length(
    target: *mut Target,
    min_length: u32,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { mut_ptr_as_ref(target) };
    target.min_length = min_length;
    ExitCode::Success
}

/// @ingroup QkTarget
/// Returns the ``pulse_alignment`` value of this ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
/// @param pulse_alignment value representing a time resolution of gate.
///
/// @return ``QkExitCode`` specifying if the operation was successful.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 1
///     qk_target_set_pulse_alignment(target, 4);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
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
/// Sets the ``acquire_alignment`` value of this ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
/// @param acquire_alignment value representing a time resolution of measure instruction
///     starting time.
///
/// @return ``QkExitCode`` specifying if the operation was successful.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 0
///     qk_target_set_acquire_alignment(target, 5);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
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
/// Creates a copy of the ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget`` to copy.
///
/// @return A pointer to the new copy of the ``QkTarget``.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_CX);
///     uint32_t qargs[2] = {0, 1};
///     qk_target_entry_add_property(entry, qargs, 2, 0.0, 0.1);
///     QkExitCode result = qk_target_add_instruction(target, entry);
///
///     QkTarget *copied = qk_target_copy(target);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_copy(target: *mut Target) -> *mut Target {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    Box::into_raw(target.clone().into())
}

/// @ingroup QkTarget
/// Free the ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget`` to free.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     qk_target_free(target);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
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

#[derive(Debug)]
enum StandardOperation {
    Gate(StandardGate),
    Instruction(StandardInstruction),
}

impl StandardOperation {
    pub fn num_qubits(&self) -> u32 {
        match &self {
            Self::Gate(gate) => gate.num_qubits(),
            Self::Instruction(inst) => inst.num_qubits(),
        }
    }
}

impl From<StandardOperation> for PackedOperation {
    fn from(value: StandardOperation) -> Self {
        match value {
            StandardOperation::Gate(gate) => gate.into(),
            StandardOperation::Instruction(inst) => inst.into(),
        }
    }
}

/// Represents the mapping between qargs and ``InstructionProperties``
#[derive(Debug)]
pub struct TargetEntry {
    operation: StandardOperation,
    params: Option<SmallVec<[Param; 3]>>,
    map: IndexMap<Qargs, Option<InstructionProperties>, ahash::RandomState>,
}

impl TargetEntry {
    pub fn new(operation: StandardGate) -> Self {
        let params = if operation.num_params() > 0 {
            Some(
                (0..operation.num_params())
                    .map(|i| {
                        let op_name = operation.name();
                        Param::ParameterExpression(Arc::new(ParameterExpression::from_symbol(
                            Symbol::new(format!("{op_name}_param_{i}").as_str(), None, None),
                        )))
                    })
                    .collect(),
            )
        } else {
            None
        };
        Self {
            operation: StandardOperation::Gate(operation),
            params,
            map: Default::default(),
        }
    }

    pub fn new_fixed(operation: StandardGate, params: SmallVec<[Param; 3]>) -> Self {
        Self {
            operation: StandardOperation::Gate(operation),
            params: Some(params),
            map: Default::default(),
        }
    }

    pub fn new_instruction(instruction: StandardInstruction) -> Self {
        Self {
            operation: StandardOperation::Instruction(instruction),
            params: None,
            map: Default::default(),
        }
    }
}

/// @ingroup QkTargetEntry
/// Creates an entry to the ``QkTarget`` based on a ``QkGate`` instance.
///
/// @param operation The ``QkGate`` whose properties this target entry defines. If the ``QkGate``
/// takes parameters (which can be checked with ``qk_gate_num_params``) it will be added as a
/// an instruction on the target which accepts any parameter value. If the gate only accepts a
/// fixed parameter value you can use ``qk_target_entry_new_fixed`` instead.
///
/// @return A pointer to the new ``QkTargetEntry``.
///
/// # Example
/// ```c
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_H);
/// ```
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_target_entry_new(operation: StandardGate) -> *mut TargetEntry {
    Box::into_raw(Box::new(TargetEntry::new(operation)))
}

/// @ingroup QkTargetEntry
/// Creates a new entry for adding a measurement instruction to a ``QkTarget``.
///
/// @return A pointer to the new ``QkTargetEntry`` for a measurement instruction.
///
/// # Example
/// ```c
///     QkTargetEntry *entry = qk_target_entry_new_measure();
///     // Add fixed duration and error rates from qubits at index 0 to 4.
///     for (uint32_t i = 0; i < 5; i++) {
///         // Measure is a single qubit instruction
///         uint32_t qargs[1] = {i};
///         qk_target_entry_add_property(entry, qargs, 1, 1.928e-10, 7.9829e-11);
///     }
///
///     // Add the entry to a target with 5 qubits
///     QkTarget *measure_target = qk_target_new(5);
///     qk_target_add_instruction(measure_target, entry);
/// ```
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_target_entry_new_measure() -> *mut TargetEntry {
    Box::into_raw(Box::new(TargetEntry::new_instruction(
        StandardInstruction::Measure,
    )))
}

/// @ingroup QkTargetEntry
/// Creates a new entry for adding a reset instruction to a ``QkTarget``.
///
/// @return A pointer to the new ``QkTargetEntry`` for a reset instruction.
///
/// # Example
/// ```c
///     QkTargetEntry *entry = qk_target_entry_new_reset();
///     // Add fixed duration and error rates from qubits at index 0 to 2.
///     for (uint32_t i = 0; i < 3; i++) {
///         // Reset is a single qubit instruction
///         uint32_t qargs[1] = {i};
///         qk_target_entry_add_property(entry, qargs, 1, 1.2e-11, 5.9e-13);
///     }
///
///     // Add the entry to a target with 3 qubits
///     QkTarget *reset_target = qk_target_new(3);
///     qk_target_add_instruction(reset_target, entry);
/// ```
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_target_entry_new_reset() -> *mut TargetEntry {
    Box::into_raw(Box::new(TargetEntry::new_instruction(
        StandardInstruction::Reset,
    )))
}

/// @ingroup QkTargetEntry
/// Creates an entry in the ``QkTarget`` based on a ``QkGate`` instance with
/// no parameters.
///
/// @note Adding a ``QkGate`` with regular parameters is not currently supported.
///
/// @param operation The ``QkGate`` whose properties this target entry defines.
/// @param params A pointer to the parameters that the instruction is calibrated for.
///
/// @return A pointer to the new ``QkTargetEntry``.
///
/// # Example
/// ```c
///     double crx_params[1] = {3.14};
///     QkTargetEntry *entry = qk_target_entry_new_fixed(QkGate_CRX, crx_params);
/// ```
///
/// # Safety
///
/// The ``params`` type is expected to be a pointer to an array of ``double`` where the length
/// matches the expectations of the ``QkGate``. If the array is insufficiently long the
/// behavior of this function is undefined as this will read outside the bounds of the array.
/// It can be a null pointer if there are no params for a given gate. You can check
/// ``qk_gate_num_params`` to determine how many qubits are required for a given gate.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_entry_new_fixed(
    operation: StandardGate,
    params: *mut f64,
) -> *mut TargetEntry {
    unsafe {
        Box::into_raw(Box::new(TargetEntry::new_fixed(
            operation,
            parse_params(operation, params),
        )))
    }
}

/// @ingroup QkTargetEntry
/// Retrieves the number of properties stored in the target entry.
///
/// @param entry The pointer to the mapping object.
///
/// @return The number of properties in the ``QkTargetEntry``.
///
/// # Example
/// ```c
///     // Create an entry for an H gate
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_H);
///     size_t props_size = qk_target_entry_num_properties(entry);
/// ```
///
/// # Safety
///
/// The behavior is undefined if ``entry`` is not a valid,
/// non-null pointer to a ``QkTargetEntry`` object.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_entry_num_properties(entry: *const TargetEntry) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let prop_map = unsafe { const_ptr_as_ref(entry) };
    prop_map.map.len()
}

/// @ingroup QkTargetEntry
/// Frees the entry.
///
/// @note An entry pointer will be freed when added to a ``QkTarget`` via
/// ``qk_target_add_instruction``, this function is only meant to be used
/// alternatively if an entry is never added to a ``QkTarget`` instance.
///
/// @param entry The pointer to the mapping object to be freed.
///
/// # Example
/// ```c
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_H);
///     qk_target_entry_free(entry);
/// ```
///
/// # Safety
///
/// The behavior is undefined if ``entry`` is not a valid,
/// non-null pointer to a ``QkTargetEntry`` object.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_entry_free(entry: *mut TargetEntry) {
    if !entry.is_null() {
        if !entry.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(entry);
        }
    }
}

/// @ingroup QkTargetEntry
/// Adds an instruction property instance based on its assigned qargs.
///
/// @param entry The pointer to the entry object.
/// @param qargs A pointer to the array of ``uint32_t`` qubit indices to add the
///     gate on, can be a null pointer to check for global properties.
/// @param num_qubits The length of the qargs array.
/// @param duration The instruction's duration in seconds on the specific set of
///     qubits.
/// @param error The instruction's average error rate on the specific set of qubits.
///
/// # Example
/// ```c
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_CX);
///     uint32_t qargs[2] = {0, 1};
///     qk_target_entry_add_property(entry, qargs, 2, 0.0, 0.1);
/// ```
///
/// # Safety
///
/// The behavior is undefined if ``entry`` is not a valid, non-null pointer
/// to a ``QkTargetEntry`` object.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_entry_add_property(
    entry: *mut TargetEntry,
    qargs: *mut u32,
    num_qubits: u32,
    duration: f64,
    error: f64,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let entry = unsafe { mut_ptr_as_ref(entry) };
    if num_qubits != entry.operation.num_qubits() {
        return ExitCode::TargetQargMismatch;
    }
    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
    let qubits: Qargs = unsafe { parse_qargs(qargs, num_qubits) };
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let duration = if duration.is_nan() {
        None
    } else {
        Some(duration)
    };
    let error = if error.is_nan() { None } else { Some(error) };
    entry
        .map
        .insert(qubits, Some(InstructionProperties::new(duration, error)));
    ExitCode::Success
}

/// @ingroup QkTarget
/// Adds a gate to the ``QkTarget`` through a ``QkTargetEntry``.
///
/// @param target A pointer to the ``QkTarget``.
/// @param target_entry A pointer to the ``QkTargetEntry``. The pointer
/// gets freed when added to the ``QkTarget``.
///
/// @return ``QkExitCode`` specifying if the operation was successful.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_CX);
///     uint32_t qargs[2] = {0, 1};
///     qk_target_entry_add_property(entry, qargs, 2, 0.0, 0.1);
///     QkExitCode result = qk_target_add_instruction(target, entry);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
///
/// Behavior is undefined if ``entry`` is not a valid, non-null pointer to a ``QkTargetEntry``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_add_instruction(
    target: *mut Target,
    target_entry: *mut TargetEntry,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let pointer_check = check_ptr(target_entry);
    if pointer_check.is_err() {
        match pointer_check {
            Err(CInputError::AlignmentError) => return ExitCode::AlignmentError,
            Err(CInputError::NullPointerError) => return ExitCode::NullPointerError,
            _ => (),
        }
    }
    let entry = unsafe { Box::from_raw(target_entry) };
    let instruction = entry.operation;

    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { mut_ptr_as_ref(target) };

    let property_map = if entry.map.is_empty() {
        None
    } else {
        Some(entry.map)
    };

    match target.add_instruction(
        instruction.into(),
        &entry.params.unwrap_or_default(),
        None,
        property_map,
    ) {
        Ok(_) => ExitCode::Success,
        Err(e) => e.into(),
    }
}

/// @ingroup QkTarget
/// Modifies the properties of a gate in the ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
/// @param instruction The instruction to modify.
/// @param qargs The pointer to the array of ``uint32_t`` values to use as
/// qargs. Can be ``NULL`` if global.
/// @param num_qubits The number of qubits of the instruction..
/// @param duration The instruction's duration in seconds on the specific set of
///     qubits.
/// @param error The instruction's average error rate on the specific set of qubits.
///
/// @return ``QkExitCode`` specifying if the operation was successful.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     double params[1] = {3.1415};
///     QkTargetEntry *entry = qk_target_entry_new_fixed(QkGate_CRX, params);
///     uint32_t qargs[2] = {0, 1};
///     qk_target_entry_add_property(entry, qargs, 2, 0.0, 0.1);
///     qk_target_add_instruction(target, entry);
///
///     qk_target_update_property(target, QkGate_CRX, qargs, 2, 0.0012, 1.1);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
///
/// The ``qargs`` type is expected to be a pointer to an array of ``uint32_t`` where the length
/// matches is specified by ``num_qubits`` and has to match the expectation of the gate. If the
/// array is insufficiently long the behavior of this function is undefined as this will read
/// outside the bounds of the array. It can be a null pointer if there are no qubits for
/// a given gate. You can check ``qk_gate_num_qubits`` to determine how many qubits are required
/// for a given gate.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_update_property(
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
/// Returns the number of instructions tracked by a ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
///
/// @return The length of the target.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     QkTargetEntry *target_entry = qk_target_entry_new(QkGate_H);
///     qk_target_add_instruction(target, target_entry);
///
///     size_t num_instructions = qk_target_num_instructions(target);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_num_instructions(target: *const Target) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    target.len()
}

/// @ingroup QkTarget
/// Return the index at which an operation is located based on its name.
///
/// @param target A pointer to the ``QkTarget``.
/// @param name A pointer to the name to get the index of.
///
/// @return the index in which the operation is located or an invalid index
///     in the case it is not in the Target.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     QkTargetEntry *target_entry = qk_target_entry_new(QkGate_H);
///     qk_target_add_instruction(target, target_entry);
///
///     size_t op_idx = qk_target_op_get_index(target, "h");
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
/// Behavior is undefined if ``name`` is not a pointer to a valid null-terminated string.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_op_get_index(
    target: *const Target,
    name: *const c_char,
) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target_borrowed = unsafe { const_ptr_as_ref(target) };
    // SAFETY: Per documentation, this should point to a valid, null-terminated
    // string.
    let name = unsafe {
        CStr::from_ptr(name)
            .to_str()
            .expect("Error extracting gate name.")
    };

    if let Some(index) = target_borrowed.get_gate_index(name) {
        index
    } else {
        usize::MAX
    }
}

/// @ingroup QkTarget
/// Return the name of the operation stored at that index in the ``QkTarget`` instance's
/// gate map.
///
/// @param target A pointer to the ``QkTarget``.
/// @param index The index at which the gate is stored.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     QkTargetEntry *target_entry = qk_target_entry_new(QkGate_H);
///     qk_target_add_instruction(target, target_entry);
///
///     char *op_name = qk_target_op_name(target, 0);
///     // Free after use
///     qk_str_free(op_name);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_op_name(target: *const Target, index: usize) -> *mut c_char {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target_borrowed = unsafe { const_ptr_as_ref(target) };
    if let Some((retrieved, _)) = target_borrowed.get_by_index(index) {
        CString::new(retrieved.clone())
            .expect("Error allocating space for string")
            .into_raw()
    } else {
        panic!("Index does not map to any operation in the Target.")
    }
}

/// @ingroup QkTarget
/// Checks whether an operation is present in the ``QkTarget`` instance by name.
///
/// @param target A pointer to the ``QkTarget``.
/// @param name A pointer to the name string which we will look up.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     QkTargetEntry *target_entry = qk_target_entry_new(QkGate_H);
///     qk_target_add_instruction(target, target_entry);
///
///     if (qk_target_contains(target, "h")) {
///         printf("'h' has been added to the Target.");
///     };
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
/// Behavior is undefined if ``name`` is not a pointer to a valid null-terminated string.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_contains(target: *const Target, name: *const c_char) -> bool {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target_borrowed = unsafe { const_ptr_as_ref(target) };
    // SAFETY: Per documentation, this should point to a valid, null-terminated
    // string.
    let name = unsafe {
        CStr::from_ptr(name)
            .to_str()
            .expect("Error extracting gate name.")
    };

    target_borrowed.contains_key(name)
}

/// @ingroup QkTarget
/// Return the number of properties defined for the specified operation in
/// the ``QkTarget`` instance, a.k.a. the length of the property map. Returns
/// an invalid number when not found.
///
/// @param target A pointer to the ``QkTarget``.
/// @param index The index in which the gate is stored.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///     QkTargetEntry *target_entry = qk_target_entry_new(QkGate_H);
///     qk_target_add_instruction(target, target_entry);
///
///     size_t num_props = qk_target_op_num_properties(target, 0);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_op_num_properties(target: *const Target, index: usize) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target_borrowed = unsafe { const_ptr_as_ref(target) };
    target_borrowed
        .get_by_index(index)
        .map(|(_, v)| v.len())
        .unwrap_or(usize::MAX)
}

/// @ingroup QkTarget
/// Checks if the instruction provided contains properties for the specified qargs.
///
/// @param target A pointer to the ``QkTarget``.
/// @param index The index in which the gate is stored.
/// @param qargs A pointer to the array of ``uint32_t`` qubit indices to add the
///     check for, can be a null pointer to check for global properties.
///
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_CX);
///     uint32_t qargs[2] = {0, 1};
///     qk_target_entry_add_property(entry, qargs, 2, 0.0, 0.1);
///
///     qk_target_add_instruction(target, entry);
///
///     bool has_0_1 = qk_target_op_has_qargs(target, 0, qargs, 2); // will be true
///     bool has_0_2 = qk_target_op_has_qargs(target, 0, (uint32_t *){0, 2}, 2); // will be false
///     bool has_global = qk_target_op_has_qargs(target, 0, NULL, 0); // will be false
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_op_has_qargs(
    target: *const Target,
    index: usize,
    qargs: *const u32,
) -> bool {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target_borrowed = unsafe { const_ptr_as_ref(target) };
    if let Some((_, inst_map)) = target_borrowed.get_by_index(index) {
        let op = target_borrowed.get_op_by_index(index).unwrap();

        // SAFETY: Per the documentation the qubits pointer is an array of the size
        // associated with the operation or a NULL pointer.
        let parsed = unsafe { parse_qargs(qargs, op.num_qubits()) };
        inst_map.contains_key(&parsed)
    } else {
        panic!("Index '{}' not present in the Target.", index)
    }
}

/// @ingroup QkTarget
/// Retrieve the index at which some qargs are stored. Returns SIZE_MAX
/// if not found.
///
/// @param target A pointer to the ``QkTarget``.
/// @param index The index at which the gate is stored.
/// @param qargs A pointer to the array of ``uint32_t`` qubit indices to
///     check for, can be a null pointer to check for global properties.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_CX);
///     uint32_t qargs[2] = {0, 1};
///     qk_target_entry_add_property(entry, qargs, 2, 0.0, 0.1);
///     qk_target_add_instruction(target, entry);
///
///     size_t idx_0_1 = qk_target_op_qargs_index(target, 0, qargs);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_op_qargs_index(
    target: *const Target,
    inst_idx: usize,
    qargs: *const u32,
) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target_borrowed = unsafe { const_ptr_as_ref(target) };
    if let Some((_, inst_map)) = target_borrowed.get_by_index(inst_idx) {
        let op = target_borrowed.get_op_by_index(inst_idx).unwrap();
        // SAFETY: Per the documentation the qubits pointer is an array of the size
        // associated with the operation or a NULL pointer.
        let parsed = unsafe { parse_qargs(qargs, op.num_qubits()) };
        inst_map.get_index_of(&parsed).unwrap_or(usize::MAX)
    } else {
        panic!("Index '{}' not present in the Target.", inst_idx)
    }
}

/// @ingroup QkTarget
/// Retrieve the qargs for the operation by index.
///
/// @param target A pointer to the ``QkTarget``.
/// @param inst_idx The index at which the gate is stored.  
/// @param qarg_idx The index at which the qargs are stored.  
/// @param qargs A pointer to write out the ``QkQargs`` instance.
///
/// @return An exit code.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_CX);
///     uint32_t qargs[2] = {0, 1};
///     qk_target_entry_add_property(entry, qargs, 2, 0.0, 0.1);
///     qk_target_add_instruction(target, entry);
///
///     QkQargs qargs_retrieved;
///     qk_target_op_get_qargs(target, 0, 0, &qargs);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
/// Behavior is undefined if ``qargs`` does not point to an address of the correct size to
/// store ``QkQargs`` in.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_op_get_qargs(
    target: *const Target,
    inst_idx: usize,
    qarg_idx: usize,
    qargs: *mut CQargs,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target_borrowed = unsafe { const_ptr_as_ref(target) };
    if let Some((_, props_map)) = target_borrowed.get_by_index(inst_idx) {
        if let Some((retrieved_qargs, _)) = props_map.get_index(qarg_idx) {
            // SAFETY: Per documentation, the pointer goes to an address
            // pre-allocated for a ``CQargs`` object.
            let (qargs_ptr, qargs_length) = qargs_to_ptr(retrieved_qargs);
            unsafe {
                write(
                    qargs,
                    CQargs {
                        qargs: qargs_ptr,
                        len: qargs_length,
                    },
                );
            }
            ExitCode::Success
        } else {
            ExitCode::IndexError
        }
    } else {
        ExitCode::IndexError
    }
}

/// @ingroup QkTarget
/// Retrieve the qargs for the operation stored in its respective indices.
///
/// @param target A pointer to the ``QkTarget``.
/// @param inst_idx The index in which the gate is stored.
/// @param qarg_idx The index in which the qargs are stored.
/// @param inst_props A pointer to write out the ``QkInstructionProperties`` instance.
///
/// @return An exit code.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_CX);
///     uint32_t qargs[2] = {0, 1};
///     qk_target_entry_add_property(entry, qargs, 2, 0.0, 0.1);
///     qk_target_add_instruction(target, entry);
///
///     QkInstructionProperties inst_props;
///     qk_target_op_get_props(target, 0, 0, &inst_props);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
/// Behavior is undefined if ``inst_props`` does not point to an address of the correct size to
/// store ``QkInstructionProperties`` in.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_op_get_props(
    target: *const Target,
    inst_idx: usize,
    qarg_idx: usize,
    inst_props: *mut CInstructionProperties,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target_borrowed = unsafe { const_ptr_as_ref(target) };
    if let Some((_, props_map)) = target_borrowed.get_by_index(inst_idx) {
        if let Some((_, retrieved_props)) = props_map.get_index(qarg_idx) {
            // SAFETY: Per documentation, the pointer goes to an address
            // pre-allocated for a ``CInstructionProperties`` object.
            unsafe {
                if let Some(props) = retrieved_props {
                    write(
                        inst_props,
                        CInstructionProperties {
                            duration: props.duration.unwrap_or(f64::NAN),
                            error: props.error.unwrap_or(f64::NAN),
                        },
                    );
                } else {
                    // If no property is found but there are qargs, clear the instruction to represent a null instance.
                    qk_target_inst_props_clear(inst_props);
                }
            }
            ExitCode::Success
        } else {
            ExitCode::IndexError
        }
    } else {
        ExitCode::IndexError
    }
}

/// A representation of a Target operation's instruction properties.
#[repr(C)]
pub struct CInstructionProperties {
    duration: f64,
    error: f64,
}

/// @ingroup QkTarget
/// Clear the data in the ``QkInstructionProperties`` object.
///
/// This function does not free the allocation for the provided ``QkInstructionProperties``
/// pointer, but it resets the internal values to ``NAN``.
///
/// @param inst_props A pointer to the ``QkInstructionProperties`` object to clear.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_CX);
///     uint32_t qargs[2] = {0, 1};
///     qk_target_entry_add_property(entry, qargs, 2, 0.0, 0.1);
///     qk_target_add_instruction(target, entry);
///
///     QkInstructionProperties inst_props;
///     qk_target_op_get_props(target, 0, 0, &inst_props);
///     
///     // free after usage
///     qk_target_inst_props_clear(&inst_props);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``inst_props`` is a valid, non-null pointer
/// to ``QkInstructionProperties``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_inst_props_clear(inst_props: *mut CInstructionProperties) {
    // SAFETY: As per documentation, the pointer is a valid pointer to ``QkInstructionProperties``
    unsafe {
        let inst_props = mut_ptr_as_ref(inst_props);
        inst_props.duration = f64::NAN;
        inst_props.error = f64::NAN;
    }
}

/// A representation of a Target operations's quantum arguments.
#[repr(C)]
pub struct CQargs {
    qargs: *mut u32,
    len: u32,
}

/// @ingroup QkTarget
/// Clear the data in the ``QkQargs`` object.
///
/// This function does not free the allocation for the provided ``QkQargs``
/// pointer, but it resets the internal values and frees the internal qarg
/// pointer.
///
/// @param qargs A pointer to the ``QkQargs`` object to clear.
///
/// # Example
/// ```c
///     QkTarget *target = qk_target_new(5);
///
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_CX);
///     uint32_t qargs[2] = {0, 1};
///     qk_target_entry_add_property(entry, qargs, 2, 0.0, 0.1);
///     qk_target_add_instruction(target, entry);
///
///     QkQargs qargs;
///     qk_target_op_get_qargs(target, 0, 0, &qargs);
///     
///     // free after usage
///     qk_target_qargs_clear(&qargs);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``qargs`` is a valid, non-null pointer
/// to ``QkQargs``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_qargs_clear(qargs: *mut CQargs) {
    // SAFETY: The data loaded from the pointer contained in a ``CQargs``
    // object should only ever be generated by rust code and constructed
    // internally from Vecs.
    unsafe {
        let qargs = mut_ptr_as_ref(qargs);
        if qargs.len > 0 && !qargs.qargs.is_null() {
            let qubits = std::slice::from_raw_parts_mut(qargs.qargs, qargs.len as usize);
            let _: Box<[u32]> = Box::from_raw(qubits as *mut [u32]);
            qargs.qargs = std::ptr::null_mut();
        }
        qargs.len = 0;
    }
}

#[repr(C)]
pub struct CTargetOperation {
    op_type: COperationKind,
    name: *const c_char,
    num_qubits: u32,
    params: *const f64,
    num_params: u32,
}

#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_op_get(
    target: *const Target,
    index: usize,
    op_kind: *mut CTargetOperation,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target_borrowed = unsafe { const_ptr_as_ref(target) };

    if let Some(operation) = target_borrowed.get_op_by_index(index) {
        let kind = match operation {
            qiskit_transpiler::target::TargetOperation::Normal(normal_operation) => {
                match normal_operation.operation.view() {
                    qiskit_circuit::operations::OperationRef::StandardGate(_) => {
                        COperationKind::Gate
                    }
                    qiskit_circuit::operations::OperationRef::StandardInstruction(
                        standard_instruction,
                    ) => match standard_instruction {
                        StandardInstruction::Barrier(_) => COperationKind::Barrier,
                        StandardInstruction::Delay(_) => COperationKind::Delay,
                        StandardInstruction::Measure => COperationKind::Measure,
                        StandardInstruction::Reset => COperationKind::Reset,
                    },
                    qiskit_circuit::operations::OperationRef::Unitary(_) => COperationKind::Unitary,
                    _ => panic!(
                        "Unsupported operation type found: {}",
                        stringify!(normal_operation.view())
                    ),
                }
            }
            qiskit_transpiler::target::TargetOperation::Variadic(_) => panic!(
                "Unsupported operation type found: {}",
                stringify!(normal_operation.view())
            ),
        };
        let name = CString::new(
            target_borrowed
                .get_by_index(index)
                .expect("Inconsistent indices")
                .0
                .clone(),
        )
        .expect("Error extracting cstring from Target name")
        .into_raw();
        let params: Vec<f64> = operation
            .params()
            .iter()
            .filter_map(|param| match param {
                Param::Float(number) => Some(*number),
                _ => None,
            })
            .collect();
        let num_params = params
            .len()
            .try_into()
            .expect("The number of parameters exceeds the alotted amount");
        let params_boxed = params.into_boxed_slice();
        unsafe {
            write(
                op_kind,
                CTargetOperation {
                    op_type: kind,
                    name,
                    num_qubits: operation.num_qubits(),
                    params: params_boxed.as_ptr(),
                    num_params,
                },
            )
        };
        let _ = Box::into_raw(params_boxed);
        ExitCode::Success
    } else {
        ExitCode::IndexError
    }
}

#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_op_try_gate(
    target: *const Target,
    index: usize,
) -> StandardGate {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target_borrowed = unsafe { const_ptr_as_ref(target) };

    match target_borrowed
        .get_op_by_index(index)
        .expect("Invalid index")
    {
        qiskit_transpiler::target::TargetOperation::Normal(normal_operation) => normal_operation
            .operation
            .try_standard_gate()
            .expect("Not a standard gate."),
        qiskit_transpiler::target::TargetOperation::Variadic(_) => panic!("Not a standard gate."),
    }
}

/// Parses qargs based on a pointer and its size.
///
/// # Arguments
///
/// * `qargs` - The pointer to the array of ``uint32_t`` values to use as
///   qargs. Can be ``NULL`` if global.
/// * `num_qubits` - The length of the array.
///
/// # Returns
///
/// A [Qargs] object parsed from the pointer.
///
/// # Safety
///
/// Behavior is undefined if the qubits pointer is non-aligned or if the specified
/// num_qubits exceeds the number of bits stored in the array.
unsafe fn parse_qargs(qargs: *const u32, num_qubits: u32) -> Qargs {
    if qargs.is_null() {
        Qargs::Global
    } else {
        // SAFETY: Per the documentation qargs points to an array of num_qubits elements
        unsafe {
            (0..num_qubits)
                .map(|idx| PhysicalQubit(*qargs.wrapping_add(idx as usize)))
                .collect()
        }
    }
}

/// Parse params based on a standard gate and a pointer to a float.
///
/// # Arguments
///
/// * `gate` - The [StandardGate] for which we will extract parameters.
/// * `params` -  A pointer to the parameters that the instruction is calibrated for.
///
/// # Returns
///
/// A collection of [Param] wrapped in a [SmallVec].
///
/// # Safety
///
/// The ``params`` type is expected to be a pointer to an array of ``double`` where the length
/// matches the expectations of the ``QkGate``. If the array is insufficiently long the
/// behavior of this function is undefined as this will read outside the bounds of the array.
/// It can be a null pointer if there are no params for a given gate. You can check
/// ``qk_gate_num_params`` to determine how many qubits are required for a given gate.
unsafe fn parse_params(gate: StandardGate, params: *mut f64) -> SmallVec<[Param; 3]> {
    // SAFETY: Per the documentation the params pointers are arrays of num_params() elements.
    unsafe {
        match gate.num_params() {
            0 => smallvec![],
            1 => smallvec![(*params.wrapping_add(0)).into()],
            2 => smallvec![
                (*params.wrapping_add(0)).into(),
                (*params.wrapping_add(1)).into(),
            ],
            3 => smallvec![
                (*params.wrapping_add(0)).into(),
                (*params.wrapping_add(1)).into(),
                (*params.wrapping_add(2)).into(),
            ],
            4 => smallvec![
                (*params.wrapping_add(0)).into(),
                (*params.wrapping_add(1)).into(),
                (*params.wrapping_add(2)).into(),
                (*params.wrapping_add(3)).into(),
            ],
            // There are no standard gates that take > 4 params
            _ => unreachable!(),
        }
    }
}

/// Converts a collection of Qargs into a mutable array pointer and its length.
///
/// # Arguments
///
/// * `qargs` - A reference to a [Qargs] object to convert
///
/// # Returns
///
/// A tuple with a mutable pointer to an array of `u32` members and the length
/// as a `u32`.
fn qargs_to_ptr(qargs: &Qargs) -> (*mut u32, u32) {
    match qargs {
        Qargs::Global => (null_mut(), 0),
        Qargs::Concrete(small_vec) => {
            let length = small_vec.len();
            let qargs: Vec<u32> = small_vec.iter().map(|bit| bit.0).collect();
            let mut ret = qargs.into_boxed_slice();
            let out_qargs = ret.as_mut_ptr();
            let _ = Box::into_raw(ret);
            (
                out_qargs,
                length
                    .try_into()
                    .expect("The length of these qargs exceeds the limit"),
            )
        }
    }
}
