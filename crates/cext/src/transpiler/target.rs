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

use crate::exit_codes::{CInputError, ExitCode};
use crate::pointers::{check_ptr, const_ptr_as_ref, mut_ptr_as_ref};
use indexmap::IndexMap;
use qiskit_circuit::operations::StandardInstruction;
use qiskit_circuit::operations::{Operation, Param, StandardGate};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::PhysicalQubit;
use qiskit_transpiler::target::{InstructionProperties, Qargs, Target};
use smallvec::{smallvec, SmallVec};

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
///     
///     QkTarget *target = qk_target_new(5);
///
#[no_mangle]
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
///     
///     QkTarget *target = qk_target_new(5);
///     uint32_t num_qubits = qk_target_num_qubits(target);
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
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
/// @return The dt value of this ``QkTarget`` or ``NaN`` if not assigned.
///
/// # Example
///
///     QkTarget *target = qk_target_new(5);
///     qk_target_set_dt(target, 10e-9);
///     double dt = qk_target_dt(target);
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
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
///
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 1
///     uint32_t granularity = qk_target_granularity(target);
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
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
///
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 1
///     size_t min_length = qk_target_min_length(target);
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
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
///
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 1
///     uint32_t pulse_alignment = qk_target_pulse_alignment(target);
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
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
///
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 0
///     uint32_t acquire_alignment = qk_target_pulse_alignment(target);
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
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
///     QkTarget *target = qk_target_new(5);
///     double dt = qk_target_set_dt(target, 10e-9);
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
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
///
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 1
///     qk_target_set_granularity(target, 2);
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
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
/// Sets the ``min_length`` value of this ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
/// @param min_length The minimum pulse gate length value in units of ``dt``.
///
/// @return ``QkExitCode`` specifying if the operation was successful.
///
/// # Example
///
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 1
///     qk_target_set_min_length(target, 3);
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
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
///
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 1
///     qk_target_set_pulse_alignment(target, 4);
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
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
/// Sets the ``acquire_alignment`` value of this ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget``.
/// @param acquire_alignment value representing a time resolution of measure instruction
///     starting time.
///
/// @return ``QkExitCode`` specifying if the operation was successful.
///
/// # Example
///     QkTarget *target = qk_target_new(5);
///     // The value defaults to 0
///     qk_target_set_acquire_alignment(target, 5);
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
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
/// Creates a copy of the ``QkTarget``.
///
/// @param target A pointer to the ``QkTarget`` to copy.
///
/// @return A pointer to the new copy of the ``QkTarget``.
///
/// # Example
///     
///     QkTarget *target = qk_target_new(5);
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_CX);
///     uint32_t qargs[2] = {0, 1};
///     qk_target_entry_add_property(entry, qargs, 2, 0.0, 0.1);
///     QkExitCode result = qk_target_add_instruction(target, entry);
///
///     QkTarget *copied = qk_target_copy(target);
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
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
///     
///     QkTarget *target = qk_target_new(5);
///     qk_target_free(target);
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
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
        Self {
            operation: StandardOperation::Gate(operation),
            params: None,
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
/// Creates an entry to the ``QkTarget`` based on a ``QkGate`` instance with
/// no parameters.
///
/// @note If the instance of ``QkGate`` uses fixed parameters, use ``qk_target_entry_new_fixed``.
/// Regular parameters are not currently supported.
///
/// @param operation The ``QkGate`` whose properties this target entry defines.
///
/// @return A pointer to the new ``QkTargetEntry``.
///
/// # Example
///
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_H);
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_target_entry_new(operation: StandardGate) -> *mut TargetEntry {
    // Fast fail if the instruction is expecting parameters.
    if operation.num_params() != 0 {
        panic!("Tried to create an non-parametric entry with a parametric gate.")
    }
    Box::into_raw(Box::new(TargetEntry::new(operation)))
}

/// @ingroup QkTargetEntry
/// Creates a new entry for adding a measurement instruction to a ``QkTarget``.
///
/// @return A pointer to the new ``QkTargetEntry`` for a measurement instruction.
///
/// # Example
///
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
#[no_mangle]
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
///
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
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_target_entry_new_reset() -> *mut TargetEntry {
    Box::into_raw(Box::new(TargetEntry::new_instruction(
        StandardInstruction::Reset,
    )))
}

/// @ingroup QkTargetEntry
/// Creates an entry to the ``QkTarget`` based on a ``QkGate`` instance with
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
///
///     double crx_params[1] = {3.14};
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_CRX, crx_params);
///
/// # Safety
///
/// The ``params`` type is expected to be a pointer to an array of ``double`` where the length
/// matches the the expectations of the ``QkGate``. If the array is insufficently long the
/// behavior of this function is undefined as this will read outside the bounds of the array.
/// It can be a null pointer if there are no params for a given gate. You can check
/// ``qk_gate_num_params`` to determine how many qubits are required for a given gate.
#[no_mangle]
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
///
///     // Create an entry for an H gate
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_H);
///     size_t props_size = qk_target_entry_num_properties(entry);
///
/// # Safety
///
/// The behavior is undefined if ``entry`` is not a valid,
/// non-null pointer to a ``QkTargetEntry`` object.
#[no_mangle]
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
///
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_H);
///     qk_target_entry_free(entry);
///
/// # Safety
///
/// The behavior is undefined if ``entry`` is not a valid,
/// non-null pointer to a ``QkTargetEntry`` object.
#[no_mangle]
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
///
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_CX);
///     uint32_t qargs[2] = {0, 1};
///     qk_target_entry_add_property(entry, qargs, 2, 0.0, 0.1);
///
/// # Safety
///
/// The behavior is undefined if ``entry`` is not a valid, non-null pointer
/// to a ``QkTargetEntry`` object.
#[no_mangle]
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
///     
///     QkTarget *target = qk_target_new(5);
///     QkTargetEntry *entry = qk_target_entry_new(QkGate_CX);
///     uint32_t qargs[2] = {0, 1};
///     qk_target_entry_add_property(entry, qargs, 2, 0.0, 0.1);
///     QkExitCode result = qk_target_add_instruction(target, entry);
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
///
/// Behavior is undefined if ``entry`` is not a valid, non-null pointer to a ``QkTargetEntry``.
#[no_mangle]
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
///     
///     QkTarget *target = qk_target_new(5);
///     double params[1] = {3.1415};
///     QkTargetEntry *entry = qk_target_entry_new_fixed(QkGate_CRX, params);
///     uint32_t qargs[2] = {0, 1};
///     qk_target_entry_add_property(entry, qargs, 2, 0.0, 0.1);
///     qk_target_add_instruction(target, entry);
///
///     qk_target_update_property(target, QkGate_CRX, qargs, 2, 0.0012, 1.1)
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
///
/// The ``qargs`` type is expected to be a pointer to an array of ``uint32_t`` where the length
/// matches is specified by ``num_qubits`` and has to match the expectation of the gate. If the
/// array is insufficently long the behavior of this function is undefined as this will read
/// outside the bounds of the array. It can be a null pointer if there are no qubits for
/// a given gate. You can check ``qk_gate_num_qubits`` to determine how many qubits are required
/// for a given gate.
#[no_mangle]
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
///     
///     QkTarget *target = qk_target_new(5);
///     QkTargetEntry *target_enty = qk_target_entry_new(QkGate_H);
///     qk_target_add_instruction(target, target_entry);
///
///     size_t num_instructions = qk_target_num_instructions(target);
///
/// # Safety
///
/// Behavior is undefined if ``QkTarget`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_target_num_instructions(target: *const Target) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    target.len()
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

/// Parse params based on a standarg gate and a pointer to a float.
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
/// matches the the expectations of the ``QkGate``. If the array is insufficently long the
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
