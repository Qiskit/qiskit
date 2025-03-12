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

use std::ffi::{c_char, CString};

use crate::exit_codes::ExitCode;
use crate::pointers::{const_ptr_as_ref, mut_ptr_as_ref};

use qiskit_circuit::bit::{ShareableClbit, ShareableQubit};
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{Operation, Param, StandardGate, StandardInstruction};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{Clbit, Qubit};

/// @ingroup QkCircuit
/// Construct a new circuit with the given number of qubits and clbits.
///
/// @param num_qubits The number of qubits the circuit contains.
/// @param num_clbits The number of clbits the circuit contains.
///
/// @return A pointer to the created circuit.
///
/// # Example
///
///     QkCircuit *empty = qk_circuit_new(100, 100);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_circuit_new(num_qubits: u32, num_clbits: u32) -> *mut CircuitData {
    let qubits = if num_qubits > 0 {
        Some(
            (0..num_qubits)
                .map(|_| ShareableQubit::new_anonymous())
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    let clbits = if num_clbits > 0 {
        Some(
            (0..num_clbits)
                .map(|_| ShareableClbit::new_anonymous())
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };

    let circuit = CircuitData::new(qubits, clbits, None, 0, (0.).into()).unwrap();
    Box::into_raw(Box::new(circuit))
}

/// @ingroup QkCircuit
/// Get the number of qubits the circuit contains.
///
/// @param circuit A pointer to the circuit.
///
/// @return The number of qubits the circuit is defined on.
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100);
///     uint32_t num_qubits = qk_circuit_num_qubits(qc);  // num_qubits==100
///
/// # Safety
///
/// Behavior is undefined ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_num_qubits(circuit: *const CircuitData) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };

    circuit.num_qubits() as u32
}

/// @ingroup QkCircuit
/// Get the number of clbits the circuit contains.
///
/// @param circuit A pointer to the circuit.
///
/// @return The number of qubits the circuit is defined on.
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100, 50);
///     uint32_t num_clbits = qk_circuit_num_clbits(qc);  // num_clbits==50
///
/// # Safety
///
/// Behavior is undefined ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_num_clbits(circuit: *const CircuitData) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };

    circuit.num_clbits() as u32
}

/// @ingroup QkCircuit
/// Free the circuit.
///
/// @param circuit A pointer to the circuit to free.
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100, 100);
///     qk_circuit_free(qc);
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not either null or a valid pointer to a
/// [CircuitData].
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_free(circuit: *mut CircuitData) {
    if !circuit.is_null() {
        if !circuit.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(circuit);
        }
    }
}

/// @ingroup QkCircuit
/// Append a standard gate to the circuit.
///
/// @param circuit A pointer to the circuit to add the gate to.
/// @param gate The StandardGate to add to the circuit.
/// @param qubits The pointer to the array of ``uint32_t`` qubit indices to add the gate on.
/// @param params The pointer to the array of ``double`` values to use for the gate parameters.
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100);
///     qk_circuit_append_standard_gate(qc, HGate, *[0], *[]);
///
/// # Safety
///
/// The ``qubits`` and ``params`` types are expected to be a non-null pointer to an array of
/// ``uint32_t`` and ``double`` respectively where the length is matching the expectations for
/// the standard gate. If the array is insufficently long the behavior of this function is
/// undefined as this will read outside the bounds of the array.
///
/// Behavior is undefined ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_append_standard_gate(
    circuit: *mut CircuitData,
    gate: StandardGate,
    qubits: *const u32,
    params: *const f64,
) -> ExitCode {
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    unsafe {
        let qargs: &[Qubit] = match gate.num_qubits() {
            0 => &[],
            1 => &[Qubit(*qubits.wrapping_add(0))],
            2 => &[
                Qubit(*qubits.wrapping_add(0)),
                Qubit(*qubits.wrapping_add(1)),
            ],
            3 => &[
                Qubit(*qubits.wrapping_add(0)),
                Qubit(*qubits.wrapping_add(1)),
                Qubit(*qubits.wrapping_add(2)),
            ],
            // There are no standard gates > 3 qubits
            _ => unreachable!(),
        };
        let params: &[Param] = match gate.num_params() {
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
            // There are no standard gates that take > 3 params
            _ => unreachable!(),
        };
        circuit.push_standard_gate(gate, params, qargs);
    }
    ExitCode::Success
}

/// @ingroup QkCircuit
/// Append a measurement to the circuit
///
/// @param circuit A pointer to the circuit to add the gate to
/// @param qubits The ``uint32_t`` for the qubit to measure
/// @param clbits The ``uint32_t`` for the clbit to store the measurement outcome in
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100);
///     qk_circuit_append_measure(qc, 0, 0);
///
/// # Safety
///
/// Behavior is undefined ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_append_measure(
    circuit: *mut CircuitData,
    qubit: u32,
    clbit: u32,
) -> ExitCode {
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    circuit.push_packed_operation(
        PackedOperation::from_standard_instruction(StandardInstruction::Measure),
        &[],
        &[Qubit(qubit)],
        &[Clbit(clbit)],
    );
    ExitCode::Success
}

/// @ingroup QkCircuit
/// Append a reset to the circuit
///
/// @param circuit A pointer to the circuit to add the reset to
/// @param qubits The ``uint32_t`` for the qubit to reset
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100);
///     qk_circuit_append_reset(qc, 0);
///
///
/// # Safety
///
/// Behavior is undefined ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_append_reset(
    circuit: *mut CircuitData,
    qubit: u32,
) -> ExitCode {
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    circuit.push_packed_operation(
        PackedOperation::from_standard_instruction(StandardInstruction::Reset),
        &[],
        &[Qubit(qubit)],
        &[],
    );
    ExitCode::Success
}

/// @ingroup QkCircuit
/// Append a barrier to the circuit
///
/// @param circuit A pointer to the circuit to add the barrier to
/// @param num_qubits The number of qubits wide the barrier is
/// @param qubits The pointer to the array of ``uint32_t`` qubit indices to add the barrier on.
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100);
///     qk_circuit_append_reset(qc, 0);
///
///
/// # Safety
/// The length of the array qubits points to must be num_qubits. If there is
/// a mismatch the behavior is undefined.
///
/// Behavior is undefined ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_append_barrier(
    circuit: *mut CircuitData,
    num_qubits: u32,
    qubits: *const u32,
) -> ExitCode {
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    let qubits: Vec<Qubit> = unsafe {
        (0..num_qubits)
            .map(|idx| Qubit(*qubits.wrapping_add(idx as usize)))
            .collect()
    };
    circuit.push_packed_operation(
        PackedOperation::from_standard_instruction(StandardInstruction::Barrier(num_qubits)),
        &[],
        &qubits,
        &[],
    );
    ExitCode::Success
}

#[repr(C)]
pub struct OpCount {
    name: *mut c_char,
    count: usize,
}

#[repr(C)]
pub struct OpCounts {
    data: *mut OpCount,
    len: usize,
}

/// @ingroup QkCircuit
/// Return a list of string names for instructions in a circuit and their counts.
///
/// @param circuit A pointer to the circuit to get the counts for.
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100);
///     qk_circuit_append_standard_gate(qc, HGate, *[0], *[]);
///     qk_circuit_count_ops(qc);
///
/// # Safety
///
/// Behavior is undefined ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_count_ops(circuit: *const CircuitData) -> OpCounts {
    let circuit = unsafe { const_ptr_as_ref(circuit) };
    let count_ops = circuit.count_ops();
    let mut output: Vec<OpCount> = count_ops
        .into_iter()
        .map(|(name, count)| OpCount {
            name: CString::new(name).unwrap().into_raw(),
            count,
        })
        .collect();
    let data = output.as_mut_ptr();
    let len = output.len();
    std::mem::forget(output);
    OpCounts { data, len }
}

/// @ingroup QkCircuit
/// Free a circuit op count list.
///
/// @param op_counts The returned op count list from ``qk_circuit_count_ops``.
///
/// # Safety
///
/// Behavior is undefined if ``op_counts`` is not the object returned by ``qk_circuit_count_ops``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_free_count_ops(op_counts: OpCounts) {
    let data = unsafe { std::slice::from_raw_parts_mut(op_counts.data, op_counts.len) };
    let data = data.as_mut_ptr();
    unsafe {
        let _ = Box::from_raw(data);
    }
}
