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

use std::ffi::{c_char, CStr, CString};

use crate::exit_codes::ExitCode;
use crate::pointers::{const_ptr_as_ref, mut_ptr_as_ref};

use qiskit_circuit::bit::{ShareableClbit, ShareableQubit};
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{DelayUnit, Operation, Param, StandardGate, StandardInstruction};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{Clbit, Qubit};

#[cfg(feature = "python_binding")]
use pyo3::ffi::PyObject;
#[cfg(feature = "python_binding")]
use pyo3::types::PyAnyMethods;
#[cfg(feature = "python_binding")]
use pyo3::{intern, Python};
#[cfg(feature = "python_binding")]
use qiskit_circuit::imports::QUANTUM_CIRCUIT;

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
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
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
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
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
/// ``QkCircuit``.
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
/// @param qubits The pointer to the array of ``uint32_t`` qubit indices to add the gate on. This
///     can be a null pointer if there are no qubits for `gate` (e.g. `QkGate_GlobalPhase`)
/// @param params The pointer to the array of ``double`` values to use for the gate parameters.
///     This can be a null pointer if there are no parameters for `gate` (e.g. `QkGate_H`).
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100);
///     qk_circuit_gate(qc, QkGate_H, *[0], *[]);
///
/// # Safety
///
/// The ``qubits`` and ``params`` types are expected to be a pointer to an array of ``uint32_t``
/// and ``double`` respectively where the length is matching the expectations for the standard
/// gate. If the array is insufficently long the behavior of this function is undefined as this
/// will read outside the bounds of the array. It can be a null pointer if there are no qubits
/// or params for a given gate. You can check `qk_gate_num_qubits` and `qk_gate_num_params` to
/// determine how many qubits and params are required for a given gate.
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_gate(
    circuit: *mut CircuitData,
    gate: StandardGate,
    qubits: *const u32,
    params: *const f64,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    // SAFETY: Per the documentation the qubits and params pointers are arrays of num_qubits()
    // and num_params() elements respectively.
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
            4 => &[
                Qubit(*qubits.wrapping_add(0)),
                Qubit(*qubits.wrapping_add(1)),
                Qubit(*qubits.wrapping_add(2)),
                Qubit(*qubits.wrapping_add(3)),
            ],
            // There are no standard gates > 4 qubits
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
            4 => &[
                (*params.wrapping_add(0)).into(),
                (*params.wrapping_add(1)).into(),
                (*params.wrapping_add(2)).into(),
                (*params.wrapping_add(3)).into(),
            ],
            // There are no standard gates that take > 4 params
            _ => unreachable!(),
        };
        circuit.push_standard_gate(gate, params, qargs);
    }
    ExitCode::Success
}

/// @ingroup QkCircuit
/// Get the number of qubits for a `QkGate`
///
/// @param gate The standard gate to get the number of qubits for
///
/// # Example
///
///     uint32_t num_qubits = qk_gate_num_qubits(QkGate_CCX);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_gate_num_qubits(gate: StandardGate) -> u32 {
    gate.num_qubits()
}

/// @ingroup QkCircuit
/// Get the number of params for a `QkGate`
///
/// @param gate The standard gate to get the number of qubits for
///
/// # Example
///
///     uint32_t num_params = qk_gate_num_params(QkGate_R);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_gate_num_params(gate: StandardGate) -> u32 {
    gate.num_params()
}

/// @ingroup QkCircuit
/// Append a measurement to the circuit
///
/// @param circuit A pointer to the circuit to add the measurement to
/// @param qubit The ``uint32_t`` for the qubit to measure
/// @param clbit The ``uint32_t`` for the clbit to store the measurement outcome in
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100, 1);
///     qk_circuit_measure(qc, 0, 0);
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_measure(
    circuit: *mut CircuitData,
    qubit: u32,
    clbit: u32,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
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
/// @param qubit The ``uint32_t`` for the qubit to reset
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100);
///     qk_circuit_reset(qc, 0);
///
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_reset(circuit: *mut CircuitData, qubit: u32) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
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
///     uint32_t qubits[5] = {0, 1, 2, 3, 4};
///     qk_circuit_barrier(qc, 5, qubits);
///
/// # Safety
///
/// The length of the array qubits points to must be num_qubits. If there is
/// a mismatch the behavior is undefined.
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_barrier(
    circuit: *mut CircuitData,
    num_qubits: u32,
    qubits: *const u32,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    // SAFETY: Per the documentation the qubits pointer is an array of num_qubits elements
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

/// An individual operation count represented by the operation name
/// and the number of instances in the circuit.
#[repr(C)]
pub struct OpCount {
    /// A nul terminated string representing the operation name
    name: *const c_char,
    /// The number of instances of this operation in the circuit
    count: usize,
}

/// An array of ``OpCount`` objects representing the total counts of all
/// the operation types in a circuit.
#[repr(C)]
pub struct OpCounts {
    /// A array of size ``len`` containing ``OpCount`` objects for each
    /// type of operation in the circuit
    data: *mut OpCount,
    /// The number of elements in ``data``
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
///     qk_circuit_gate(qc, HGate, *[0], *[]);
///     qk_circuit_count_ops(qc);
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_count_ops(circuit: *const CircuitData) -> OpCounts {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
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
/// Return the number of instructions in the circuit
///
/// @param circuit A pointer to the circuit to get the total number of instructions for.
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100);
///     qk_circuit_gate(qc, QkGate_H, *[0], *[]);
///     qk_circuit_num_instructions(qc); // 1
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_num_instructions(circuit: *const CircuitData) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };
    circuit.__len__()
}

/// A circuit instruction representation.
///
/// This struct represents the data contained in an individual instruction in a ``QkCircuit``.
/// It is not a pointer to the underlying object, but contains a copy of the properties of the
/// instruction for inspection.
#[repr(C)]
pub struct CInstruction {
    /// The instruction name
    name: *const c_char,
    /// The number of qubits for this instruction.
    num_qubits: u32,
    /// A pointer to an array of qubit indices this instruction operates on.
    qubits: *mut u32,
    /// The number of clbits for this instruction.
    num_clbits: u32,
    /// A pointer to an array of clbit indices this instruction operates on.
    clbits: *mut u32,
    /// The number of parameters for this instruction.
    num_params: u32,
    /// A pointer to an array of parameter values for this instruction.
    params: *mut f64,
}

/// @ingroup QkCircuit
/// Return the instruction details for an instruction in the circuit
///
/// This function is used to get the instruction details for a given instruction in
/// the circuit.
///
/// @param circuit A pointer to the circuit to get the instruction details for.
/// @param index The instruction index to get the instruction details of.
///
/// @return The instruction details for the specified instructions
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(100);
///     qk_circuit_gate(qc, QkGate_H, *[0], *[]);
///     QkCircuitInstruction inst = qk_circuit_get_instruction(qc, 0);
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``. The
/// value for ``index`` must be less than the value returned by `qk_circuit_num_instructions`
/// otherwise this function will panic
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_get_instruction(
    circuit: *const CircuitData,
    index: usize,
) -> CInstruction {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };
    if index >= circuit.__len__() {
        panic!("Invalid index")
    }
    let packed_inst = &circuit.data()[index];
    let qargs = circuit.get_qargs(packed_inst.qubits);
    let mut qargs_vec: Vec<u32> = qargs.iter().map(|x| x.0).collect();
    let cargs = circuit.get_cargs(packed_inst.clbits);
    let mut cargs_vec: Vec<u32> = cargs.iter().map(|x| x.0).collect();
    let params = packed_inst.params_view();
    let mut params_vec: Vec<f64> = params
        .iter()
        .map(|x| match x {
            Param::Float(val) => *val,
            _ => unreachable!("Invalid parameter on instruction"),
        })
        .collect();
    let out_qargs = qargs_vec.as_mut_ptr();
    std::mem::forget(qargs_vec);
    let out_cargs = cargs_vec.as_mut_ptr();
    std::mem::forget(cargs_vec);
    let out_params = params_vec.as_mut_ptr();
    std::mem::forget(params_vec);

    CInstruction {
        name: CString::new(packed_inst.op.name()).unwrap().into_raw(),
        num_qubits: qargs.len() as u32,
        qubits: out_qargs,
        num_clbits: cargs.len() as u32,
        clbits: out_cargs,
        num_params: params.len() as u32,
        params: out_params,
    }
}

/// @ingroup QkCircuit
/// Free a circuit instruction object
///
/// @param inst The instruction to free
///
/// # Safety
/// Behavior is undefined if ``inst`` is not an object returned by ``qk_circuit_get_instruction``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_instruction_free(inst: CInstruction) {
    // SAFETY: Loading the data from pointers contained in a CInstruction. These should only be
    // created by rust code and are constructed from Vecs internally or CStrings.
    unsafe {
        if inst.num_qubits > 0 {
            let qubits = std::slice::from_raw_parts_mut(inst.qubits, inst.num_qubits as usize);
            let _ = Box::from_raw(qubits.as_mut_ptr());
        }
        if inst.num_clbits > 0 {
            let clbits = std::slice::from_raw_parts_mut(inst.clbits, inst.num_clbits as usize);
            let _ = Box::from_raw(clbits.as_mut_ptr());
        }
        if inst.num_params > 0 {
            let params = std::slice::from_raw_parts_mut(inst.params, inst.num_params as usize);
            let _ = Box::from_raw(params.as_mut_ptr());
        }
        let _: Box<CStr> = Box::from(CStr::from_ptr(inst.name));
    }
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
pub unsafe extern "C" fn qk_opcounts_free(op_counts: OpCounts) {
    // SAFETY: Loading data contained in OpCounts as a slice which was constructed from a Vec
    let data = unsafe { std::slice::from_raw_parts_mut(op_counts.data, op_counts.len) };
    let data = data.as_mut_ptr();
    // SAFETY: Loading a box from the slice pointer created above
    unsafe {
        let _ = Box::from_raw(data);
    }
}

/// @ingroup QkCircuit
/// Convert to a Python-space ``QuantumCircuit``.
///
/// This function takes ownership of the pointer and gives it to Python. Using
/// the input ``circuit`` pointer after it's passed to this function is
/// undefined behavior. In particular, ``qk_circuit_free`` should not be called
/// on this pointer anymore.
///
/// @param circuit The C-space ``QkCircuit`` pointer.
///
/// @return A Python ``QuantumCircuit`` object.
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to
/// a ``QkCircuit``
///
/// It is assumed that the thread currently executing this function holds the
/// Python GIL. This is required to create the Python object returned by this
/// function.
#[no_mangle]
#[cfg(feature = "python_binding")]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_to_python(circuit: *mut CircuitData) -> *mut PyObject {
    unsafe {
        let circuit = Box::from_raw(mut_ptr_as_ref(circuit));
        let py = Python::assume_gil_acquired();
        QUANTUM_CIRCUIT
            .get_bound(py)
            .call_method1(intern!(py, "_from_circuit_data"), (*circuit,))
            .expect("Unabled to create a Python circuit")
            .into_ptr()
    }
}

/// @ingroup QkCircuit
///
/// Units for circuit delays.
#[repr(u8)]
pub enum QkDelayUnit {
    /// Seconds.
    S = 0,
    /// Milliseconds.
    MS = 1,
    /// Microseconds.
    US = 2,
    /// Nanoseconds.
    NS = 3,
    /// Picoseconds.
    PS = 4,
    /// Device-native time unit ``dt``.
    DT = 5,
}

/// @ingroup QkCircuit
/// Append a delay instruction to the circuit.
///
/// @param circuit A pointer to the circuit to add the delay to.
/// @param qubit The ``uint32_t`` index of the qubit to apply the delay to.
/// @param duration The duration of the delay.
/// @param unit An enum representing the unit of the duration.
///
/// @return An exit code.
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(1, 0);
///     qk_circuit_delay(qc, 0, 100.0, QkDelayUnit_NS);
///     
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_delay(
    circuit: *mut CircuitData,
    qubit: u32,
    duration: f64,
    unit: QkDelayUnit,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };

    let delay_unit_variant = match unit {
        QkDelayUnit::S => DelayUnit::S,
        QkDelayUnit::MS => DelayUnit::MS,
        QkDelayUnit::US => DelayUnit::US,
        QkDelayUnit::NS => DelayUnit::NS,
        QkDelayUnit::PS => DelayUnit::PS,
        QkDelayUnit::DT => DelayUnit::DT,
    };

    let duration_param: Param = duration.into();
    let delay_instruction = StandardInstruction::Delay(delay_unit_variant);

    circuit.push_packed_operation(
        PackedOperation::from_standard_instruction(delay_instruction),
        &[duration_param],
        &[Qubit(qubit)],
        &[],
    );

    ExitCode::Success
}
