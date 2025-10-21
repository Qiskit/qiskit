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
use smallvec::smallvec;

use crate::exit_codes::ExitCode;
use qiskit_circuit::Qubit;
use qiskit_circuit::bit::{ClassicalRegister, QuantumRegister};
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::operations::{Operation, StandardGate};

use rustworkx_core::petgraph::stable_graph::NodeIndex;

/// @ingroup QkDag
/// Construct a new empty DAG.
///
/// You must free the returned DAG with qk_dag_free when done with it.
///
/// @return A pointer to the created DAG.
///
/// # Example
/// ```c
/// QkDag *empty = qk_dag_new();
/// ```
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_dag_new() -> *mut DAGCircuit {
    let dag = DAGCircuit::new();
    Box::into_raw(Box::new(dag))
}

/// @ingroup QkDag
/// Add a quantum register to the DAG.
///
/// @param dag A pointer to the DAG.
/// @param reg A pointer to the quantum register.
///
/// # Example
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(1024, "my_register");
/// qk_dag_add_quantum_register(dag, qr);
/// qk_quantum_register_free(qr);
/// qk_dag_free(dag);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag`` and
/// if ``reg`` is not a valid, non-null pointer to a ``QkQuantumRegister``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_add_quantum_register(
    dag: *mut DAGCircuit,
    reg: *const QuantumRegister,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let dag = unsafe { mut_ptr_as_ref(dag) };
    let qreg = unsafe { const_ptr_as_ref(reg) };

    dag.add_qreg(qreg.clone())
        .expect("Invalid register unable to be added to DAG");
}

/// @ingroup QkDag
/// Add a classical register to the DAG.
///
/// @param dag A pointer to the DAG.
/// @param reg A pointer to the classical register.
///
/// # Example
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkClassicalRegister *cr = qk_classical_register_new(24, "my_register");
/// qk_dag_add_classical_register(dag, cr);
/// qk_classical_register_free(cr);
/// qk_dag_free(dag);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag`` and
/// if ``reg`` is not a valid, non-null pointer to a ``QkClassicalRegister``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_add_classical_register(
    dag: *mut DAGCircuit,
    reg: *const ClassicalRegister,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let dag = unsafe { mut_ptr_as_ref(dag) };
    let creg = unsafe { const_ptr_as_ref(reg) };

    dag.add_creg(creg.clone())
        .expect("Invalid register unable to be added to DAG");
}

/// @ingroup QkDag
/// Get the number of qubits the DAG contains.
///
/// @param dag A pointer to the DAG.
///
/// @return The number of qubits the DAG is defined on.
///
/// # Example
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(24, "my_register");
/// qk_dag_add_quantum_register(dag, qr);
/// uint32_t num_qubits = qk_dag_num_qubits(dag);  // num_qubits==24
/// qk_quantum_register_free(qr);
/// qk_dag_free(dag);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_num_qubits(dag: *const DAGCircuit) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let dag = unsafe { const_ptr_as_ref(dag) };

    dag.num_qubits() as u32
}

/// @ingroup QkDag
/// Get the number of clbits the DAG contains.
///
/// @param dag A pointer to the DAG.
///
/// @return The number of clbits the DAG is defined on.
///
/// # Example
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkClassicalRegister *cr = qk_classical_register_new(24, "my_register");
/// qk_dag_add_classical_register(dag, cr);
/// uint32_t num_clbits = qk_dag_num_clbits(dag);  // num_clbits==24
/// qk_classical_register_free(cr);
/// qk_dag_free(dag);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_num_clbits(dag: *const DAGCircuit) -> u32 {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let dag = unsafe { const_ptr_as_ref(dag) };

    dag.num_clbits() as u32
}

/// @ingroup QkDag
/// Return the total number of operation nodes in the DAG.
///
/// @param dag A pointer to the DAG.
///
/// @return The total number of instructions in the DAG.
///
/// # Example
/// ```c
///     QkDag *dag = qk_dag_new();
///     QkQuantumRegister *qr = qk_quantum_register_new(1, "my_register");
///     qk_dag_add_quantum_register(dag, qr);
///
///     uint32_t qubit[1] = {0};
///     qk_dag_apply_gate(dag, QkGate_H, qubit, NULL, false);
///     size_t num = qk_dag_num_op_nodes(dag); // 1
///
///     qk_dag_free(dag);
///     qk_quantum_register_free(qr);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_num_op_nodes(dag: *const DAGCircuit) -> usize {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let dag = unsafe { const_ptr_as_ref(dag) };
    dag.num_ops()
}

/// @ingroup QkDag
///
/// DAG node type.
#[repr(u8)]
pub enum CDagNodeType {
    /// Operation node.
    Operation = 0,
    /// Qubit wire start node.
    QubitIn = 1,
    /// Qubit wire end node.
    QubitOut = 2,
    /// Clbit wire start node.
    ClbitIn = 3,
    /// Clbit wire end node.
    ClbitOut = 4,
    /// Classical variable wire start node.
    VarIn = 5,
    /// Classical variable wire end node.
    VarOut = 6,
}

/// @ingroup QkDag
/// Get the type of the specified node.
///
/// The result can be used in a switch statement to dispatch proper handling
/// when iterating over nodes of unknown type.
///
/// @param dag A pointer to the DAG.
/// @param qubit The node to get the type of.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_node_type(dag: *const DAGCircuit, node: u32) -> CDagNodeType {
    let dag = unsafe { const_ptr_as_ref(dag) };
    match dag
        .dag()
        .node_weight(NodeIndex::new(node as usize))
        .unwrap()
    {
        NodeType::QubitIn(_) => CDagNodeType::QubitIn,
        NodeType::QubitOut(_) => CDagNodeType::QubitOut,
        NodeType::ClbitIn(_) => CDagNodeType::ClbitIn,
        NodeType::ClbitOut(_) => CDagNodeType::ClbitOut,
        NodeType::VarIn(_) => CDagNodeType::VarIn,
        NodeType::VarOut(_) => CDagNodeType::VarOut,
        NodeType::Operation(_) => CDagNodeType::Operation,
    }
}

/// @ingroup QkDag
/// Retrieve the index of the input node of the wire corresponding to the given qubit.
///
/// @param dag A pointer to the DAG.
/// @param qubit The qubit to get the input node index of.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_qubit_in_node(dag: *const DAGCircuit, qubit: u32) -> u32 {
    let dag = unsafe { const_ptr_as_ref(dag) };
    dag.qubit_io_map()[qubit as usize][0].index() as u32
}

/// @ingroup QkDag
/// Retrieve the index of the output node of the wire corresponding to the given qubit.
///
/// @param dag A pointer to the DAG.
/// @param qubit The qubit to get the output node index of.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_clbit_out_node(dag: *const DAGCircuit, clbit: u32) -> u32 {
    let dag = unsafe { const_ptr_as_ref(dag) };
    dag.clbit_io_map()[clbit as usize][1].index() as u32
}

/// @ingroup QkDag
/// Retrieve the index of the input node of the wire corresponding to the given clbit.
///
/// @param dag A pointer to the DAG.
/// @param clbit The clbit to get the input node index of.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_clbit_in_node(dag: *const DAGCircuit, clbit: u32) -> u32 {
    let dag = unsafe { const_ptr_as_ref(dag) };
    dag.clbit_io_map()[clbit as usize][0].index() as u32
}

/// @ingroup QkDag
/// Retrieve the index of the output node of the wire corresponding to the given qubit.
///
/// @param dag A pointer to the DAG.
/// @param qubit The qubit to get the output node index of.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_qubit_out_node(dag: *const DAGCircuit, qubit: u32) -> u32 {
    let dag = unsafe { const_ptr_as_ref(dag) };
    dag.qubit_io_map()[qubit as usize][1].index() as u32
}

/// @ingroup QkDag
/// Retrieve the value of a wire endpoint node.
///
/// @param dag A pointer to the DAG.
/// @param qubit The endpoint node to get the wire value of.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_endpoint_node_value(dag: *const DAGCircuit, node: u32) -> u32 {
    let dag = unsafe { const_ptr_as_ref(dag) };
    (match dag
        .dag()
        .node_weight(NodeIndex::new(node as usize))
        .unwrap()
    {
        NodeType::QubitIn(value) => value.index(),
        NodeType::QubitOut(value) => value.index(),
        NodeType::ClbitIn(value) => value.index(),
        NodeType::ClbitOut(value) => value.index(),
        NodeType::VarIn(value) => value.index(),
        NodeType::VarOut(value) => value.index(),
        NodeType::Operation(_) => panic!("Specified node is not a wire endpoint node."),
    }) as u32
}

/// @ingroup QkDag
/// Apply a ``QkGate`` to the DAG.
///
/// @param dag A pointer to the DAG to apply the gate to.
/// @param gate The StandardGate to apply.
/// @param qubits The pointer to the array of ``uint32_t`` qubit indices to add the gate on. This
///     can be a null pointer if there are no qubits for ``gate`` (e.g. ``QkGate_GlobalPhase``).
/// @param params The pointer to the array of ``double`` values to use for the gate parameters.
///     This can be a null pointer if there are no parameters for ``gate`` (e.g. ``QkGate_H``).
/// @param front If ``true``, the gate is applied as the first operation on the specified qubits,
///     rather than as the last.
/// @param out_node A pointer where the newly added node's index will be written on success, or
///     NULL if you don't need it.
///
/// @return An exit code.
///
/// # Example
/// ```c
///     QkDag *dag = qk_dag_new();
///     QkQuantumRegister *qr = qk_quantum_register_new(1, "my_register");
///     qk_dag_add_quantum_register(dag, qr);
///
///     uint32_t qubit[1] = {0};
///     qk_dag_apply_gate(dag, QkGate_H, qubit, NULL, false, NULL);
///
///     qk_dag_free(dag);
///     qk_quantum_register_free(qr);
/// ```
///
/// # Safety
///
/// The ``qubits`` and ``params`` types are expected to be a pointer to an array of ``uint32_t``
/// and ``double`` respectively where the length is matching the expectations for the standard
/// gate. If the array is insufficiently long the behavior of this function is undefined as this
/// will read outside the bounds of the array. It can be a null pointer if there are no qubits
/// or params for a given gate. You can check ``qk_gate_num_qubits`` and ``qk_gate_num_params`` to
/// determine how many qubits and params are required for a given gate.
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_apply_gate(
    dag: *mut DAGCircuit,
    gate: StandardGate,
    qubits: *const u32,
    params: *const f64,
    front: bool,
    out_node: *mut u32,
) -> ExitCode {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let dag = unsafe { mut_ptr_as_ref(dag) };
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
            // There are no ``QkGate``s > 4 qubits
            _ => panic!(),
        };
        let params = match gate.num_params() {
            0 => None,
            1 => Some(smallvec![(*params.wrapping_add(0)).into()]),
            2 => Some(smallvec![
                (*params.wrapping_add(0)).into(),
                (*params.wrapping_add(1)).into(),
            ]),
            3 => Some(smallvec![
                (*params.wrapping_add(0)).into(),
                (*params.wrapping_add(1)).into(),
                (*params.wrapping_add(2)).into(),
            ]),
            4 => Some(smallvec![
                (*params.wrapping_add(0)).into(),
                (*params.wrapping_add(1)).into(),
                (*params.wrapping_add(2)).into(),
                (*params.wrapping_add(3)).into(),
            ]),
            // There are no ``QkGate``s that take > 4 params
            _ => panic!(),
        };
        let new_node = if front {
            dag.apply_operation_front(
                gate.into(),
                qargs,
                &[],
                params,
                None,
                #[cfg(feature = "cache_pygates")]
                None,
            )
            .unwrap()
        } else {
            dag.apply_operation_back(
                gate.into(),
                qargs,
                &[],
                params,
                None,
                #[cfg(feature = "cache_pygates")]
                None,
            )
            .unwrap()
        };
        if !out_node.is_null() {
            *out_node = new_node.index() as u32;
        }
    }
    ExitCode::Success
}

/// @ingroup QkDag
/// Free the DAG.
///
/// @param dag A pointer to the DAG to free.
///
/// # Example
/// ```c
/// QkDag *dag = qk_dag_new();
/// qk_dag_free(dag);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not either null or a valid pointer to a
/// ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_free(dag: *mut DAGCircuit) {
    if !dag.is_null() {
        if !dag.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(dag);
        }
    }
}
