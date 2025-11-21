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

use qiskit_circuit::Qubit;
use qiskit_circuit::bit::{ClassicalRegister, QuantumRegister};
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeIndex, NodeType};
use qiskit_circuit::operations::{
    Operation, OperationRef, Param, StandardGate, StandardInstruction,
};

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
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { mut_ptr_as_ref(dag) };
    // SAFETY: Per documentation, the pointer is to valid data.
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
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { mut_ptr_as_ref(dag) };
    // SAFETY: Per documentation, the pointer is to valid data.
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
    // SAFETY: Per documentation, the pointer is to valid data.
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
    // SAFETY: Per documentation, the pointer is to valid data.
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
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(1, "my_register");
/// qk_dag_add_quantum_register(dag, qr);
///
/// uint32_t qubit[1] = {0};
/// qk_dag_apply_gate(dag, QkGate_H, qubit, NULL, false);
/// size_t num = qk_dag_num_op_nodes(dag); // 1
///
/// qk_dag_free(dag);
/// qk_quantum_register_free(qr);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_num_op_nodes(dag: *const DAGCircuit) -> usize {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    dag.num_ops()
}

/// The type of node in a ``QkDag``.
///
/// Operation nodes represent an applied instruction. The rest of the nodes are
/// considered "wire" nodes and represent the endpoints of the DAG's data dependency
/// chains.
#[derive(Copy, Clone, Debug)]
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
/// @param node The node to get the type of.
///
/// @return The type of the node.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_node_type(dag: *const DAGCircuit, node: u32) -> CDagNodeType {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    match dag.dag()[NodeIndex::new(node as usize)] {
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
/// @return The input node of the qubit wire.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_qubit_in_node(dag: *const DAGCircuit, qubit: u32) -> u32 {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    dag.qubit_io_map()[qubit as usize][0].index() as u32
}

/// @ingroup QkDag
/// Retrieve the index of the output node of the wire corresponding to the given qubit.
///
/// @param dag A pointer to the DAG.
/// @param qubit The qubit to get the output node index of.
///
/// @return The output node of the qubit wire.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_qubit_out_node(dag: *const DAGCircuit, qubit: u32) -> u32 {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    dag.qubit_io_map()[qubit as usize][1].index() as u32
}

/// @ingroup QkDag
/// Retrieve the index of the input node of the wire corresponding to the given clbit.
///
/// @param dag A pointer to the DAG.
/// @param clbit The clbit to get the input node index of.
///
/// @return The input node of the clbit wire.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_clbit_in_node(dag: *const DAGCircuit, clbit: u32) -> u32 {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    dag.clbit_io_map()[clbit as usize][0].index() as u32
}

/// @ingroup QkDag
/// Retrieve the index of the output node of the wire corresponding to the given clbit.
///
/// @param dag A pointer to the DAG.
/// @param clbit The clbit to get the output node index of.
///
/// @return The output node of the clbit wire.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_clbit_out_node(dag: *const DAGCircuit, clbit: u32) -> u32 {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    dag.clbit_io_map()[clbit as usize][1].index() as u32
}

/// @ingroup QkDag
/// Retrieve the value of a wire endpoint node.
///
/// @param dag A pointer to the DAG.
/// @param node The endpoint node to get the wire value of.
///
/// @return The value (e.g. qubit, clbit, or var) within the endpoint node.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_wire_node_value(dag: *const DAGCircuit, node: u32) -> u32 {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    (match dag.dag()[NodeIndex::new(node as usize)] {
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
/// Gets the number of qubits of the specified operation node.
///
/// Panics if the node is not an operation.
///
/// @param dag A pointer to the DAG.
/// @param node The operation node to get the number of qubits of.
///
/// @return The number of qubits of the operation.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_op_node_num_qubits(dag: *const DAGCircuit, node: u32) -> u32 {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    let instr = dag.dag()[NodeIndex::new(node as usize)].unwrap_operation();
    instr.op.num_qubits()
}

/// @ingroup QkDag
/// Gets the number of clbits of the specified operation node.
///
/// Panics if the node is not an operation.
///
/// @param dag A pointer to the DAG.
/// @param node The operation node to get the number of clbits of.
///
/// @return The number of clbits of the operation.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_op_node_num_clbits(dag: *const DAGCircuit, node: u32) -> u32 {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    let instr = dag.dag()[NodeIndex::new(node as usize)].unwrap_operation();
    instr.op.num_clbits()
}

/// @ingroup QkDag
/// Gets the number of params of the specified operation node.
///
/// Panics if the node is not an operation.
///
/// @param dag A pointer to the DAG.
/// @param node The operation node to get the number of params of.
///
/// @return The number of params of the operation.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_op_node_num_params(dag: *const DAGCircuit, node: u32) -> u32 {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    let instr = dag.dag()[NodeIndex::new(node as usize)].unwrap_operation();
    instr.op.num_params()
}

/// @ingroup QkDag
/// Retrieve the qubits of the specified operation node.
///
/// Panics if the node is not an operation.
///
/// @param dag A pointer to the DAG.
/// @param node The operation node to get the qubits of.
///
/// @return A pointer to the qubits. Use ``qk_dag_op_node_num_qubits`` to determine the number of
///     elements.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_op_node_qubits(dag: *const DAGCircuit, node: u32) -> *const u32 {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    let instr = dag.dag()[NodeIndex::new(node as usize)].unwrap_operation();
    dag.qargs_interner().get(instr.qubits).as_ptr().cast()
}

/// @ingroup QkDag
/// Retrieve the clbits of the specified operation node.
///
/// Panics if the node is not an operation.
///
/// @param dag A pointer to the DAG.
/// @param node The operation node to get the clbits of.
///
/// @return A pointer to the clbits. Use ``qk_dag_op_node_num_clbits`` to determine the number of
///     elements.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_op_node_clbits(dag: *const DAGCircuit, node: u32) -> *const u32 {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    let instr = dag.dag()[NodeIndex::new(node as usize)].unwrap_operation();
    dag.cargs_interner().get(instr.clbits).as_ptr().cast()
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
///
/// @return The index of the newly added operation node.
///
/// # Example
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(1, "my_register");
/// qk_dag_add_quantum_register(dag, qr);
///
/// uint32_t qubit[1] = {0};
/// qk_dag_apply_gate(dag, QkGate_H, qubit, NULL, false);
///
/// qk_dag_free(dag);
/// qk_quantum_register_free(qr);
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
) -> u32 {
    // SAFETY: Per documentation, the pointer is to valid data.
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
        new_node.index() as u32
    }
}

/// @ingroup QkDag
/// Retrieve the standard gate of the specified node.
///
/// Panics if the node is not a standard gate operation.
///
/// @param dag A pointer to the DAG.
/// @param node The operation node to get the standard gate of.
/// @param out_params A buffer to be filled with the gate's params or NULL
///     if they're not wanted.
///
/// @return The gate value.
///
/// # Example
/// ```c
/// QkDag *dag = qk_dag_new();
/// QkQuantumRegister *qr = qk_quantum_register_new(1, "my_register");
/// qk_dag_add_quantum_register(dag, qr);
///
/// uint32_t qubit[1] = {0};
/// uint32_t h_gate_idx = qk_dag_apply_gate(dag, QkGate_H, qubit, NULL, false);
///
/// QkGate gate = qk_dag_op_node_gate_op(dag, h_gate_idx, NULL);
///
/// qk_dag_free(dag);
/// qk_quantum_register_free(qr);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a
/// ``QkDag``.
/// If ``out_params`` is non-NULL, it must point to a buffer large enough to
/// hold all the gate's params, otherwise behavior is undefined as this
/// function will write beyond its bounds.
/// You can check ``qk_dag_op_node_num_params`` to determine how many params
/// are required for any given operation node.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_op_node_gate_op(
    dag: *const DAGCircuit,
    node: u32,
    out_params: *mut f64,
) -> StandardGate {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    let instr = dag.dag()[NodeIndex::new(node as usize)].unwrap_operation();
    if !out_params.is_null() {
        let params = instr.params_view();
        for (i, param) in params
            .iter()
            .map(|x| match x {
                Param::Float(val) => *val,
                _ => panic!("Invalid parameter on instruction"),
            })
            .enumerate()
        {
            // SAFETY: Per documentation, the buffer is large enough to hold all params.
            unsafe {
                out_params.add(i).write(param);
            }
        }
    }
    instr.standard_gate().unwrap()
}

/// The operation's kind.
///
/// This is returned when querying a particular node in the graph with ``qk_dag_op_node_kind``,
/// and is intended to allow the caller to dispatch (e.g. via a "switch") calls specific to
/// the contained operation's kind.
#[derive(Copy, Clone, Debug)]
#[repr(u8)]
pub enum COperationKind {
    Gate = 0,
    Barrier = 1,
    Delay = 2,
    Measure = 3,
    Reset = 4,
    Unitary = 5,
    PauliProductMeasurement = 6,
}

/// @ingroup QkDag
/// Get the "kind" of an operation node.
///
/// The result can be used in a switch statement to dispatch proper handling
/// when iterating over operation nodes.
///
/// Panics if ``node`` is not an operation node.
///
/// @param dag A pointer to the DAG.
/// @param node The operation node to get the "kind" of.
///
/// @return The "kind" of the node.
///
/// # Safety
///
/// Behavior is undefined if ``dag`` is not a valid, non-null pointer to a ``QkDag``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_dag_op_node_kind(dag: *const DAGCircuit, node: u32) -> COperationKind {
    // SAFETY: Per documentation, the pointer is to valid data.
    let dag = unsafe { const_ptr_as_ref(dag) };
    match dag.dag()[NodeIndex::new(node as usize)]
        .unwrap_operation()
        .op
        .view()
    {
        OperationRef::StandardGate(_) => COperationKind::Gate,
        OperationRef::StandardInstruction(instr) => match instr {
            StandardInstruction::Barrier(_) => COperationKind::Barrier,
            StandardInstruction::Delay(_) => COperationKind::Delay,
            StandardInstruction::Measure => COperationKind::Measure,
            StandardInstruction::Reset => COperationKind::Reset,
        },
        OperationRef::Unitary(_) => COperationKind::Unitary,
        OperationRef::PauliProductMeasurement(_) => COperationKind::PauliProductMeasurement,
        OperationRef::Gate(_) | OperationRef::Instruction(_) | OperationRef::Operation(_) => {
            panic!("Python instances are not supported via the C API");
        }
    }
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
