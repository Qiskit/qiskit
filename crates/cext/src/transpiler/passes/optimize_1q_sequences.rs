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
use qiskit_circuit::{
    circuit_data::CircuitData, converters::dag_to_circuit, dag_circuit::DAGCircuit,
};
use qiskit_transpiler::{passes::run_optimize_1q_gates_decomposition, target::Target};

/// @ingroup QkTranspilerPasses
/// Runs the Optimize1qGatesDecomposition pass in standalone mode on a circuit.
///
/// Optimize1qGatesDecomposition optimizes single-qubit gate sequences by re-synthesizing
/// the unitary under the constraints of the target's basis gates and error rates.
///
/// The decision of whether to replace the original chain depends on:
/// - If the original chain was out of basis.
/// - If the original chain was in basis but the replacement has lower error rates.
/// - If the original chain is an identity (chain gets removed).
///
/// The error is the combined multiplication of the errors of individual gates on the
/// qubit it operates on.
///
/// @param circuit A pointer to the ``QkCircuit`` object to transform.
/// @param target A pointer to the ``QkTarget`` object or a null pointer.
/// In the case a null pointer is provided and gate errors are unknown
/// the pass will choose the sequence with the least amount of gates,
/// and will support all basis gates on its Euler basis set.
///
/// # Example
///
///     QkTarget *target = qk_target_new(1);
///     double u_errors[3] = {0., 1e-4, 1e-4};
///     for (int idx = 0; idx < 3; idx++) {
///         QkTargetEntry *u_entry = qk_target_entry_new(QkGate_U);
///         uint32_t qargs[1] = {
///             0,
///         };
///         qk_target_entry_add_property(u_entry, qargs, 1, NAN, u_errors[idx]);
///         qk_target_add_instruction(target, u_entry);
///     }
///
///     // Build circuit
///     QkCircuit *circuit = qk_circuit_new(1, 0);
///     uint32_t qubits[1] = {0};
///     for (int iter = 0; iter < 3; iter++) {
///         qk_circuit_gate(circuit, QkGate_H, qubits, NULL);
///     }
///
///     // Run transpiler pass
///     qk_transpiler_standalone_optimize_1q_sequences(circuit, target);
///
///     // Clean up
///     qk_target_free(target);
///     qk_circuit_free(circuit);
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit`` and
/// if ``target`` is not a valid pointer to a ``QkTarget``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpiler_standalone_optimize_1q_sequences(
    circuit: *mut CircuitData,
    target: *const Target,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe {
        if target.is_null() {
            None
        } else {
            Some(const_ptr_as_ref(target))
        }
    };
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };

    // Convert the circuit to a DAG.
    let mut circuit_as_dag = DAGCircuit::from_circuit_data(circuit, false, None, None, None, None)
        .expect("Error while converting the circuit to a dag.");

    // Run the pass
    run_optimize_1q_gates_decomposition(&mut circuit_as_dag, target, None, None)
        .expect("Error while running the pass.");

    // Convert the DAGCircuit back to an instance of CircuitData
    let dag_to_circuit = dag_to_circuit(&circuit_as_dag, false)
        .expect("Error while converting the dag to a circuit.");

    // Convert to pointer.
    *circuit = dag_to_circuit;
}
