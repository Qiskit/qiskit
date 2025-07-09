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

use crate::pointers::const_ptr_as_ref;
use qiskit_circuit::{
    circuit_data::CircuitData, converters::dag_to_circuit, dag_circuit::DAGCircuit,
};
use qiskit_transpiler::{passes::run_optimize_1q_gates_decomposition, target::Target};

/// @ingroup QkTranspilerPasses
/// Runs the Optimize1qGatesDecomposition pass in standalone mode on a circuit.
///
/// Optimize1qGatesDecomposition, as its name implies, optimizes chains of single-qubit
/// gates by combining them into a single gate.
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
/// @param target A pointer to the ``QkTarget`` object.
///
/// @return The circuit after applying the optimizations.
///
/// # Example
///
///     QkTarget *target_u1_u2_u3 = qk_target_new(1);
///
///     double u_errors[3] = {0., 1e-4, 1e-4};
///     QkGate u_gates[3] = {QkGate_U1, QkGate_U2, QkGate_U3};
///     // TODO: Update this part to use parameters once we support them.
///     double u1_params[1] = {3.14};
///     double u2_params[2] = {3.14, 3.14 / 2.};
///     double u3_params[3] = {3.14, 3.14 / 2., 3.14 / 4.};
///
///     double **u_params = {u1_params, u2_params, u3_params};
///     for (int idx = 0; idx < 3; idx++) {
///         QkTargetEntry *u_entry = qk_target_entry_new_fixed(u_gates[idx], u_params[idx]);
///         uint32_t qargs[1] = {
///             0,
///         };
///         qk_target_entry_add_property(u_entry, qargs, 1, NAN, u_errors[idx]);
///         qk_target_add_instruction(target_u1_u2_u3, u_entry);
///     }
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit`` and
/// if ``target`` is not a valid, non-null pointer to a ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpiler_standalone_optimize_1q_gates_decomposition(
    circuit: *const CircuitData,
    target: *const Target,
) -> *mut CircuitData {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };

    // Convert the circuit to a DAG.
    let mut circuit_as_dag = DAGCircuit::from_circuit_data(circuit, false, None, None, None, None)
        .expect("Error while converting the circuit to a dag.");

    // Run the pass
    run_optimize_1q_gates_decomposition(&mut circuit_as_dag, Some(target), None, None)
        .expect("Error while running the pass.");

    // Convert the DAGCircuit back to an instance of CircuitData
    let dag_to_circuit = dag_to_circuit(&circuit_as_dag, false)
        .expect("Error while converting the dag to a circuit.");

    // Convert to pointer.
    Box::into_raw(Box::new(dag_to_circuit))
}
