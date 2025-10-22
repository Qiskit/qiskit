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

use crate::pointers::mut_ptr_as_ref;

use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::converters::dag_to_circuit;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_transpiler::passes::run_split_2q_unitaries;
use qiskit_transpiler::transpile_layout::TranspileLayout;

/// @ingroup QkTranspilerPasses
/// Run the Split2QUnitaries transpiler pass
///
/// @param circuit A mutable pointer to the circuit to run Split2QUnitaries on. This will be
///     replaced with the new circuit if any gates are optimized and the original will be freed.
/// @param requested_fidelity Allowed tolerance for splitting two-qubit unitaries and gate decompositions.
/// @param split_swaps Whether to attempt to split swap gates, resulting in a permutation of the qubits.
///
/// @return If any swap equivalent unitaries are split this function returns a pointer to a ``TranspileLayout``
///     that contains the permutation induced by this circuit optimization. If no swap equivalent
///     unitaries are split this will be a null pointer.
///
/// # Example
///
/// ```c
/// QkCircuit *qc = qk_circuit_new(4, 0);
/// for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i++) {
///     uint32_t qargs[2] = {i, i + 1};
///     for (uint32_t j = 0; j<i+1; j++) {
///         qk_circuit_gate(qc, QkGate_CX, qargs, NULL);
///     }
/// }
/// QkTranspileLayout *result = qk_transpiler_pass_standalone_split_2q_unitaries(qc, 1e-12, true)
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_split_2q_unitaries(
    circuit: *mut CircuitData,
    requested_fidelity: f64,
    split_swaps: bool,
) -> *mut TranspileLayout {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    let mut dag = match DAGCircuit::from_circuit_data(circuit, false, None, None, None, None) {
        Ok(dag) => dag,
        Err(_e) => panic!("Internal circuit -> DAG conversion failed."),
    };
    let result = run_split_2q_unitaries(&mut dag, requested_fidelity, split_swaps)
        .unwrap_or_else(|_| panic!("Running the Split2qUnitaries pass failed"));
    match result {
        Some((out_dag, permutation)) => {
            let out_circuit = match dag_to_circuit(&out_dag, false) {
                Ok(qc) => qc,
                Err(_e) => panic!("Internal DAG -> circuit conversion failed."),
            };
            let num_input_qubits = circuit.num_qubits() as u32;
            let qubits = out_circuit.qubits().objects().clone();
            let qregs = out_circuit.qregs().to_vec();
            *circuit = out_circuit;
            Box::into_raw(Box::new(TranspileLayout::new(
                None,
                Some(permutation.into_iter().map(Qubit::new).collect()),
                qubits,
                num_input_qubits,
                qregs,
            )))
        }
        None => {
            let out_circuit = match dag_to_circuit(&dag, false) {
                Ok(qc) => qc,
                Err(_e) => panic!("Internal DAG -> circuit conversion failed."),
            };
            *circuit = out_circuit;
            std::ptr::null_mut()
        }
    }
}
