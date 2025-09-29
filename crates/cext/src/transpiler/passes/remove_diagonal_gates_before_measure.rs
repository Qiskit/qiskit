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
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::converters::dag_to_circuit;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_transpiler::passes::run_remove_diagonal_before_measure;

/// @ingroup QkTranspilerPasses
/// Run the ``RemoveDiagonalGatesBeforeMeasure`` pass on a circuit.
///
/// Transpiler pass to remove diagonal gates (like RZ, T, Z, etc) before
/// a measurement. Including diagonal 2Q gates.
///
/// @param circuit A pointer to the circuit to run this pass on
///
/// # Example
///
/// ```c
///     QkCircuit *qc = qk_circuit_new(1, 1);
///     qk_circuit_gate(qc, QkGate_Z, {0}, NULL);
///     qk_circuit_measure(qc, 0, 0);
///     qk_transpiler_pass_standalone_remove_diagonal_gates_before_measure(qc);
///     // ...
///     qk_circuit_free(qc);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_remove_diagonal_gates_before_measure(
    circuit: *mut CircuitData,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    let mut dag = DAGCircuit::from_circuit_data(circuit, false, None, None, None, None)
        .expect("Circuit to DAG conversion failed");
    run_remove_diagonal_before_measure(&mut dag);
    let result = dag_to_circuit(&dag, false).expect("DAG to Circuit conversion failed");
    *circuit = result;
}
