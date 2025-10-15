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
use qiskit_transpiler::passes::run_elide_permutations;
use qiskit_transpiler::transpile_layout::TranspileLayout;

/// @ingroup QkTranspilerPasses
/// Run the ElidePermutations transpiler pass on a circuit.
///
/// The ElidePermutations transpiler pass removes any permutation operations from a pre-layout
/// circuit.
///
/// This pass is intended to be run before a layout (mapping virtual qubits to physical qubits) is
/// set during the transpilation pipeline. This pass iterates over the circuit
/// and when a Swap gate is encountered it permutes the virtual qubits in
/// the circuit and removes the swap gate. This will effectively remove any
/// swap gates in the circuit prior to running layout. This optimization is
/// not valid after a layout has been set and should not be run in this case.
///
/// @param circuit A pointer to the circuit to run ElidePermutations on. If there are changes made
///     the object pointed to is changed in place. In case of gates being elided the original circuit's
///     allocations are freed by this function.
///
/// @return the layout object containing the output permutation induced by the elided gates in the
///         circuit. If no elisions are performed this will be a null pointer and the input circuit
///         is unchanged.
///
/// # Example
///
/// ```c
///     QkCircuit *qc = qk_circuit_new(4, 0);
///     for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i++) {
///         uint32_t qargs[2] = {i, i + 1};
///         for (uint32_t j = 0; j<i+1; j++) {
///             qk_circuit_gate(qc, QkGate_CX, qargs, NULL);
///         }
///     }
///     QkTranspileLayout *elide_result = qk_transpiler_pass_standalone_elide_permutations(qc);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit``  is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_elide_permutations(
    circuit: *mut CircuitData,
) -> *mut TranspileLayout {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { mut_ptr_as_ref(circuit) };
    let dag = match DAGCircuit::from_circuit_data(circuit, false, None, None, None, None) {
        Ok(dag) => dag,
        Err(_e) => panic!("Internal circuit to DAG conversion failed."),
    };
    let res = run_elide_permutations(&dag).unwrap();
    match res {
        Some(res) => {
            let out_circuit =
                dag_to_circuit(&res.0, false).expect("Internal DAG to Circuit conversion failed.");
            let num_input_qubits = circuit.num_qubits() as u32;
            *circuit = out_circuit;
            Box::into_raw(Box::new(TranspileLayout::new(
                None,
                Some(res.1.into_iter().map(Qubit::new).collect()),
                circuit.qubits().objects().clone(),
                num_input_qubits,
                circuit.qregs().to_vec(),
            )))
        }
        None => std::ptr::null_mut(),
    }
}
