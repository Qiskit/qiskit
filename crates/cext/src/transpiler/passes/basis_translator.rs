// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::pointers::{const_ptr_as_ref, mut_ptr_as_ref};
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::converters::dag_to_circuit;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_transpiler::passes::run_basis_translator;
use qiskit_transpiler::standard_equivalence_library::generate_standard_equivalence_library;
use qiskit_transpiler::target::Target;

/// @ingroup QkTranspilerPasses
/// Run the BasisTranslator transpiler pass on a circuit.
///
/// Refer to the ``qk_transpiler_pass_basis_translator`` function for more details about the pass.
///
/// @param circuit A pointer to the circuit to run BasisTranslator on.
/// @param target The target where we will obtain basis gates from.
/// @param min_qubits The minimum number of qubits for operations in the input circuit to translate.
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` and/or ``target`` are not valid, non-null
/// pointers to a ``QkCircuit`` or ``QkTarget``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_basis_translator(
    circuit: *mut CircuitData,
    target: *const Target,
    min_qubits: usize,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circ_from_ptr = unsafe { mut_ptr_as_ref(circuit) };
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    let dag = DAGCircuit::from_circuit_data(circ_from_ptr, false, None, None, None, None)
        .expect("Circuit to DAG conversion failed");

    let mut equiv_lib = generate_standard_equivalence_library();

    let result_dag =
        match run_basis_translator(&dag, &mut equiv_lib, min_qubits, Some(target), None) {
            Ok(Some(dag)) => dag,
            Ok(None) => return,
            Err(e) => panic!("{}", e),
        };
    let result_circ = dag_to_circuit(&result_dag, false).expect("DAG to Circuit conversion failed");
    *circ_from_ptr = result_circ;
}

/// @ingroup QkTranspilerPasses
/// Run the BasisTranslator transpiler pass on a DAG Circuit.
///
/// The BasisTranslator transpiler pass translates gates to a target basis by
/// searching for a set of translations from the standard EquivalenceLibrary.
///
/// @param dag A pointer to the DAG Circuit to run BasisTranslator on.
/// The DAG Circuit will be replaced in-place, unless it is already in the target basis,
/// in which case the DAG Circuit remains unchanged.
/// @param target The target where we will obtain basis gates from.
/// @param min_qubits The minimum number of qubits for operations in the input
/// DAG to translate.
///
/// # Example
///
/// ```c
///    #include <qiskit.h>
///
///    QkDag *dag = qk_dag_new();
///    QkQuantumRegister *qr = qk_quantum_register_new(3, "qr");
///    qk_dag_add_quantum_register(dag, qr);
///    qk_dag_apply_gate(dag, QkGate_CCX, (uint32_t[3]){0, 1, 2}, NULL, false);
///
///    // Create a Target with global properties.
///    QkTarget *target = qk_target_new(3);
///    qk_target_add_instruction(target, qk_target_entry_new(QkGate_H));
///    qk_target_add_instruction(target, qk_target_entry_new(QkGate_T));
///    qk_target_add_instruction(target, qk_target_entry_new(QkGate_Tdg));
///    qk_target_add_instruction(target, qk_target_entry_new(QkGate_CX));
///
///    // Run pass
///    qk_transpiler_pass_basis_translator(dag, target, 0);
///
///    // Free the dag, register, and target pointers once you're done
///    qk_dag_free(dag);
///    qk_quantum_register_free(qr);
///    qk_target_free(target);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``dag`` and/or ``target`` are not valid, non-null
/// pointers to a ``QkDag`` or ``QkTarget``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpiler_pass_basis_translator(
    dag: *mut DAGCircuit,
    target: *const Target,
    min_qubits: usize,
) {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let dag_from_ptr = unsafe { mut_ptr_as_ref(dag) };
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let target = unsafe { const_ptr_as_ref(target) };

    let mut equiv_lib = generate_standard_equivalence_library();

    let result_dag =
        match run_basis_translator(dag_from_ptr, &mut equiv_lib, min_qubits, Some(target), None) {
            Ok(Some(new_dag)) => new_dag,
            Ok(None) => return,
            Err(e) => panic!("{}", e),
        };
    *dag_from_ptr = result_dag;
}
