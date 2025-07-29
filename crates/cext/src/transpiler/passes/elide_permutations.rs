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

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::converters::dag_to_circuit;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_transpiler::passes::run_elide_permutations;

/// The result from ``qk_transpiler_pass_standalone_elide_permutations()``.
pub struct ElidePermutationsResult {
    elided: bool,
    circuit: Option<CircuitData>,
    permutation: Option<Vec<usize>>,
}

/// @ingroup QkElidePermutationsResult
/// Check whether the elide permutations was able to elide anything
///
/// @param result a pointer to the result
///
/// @returns ``true`` if the ``qk_transpiler_pass_standalone_elide_permutations()`` run elided any gates
///
/// # Safety
///
/// Behavior is undefined if ``result`` is not a valid, non-null pointer to a
/// ``QkElidePermutationsResult``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_elide_permutations_result_elided_gates(
    result: *const ElidePermutationsResult,
) -> bool {
    let result = unsafe { const_ptr_as_ref(result) };
    result.elided
}

/// @ingroup QkElidePermutationsResult
/// Get the circuit from the elide permutations result
///
/// @param result a pointer to the result of the pass. It must have elided gates as checked by
/// ``qk_elide_permutations_result_elided_gates()``
///
/// @returns A pointer to the circuit with the permutation gates elided
///
/// # Safety
///
/// Behavior is undefined if ``result`` is not a valid, non-null pointer to a
/// ``QkElidePermutationsResult``. The pointer to the returned circuit is owned by the
/// result object, it should not be passed to ``qk_circuit_free()`` as it will
/// be freed by ``qk_elide_permutations_result_free()``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_elide_permutations_result_circuit(
    result: *const ElidePermutationsResult,
) -> *const CircuitData {
    let result = unsafe { const_ptr_as_ref(result) };
    result
        .circuit
        .as_ref()
        .map(|x| x as *const _)
        .expect("Result didn't elide any gates")
}

/// @ingroup QkElidePermutationsResult
/// Get the permutation array for the elided gates
///
/// @param result a pointer to the result of the pass. It must have elided gates as checked by
/// ``qk_elide_permutations_result_elided_gates()``
///
/// @returns A pointer to the permutation array caused by the swap elision performed by the
/// pass. This array has a length equal to the number of qubits of the circuit returned by
/// ``qk_elide_permutations_result_elided_gates()`` (or the circuit passed into
/// ``qk_transpiler_pass_standalone_elide_permutations()``). The permutation array maps the
/// virtual qubit in the original circuit at each index to its new output position after all
/// the elision performed by the pass. For example, and array of ``[2, 1, 0]`` means that
/// qubit 0 is now in qubit 2 on the output of the circuit and qubit 2's is now at 0 (qubit 1
/// remains unchanged)
///
/// # Safety
///
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a
/// ``QkElidePermutationsResult``. Also qubit must be a valid qubit for the circuit and
/// there must be a result found. The pointer to the permutation array is owned by the
/// result object, it should not be passed to ``free()`` as it will be freed by
/// ``qk_elide_permutations_result_free()`` when that is called.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_elide_permutations_result_permutation(
    result: *const ElidePermutationsResult,
) -> *const usize {
    let result = unsafe { const_ptr_as_ref(result) };
    result
        .permutation
        .as_ref()
        .map(|x| x.as_ptr())
        .expect("Result didn't elide any gates")
}

/// @ingroup QkElidePermutationsResult
/// Free a ``QkElidePermutationsResult`` object
///
/// @param result a pointer to the result object to free
///
/// # Safety
///
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a
/// ``QkElidePermutationsResult``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_elide_permutations_result_free(result: *mut ElidePermutationsResult) {
    if !result.is_null() {
        if !result.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }
        // SAFETY: We have verified the pointer is non-null and aligned, so
        // it should be readable by Box.
        unsafe {
            let _ = Box::from_raw(result);
        }
    }
}

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
/// swap gates in the cirucit prior to running layout. This optimization is
/// not valid after a layout has been set and should not be run in this case.
///
/// @param circuit A pointer to the circuit to run ElidePermutations on
///
/// @return QkElidePermutationsResult object that contains the results of the pass
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
///     QkElidePermutationsResult *elide_result = qk_transpiler_pass_standalone_elide_permutations(qc);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit``  is not a valid, non-null pointer to a ``QkCircuit``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_elide_permutations(
    circuit: *const CircuitData,
) -> *mut ElidePermutationsResult {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };
    let dag = match DAGCircuit::from_circuit_data(circuit, false, None, None, None, None) {
        Ok(dag) => dag,
        Err(e) => panic!("{}", e),
    };
    let res = match run_elide_permutations(&dag) {
        Ok(res) => res,
        Err(e) => panic!("{}", e),
    };
    match res {
        Some(res) => Box::into_raw(Box::new(ElidePermutationsResult {
            elided: true,
            circuit: Some(dag_to_circuit(&res.0, false).expect("DAG to Circuit conversion failed")),
            permutation: Some(res.1),
        })),
        None => Box::into_raw(Box::new(ElidePermutationsResult {
            elided: false,
            circuit: None,
            permutation: None,
        })),
    }
}
