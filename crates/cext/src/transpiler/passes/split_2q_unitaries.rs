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

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::converters::dag_to_circuit;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_transpiler::passes::run_split_2q_unitaries;

/// The result from ``qk_transpiler_pass_standalone_split_2q_unitaries()``.
pub struct Split2qUnitariesResult {
    circuit: CircuitData,
    permutation: Vec<u32>,
}

/// @ingroup QkSplit2qUnitariesResult
/// Get the modified circuit if any unitaries were split
///
/// @param result a pointer to the result.
///
/// @returns A pointer to the output circuit.
///
/// # Safety
///
/// Behavior is undefined if ``result`` is not a valid, non-null pointer to a
/// ``QkSplit2qUnitariesResult``. The pointer to the returned circuit is owned by
/// the result object, it should not be passed to ``qk_circuit_free()`` as it
/// will be freed by ``qk_split_2q_unitaries_result_free()``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_split_2q_unitaries_result_circuit(
    result: *const Split2qUnitariesResult,
) -> *const CircuitData {
    let result = unsafe { const_ptr_as_ref(result) };
    &result.circuit as *const _
}

/// @ingroup QkSplit2qUnitariesResult
/// Get the length of the permutation array of the permutation caused by swap equivalent unitaries
/// that were split
///
/// @param result a pointer to the result.
///
/// @returns The length of the permutation array stored in the result. This will be 0 if
/// ``qk_transpiler_pass_standalone_split_2q_unitaries()`` is called with ``split_swaps`` set to
/// ``false`` or there were no swap equivalent unitaries in the circuit.
///
/// # Safety
///
/// Behavior is undefined if ``result`` is not a valid, non-null pointer to a
/// ``QkSplit2qUnitariesResult``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_split_2q_unitaries_result_permutation_len(
    result: *const Split2qUnitariesResult,
) -> usize {
    let result = unsafe { const_ptr_as_ref(result) };
    result.permutation.len()
}

/// @ingroup QkSplit2qUnitariesResult
/// Get the length of the permutation array of the permutation caused by swap equivalent unitaries
/// that were split
///
/// @param result a pointer to the result.
///
/// @returns A pointer to the permutation array with the length that can be found with
/// ``qk_split_2q_unitaries_result_permutation_len()``.
///
/// # Safety
///
/// Behavior is undefined if ``result`` is not a valid, non-null pointer to a
/// ``QkSplit2qUnitariesResult``. The pointer to the returned permutation array is owned by
/// the result object, it should not be passed to ``free()`` as it will be freed by
/// ``qk_split_2q_unitaries_result_free()``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_split_2q_unitaries_result_permutation(
    result: *const Split2qUnitariesResult,
) -> *const u32 {
    let result = unsafe { const_ptr_as_ref(result) };
    result.permutation.as_ptr()
}

/// @ingroup QkSplit2qUnitariesResult
/// Free a ``QkSplit2qUnitariesResult`` object
///
/// @param layout a pointer to the result to free
///
/// # Safety
///
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a
/// ``QkSplit2qUnitariesResult``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_split_2q_unitaries_result_free(result: *mut Split2qUnitariesResult) {
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
/// Run the Split2QUnitaries transpiler pass
///
///
/// @param circuit A pointer to the circuit to run Split2QUnitaries on
/// @param requested_fidelity Allowed tolerance for splitting two-qubit unitaries and gate decompositions.
/// @param split_swaps Whether to attempt to split swap gates, resulting in a permutation of the qubits.
///
/// @return QkSplit2qUnitariesResult object that contains the results of the pass
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
///     QkSplit2qUnitariesResult *result = qk_transpiler_pass_standalone_split_2q_unitaries(qc, 1.0 - 1e-16, true)
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` or ``target`` is not a valid, non-null pointer to a ``QkCircuit`` and ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_split_2q_unitaries(
    circuit: *const CircuitData,
    requested_fidelity: f64,
    split_swaps: bool,
) -> *mut Split2qUnitariesResult {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };
    let mut dag = match DAGCircuit::from_circuit_data(circuit, false, None, None, None, None) {
        Ok(dag) => dag,
        Err(e) => panic!("{}", e),
    };
    let result = match run_split_2q_unitaries(&mut dag, requested_fidelity, split_swaps) {
        Ok(result) => result,
        Err(e) => panic!("{}", e),
    };
    match result {
        Some((out_dag, permutation)) => {
            let out_circuit = match dag_to_circuit(&out_dag, false) {
                Ok(qc) => qc,
                Err(e) => panic!("{}", e),
            };
            Box::into_raw(Box::new(Split2qUnitariesResult {
                circuit: out_circuit,
                permutation: permutation.into_iter().map(|x| x as u32).collect(),
            }))
        }
        None => {
            let out_circuit = match dag_to_circuit(&dag, false) {
                Ok(qc) => qc,
                Err(e) => panic!("{}", e),
            };
            Box::into_raw(Box::new(Split2qUnitariesResult {
                circuit: out_circuit,
                permutation: Vec::new(),
            }))
        }
    }
}
