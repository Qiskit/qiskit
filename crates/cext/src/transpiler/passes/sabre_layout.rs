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

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::converters::dag_to_circuit;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::nlayout::NLayout;
use qiskit_circuit::VirtualQubit;
use qiskit_transpiler::passes::sabre::heuristic;
use qiskit_transpiler::passes::sabre::sabre_layout_and_routing;
use qiskit_transpiler::target::Target;

/// The result from ``qk_transpiler_pass_standalone_sabre_layout()``
pub struct SabreLayoutResult {
    circuit: CircuitData,
    initial_layout: NLayout,
    final_layout: NLayout,
}

/// @ingroup QkSabreLayoutResult
/// Get the circuit from sabre layout result
///
/// @param result a pointer to the result of the pass.
///
/// @returns A pointer to the output circuit
///
/// # Safety
///
/// Behavior is undefined if ``result`` is not a valid, non-null pointer to a
/// ``QkSabreLayoutResult``. The pointer to the returned circuit is owned by
/// the result object, it should not be passed to ``qk_circuit_free()`` as it
/// will be freed by ``qk_sabre_layout_result_free()``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_sabre_layout_result_circuit(
    result: *mut SabreLayoutResult,
) -> *mut CircuitData {
    let result = unsafe { mut_ptr_as_ref(result) };
    (&mut result.circuit) as *mut _
}

/// @ingroup QkSabreLayoutResult
/// Get the number of qubits in the initial layout
///
/// @param result a pointer to the result
///
/// @returns The number of qubits in the initial layout
///
/// # Safety
///
/// Behavior is undefined if ``result`` is not a valid, non-null pointer to a
/// ``QkSabreLayoutResult``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_sabre_layout_result_initial_layout_num_qubits(
    result: *const SabreLayoutResult,
) -> u32 {
    let result = unsafe { const_ptr_as_ref(result) };
    result.initial_layout.num_qubits() as u32
}

/// @ingroup QkSabreLayoutResult
/// Get the physical qubit for a given virtual qubit from the initial layout
///
/// @param result a pointer to the result
/// @param qubit the virtual qubit to get the physical qubit. This must
/// be a valid qubit in the circuit
///
/// @returns The physical qubit mapped to in the initial layout
///
/// # Safety
///
/// Behavior is undefined if ``result`` is not a valid, non-null pointer to a
/// ``QkSabreLayoutResult``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_sabre_layout_result_map_virtual_qubit_initial_layout(
    result: *const SabreLayoutResult,
    qubit: u32,
) -> u32 {
    let result = unsafe { const_ptr_as_ref(result) };
    result
        .initial_layout
        .virtual_to_physical(VirtualQubit::new(qubit))
        .0
}

/// @ingroup QkSabreLayoutResult
/// Get the number of qubits in the final layout
///
/// @param result a pointer to the result
///
/// @returns The number of qubits in the final layout
///
/// # Safety
///
/// Behavior is undefined if ``result`` is not a valid, non-null pointer to a
/// ``QkSabreLayoutResult``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_sabre_layout_result_final_layout_num_qubits(
    result: *const SabreLayoutResult,
) -> u32 {
    let result = unsafe { const_ptr_as_ref(result) };
    result.final_layout.num_qubits() as u32
}

/// @ingroup QkSabreLayoutResult
/// Get the output position for a qubit in the circuit from the final layout
///
/// @param result a pointer to the result
/// @param qubit the qubit to get the final position of after the permutations
/// from the inserted swaps in the circuit. This must be a valid qubit in the circuit
///
/// @returns The final position of the qubit mapped in the final layout
///
/// # Safety
///
/// Behavior is undefined if ``result`` is not a valid, non-null pointer to a
/// ``QkSabreLayoutResult``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_sabre_layout_result_map_qubit_final_layout(
    result: *const SabreLayoutResult,
    qubit: u32,
) -> u32 {
    let result = unsafe { const_ptr_as_ref(result) };
    result
        .final_layout
        .virtual_to_physical(VirtualQubit::new(qubit))
        .0
}

/// @ingroup QkSabreLayoutResult
/// Free a ``QkSabreLayoutResult`` object
///
/// @param result a pointer to the result to free
///
/// # Safety
///
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a
/// ``QkSabreLayoutResult``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_sabre_layout_result_free(result: *mut SabreLayoutResult) {
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
/// Run the SabreLayout transpiler pass on a circuit.
///
/// The SabreLayout pass chooses a layout via an iterative bidirectional routing of the input
/// circuit.
///
/// Starting with a random initial Layout, the algorithm does a full routing of the circuit to end up with a final_layout.
/// This final_layout is then used as the initial_layout for routing the reverse circuit. The algorithm iterates a number
/// of times until it finds an initial_layout that reduces full routing cost.
///
/// This method exploits the reversibility of quantum circuits, and tries to include global circuit information in the
/// choice of initial_layout.
///
/// This pass will run both layout and routing and will transform the circuit so that the layout is applied to the input
/// (meaning that the output circuit will have ancilla qubits allocated for unused qubits on the coupling map and the
/// qubits will be reordered to match the mapped physical qubits) and then routing will be applied. This is done because
/// the pass will run parallel seed trials with different random seeds for selecting the random initial layout and then
/// selecting the routed output which results in the least number of swap gates needed. This final
/// swap calculation is the same as performing a final routing, so it's more efficient to apply it
/// after computing it.
///
/// # References
///
/// [1] Henry Zou and Matthew Treinish and Kevin Hartman and Alexander Ivrii and Jake Lishman.
/// "LightSABRE: A Lightweight and Enhanced SABRE Algorithm"
/// [arXiv:2409.08368](https://doi.org/10.48550/arXiv.2409.08368)
///
/// [2] Li, Gushu, Yufei Ding, and Yuan Xie. "Tackling the qubit mapping problem
/// for NISQ-era quantum devices." ASPLOS 2019.
/// [`arXiv:1809.02573](https://arxiv.org/pdf/1809.02573.pdf)
///
/// @param circuit A pointer to the circuit to run SabreLayout on
///
/// @return QkElidePermutationsResult object that contains the results of the pass
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` or ``target`` is not a valid, non-null pointer to a ``QkCircuit`` and ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_sabre_layout(
    circuit: *const CircuitData,
    target: *const Target,
    max_iterations: usize,
    num_swap_trials: usize,
    num_random_trials: usize,
    seed: i64,
) -> *mut SabreLayoutResult {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };
    let target = unsafe { const_ptr_as_ref(target) };
    let mut dag = match DAGCircuit::from_circuit_data(circuit, false, None, None, None, None) {
        Ok(dag) => dag,
        Err(e) => panic!("{}", e),
    };
    let seed = if seed < 0 { None } else { Some(seed as u64) };
    let heuristic = heuristic::Heuristic::new(
        Some(heuristic::BasicHeuristic::new(
            1.0,
            heuristic::SetScaling::Constant,
        )),
        Some(heuristic::LookaheadHeuristic::new(
            0.5,
            20,
            heuristic::SetScaling::Size,
        )),
        Some(heuristic::DecayHeuristic::new(0.001, 5)),
        Some(10 * target.num_qubits.unwrap() as usize),
        1e-10,
    );
    let (result, initial_layout, final_layout) = match sabre_layout_and_routing(
        &mut dag,
        target,
        &heuristic,
        max_iterations,
        num_swap_trials,
        num_random_trials,
        seed,
        Vec::new(),
        false,
    ) {
        Ok(res) => res,
        Err(e) => panic!("{}", e),
    };
    let out_circuit = match dag_to_circuit(&result, false) {
        Ok(qc) => qc,
        Err(e) => panic!("{}", e),
    };
    Box::into_raw(Box::new(SabreLayoutResult {
        circuit: out_circuit,
        initial_layout,
        final_layout,
    }))
}
