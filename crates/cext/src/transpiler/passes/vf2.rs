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
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::VirtualQubit;
use qiskit_transpiler::passes::vf2::{
    vf2_layout_pass_average, Vf2PassConfiguration, Vf2PassReturn,
};
use qiskit_transpiler::target::Target;

/// The result from ``qk_transpiler_pass_standalone_vf2_layout()``.
pub struct VF2LayoutResult(Vf2PassReturn);

/// @ingroup QkVF2LayoutResult
/// Check whether a result was found.
///
/// A ``true`` value includes the situation where the configuration specified to try the "trivial"
/// layout and it was found to be the best (and consequently no qubit relabelling is necessary,
/// other than ancilla expansion if appropriate).  See ``qk_vf2_layout_result_has_improvement`` to
/// distinguish whether an explicit remapping is stored.
///
/// @param layout a pointer to the layout
///
/// @returns ``true`` if the VF2-based layout pass found any match.
///
/// # Safety
///
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a
/// ``QkVF2LayoutResult``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_vf2_layout_result_has_match(layout: *const VF2LayoutResult) -> bool {
    let layout = unsafe { const_ptr_as_ref(layout) };
    layout.0 != Vf2PassReturn::NoSolution
}

/// @ingroup QkVF2LayoutResult
/// Check whether the result is an improvement to the trivial layout.
///
/// @param layout a pointer to the layout
///
/// @returns ``true`` if the VF2-based layout pass found an improved match.
///
/// # Safety
///
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a
/// ``QkVF2LayoutResult``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_vf2_layout_result_has_improvement(
    layout: *const VF2LayoutResult,
) -> bool {
    let layout = unsafe { const_ptr_as_ref(layout) };
    matches!(layout.0, Vf2PassReturn::Solution(_))
}

/// @ingroup QkVF2LayoutResult
/// Get the physical qubit for a given virtual qubit
///
/// @param layout a pointer to the layout
/// @param qubit the virtual qubit to get the physical qubit of
///
/// @returns The physical qubit mapped to by the specified virtual qubit
///
/// # Safety
///
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a
/// ``QkVF2LayoutResult`` containing a result, or if the qubit is out of range for the initial
/// circuit.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_vf2_layout_result_map_virtual_qubit(
    layout: *const VF2LayoutResult,
    qubit: u32,
) -> u32 {
    let layout = unsafe { const_ptr_as_ref(layout) };
    match &layout.0 {
        Vf2PassReturn::NoSolution => panic!("There was no layout found!"),
        // It's undefined behaviour to pass a qubit that's out of range, so it's not our problem
        // that we can't tell if this is valid.
        Vf2PassReturn::NoImprovement => qubit,
        Vf2PassReturn::Solution(mapping) => mapping[&VirtualQubit::new(qubit)].0,
    }
}

/// @ingroup QkVF2LayoutResult
/// Free a ``QkVF2LayoutResult`` object
///
/// @param layout a pointer to the layout to free
///
/// # Example
///
///     QkCircuit *qc = qk_circuit_new(1, 0);
///
/// # Safety
///
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a ``QkVF2Layout``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_vf2_layout_result_free(layout: *mut VF2LayoutResult) {
    if !layout.is_null() {
        if !layout.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }
        // SAFETY: We have verified the pointer is non-null and aligned, so
        // it should be readable by Box.
        unsafe {
            let _ = Box::from_raw(layout);
        }
    }
}

/// A set of configurations for the VF2 layout passes.
///
/// See the setter methods associated with this `struct` for the available configurations.
pub struct VF2LayoutConfiguration(Vf2PassConfiguration);
/// @ingroup QkVF2LayoutConfiguration
/// Create a new configuration for the VF2 passes that runs everything completely unbounded.
///
/// Call ``qk_vf2_layout_configuration_free`` with the return value to free the memory when done.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_vf2_layout_configuration_new() -> *mut VF2LayoutConfiguration {
    Box::into_raw(Box::new(VF2LayoutConfiguration(
        Vf2PassConfiguration::default_unbounded(),
    )))
}
/// @ingroup QkVF2LayoutConfiguration
/// Create a new configuration for the VF2 passes that runs everything completely unbounded.
///
/// # Safety
///
/// Behavior is undefined if ``config`` is a non-null pointer, but does not point to a valid,
/// aligned ``VF2LayoutConfiguration`` object.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_vf2_layout_configuration_free(config: *mut VF2LayoutConfiguration) {
    if !config.is_null() {
        if !config.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.");
        }
        let _ = unsafe { Box::from_raw(config) };
    }
}
/// @ingroup QkVF2LayoutConfiguration
/// Limit the numbers of times that the VF2 algorithm will attempt to extend its mapping before and
/// after it finds the first match.
///
/// @param config The configuration to update.
/// @param before The number of attempts to allow before the first match is found.  Set to a
///     negative number to have no bound.
/// @param after The number of attempts to allow after the first match (if any) is found.  Set to a
///     negative number to have no bound.
///
/// # Safety
///
/// Behavior is undefined if `config` is not a valid, aligned, non-null pointer to a
/// `VF2LayoutConfiguration`.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_vf2_layout_configuration_call_limit(
    config: *mut VF2LayoutConfiguration,
    before: i64,
    after: i64,
) {
    let lift = |limit: i64| -> Option<usize> {
        (limit > 0).then(|| limit.try_into().unwrap_or(usize::MAX))
    };
    unsafe { (*config).0.call_limit = (lift(before), lift(after)) };
}
/// @ingroup QkVF2LayoutConfiguration
/// Limit the runtime of the VF2 search.
///
/// This is not a hard limit; it is only checked when an improved layout is encountered.  Using this
/// option also makes the pass non-deterministic. It is generally recommended to use
/// ``qk_vf2_layout_configuration_call_limit`` instead.
///
/// @param config The configuration to update.
/// @param limit The time in seconds to allow.  Set to a non-positive value to run with no limit.
///
/// # Safety
///
/// Behavior is undefined if `config` is not a valid, aligned, non-null pointer to a
/// `VF2LayoutConfiguration`.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_vf2_layout_configuration_time_limit(
    config: *mut VF2LayoutConfiguration,
    limit: f64,
) {
    unsafe { (*config).0.time_limit = (limit > 0.0).then_some(limit) };
}
/// @ingroup QkVF2LayoutConfiguration
/// Limit the total number of complete improvements found.
///
/// Since the VF2 search tree is pruned on-the-fly based on scoring in the ``Target``, this limit
/// is not especially powerful.  See ``qk_vf2_layout_configuration_call_limit`` for a tighter bound.
///
/// @param config The configuration to update.
/// @param limit The number of complete layouts to allow before terminating.  Set to 0 to run
///     unbounded.
///
/// # Safety
///
/// Behavior is undefined if `config` is not a valid, aligned, non-null pointer to a
/// `VF2LayoutConfiguration`.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_vf2_layout_configuration_max_trials(
    config: *mut VF2LayoutConfiguration,
    limit: u64,
) {
    unsafe { (*config).0.max_trials = Some(limit.try_into().unwrap_or(usize::MAX)) };
}
/// @ingroup QkVF2LayoutConfiguration
/// Activate node shuffling of the input graphs with a given seed.
///
/// This effectively drives a modification of the matching order of VF2, which in theory means that
/// the space of a bounded search is not biased based on the node indices.  In practice, Qiskit uses
/// the VF2++ ordering improvements when running in "average" mode (corresponding to initial layout
/// search), and starts from the identity mapping in "exact" made.  Both of these ordering
/// heuristics are typically far more likely to find results for the given problem than
/// randomization.
///
/// If this function was not called, no node shuffling takes place.
///
/// @param config The configuration to update.
/// @param seed The seed to use for the activated shuffling.
///
/// # Safety
///
/// Behavior is undefined if `config` is not a valid, aligned, non-null pointer to a
/// `VF2LayoutConfiguration`.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_vf2_layout_configuration_shuffle_nodes(
    config: *mut VF2LayoutConfiguration,
    seed: u64,
) {
    unsafe { (*config).0.shuffle_seed = Some(seed) };
}
/// @ingroup QkVF2LayoutConfiguration
/// Whether to eagerly score the initial "trivial" layout of the interaction graph.
///
/// You typically want to set this ``true`` if you are using the VF2 passes to improve a circuit
/// that is already lowered to hardware, in order to set a baseline for the score-based pruning.  If
/// not, you can leave this as ``false`` (the default), to avoid a calculation that likely will not
/// have any impact.
///
/// @param config The configuration to update.
/// @param score_inital Whether to eagerly score the initial trivial layout.
///
/// # Safety
///
/// Behavior is undefined if `config` is not a valid, aligned, non-null pointer to a
/// `VF2LayoutConfiguration`.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_vf2_layout_configuration_score_initial(
    config: *mut VF2LayoutConfiguration,
    score_initial: bool,
) {
    unsafe { (*config).0.score_initial_layout = score_initial };
}

/// @ingroup QkTranspilerPasses
/// Use the VF2 algorithm to choose a layout (if possible) for the input circuit, using a
/// noise-aware scoring heuristic based only on hardware error rates, and not the specific gates in
/// the circuit.
///
/// This function corresponds to the Python-space ``VF2Layout`` pass.
///
/// This function is suitable for use on circuits that have not yet been fully lowered to hardware.
/// If this pass finds a solution that means there is a "perfect layout" and that no
/// further swap mapping or routing is needed. However, there is not always a possible
/// solution, or a solution might exist but it is not found within the limits specified
/// when the pass is called.
///
/// @param circuit A pointer to the circuit to run VF2Layout on
/// @param target A pointer to the target to run the VF2Layout pass on
/// @param config A pointer to the ``QkVF2LayoutConfiguration`` configuration structure.  If this
///     pointer is null, the pass defaults are used.
/// @param strict_direction If true, the pass will consider the edge direction in the
///     connectivity described in the ``target``. Typically, setting this to ``false``
///     is desireable as the error heuristic is already very approximate, and two-qubit gates can
///     almost invariably be synthesised to "flip" direction using only local one-qubit gates and
///     the native-direction two-qubit gate.
///
/// @return QkVF2LayoutResult A pointer to a result object that contains the
/// results of the pass. This object is heap allocated and will need to be freed with the
/// ``qk_vf2_layout_result_free()`` function.
///
/// # Example
///
/// ```c
///     QkTarget *target = qk_target_new(5);
///     uint32_t current_num_qubits = qk_target_num_qubits(target);
///     QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
///     for (uint32_t i = 0; i < current_num_qubits - 1; i++) {
///         uint32_t qargs[2] = {i, i + 1};
///         double inst_error = 0.0090393 * (current_num_qubits - i);
///         double inst_duration = 0.020039;
///         qk_target_entry_add_property(cx_entry, qargs, 2, inst_duration, inst_error);
///     }
///     QkExitCode result_cx = qk_target_add_instruction(target, cx_entry);
///     QkCircuit *qc = qk_circuit_new(4, 0);
///     for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i++) {
///         uint32_t qargs[2] = {i, i + 1};
///         for (uint32_t j = 0; j<i+1; j++) {
///             qk_circuit_gate(qc, QkGate_CX, qargs, NULL);
///         }
///     }
///     QkVF2LayoutConfiguration *config = qk_vf2_layout_configuration_new();
///     qk_vf2_layout_configuration_call_limit(config, 10000, 10000);
///     QkVF2LayoutResult *layout_result = qk_transpiler_pass_standalone_vf2_layout(qc, target, config, false);
///     qk_vf2_layout_result_free(layout_result);
///     qk_vf2_layout_configuration_free(config);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` or ``target`` is not a valid, non-null pointer to a
/// ``QkCircuit`` and ``QkTarget``.  Behavior is undefined if ``config`` is a non-null pointer that
/// does not point to a valid ``QkVF2LayoutConfiguration`` object (but a null pointer is fine).
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_vf2_layout(
    circuit: *const CircuitData,
    target: *const Target,
    config: *const VF2LayoutConfiguration,
    strict_direction: bool,
) -> *mut VF2LayoutResult {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };
    let target = unsafe { const_ptr_as_ref(target) };
    let dag = match DAGCircuit::from_circuit_data(circuit, false, None, None, None, None) {
        Ok(dag) => dag,
        Err(e) => panic!("{}", e),
    };
    let config_default = Vf2PassConfiguration::default_abstract();
    let config = if config.is_null() {
        &config_default
    } else {
        unsafe { &const_ptr_as_ref(config).0 }
    };
    vf2_layout_pass_average(&dag, target, config, strict_direction, None)
        .map(|result| Box::into_raw(Box::new(VF2LayoutResult(result))))
        .unwrap()
}
