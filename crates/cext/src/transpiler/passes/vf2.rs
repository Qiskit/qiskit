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

use hashbrown::HashMap;

use crate::pointers::const_ptr_as_ref;

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::{PhysicalQubit, VirtualQubit};
use qiskit_transpiler::passes::vf2_layout_pass;
use qiskit_transpiler::target::Target;

/// The result from ``qk_transpiler_pass_standalone_vf2_layout()``.
pub struct VF2LayoutResult(Option<HashMap<VirtualQubit, PhysicalQubit>>);

/// @ingroup QkVF2LayoutResult
/// Check whether a result was found.
///
/// @param layout a pointer to the layout
///
/// @returns ``true`` if the ``qk_transpiler_pass_standalone_vf2_layout()`` run found a layout
///
/// # Safety
///
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a
/// ``QkVF2LayoutResult``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_vf2_layout_result_has_match(layout: *const VF2LayoutResult) -> bool {
    let layout = unsafe { const_ptr_as_ref(layout) };
    layout.0.is_some()
}

/// @ingroup QkVF2LayoutResult
/// Get the number of virtual qubits in the layout.
///
/// @param layout a pointer to the layout
///
/// @returns The number of virtual qubits in the layout
///
/// # Safety
///
/// Behavior is undefined if ``layout`` is not a valid, non-null pointer to a
/// ``QkVF2LayoutResult``. The result must have a layout found.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_vf2_layout_result_num_qubits(layout: *const VF2LayoutResult) -> u32 {
    let layout = unsafe { const_ptr_as_ref(layout) };
    let Some(ref layout) = layout.0 else {
        panic!("Invalid call for empty layout result");
    };
    layout.len() as u32
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
/// ``QkVF2LayoutResult``. Also qubit must be a valid qubit for the circuit and
/// there must be a result found.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_vf2_layout_result_map_virtual_qubit(
    layout: *const VF2LayoutResult,
    qubit: u32,
) -> u32 {
    let layout = unsafe { const_ptr_as_ref(layout) };
    let Some(ref layout) = layout.0 else {
        panic!("There was no layout found");
    };
    match layout.get(&VirtualQubit(qubit)) {
        Some(physical) => physical.0,
        None => panic!("The specified qubit is not in the layout: {qubit}"),
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

/// @ingroup QkTranspilerPasses
/// Run the VF2Layout pass on a circuit.
///
/// VF2Layout is a pass for choosing a layout of a circuit onto a connectivity graph as
/// a subgraph isomorphism problem solved by VF2.
///
/// If this pass finds a solution that means there is a "perfect layout" and that no
/// further swap mapping or routing is needed. However, there is not always a possible
/// solution, or a solution might exist but it is not found within the limits specified
/// when the pass is called.
///
/// By default, this pass will construct a heuristic scoring map based on the error rates
/// in the provided ``target`` argument. The function will continue searching for layouts
/// and use the heuristic scoring to return the layout which will run with the best estimated
/// fidelity.
///
/// @param circuit A pointer to the circuit to run VF2Layout on
/// @param target A pointer to the target to run the VF2Layout pass on
/// @param strict_direction If true the pass will consider the edge direction in the
///     connectivity described in the ``target``. Typically setting this to ``false``
///     is desireable as an undirected search has more degrees of freedom and is more likely
///     to find a layout (or a better layout if there are multiple choices) and correcting
///     directionality is a simple operation for later transpilation stages.
/// @param call_limit The number of state visits to attempt in each execution of the VF2 algorithm.
///     If the value is set to a negative value the VF2 algorithm will run without any limit.
/// @param time_limit The total time in seconds to run for ``VF2Layout``. This is checked after
///     each layout search so it is not a hard time limit, but a soft limit that when checked
///     if the set time has elapsed the function will return the best layout it has found so
///     far. Set this to a value less than or equal to 0.0 to run without any time limit.
/// @param max_trials The maximum number of trials to run the VF2 algorithm to try and find
///     layouts. If the value is negative this will be treated as unbounded which means the
///     algorithm will run until all possible layouts are scored. If the value is 0 the number
///     of trials will be limited based on the number of edges in the interaction or the coupling
///     graph (whichever is larger).
///
/// @return QkVF2LayoutResult A pointer to a result object that contains the
/// results of the pass. This object is heap allocated and will need to be freed with the
/// ``qk_vf2_layout_result_free()`` function.
///
/// # Example
///
/// ```c
///     QkTarget *target = qk_target_new(5)
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
///     QkVF2LayoutResult *layout_result = qk_transpiler_pass_standalone_vf2_layout(qc, target, false, -1, NAN, -1);
///     qk_vf2_layout_result_free(layout_result);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` or ``target`` is not a valid, non-null pointer to a ``QkCircuit`` and ``QkTarget``.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_vf2_layout(
    circuit: *const CircuitData,
    target: *const Target,
    strict_direction: bool,
    call_limit: i64,
    time_limit: f64,
    max_trials: isize,
) -> *mut VF2LayoutResult {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { const_ptr_as_ref(circuit) };
    let target = unsafe { const_ptr_as_ref(target) };
    let dag = match DAGCircuit::from_circuit_data(circuit, false, None, None, None, None) {
        Ok(dag) => dag,
        Err(e) => panic!("{}", e),
    };
    let call_limit = if call_limit < 0 {
        None
    } else {
        Some(call_limit as usize)
    };
    let time_limit = if time_limit <= 0.0 {
        None
    } else {
        Some(time_limit)
    };
    let max_trials = if max_trials < 0 {
        Some(0)
    } else if max_trials == 0 {
        None
    } else {
        Some(max_trials)
    };
    let layout = match vf2_layout_pass(
        &dag,
        target,
        strict_direction,
        call_limit,
        time_limit,
        max_trials,
        None,
    ) {
        Ok(layout) => layout,
        Err(e) => panic!("{}", e),
    };
    Box::into_raw(Box::new(VF2LayoutResult(layout)))
}
