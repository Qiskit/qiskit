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

use pyo3::prelude::*;
use std::f64::consts::{PI, TAU};
use std::ops::Rem;

use qiskit_circuit::dag_circuit::DAGCircuit;
// use qiskit_circuit::dag_circuit::add_global_phase;
// use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{OperationRef, Param, StandardGate};
use qiskit_synthesis::ross_selinger::py_gridsynth_rz;

static ROTATION_GATE_NAMES: &str = "rz";

const EPSILON: f64 = 1e-10;
const FOUR_PI: f64 = 2.0 * TAU;

/// Rz(theta+2*n*pi) = (-1)^n Rz(theta)
/// Normalize angle first (to handle negative angles as well)
/// Check whether theta/2*pi is odd or even
/// If odd, synthesize Rz(theta) and update global phase by pi
/// If even, synthesize Rz(theta) as is
/// Add tolerance to this after working out approximation rules

fn rz_cyclicity(angle: f64) -> (f64, f64) {
    
    let angle_normalized = angle.rem(FOUR_PI);

    if angle_normalized >= TAU {
        // Rz(theta) where theta in [2*pi, 4*pi)
        // = -Rz(theta - 2*pi) = e^(i*pi) Rz(theta - 2*pi)
        let angle = angle_normalized - TAU; // Map to [0, 2*pi)
        let phase_update = PI; // Global phase
        (angle, phase_update)
    } else {
        // Already in [0, 2*pi)
        (angle_normalized, 0.0)
    }
}

// fn collect_sort_angles(
//     angle_list: &mut Vec<f64>
// )

/// Takes the angle and error of an Rz gate as the input and outputs
/// an equivalent set of Clifford+T gates synthesized via gridsynth
///  in addition to the global phase to be updated
///  based on which range the angle lies in (as in fn rz_cyclicity).

fn synthesize_rz_gate_via_gridsynth(
    angle: f64,
    _epsilon: f64,
) -> PyResult<(Vec<StandardGate>, Param)> {
    let (angle_normalized, phase_update) = rz_cyclicity(angle);
    let _epsilon = EPSILON;

    let synth_circ = py_gridsynth_rz(angle_normalized, _epsilon);

    let mut circ_data = synth_circ?;
    circ_data.add_global_phase(&Param::Float(phase_update))?;

    // get sequence of standard gates

    let sequence: Vec<StandardGate> = circ_data
        .data()
        .iter()
        .map(|inst| {
            if let OperationRef::StandardGate(gate) = inst.op.view() {
                gate
            } else {
                panic!("Non-standard gate found in synthesized circuit");
            }
        })
        .collect();

    let final_phase_update = circ_data.global_phase().clone();

    Ok((sequence, final_phase_update))
}

// Synthesize based on Rz(theta+2*n*pi) = (-1)^n Rz(theta)

// Iterate over nodes in the DAG and collect nodes that
// are Rz gates. For now, we just plainly replace the nodes.
// In the next iteration we will have three more additions:
// a function to collect and sort the angles from the DAG
// a function that can identify angles in the dag that can
// reuse the same synthesized circuit, whether because the
// gates are identical, or because we round off some angles
// based on a userconfigurable tolerance (set a default that
// can mathematically agree with the gridsynth error as well)
// Next, use this function, to check before synthesizing an
// angle. If it exists, reuse, else synthesize. This way we
// avoid redundant synthesis calls.

#[pyfunction]
#[pyo3(name = "synthesize_rz_rotations")]
pub fn py_run_synthesize_rz_rotations(dag: &mut DAGCircuit, epsilon: f64) -> PyResult<()> {
    
    // Skip the pass if there are no RZ rotation gates.

    if dag
        .get_op_counts()
        .keys()
        .all(|k| !ROTATION_GATE_NAMES.contains(k.as_str()))
    {
        return Ok(());
    }

    // Iterate over nodes in the DAG and collect nodes that have RZ gates. 

    let candidates: Vec<_> = dag
        .op_nodes(false)
        .filter_map(|(node_index, inst)| {
            if let OperationRef::StandardGate(StandardGate::RZ) = inst.op.view() {
                if let Param::Float(angle) = inst.params_view()[0] {
                    Some((node_index, StandardGate::RZ, angle))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    for (node_index, _gate, angle) in candidates {
        let (sequence, phase_update) = synthesize_rz_gate_via_gridsynth(angle, epsilon)?;

        // we should remove the original gate, and instead add the sequence of gates
        for new_gate in &sequence {
            dag.insert_1q_on_incoming_qubit((*new_gate, &[]), node_index);
        }
        dag.remove_1q_sequence(&[node_index]);
        dag.add_global_phase(&phase_update)?;
    }

    Ok(())
}

pub fn synthesize_rz_rotations_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_synthesize_rz_rotations))?;
    Ok(())
}
