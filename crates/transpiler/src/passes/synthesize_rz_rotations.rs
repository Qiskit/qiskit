// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;
use rand::seq::index;
use std::f64::consts::PI;

use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::operations::{OperationRef, Param, StandardGate};
use qiskit_synthesis::ross_selinger::py_gridsynth_rz;

// static ROTATION_GATE_NAMES: &str = "rz";
// const EPSILON: f64 = 1e-10;
const PI_TWO: f64 = PI / 2.0;
const THREE_PI_TWO: f64 = 3.0 * PI / 2.0;
const TWO_PI: f64 = 2.0 * PI;
const FIVE_PI_TWO: f64 = 5.0 * PI / 2.0;
const THREE_PI: f64 = 3.0 * PI;
const SEVEN_PI_TWO: f64 = 5.0 * PI / 2.0;
const FOUR_PI: f64 = 4.0 * PI;

// look up table for phase update and gates added during
// canonicalization of angles

static PHASE_GATE_LUT: [(f64, Option<StandardGate>); 8] = [
    (0.0, None),
    (0.0, Some(StandardGate::S)),
    (0.0, Some(StandardGate::Z)),
    (0.0, Some(StandardGate::Sdg)),
    (PI, None),
    (PI, Some(StandardGate::S)),
    (PI, Some(StandardGate::Z)),
    (PI, Some(StandardGate::Sdg)),
];

static PI_INTERVALS: [f64; 8] = [
    0.0,
    PI_TWO,
    PI,
    THREE_PI_TWO,
    TWO_PI,
    FIVE_PI_TWO,
    THREE_PI,
    SEVEN_PI_TWO,
];

/// Rz(theta+2*n*pi) = (-1)^n Rz(theta)
/// Normalize angle first (to handle negative angles as well)
/// Check whether theta/2*pi is odd or even
/// If odd, synthesize Rz(theta) and update global phase by pi
/// If even, synthesize Rz(theta) as is
// Add tolerance to this after working out approximation rules
fn rz_cyclicity(angle: f64) -> (f64, f64) {
    let angle_normalized = angle.rem_euclid(FOUR_PI);
    if angle_normalized >= TWO_PI {
        // Rz(theta) where theta in [2*pi, 4*pi)
        // = -Rz(theta - 2*pi) = e^(i*pi) Rz(theta - 2*pi)
        // Map to [0, 2*pi) and update global phase
        (angle_normalized - TWO_PI, PI)
    } else {
        // Already in [0, 2*pi)
        (angle_normalized, 0.0)
    }
}

fn canonicalize_angle(angle: f64) -> (f64, f64, Option<StandardGate>) {
    // canonicalize_angle
    let angle_normalized = angle.rem_euclid(FOUR_PI);
    // We want to use the following properties of Rz(theta)
    // Rz(theta + pi/2) = Rz(theta).S
    // Rz(theta + pi) = Rz(theta).Z
    // Rz(theta + 2*pi) = -Rz(theta)
    // Divide the domain [0, 4*pi) into eight intervals of length
    // pi/2 and define the angle mapping, gate sequence, and phase update, accordingly.

    // Determine interval of angle via ratio
    let interval = (angle_normalized / PI_TWO).floor() as u32;

    // If theta in [0, pi/2) return as is
    match interval {
        0 =>
        // [0,pi/2): Rz(theta) = Rz(theta)
        {
            (angle_normalized, 0.0, None)
        }
        1 =>
        // [pi/2, pi): Rz(pi/2 + theta) = Rz(theta) · S
        {
            (angle_normalized - PI_TWO, 0.0, Some(StandardGate::S))
        }
        2 =>
        // [pi, 3*pi/2)): Rz(pi + theta) = Rz(theta) · Z
        {
            (angle_normalized - THREE_PI_TWO, 0.0, Some(StandardGate::Z))
        }
        3 =>
        // [3*pi/2, 2pi): Rz(3*pi/2 + θ) = Rz(θ) · S · Z = Rz(θ) · Sdg
        {
            (angle_normalized - TWO_PI, 0.0, Some(StandardGate::Sdg))
        }
        4 =>
        // [2*pi,5*pi/2): Rz(2*pi+ theta) = Rz(theta)
        {
            (angle_normalized - FIVE_PI_TWO, PI, None)
        }
        5 =>
        // [5*pi/2, 3*pi): Rz(5*pi/2 + theta) = -Rz(theta) · S
        {
            (angle_normalized - THREE_PI, PI, Some(StandardGate::S))
        }
        6 =>
        // [3*pi, 7*pi/2): Rz(3*pi + theta) = -Rz(theta) · Z
        {
            (angle_normalized - SEVEN_PI_TWO, PI, Some(StandardGate::Z))
        }
        7 =>
        // [7*pi/2, 4*pi): Rz(7*pi/2 + theta) = -Rz(theta) · Sdg
        {
            (angle_normalized - FOUR_PI, PI, Some(StandardGate::Sdg))
        }
        _ => unreachable!("Check angle input, interval out of bounds"),
    }
}

fn canonicalize_angle_2(angle: f64) -> (u8, f64) {
    // canonicalize_angle
    let angle_normalized = angle.rem_euclid(FOUR_PI);
    // We want to use the following properties of Rz(theta)
    // Rz(theta + pi/2) = Rz(theta).S
    // Rz(theta + pi) = Rz(theta).Z
    // Rz(theta + 2*pi) = -Rz(theta)
    // Divide the domain [0, 4*pi) into eight intervals of length
    // pi/2 and define the angle mapping, gate sequence, and phase update, accordingly.

    // Determine interval of angle via ratio
    let interval = (angle_normalized / PI_TWO).floor() as u8;
    // Is it faster to look up or multiply?
    (interval, angle_normalized - PI_INTERVALS[interval as usize])
    // (interval, angle_normalized - (interval as f64) * PI_TWO)
}

// fn collect_sort_angles(
//     angle_list: &mut Vec<f64>
// )

/// Takes the angle and error of an Rz gate as the input and outputs
/// an equivalent set of Clifford+T gates synthesized via gridsynth
///  in addition to the global phase to be updated
///  based on which range the angle lies in (as in fn rz_cyclicity).
fn synthesize_rz_gate_via_gridsynth_(
    angle: f64,
    _epsilon: f64,
) -> PyResult<(Vec<StandardGate>, Param)> {
    let (angle_normalized, phase_update, gate) = canonicalize_angle(angle);
    // let _epsilon = _epsilon;

    let mut circ_data = py_gridsynth_rz(angle_normalized, _epsilon)?;
    circ_data.add_global_phase(&Param::Float(phase_update))?;

    // get sequence of standard gates

    let mut sequence: Vec<StandardGate> = circ_data
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
    sequence.extend(gate);
    let final_phase_update = circ_data.global_phase().clone();

    Ok((sequence, final_phase_update))
}

fn synthesize_rz_gate_via_gridsynth_2(
    angle: f64,
    _epsilon: f64,
) -> PyResult<(Vec<StandardGate>, Param)> {
    let (index, angle_normalized) = canonicalize_angle_2(angle);
    // let _epsilon = _epsilon;

    let mut circ_data = py_gridsynth_rz(angle_normalized, _epsilon)?;

    // get phase and update
    circ_data.add_global_phase(&Param::Float(PHASE_GATE_LUT[index as usize].0))?;

    let phase = circ_data.global_phase().clone();

    // get sequence of standard gates

    let mut sequence: Vec<StandardGate> = circ_data
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
    // updatesequence based on LUT
    sequence.extend(PHASE_GATE_LUT[index as usize].1);

    Ok((sequence, phase))
}

// remove canonicalization in this step as it is done during dag traversal
fn synthesize_rz_gate_via_gridsynth_3(
    angle: f64,
    _epsilon: f64,
) -> PyResult<(Vec<StandardGate>, Param)> {
    let circ_data = py_gridsynth_rz(angle, _epsilon)?;

    // get phase and update
    //  circ_data.add_global_phase(
    //     &Param::Float(PHASE_GATE_LUT[index as usize].0))?;

    let phase = circ_data.global_phase().clone();

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
    // updatesequence based on LUT
    // sequence.extend(PHASE_GATE_LUT[index as usize].1);
    // return sequence and phase as is now, will update it during dag traversal
    Ok((sequence, phase))
}

fn synthesize_rz_gate_via_gridsynth(
    angle: f64,
    _epsilon: f64,
) -> PyResult<(Vec<StandardGate>, Param)> {
    let (angle_normalized, phase_update) = rz_cyclicity(angle);
    // let _epsilon = _epsilon;

    let mut circ_data = py_gridsynth_rz(angle_normalized, _epsilon)?;
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
    if dag.get_op_counts().keys().all(|k| k != "rz") {
        return Ok(());
    }

    // Iterate over nodes in the DAG and collect nodes that have RZ gates.
    // Canonicalize angles already at this stage, so that we can use them for sorting.
    let mut candidates: Vec<_> = dag
        .op_nodes(false)
        .filter_map(|(node_index, inst)| {
            if let OperationRef::StandardGate(StandardGate::RZ) = inst.op.view() {
                if let Param::Float(angle) = inst.params_view()[0] {
                    let (interval_index, canonical_angle) = canonicalize_angle_2(angle);
                    Some((node_index, canonical_angle, interval_index))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    // Sort candidates based on the (canonicalized) angles
    candidates.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut prev_result: Option<(f64, (Vec<StandardGate>, Param))> = None;

    for (node_index, angle, interval_index) in candidates {
        // Get or compute the sequence and phase update.
        // Right now we just check if the current angle is exactly equal to the previous angle, but
        // todo: allow delta-difference with delta depending on epsilon
        let should_recompute = prev_result
            .as_ref()
            .is_none_or(|(prev_angle, _)| *prev_angle != angle);

        if should_recompute {
            let (sequence, phase_update) = synthesize_rz_gate_via_gridsynth_3(angle, epsilon)?;

            prev_result = Some((angle, (sequence, phase_update)));
        }

        let (sequence, phase_update) = &prev_result.as_ref().unwrap().1;
        // if i do this without making a copy here will it modify sequence as saved in prev_result and
        // thus affect future iterations that reuse the same sequence? i am assuming it might,
        // so making a copy to be safe
        let mut sequence_copy = sequence.clone();
        sequence_copy.extend(PHASE_GATE_LUT[interval_index as usize].1);
        for new_gate in sequence_copy {
            dag.insert_1q_on_incoming_qubit((new_gate, &[]), node_index);
        }
        dag.remove_1q_sequence(&[node_index]);
        dag.add_global_phase(phase_update)?;
        dag.add_global_phase(&Param::Float(PHASE_GATE_LUT[interval_index as usize].0))?;
    }

    Ok(())
}

// #[pyfunction]
// #[pyo3(name = "synthesize_rz_rotations")]
// pub fn py_run_synthesize_rz_rotations(dag: &mut DAGCircuit, epsilon: f64) -> PyResult<()> {
//     // Skip the pass if there are no RZ rotation gates.

//     if dag.get_op_counts().keys().all(|k| k != "rz") {
//         return Ok(());
//     }

//     // Iterate over nodes in the DAG and collect nodes that have RZ gates.

//     let candidates: Vec<_> = dag
//         .op_nodes(false)
//         .filter_map(|(node_index, inst)| {
//             if let OperationRef::StandardGate(StandardGate::RZ) = inst.op.view() {
//                 if let Param::Float(angle) = inst.params_view()[0] {
//                     Some((node_index, angle))
//                 } else {
//                     None
//                 }
//             } else {
//                 None
//             }
//         })
//         .collect();

//     for (node_index, angle) in candidates {
//         let (sequence, phase_update) = synthesize_rz_gate_via_gridsynth(angle, epsilon)?;

//         // we should remove the original gate, and instead add the sequence of gates
//         for new_gate in &sequence {
//             dag.insert_1q_on_incoming_qubit((*new_gate, &[]), node_index);
//         }
//         dag.remove_1q_sequence(&[node_index]);
//         dag.add_global_phase(&phase_update)?;
//     }

//     Ok(())
// }

pub fn synthesize_rz_rotations_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_synthesize_rz_rotations))?;
    Ok(())
}
