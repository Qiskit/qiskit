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
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::operations::{OperationRef, Param, StandardGate};
use qiskit_synthesis::ross_selinger::py_gridsynth_rz;

const MINIMUM_EPSILON: f64 = 1e-2; // minimum epsilon for synthesis
const MAXIMUM_EPSILON: f64 = 1e-12; // maximum epsilon for synthesis

/// look up table for phase update and gates added during
/// canonicalization of angles

static PHASE_GATE_LUT: [(f64, Option<StandardGate>); 8] = [
    (0.0, None),
    (-FRAC_PI_4, Some(StandardGate::S)),
    (-FRAC_PI_2, Some(StandardGate::Z)),
    (-3. * FRAC_PI_4, Some(StandardGate::Sdg)),
    (PI, None),
    (-5. * FRAC_PI_4, Some(StandardGate::S)),
    (-6. * FRAC_PI_4, Some(StandardGate::Z)),
    (-7. * FRAC_PI_4, Some(StandardGate::Sdg)),
];

// Takes angle as f64 and returns \(interval,canonical_angle)
// as \(u8,f64). 
// We want to use the following properties of Rz(theta)
// Rz(theta + pi/2) = Rz(theta).S
// Rz(theta + pi) = Rz(theta).Z
// Rz(theta + 2*pi) = -Rz(theta)
// Divide the domain [0, 4*pi) into eight intervals of length
// pi/2 and define the angle mapping, gate sequence, and phase update, accordingly.
// Any angle theta canonicalized as theta_prime from [0,4*pi) -> [0,pi/2)
// can be uniquely represented by the combination of (theta_prime, interval)
// where interval is the floor of theta/(pi/2) and is a u8 bit that functions
// as an index to the static table that uniquely determines the combination of 
// phase_update and angle to be added to the DAG after synthesis using the 
// canonical_angle. The limits of this uniqueness is determined by 
// angle_normalized = angle.rem_euclid(FOUR_PI); since it is f64, any
// angle that differs in decimal places beyond f64 will have a non-unique
// representation.

/// Finds a canonical representation of an angle.
///
/// Given `angle`, this returns `(interval, angle_normalized)` such that
/// angle_normalized = angle (mod pi/2)
/// (angle - angle_normalized - interval * pi/2) = 0 (mod 4pi)
fn canonicalize_angle(angle: f64) -> (u8, f64) {
    let angle_normalized = angle.rem_euclid(FRAC_PI_2);
    debug_assert!((0.0..FRAC_PI_2).contains(&angle_normalized));
    let interval = ((angle - angle_normalized) / FRAC_PI_2)
        .round()
        .rem_euclid(8.) as u8;
    (interval, angle_normalized)
}

/// Approximates RZ-rotation using gridsynth. 
///
/// Returns the sequence of gates in the  synthesized circuit and 
/// an update to the global phase.
fn synthesize_rz_gate_via_gridsynth(
    angle: f64,
    epsilon: f64,
) -> PyResult<(Vec<StandardGate>, Param)> {
    let circ_data = py_gridsynth_rz(angle, epsilon)?;

    // obtain phase from circuit data
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
    Ok((sequence, phase))
}

#[pyfunction]
#[pyo3(name = "synthesize_rz_rotations")]
pub fn py_run_synthesize_rz_rotations(
    dag: &mut DAGCircuit,
    approximation_degree: f64,
) -> PyResult<()> {
    
    // Skip the pass if there are no RZ rotation gates.
    if dag.get_op_counts().keys().all(|k| k != "rz") {
        return Ok(());
    }
    // bound epsilon between minimum and max values to ensure fidelity of synthesis
    let epsilon = MAXIMUM_EPSILON.max(MINIMUM_EPSILON.min(1. - approximation_degree));
    // Iterate over nodes in the DAG and collect nodes that have RZ gates.
    // Canonicalize angles already at this stage, so that we can use them for sorting.
    let mut candidates: Vec<_> = dag
        .op_nodes(false)
        .filter_map(|(node_index, inst)| {
            if let OperationRef::StandardGate(StandardGate::RZ) = inst.op.view() {
                if let Param::Float(angle) = inst.params_view()[0] {
                    let (interval_index, canonical_angle) = canonicalize_angle(angle);
                    Some((node_index, canonical_angle, interval_index))
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    // Sort candidates based on the \(canonicalized) angles
    candidates.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut prev_result: Option<(f64, (Vec<StandardGate>, Param))> = None;

    for (node_index, angle, interval_index) in candidates {
        // Get or compute the sequence and phase update.
        // Using triangle inequality it can be shown that if the current angle is within epsilon/2 
        // of the previous angle, we can reuse the synthesis of previous angle by calling gridsynth
        // with epsilon/2
        let should_recompute = prev_result
            .as_ref()
            .is_none_or(|(prev_angle, _)| *prev_angle + epsilon / 2. < angle);

        if should_recompute {
            let (sequence, phase_update) = synthesize_rz_gate_via_gridsynth(angle, epsilon / 2.)?;

            prev_result = Some((angle, (sequence, phase_update)));
        }

        let (sequence, phase_update) = &prev_result.as_ref().unwrap().1;

        // Add the gates and phase update to DAG, remove old node

        for new_gate in sequence {
            dag.insert_1q_on_incoming_qubit((*new_gate, &[]), node_index);
        }
        if let Some(gate) = PHASE_GATE_LUT[interval_index as usize].1 {
            dag.insert_1q_on_incoming_qubit((gate, &[]), node_index);
        }
        dag.remove_1q_sequence(&[node_index]);

        dag.add_global_phase(phase_update)?;
        dag.add_global_phase(&Param::Float(PHASE_GATE_LUT[interval_index as usize].0))?;
    }

    Ok(())
}

pub fn synthesize_rz_rotations_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_synthesize_rz_rotations))?;
    Ok(())
}
