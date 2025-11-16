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
use pyo3::wrap_pyfunction;
use std::f64::consts::PI;

use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::operations::{OperationRef, Param, StandardGate};
use rustworkx_core::petgraph::stable_graph::NodeIndex;

const ROTATION_GATE_NAMES: &[&str; 3] = &["rx", "ry", "rz"];

const PI8: f64 = PI / 8.0;
const DEFAULT_ATOL: f64 = 1e-10;

/// For an angle, if it is a multiple of PI/8, calculate the multiplicity mod 16,
/// Otherwise, return None.
fn is_angle_close_to_multiple_of_2pi_8(angle: f64) -> Option<usize> {
    let closest_ratio = angle * 4.0 / PI;
    let closest_integer = closest_ratio.round();
    if (closest_ratio - closest_integer).abs() < DEFAULT_ATOL {
        Some((closest_integer as usize) % 16)
    } else {
        None
    }
}

/// Gets a rotation gate (RX/RY/RZ) and outputs an equivalent vector of standard gates and
/// a global phase, when the gate is sufficiently close to Clifford+T/Tdg.
/// Otherwise, return None.
fn try_replace_rotation_by_discrete(
    gate: StandardGate,
    angle: f64,
) -> Option<(Vec<StandardGate>, f64)> {
    let multiple = is_angle_close_to_multiple_of_2pi_8(angle);
    let mut discrete_sequence = Vec::<StandardGate>::with_capacity(4);

    let global_phase = match (gate, multiple) {
        (StandardGate::RZ, Some(0)) => {
            discrete_sequence.push(StandardGate::I);
            0.0
        }
        (StandardGate::RZ, Some(1)) => {
            discrete_sequence.push(StandardGate::T);
            -1.0 * PI8
        }
        (StandardGate::RZ, Some(2)) => {
            discrete_sequence.push(StandardGate::S);
            -2.0 * PI8
        }
        (StandardGate::RZ, Some(3)) => {
            discrete_sequence.push(StandardGate::S);
            discrete_sequence.push(StandardGate::T);
            -3.0 * PI8
        }
        (StandardGate::RZ, Some(4)) => {
            discrete_sequence.push(StandardGate::Z);
            -4.0 * PI8
        }
        (StandardGate::RZ, Some(5)) => {
            discrete_sequence.push(StandardGate::Z);
            discrete_sequence.push(StandardGate::T);
            -5.0 * PI8
        }
        (StandardGate::RZ, Some(6)) => {
            discrete_sequence.push(StandardGate::Sdg);
            -6.0 * PI8
        }
        (StandardGate::RZ, Some(7)) => {
            discrete_sequence.push(StandardGate::Tdg);
            -7.0 * PI8
        }
        (StandardGate::RZ, Some(8)) => {
            discrete_sequence.push(StandardGate::I);
            -1.0 * PI
        }
        (StandardGate::RZ, Some(9)) => {
            discrete_sequence.push(StandardGate::T);
            -9.0 * PI8
        }
        (StandardGate::RZ, Some(10)) => {
            discrete_sequence.push(StandardGate::S);
            -10.0 * PI8
        }
        (StandardGate::RZ, Some(11)) => {
            discrete_sequence.push(StandardGate::S);
            discrete_sequence.push(StandardGate::T);
            -11.0 * PI8
        }
        (StandardGate::RZ, Some(12)) => {
            discrete_sequence.push(StandardGate::Z);
            -12.0 * PI8
        }
        (StandardGate::RZ, Some(13)) => {
            discrete_sequence.push(StandardGate::Z);
            discrete_sequence.push(StandardGate::T);
            -13.0 * PI8
        }
        (StandardGate::RZ, Some(14)) => {
            discrete_sequence.push(StandardGate::Sdg);
            -14.0 * PI8
        }
        (StandardGate::RZ, Some(15)) => {
            discrete_sequence.push(StandardGate::Tdg);
            -15.0 * PI8
        }
        (StandardGate::RX, Some(0)) => {
            discrete_sequence.push(StandardGate::I);
            0.0
        }
        (StandardGate::RX, Some(1)) => {
            discrete_sequence.push(StandardGate::H);
            discrete_sequence.push(StandardGate::T);
            discrete_sequence.push(StandardGate::H);
            -1.0 * PI8
        }
        (StandardGate::RX, Some(2)) => {
            discrete_sequence.push(StandardGate::SX);
            -2.0 * PI8
        }
        (StandardGate::RX, Some(3)) => {
            discrete_sequence.push(StandardGate::SX);
            discrete_sequence.push(StandardGate::H);
            discrete_sequence.push(StandardGate::T);
            discrete_sequence.push(StandardGate::H);
            -3.0 * PI8
        }
        (StandardGate::RX, Some(4)) => {
            discrete_sequence.push(StandardGate::X);
            -4.0 * PI8
        }
        (StandardGate::RX, Some(5)) => {
            discrete_sequence.push(StandardGate::X);
            discrete_sequence.push(StandardGate::H);
            discrete_sequence.push(StandardGate::T);
            discrete_sequence.push(StandardGate::H);
            -5.0 * PI8
        }
        (StandardGate::RX, Some(6)) => {
            discrete_sequence.push(StandardGate::SXdg);
            -6.0 * PI8
        }
        (StandardGate::RX, Some(7)) => {
            discrete_sequence.push(StandardGate::H);
            discrete_sequence.push(StandardGate::Tdg);
            discrete_sequence.push(StandardGate::H);
            -7.0 * PI8
        }
        (StandardGate::RX, Some(8)) => {
            discrete_sequence.push(StandardGate::I);
            -1.0 * PI
        }
        (StandardGate::RX, Some(9)) => {
            discrete_sequence.push(StandardGate::H);
            discrete_sequence.push(StandardGate::T);
            discrete_sequence.push(StandardGate::H);
            -9.0 * PI8
        }
        (StandardGate::RX, Some(10)) => {
            discrete_sequence.push(StandardGate::SX);
            -10.0 * PI8
        }
        (StandardGate::RX, Some(11)) => {
            discrete_sequence.push(StandardGate::SX);
            discrete_sequence.push(StandardGate::H);
            discrete_sequence.push(StandardGate::T);
            discrete_sequence.push(StandardGate::H);
            -11.0 * PI8
        }
        (StandardGate::RX, Some(12)) => {
            discrete_sequence.push(StandardGate::X);
            -12.0 * PI8
        }
        (StandardGate::RX, Some(13)) => {
            discrete_sequence.push(StandardGate::X);
            discrete_sequence.push(StandardGate::H);
            discrete_sequence.push(StandardGate::T);
            discrete_sequence.push(StandardGate::H);
            -13.0 * PI8
        }
        (StandardGate::RX, Some(14)) => {
            discrete_sequence.push(StandardGate::SXdg);
            -14.0 * PI8
        }
        (StandardGate::RX, Some(15)) => {
            discrete_sequence.push(StandardGate::H);
            discrete_sequence.push(StandardGate::Tdg);
            discrete_sequence.push(StandardGate::H);
            -15.0 * PI8
        }
        (StandardGate::RY, Some(0)) => {
            discrete_sequence.push(StandardGate::I);
            0.0
        }
        (StandardGate::RY, Some(1)) => {
            discrete_sequence.push(StandardGate::SX);
            discrete_sequence.push(StandardGate::T);
            discrete_sequence.push(StandardGate::SXdg);
            -1.0 * PI8
        }
        (StandardGate::RY, Some(2)) => {
            discrete_sequence.push(StandardGate::Z);
            discrete_sequence.push(StandardGate::H);
            0.0
        }
        (StandardGate::RY, Some(3)) => {
            discrete_sequence.push(StandardGate::SX);
            discrete_sequence.push(StandardGate::T);
            discrete_sequence.push(StandardGate::S);
            discrete_sequence.push(StandardGate::SXdg);
            -3.0 * PI8
        }
        (StandardGate::RY, Some(4)) => {
            discrete_sequence.push(StandardGate::Y);
            -4.0 * PI8
        }
        (StandardGate::RY, Some(5)) => {
            discrete_sequence.push(StandardGate::Y);
            discrete_sequence.push(StandardGate::SX);
            discrete_sequence.push(StandardGate::T);
            discrete_sequence.push(StandardGate::SXdg);
            -5.0 * PI8
        }
        (StandardGate::RY, Some(6)) => {
            discrete_sequence.push(StandardGate::H);
            discrete_sequence.push(StandardGate::Z);
            -1.0 * PI
        }
        (StandardGate::RY, Some(7)) => {
            discrete_sequence.push(StandardGate::SX);
            discrete_sequence.push(StandardGate::Tdg);
            discrete_sequence.push(StandardGate::SXdg);
            -7.0 * PI8
        }
        (StandardGate::RY, Some(8)) => {
            discrete_sequence.push(StandardGate::I);
            -1.0 * PI
        }
        (StandardGate::RY, Some(9)) => {
            discrete_sequence.push(StandardGate::SX);
            discrete_sequence.push(StandardGate::T);
            discrete_sequence.push(StandardGate::SXdg);
            -9.0 * PI8
        }
        (StandardGate::RY, Some(10)) => {
            discrete_sequence.push(StandardGate::Z);
            discrete_sequence.push(StandardGate::H);
            -1.0 * PI
        }
        (StandardGate::RY, Some(11)) => {
            discrete_sequence.push(StandardGate::SX);
            discrete_sequence.push(StandardGate::T);
            discrete_sequence.push(StandardGate::S);
            discrete_sequence.push(StandardGate::SXdg);
            -11.0 * PI8
        }
        (StandardGate::RY, Some(12)) => {
            discrete_sequence.push(StandardGate::Y);
            -12.0 * PI8
        }
        (StandardGate::RY, Some(13)) => {
            discrete_sequence.push(StandardGate::Y);
            discrete_sequence.push(StandardGate::SX);
            discrete_sequence.push(StandardGate::T);
            discrete_sequence.push(StandardGate::SXdg);
            -13.0 * PI8
        }
        (StandardGate::RY, Some(14)) => {
            discrete_sequence.push(StandardGate::H);
            discrete_sequence.push(StandardGate::Z);
            -0.0
        }
        (StandardGate::RY, Some(15)) => {
            discrete_sequence.push(StandardGate::SX);
            discrete_sequence.push(StandardGate::Tdg);
            discrete_sequence.push(StandardGate::SXdg);
            -15.0 * PI8
        }
        _ => {
            return None;
        }
    };

    Some((discrete_sequence, global_phase))
}

#[pyfunction]
#[pyo3(name = "discretize_rotations")]
pub fn run_discretize_rotations(dag: &mut DAGCircuit) -> PyResult<()> {
    let op_counts = dag.get_op_counts();

    // Skip the pass if there are no RX/RY/RZ rotation gates.
    if op_counts
        .keys()
        .all(|k| !ROTATION_GATE_NAMES.contains(&k.as_str()))
    {
        return Ok(());
    }

    // Iterate over nodes in the DAG and collect nodes that are of the form
    // RX/RY/RZ with an angle that is a multiply of 2*pi/8
    let mut candidates: Vec<(NodeIndex, StandardGate, f64)> = Vec::new();

    for (node_index, inst) in dag.op_nodes(true) {
        match inst.op.view() {
            OperationRef::StandardGate(gate) => {
                if gate == StandardGate::RX || gate == StandardGate::RY || gate == StandardGate::RZ
                {
                    match inst.params_view()[0] {
                        Param::Float(angle) => {
                            candidates.push((node_index, gate, angle));
                        }
                        _ => {
                            continue;
                        }
                    }
                }
            }
            _ => {
                continue;
            }
        }
    }

    let mut global_phase_update: f64 = 0.;

    for (node_index, gate, angle) in candidates {
        let result = try_replace_rotation_by_discrete(gate, angle);

        if let Some((sequence, phase_update)) = result {
            // we should remove the original gate, and instead add the sequence of gates
            for new_gate in sequence {
                dag.insert_1q_on_incoming_qubit((new_gate, &[]), node_index);
            }
            dag.remove_1q_sequence(&[node_index]);
            global_phase_update += phase_update;
        }
    }

    dag.add_global_phase(&Param::Float(global_phase_update))?;

    Ok(())
}

pub fn discretize_rotations_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_discretize_rotations))?;
    Ok(())
}
