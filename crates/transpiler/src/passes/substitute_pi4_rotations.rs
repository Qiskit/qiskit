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
use std::f64::consts::{FRAC_PI_8, PI};

use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::operations::{OperationRef, Param, StandardGate};
use rustworkx_core::petgraph::stable_graph::NodeIndex;

static ROTATION_GATE_NAMES: [&str; 3] = ["rx", "ry", "rz"];

const MINIMUM_TOL: f64 = 1e-12;

/// Table for RZ(k * pi / 4) substitutions, with 0 <= k < 15
static RZ_SUBSTITUTIONS: [(&[StandardGate], f64); 16] = [
    (&[], 0.0),
    (&[StandardGate::T], -FRAC_PI_8),
    (&[StandardGate::S], -2.0 * FRAC_PI_8),
    (&[StandardGate::S, StandardGate::T], -3.0 * FRAC_PI_8),
    (&[StandardGate::Z], -4.0 * FRAC_PI_8),
    (&[StandardGate::Z, StandardGate::T], -5.0 * FRAC_PI_8),
    (&[StandardGate::Sdg], -6.0 * FRAC_PI_8),
    (&[StandardGate::Tdg], -7.0 * FRAC_PI_8),
    (&[], -PI),
    (&[StandardGate::T], -9.0 * FRAC_PI_8),
    (&[StandardGate::S], -10. * FRAC_PI_8),
    (&[StandardGate::S, StandardGate::T], -11.0 * FRAC_PI_8),
    (&[StandardGate::Z], -12.0 * FRAC_PI_8),
    (&[StandardGate::Z, StandardGate::T], -13.0 * FRAC_PI_8),
    (&[StandardGate::Sdg], -14.0 * FRAC_PI_8),
    (&[StandardGate::Tdg], -15.0 * FRAC_PI_8),
];

/// Table for RX(k * pi / 4) substitutions, with 0 <= k < 15
static RX_SUBSTITUTIONS: [(&[StandardGate], f64); 16] = [
    (&[], 0.0),
    (
        &[StandardGate::H, StandardGate::T, StandardGate::H],
        -FRAC_PI_8,
    ),
    (&[StandardGate::SX], -2.0 * FRAC_PI_8),
    (
        &[
            StandardGate::SX,
            StandardGate::H,
            StandardGate::T,
            StandardGate::H,
        ],
        -3.0 * FRAC_PI_8,
    ),
    (&[StandardGate::X], -4.0 * FRAC_PI_8),
    (
        &[
            StandardGate::X,
            StandardGate::H,
            StandardGate::T,
            StandardGate::H,
        ],
        -5.0 * FRAC_PI_8,
    ),
    (&[StandardGate::SXdg], -6.0 * FRAC_PI_8),
    (
        &[StandardGate::H, StandardGate::Tdg, StandardGate::H],
        -7.0 * FRAC_PI_8,
    ),
    (&[], -PI),
    (
        &[StandardGate::H, StandardGate::T, StandardGate::H],
        -9.0 * FRAC_PI_8,
    ),
    (&[StandardGate::SX], -10.0 * FRAC_PI_8),
    (
        &[
            StandardGate::SX,
            StandardGate::H,
            StandardGate::T,
            StandardGate::H,
        ],
        -11.0 * FRAC_PI_8,
    ),
    (&[StandardGate::X], -12.0 * FRAC_PI_8),
    (
        &[
            StandardGate::X,
            StandardGate::H,
            StandardGate::T,
            StandardGate::H,
        ],
        -13.0 * FRAC_PI_8,
    ),
    (&[StandardGate::SXdg], -14.0 * FRAC_PI_8),
    (
        &[StandardGate::H, StandardGate::Tdg, StandardGate::H],
        -15.0 * FRAC_PI_8,
    ),
];

/// Table for RY(k * pi / 4) substitutions, with 0 <= k < 15
static RY_SUBSTITUTIONS: [(&[StandardGate], f64); 16] = [
    (&[], 0.0),
    (
        &[StandardGate::SX, StandardGate::T, StandardGate::SXdg],
        -FRAC_PI_8,
    ),
    (&[StandardGate::Z, StandardGate::H], 0.0),
    (
        &[
            StandardGate::SX,
            StandardGate::T,
            StandardGate::S,
            StandardGate::SXdg,
        ],
        -3.0 * FRAC_PI_8,
    ),
    (&[StandardGate::Y], -4.0 * FRAC_PI_8),
    (
        &[
            StandardGate::Y,
            StandardGate::SX,
            StandardGate::T,
            StandardGate::SXdg,
        ],
        -5.0 * FRAC_PI_8,
    ),
    (&[StandardGate::H, StandardGate::Z], -PI),
    (
        &[StandardGate::SX, StandardGate::Tdg, StandardGate::SXdg],
        -7.0 * FRAC_PI_8,
    ),
    (&[], -PI),
    (
        &[StandardGate::SX, StandardGate::T, StandardGate::SXdg],
        -9.0 * FRAC_PI_8,
    ),
    (&[StandardGate::Z, StandardGate::H], -PI),
    (
        &[
            StandardGate::SX,
            StandardGate::T,
            StandardGate::S,
            StandardGate::SXdg,
        ],
        -11.0 * FRAC_PI_8,
    ),
    (&[StandardGate::Y], -12.0 * FRAC_PI_8),
    (
        &[
            StandardGate::Y,
            StandardGate::SX,
            StandardGate::T,
            StandardGate::SXdg,
        ],
        -13.0 * FRAC_PI_8,
    ),
    (&[StandardGate::H, StandardGate::Z], 0.0),
    (
        &[StandardGate::SX, StandardGate::Tdg, StandardGate::SXdg],
        -15.0 * FRAC_PI_8,
    ),
];

/// For a given angle, if it is a multiple of PI/4, calculate the multiple mod 16,
/// Otherwise, return `None`.
fn is_angle_close_to_multiple_of_pi_4(angle: f64, tol: f64) -> Option<usize> {
    let closest_ratio = angle * 4.0 / PI;
    let closest_integer = closest_ratio.round();
    let closest_angle = closest_integer * PI / 4.0;
    let theta = angle - closest_angle;

    // Explicit trace calculation of RX/RY/RZ matrices
    let tr_over_dim = (theta / 2.).cos();
    let dim = 2.;

    // fidelity-based tolerance
    let f_pro = tr_over_dim.abs().powi(2);
    let gate_fidelity = (dim * f_pro + 1.) / (dim + 1.);
    if (1. - gate_fidelity).abs() < tol {
        Some((closest_integer as i64).rem_euclid(16) as usize)
    } else {
        None
    }
}

/// Gets a rotation gate (RX/RY/RZ) and outputs an equivalent vector of standard gates and
/// a global phase, when the gate is sufficiently close to Clifford+T/Tdg.
/// Otherwise, return None.
/// Note that odd multiples of pi/4 require a single T or Tdg gate
//  as well as some Clifford gates,
//  while even multiples of pi/4, or equivalently, integer multiples of pi/2,
//  can be written using only Clifford gates.
/// The output contains at most one T or Tdg gate, and an optimal number of
/// Clifford gates.
fn try_replace_rotation_by_discrete(
    gate: StandardGate,
    angle: f64,
    tol: f64,
) -> Option<(&'static [StandardGate], f64)> {
    let multiple = is_angle_close_to_multiple_of_pi_4(angle, tol)?;

    match gate {
        StandardGate::RX => Some(RX_SUBSTITUTIONS[multiple]),
        StandardGate::RY => Some(RY_SUBSTITUTIONS[multiple]),
        StandardGate::RZ => Some(RZ_SUBSTITUTIONS[multiple]),
        _ => None,
    }
}

#[pyfunction]
#[pyo3(name = "substitute_pi4_rotations")]
pub fn py_run_substitute_pi4_rotations(
    dag: &mut DAGCircuit,
    approximation_degree: f64,
) -> PyResult<()> {
    // Skip the pass if there are no RX/RY/RZ rotation gates.
    if dag
        .get_op_counts()
        .keys()
        .all(|k| !ROTATION_GATE_NAMES.contains(&k.as_str()))
    {
        return Ok(());
    }

    // Iterate over nodes in the DAG and collect nodes that are of the form
    // RX/RY/RZ with an angle that is a multiple of pi/4
    let mut candidates: Vec<(NodeIndex, StandardGate, f64)> = Vec::new();

    for (node_index, inst) in dag.op_nodes(false) {
        if let OperationRef::StandardGate(gate) = inst.op.view() {
            if matches!(gate, StandardGate::RX | StandardGate::RY | StandardGate::RZ) {
                if let Param::Float(angle) = inst.params_view()[0] {
                    candidates.push((node_index, gate, angle));
                }
            }
        }
    }

    let mut global_phase_update: f64 = 0.;
    let tol = MINIMUM_TOL.max(1.0 - approximation_degree);

    for (node_index, gate, angle) in candidates {
        if let Some((sequence, phase_update)) = try_replace_rotation_by_discrete(gate, angle, tol) {
            // we should remove the original gate, and instead add the sequence of gates
            for new_gate in sequence {
                dag.insert_1q_on_incoming_qubit((*new_gate, &[]), node_index);
            }
            dag.remove_1q_sequence(&[node_index]);
            global_phase_update += phase_update;
        }
    }

    dag.add_global_phase(&Param::Float(global_phase_update))?;

    Ok(())
}

pub fn substitute_pi4_rotations_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_substitute_pi4_rotations))?;
    Ok(())
}
