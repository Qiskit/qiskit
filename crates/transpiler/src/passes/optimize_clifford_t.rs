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

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::operations::Param;
use qiskit_circuit::operations::StandardGate::{H, S, X, Y, Z};
use qiskit_circuit::operations::{OperationRef, StandardGate};
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use crate::TranspilerError;

// List of 1-qubit Clifford+T gate names.
const CLIFFORD_T_GATE_NAMES: &[&str; 18] = &[
    "id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "cx", "cz", "cy", "swap", "iswap", "ecr",
    "dcx", "t", "tdg",
];

#[derive(Clone, Copy, PartialEq)]
enum Pauli1q {
    X,
    Y,
    Z,
}

struct Clifford1q {
    idx: usize, // single-qubit clifford tableaus: takes values 0..24
    w: usize,   // phase of the form 2 w pi / 4: takes values 0..8
}

impl Clifford1q {
    fn identity() -> Self {
        Self { idx: 0, w: 0 }
    }

    fn append_clifford_gate(&mut self, gate: StandardGate) {
        match gate {
            StandardGate::I => {}
            StandardGate::H => {
                let (new_idx, w_update) = APPEND_H[self.idx];
                (self.idx, self.w) = (new_idx, (self.w + w_update) % 8);
            }
            StandardGate::S => {
                let (new_idx, w_update) = APPEND_S[self.idx];
                (self.idx, self.w) = (new_idx, (self.w + w_update) % 8);
            }
            StandardGate::Sdg => {
                self.append_clifford_gate(StandardGate::S);
                self.append_clifford_gate(StandardGate::S);
                self.append_clifford_gate(StandardGate::S);
            }
            StandardGate::Z => {
                self.append_clifford_gate(StandardGate::S);
                self.append_clifford_gate(StandardGate::S);
            }
            StandardGate::SX => {
                self.append_clifford_gate(StandardGate::H);
                self.append_clifford_gate(StandardGate::S);
                self.append_clifford_gate(StandardGate::H);
            }
            StandardGate::SXdg => {
                self.append_clifford_gate(StandardGate::H);
                self.append_clifford_gate(StandardGate::Sdg);
                self.append_clifford_gate(StandardGate::H);
            }
            StandardGate::X => {
                self.append_clifford_gate(StandardGate::SX);
                self.append_clifford_gate(StandardGate::SX);
            }
            StandardGate::Y => {
                self.append_clifford_gate(StandardGate::X);
                self.append_clifford_gate(StandardGate::Z);
                self.w = (self.w + 6) % 8;
            }
            _ => unreachable!("should not be here"),
        }
    }

    fn prepend_clifford_gate(&mut self, gate: StandardGate) {
        match gate {
            StandardGate::I => {}
            StandardGate::H => {
                let (new_idx, w_update) = PREPEND_H[self.idx];
                (self.idx, self.w) = (new_idx, (self.w + w_update) % 8);
            }
            StandardGate::S => {
                let (new_idx, w_update) = PREPEND_S[self.idx];
                (self.idx, self.w) = (new_idx, (self.w + w_update) % 8);
            }
            StandardGate::Sdg => {
                self.prepend_clifford_gate(StandardGate::S);
                self.prepend_clifford_gate(StandardGate::S);
                self.prepend_clifford_gate(StandardGate::S);
            }
            StandardGate::Z => {
                self.prepend_clifford_gate(StandardGate::S);
                self.prepend_clifford_gate(StandardGate::S);
            }
            StandardGate::SX => {
                self.prepend_clifford_gate(StandardGate::H);
                self.prepend_clifford_gate(StandardGate::S);
                self.prepend_clifford_gate(StandardGate::H);
            }
            StandardGate::SXdg => {
                self.prepend_clifford_gate(StandardGate::H);
                self.prepend_clifford_gate(StandardGate::Sdg);
                self.prepend_clifford_gate(StandardGate::H);
            }
            StandardGate::X => {
                self.prepend_clifford_gate(StandardGate::SX);
                self.prepend_clifford_gate(StandardGate::SX);
            }
            StandardGate::Y => {
                self.prepend_clifford_gate(StandardGate::Z);
                self.prepend_clifford_gate(StandardGate::X);
                self.w = (self.w + 6) % 8;
            }
            _ => unreachable!("should not be here"),
        }
    }

    // add for all other crap
    fn evolve_pauli(&self, pauli: Pauli1q, sign: bool) -> (Pauli1q, bool) {
        let (new_pauli, new_sign) = match pauli {
            Pauli1q::X => EVOLVE_X[self.idx],
            Pauli1q::Y => EVOLVE_Y[self.idx],
            Pauli1q::Z => EVOLVE_Z[self.idx],
        };
        (new_pauli, sign ^ new_sign)
    }

    fn to_circuit(&self) -> (Vec<StandardGate>, f64) {
        let circuit: Vec<StandardGate> = CIRCUIT[self.idx].to_vec();
        let phase: f64 = (self.w as f64) * PI / 4.;
        (circuit, phase)
    }
}

fn optimize_clifford_t_1q(
    dag: &DAGCircuit,
    raw_run: &[NodeIndex],
) -> Option<(Vec<StandardGate>, f64)> {
    let mut is_reduced = false; // was any reduction applied
    let mut clifford1q = Clifford1q::identity(); // current 1q-clifford including global phase
    let mut rotations: Vec<(Pauli1q, bool)> = Vec::new(); // current list of rotations
    let mut global_phase: f64 = 0.; // current update to the global phase
    let num_nodes = raw_run.len();
    for idx in 0..num_nodes {
        let cur_node = &dag[raw_run[idx]];
        let cur_gate = if let NodeType::Operation(inst) = cur_node {
            if let OperationRef::StandardGate(gate) = inst.op.view() {
                gate
            } else {
                unreachable!("Can only have Clifford+T gates at this point");
            }
        } else {
            unreachable!("Can only have op nodes here")
        };

        if cur_gate == StandardGate::T {
            global_phase += PI / 8.;
            let evolved_rotation: (Pauli1q, bool) = clifford1q.evolve_pauli(Pauli1q::Z, false);
            rotations.push(evolved_rotation);
        } else if cur_gate == StandardGate::Tdg {
            global_phase -= PI / 8.;
            let evolved_rotation: (Pauli1q, bool) = clifford1q.evolve_pauli(Pauli1q::Z, true);
            rotations.push(evolved_rotation);
        } else {
            clifford1q.append_clifford_gate(cur_gate);
            continue;
        }

        // Check if the new rotation can be canceled or merged with the previous one,
        // which happens exactly when they correspond to the same Pauli.
        if rotations.len() >= 2 {
            let (pauli2, sign2) = rotations[rotations.len() - 2];
            let (pauli1, sign1) = rotations[rotations.len() - 1];

            if pauli1 == pauli2 {
                is_reduced = true;

                rotations.pop();
                rotations.pop();

                if sign1 != sign2 {
                    // nothing more to do
                } else {
                    match (pauli1, sign1) {
                        (Pauli1q::Z, false) => {
                            global_phase -= PI / 4.;
                            clifford1q.prepend_clifford_gate(StandardGate::S);
                        }
                        (Pauli1q::Z, true) => {
                            global_phase += PI / 4.;
                            clifford1q.prepend_clifford_gate(StandardGate::Sdg);
                        }
                        (Pauli1q::X, false) => {
                            global_phase -= PI / 4.;
                            clifford1q.prepend_clifford_gate(StandardGate::SX);
                        }
                        (Pauli1q::X, true) => {
                            global_phase += PI / 4.;
                            clifford1q.prepend_clifford_gate(StandardGate::SXdg);
                        }
                        (Pauli1q::Y, false) => {
                            global_phase -= PI / 4.;
                            clifford1q.prepend_clifford_gate(StandardGate::SXdg);
                            clifford1q.prepend_clifford_gate(StandardGate::S);
                            clifford1q.prepend_clifford_gate(StandardGate::SX);
                        }
                        (Pauli1q::Y, true) => {
                            global_phase += PI / 4.;
                            clifford1q.prepend_clifford_gate(StandardGate::SXdg);
                            clifford1q.prepend_clifford_gate(StandardGate::Sdg);
                            clifford1q.prepend_clifford_gate(StandardGate::SX);
                        }
                    }
                }
            }
        }
    }

    let mut optimized_sequence = Vec::<StandardGate>::with_capacity(num_nodes);

    // Insert rotations, converting them to Clifford+T/Tdg gates, while appropriately updating the
    // global phase.
    for (pauli, sign) in rotations {
        match (pauli, sign) {
            (Pauli1q::X, false) => {
                optimized_sequence.push(StandardGate::H);
                optimized_sequence.push(StandardGate::T);
                optimized_sequence.push(StandardGate::H);
                global_phase -= PI / 8.;
            }
            (Pauli1q::X, true) => {
                optimized_sequence.push(StandardGate::H);
                optimized_sequence.push(StandardGate::Tdg);
                optimized_sequence.push(StandardGate::H);
                global_phase += PI / 8.;
            }
            (Pauli1q::Y, false) => {
                optimized_sequence.push(StandardGate::SX);
                optimized_sequence.push(StandardGate::T);
                optimized_sequence.push(StandardGate::SXdg);
                global_phase -= PI / 8.;
            }
            (Pauli1q::Y, true) => {
                optimized_sequence.push(StandardGate::SX);
                optimized_sequence.push(StandardGate::Tdg);
                optimized_sequence.push(StandardGate::SXdg);
                global_phase += PI / 8.;
            }
            (Pauli1q::Z, false) => {
                optimized_sequence.push(StandardGate::T);
                global_phase -= PI / 8.;
            }
            (Pauli1q::Z, true) => {
                optimized_sequence.push(StandardGate::Tdg);
                global_phase += PI / 8.;
            }
        }
    }

    // Insert remaining Clifford gates
    let (clifford_gates, phase_update) = clifford1q.to_circuit();
    for gate in clifford_gates {
        optimized_sequence.push(gate);
    }
    global_phase += phase_update;

    is_reduced.then_some((optimized_sequence, global_phase))
}

#[pyfunction]
#[pyo3(name = "optimize_clifford_t")]
pub fn run_optimize_clifford_t(dag: &mut DAGCircuit) -> PyResult<()> {
    let op_counts = dag.get_op_counts();

    // Stop the pass if there are unsupported gates.
    if !op_counts
        .keys()
        .all(|k| CLIFFORD_T_GATE_NAMES.contains(&k.as_str()))
    {
        let unsupported: Vec<_> = op_counts
            .keys()
            .filter(|k| !CLIFFORD_T_GATE_NAMES.contains(&k.as_str()))
            .collect();

        return Err(TranspilerError::new_err(format!(
            "Unable to run Clifford+T optimization as the circuit contains gates not supported by the pass: {:?}",
            unsupported
        )));
    }

    let runs: Vec<Vec<NodeIndex>> = dag.collect_1q_runs().unwrap().collect();

    for raw_run in runs {
        let optimized_sequence = optimize_clifford_t_1q(dag, &raw_run);
        match optimized_sequence {
            Some((optimized_sequence, global_phase_update)) => {
                for gate in optimized_sequence {
                    dag.insert_1q_on_incoming_qubit((gate, &[]), raw_run[0]);
                }

                dag.add_global_phase(&Param::Float(global_phase_update))?;
                dag.remove_1q_sequence(&raw_run);
            }
            None => {
                // No reductions were performed.
            }
        }
    }
    Ok(())
}

pub fn optimize_clifford_t_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_optimize_clifford_t))?;
    Ok(())
}

const CIRCUIT: &[&[StandardGate]; 24] = &[
    &[],
    &[H],
    &[S],
    &[H, S],
    &[S, H],
    &[S, H, S],
    &[X],
    &[H, X],
    &[S, X],
    &[H, S, X],
    &[S, H, X],
    &[S, H, S, X],
    &[Z],
    &[H, Z],
    &[S, Z],
    &[H, S, Z],
    &[S, H, Z],
    &[S, H, S, Z],
    &[Y],
    &[H, Y],
    &[S, Y],
    &[H, S, Y],
    &[S, H, Y],
    &[S, H, S, Y],
];

const EVOLVE_X: &[(Pauli1q, bool); 24] = &[
    (Pauli1q::X, false),
    (Pauli1q::Z, false),
    (Pauli1q::Y, true),
    (Pauli1q::Y, false),
    (Pauli1q::Z, false),
    (Pauli1q::X, false),
    (Pauli1q::X, false),
    (Pauli1q::Z, false),
    (Pauli1q::Y, true),
    (Pauli1q::Y, false),
    (Pauli1q::Z, false),
    (Pauli1q::X, false),
    (Pauli1q::X, true),
    (Pauli1q::Z, true),
    (Pauli1q::Y, false),
    (Pauli1q::Y, true),
    (Pauli1q::Z, true),
    (Pauli1q::X, true),
    (Pauli1q::X, true),
    (Pauli1q::Z, true),
    (Pauli1q::Y, false),
    (Pauli1q::Y, true),
    (Pauli1q::Z, true),
    (Pauli1q::X, true),
];
const EVOLVE_Y: &[(Pauli1q, bool); 24] = &[
    (Pauli1q::Y, false),
    (Pauli1q::Y, true),
    (Pauli1q::X, false),
    (Pauli1q::Z, false),
    (Pauli1q::X, true),
    (Pauli1q::Z, false),
    (Pauli1q::Y, true),
    (Pauli1q::Y, false),
    (Pauli1q::X, true),
    (Pauli1q::Z, true),
    (Pauli1q::X, false),
    (Pauli1q::Z, true),
    (Pauli1q::Y, true),
    (Pauli1q::Y, false),
    (Pauli1q::X, true),
    (Pauli1q::Z, true),
    (Pauli1q::X, false),
    (Pauli1q::Z, true),
    (Pauli1q::Y, false),
    (Pauli1q::Y, true),
    (Pauli1q::X, false),
    (Pauli1q::Z, false),
    (Pauli1q::X, true),
    (Pauli1q::Z, false),
];
const EVOLVE_Z: &[(Pauli1q, bool); 24] = &[
    (Pauli1q::Z, false),
    (Pauli1q::X, false),
    (Pauli1q::Z, false),
    (Pauli1q::X, false),
    (Pauli1q::Y, true),
    (Pauli1q::Y, true),
    (Pauli1q::Z, true),
    (Pauli1q::X, true),
    (Pauli1q::Z, true),
    (Pauli1q::X, true),
    (Pauli1q::Y, false),
    (Pauli1q::Y, false),
    (Pauli1q::Z, false),
    (Pauli1q::X, false),
    (Pauli1q::Z, false),
    (Pauli1q::X, false),
    (Pauli1q::Y, true),
    (Pauli1q::Y, true),
    (Pauli1q::Z, true),
    (Pauli1q::X, true),
    (Pauli1q::Z, true),
    (Pauli1q::X, true),
    (Pauli1q::Y, false),
    (Pauli1q::Y, false),
];
const APPEND_S: &[(usize, usize); 24] = &[
    (2, 0),
    (3, 0),
    (12, 0),
    (13, 0),
    (5, 0),
    (16, 0),
    (20, 0),
    (21, 0),
    (6, 2),
    (7, 2),
    (23, 0),
    (10, 2),
    (14, 0),
    (15, 0),
    (0, 0),
    (1, 0),
    (17, 0),
    (4, 0),
    (8, 4),
    (9, 4),
    (18, 2),
    (19, 2),
    (11, 4),
    (22, 2),
];
const PREPEND_S: &[(usize, usize); 24] = &[
    (2, 0),
    (4, 0),
    (12, 0),
    (5, 0),
    (7, 0),
    (21, 0),
    (8, 0),
    (10, 0),
    (18, 6),
    (11, 0),
    (1, 0),
    (15, 2),
    (14, 0),
    (16, 0),
    (0, 0),
    (17, 0),
    (19, 2),
    (9, 6),
    (20, 0),
    (22, 0),
    (6, 2),
    (23, 0),
    (13, 6),
    (3, 0),
];
const APPEND_H: &[(usize, usize); 24] = &[
    (1, 0),
    (0, 0),
    (4, 0),
    (11, 7),
    (2, 0),
    (15, 1),
    (13, 0),
    (12, 0),
    (16, 0),
    (23, 1),
    (14, 0),
    (3, 1),
    (7, 0),
    (6, 0),
    (10, 0),
    (5, 7),
    (8, 0),
    (21, 7),
    (19, 4),
    (18, 4),
    (22, 4),
    (17, 1),
    (20, 4),
    (9, 7),
];
const PREPEND_H: &[(usize, usize); 24] = &[
    (1, 0),
    (0, 0),
    (3, 0),
    (2, 0),
    (11, 7),
    (10, 1),
    (7, 0),
    (6, 0),
    (9, 0),
    (8, 0),
    (5, 7),
    (4, 1),
    (13, 0),
    (12, 0),
    (15, 0),
    (14, 0),
    (23, 1),
    (22, 3),
    (19, 0),
    (18, 0),
    (21, 0),
    (20, 0),
    (17, 5),
    (16, 7),
];
