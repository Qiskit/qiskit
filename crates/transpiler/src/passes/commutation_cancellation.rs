// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::{Bound, PyResult, pyfunction, wrap_pyfunction};

use indexmap::IndexMap;
use smallvec::{SmallVec, smallvec};

use super::analyze_commutations;
use crate::commutation_checker::CommutationChecker;
use qiskit_circuit::Qubit;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType, Wire};
use qiskit_circuit::operations::{Operation, Param, StandardGate};
use qiskit_synthesis::QiskitError;

const _CUTOFF_PRECISION: f64 = 1e-5;
static ROTATION_GATES: [&str; 4] = ["p", "u1", "rz", "rx"];

static VAR_Z_MAP: [(&str, StandardGate); 3] = [
    ("rz", StandardGate::RZ),
    ("p", StandardGate::Phase),
    ("u1", StandardGate::U1),
];
static Z_ROTATIONS: [StandardGate; 6] = [
    StandardGate::Phase,
    StandardGate::Z,
    StandardGate::U1,
    StandardGate::RZ,
    StandardGate::T,
    StandardGate::S,
];
static X_ROTATIONS: [StandardGate; 2] = [StandardGate::X, StandardGate::RX];
static SUPPORTED_GATES: [StandardGate; 5] = [
    StandardGate::CX,
    StandardGate::CY,
    StandardGate::CZ,
    StandardGate::H,
    StandardGate::Y,
];

#[derive(Hash, Eq, PartialEq, Debug)]
enum GateOrRotation {
    Gate(StandardGate),
    ZRotation,
    XRotation,
}
#[derive(Hash, Eq, PartialEq, Debug)]
struct CancellationSetKey {
    gate: GateOrRotation,
    qubits: SmallVec<[Qubit; 2]>,
    com_set_index: usize,
    second_index: Option<usize>,
}

#[pyfunction]
#[pyo3(signature = (dag, commutation_checker, basis_gates=None, approximation_degree=1.))]
pub fn cancel_commutations(
    dag: &mut DAGCircuit,
    commutation_checker: &mut CommutationChecker,
    basis_gates: Option<Vec<String>>,
    approximation_degree: f64,
) -> PyResult<()> {
    let basis = basis_gates.unwrap_or_default();
    let z_var_gate = dag
        .get_op_counts()
        .keys()
        .find_map(|g| {
            VAR_Z_MAP
                .iter()
                .find(|(key, _)| *key == g.as_str())
                .map(|(_, gate)| gate)
        })
        .or_else(|| {
            // Fallback to the first matching key from basis if there is no match in dag.op_names
            basis.iter().find_map(|g| {
                VAR_Z_MAP
                    .iter()
                    .find(|(key, _)| *key == g.as_str())
                    .map(|(_, gate)| gate)
            })
        });

    // RZ and P/U1 have a phase difference of angle/2, which we need to account for
    let z_phase_shift = match z_var_gate {
        Some(z_var_gate) => {
            if z_var_gate == &StandardGate::RZ {
                |gate_name: &str, angle: f64| -> f64 {
                    if ["u1", "p"].contains(&gate_name) {
                        angle / 2.
                    } else {
                        0.
                    }
                }
            } else {
                |gate_name: &str, angle: f64| -> f64 {
                    if gate_name == "rz" { -angle / 2. } else { 0. }
                }
            }
        }
        // if there's no z_var_gate detected, Z rotations are not merged, so we have no phase shifts
        None => |_name: &str, _angle: f64| -> f64 { 0. },
    };

    // Gate sets to be cancelled
    /* Traverse each qubit to generate the cancel dictionaries
     Cancel dictionaries:
      - For 1-qubit gates the key is (gate_type, qubit_id, commutation_set_id),
        the value is the list of gates that share the same gate type, qubit, commutation set.
      - For 2qbit gates the key: (gate_type, first_qbit, sec_qbit, first commutation_set_id,
        sec_commutation_set_id), the value is the list gates that share the same gate type,
        qubits and commutation sets.
    */
    let (commutation_set, node_indices) =
        analyze_commutations(dag, commutation_checker, approximation_degree)?;
    let mut cancellation_sets = IndexMap::with_hasher(::ahash::RandomState::new());

    (0..dag.num_qubits() as u32).for_each(|qubit| {
        let wire = Qubit(qubit);
        if let Some(wire_commutation_set) = commutation_set.get(&Wire::Qubit(wire)) {
            for (com_set_idx, com_set) in wire_commutation_set.iter().enumerate() {
                if let Some(&nd) = com_set.first() {
                    if !matches!(dag[nd], NodeType::Operation(_)) {
                        continue;
                    }
                } else {
                    continue;
                }
                for node in com_set.iter() {
                    let instr = match &dag[*node] {
                        NodeType::Operation(instr) => instr,
                        _ => panic!("Unexpected type in commutation set."),
                    };
                    let num_qargs = dag.get_qargs(instr.qubits).len();
                    // no support for cancellation of parameterized gates
                    if instr.is_parameterized() {
                        continue;
                    }
                    if let Some(op_gate) = instr.op.try_standard_gate() {
                        if num_qargs == 1 && SUPPORTED_GATES.contains(&op_gate) {
                            cancellation_sets
                                .entry(CancellationSetKey {
                                    gate: GateOrRotation::Gate(op_gate),
                                    qubits: smallvec![wire],
                                    com_set_index: com_set_idx,
                                    second_index: None,
                                })
                                .or_insert_with(Vec::new)
                                .push(*node);
                        }

                        if num_qargs == 1 && Z_ROTATIONS.contains(&op_gate) {
                            cancellation_sets
                                .entry(CancellationSetKey {
                                    gate: GateOrRotation::ZRotation,
                                    qubits: smallvec![wire],
                                    com_set_index: com_set_idx,
                                    second_index: None,
                                })
                                .or_insert_with(Vec::new)
                                .push(*node);
                        }
                        if num_qargs == 1 && X_ROTATIONS.contains(&op_gate) {
                            cancellation_sets
                                .entry(CancellationSetKey {
                                    gate: GateOrRotation::XRotation,
                                    qubits: smallvec![wire],
                                    com_set_index: com_set_idx,
                                    second_index: None,
                                })
                                .or_insert_with(Vec::new)
                                .push(*node);
                        }
                        // Don't deal with Y rotation, because Y rotation doesn't commute with
                        // CNOT, so it should be dealt with by optimized1qgate pass
                        if num_qargs == 2 && dag.get_qargs(instr.qubits)[0] == wire {
                            let second_qarg = dag.get_qargs(instr.qubits)[1];
                            cancellation_sets
                                .entry(CancellationSetKey {
                                    gate: GateOrRotation::Gate(op_gate),
                                    qubits: smallvec![wire, second_qarg],
                                    com_set_index: com_set_idx,
                                    second_index: node_indices
                                        .get(&(*node, Wire::Qubit(second_qarg)))
                                        .copied(),
                                })
                                .or_insert_with(Vec::new)
                                .push(*node);
                        }
                    }
                }
            }
        }
    });

    for (cancel_key, cancel_set) in &cancellation_sets {
        if cancel_set.len() > 1 {
            if let GateOrRotation::Gate(g) = cancel_key.gate {
                if SUPPORTED_GATES.contains(&g) {
                    for &c_node in &cancel_set[0..(cancel_set.len() / 2) * 2] {
                        dag.remove_op_node(c_node);
                    }
                }
                continue;
            }
            if matches!(cancel_key.gate, GateOrRotation::ZRotation) && z_var_gate.is_none() {
                continue;
            }
            if matches!(
                cancel_key.gate,
                GateOrRotation::ZRotation | GateOrRotation::XRotation
            ) {
                let mut total_angle: f64 = 0.0;
                let mut total_phase: f64 = 0.0;
                for current_node in cancel_set {
                    let node_op = match &dag[*current_node] {
                        NodeType::Operation(instr) => instr,
                        _ => panic!("Unexpected type in commutation set run."),
                    };
                    let node_op_name = node_op.op.name();

                    let (node_angle, phase_shift) = if ROTATION_GATES.contains(&node_op_name) {
                        let node_angle = match node_op.params_view().first() {
                            Some(Param::Float(f)) => *f,
                            _ => {
                                return Err(QiskitError::new_err(format!(
                                    "Rotational gate with parameter expression encountered in cancellation {:?}",
                                    node_op.op
                                )));
                            }
                        };
                        let phase_shift = z_phase_shift(node_op_name, node_angle);
                        Ok((node_angle, phase_shift))
                    } else {
                        match node_op_name {
                            "t" => Ok((FRAC_PI_4, z_phase_shift("p", FRAC_PI_4))),
                            "s" => Ok((FRAC_PI_2, z_phase_shift("p", FRAC_PI_2))),
                            "z" => Ok((PI, z_phase_shift("p", PI))),
                            "x" => Ok((PI, FRAC_PI_2)),
                            _ => Err(PyRuntimeError::new_err(format!(
                                "Angle for operation {node_op_name} is not defined"
                            ))),
                        }
                    }?;
                    total_angle += node_angle;
                    total_phase += phase_shift;
                }

                let new_op = match cancel_key.gate {
                    GateOrRotation::ZRotation => z_var_gate.unwrap(),
                    GateOrRotation::XRotation => &StandardGate::RX,
                    _ => unreachable!(),
                };

                let pi_multiple = total_angle / PI;

                let mod4 = pi_multiple.rem_euclid(4.);
                if mod4 < _CUTOFF_PRECISION || (4. - mod4) < _CUTOFF_PRECISION {
                    // if the angle is close to a 4-pi multiple (from above or below), then the
                    // operator is equal to the identity
                } else if (mod4 - 2.).abs() < _CUTOFF_PRECISION {
                    // a 2-pi multiple has a phase of pi: RX(2pi) = RZ(2pi) = -I = I exp(i pi)
                    total_phase -= PI;
                } else {
                    // any other is not the identity and we add the gate
                    dag.insert_1q_on_incoming_qubit((*new_op, &[total_angle]), cancel_set[0]);
                }

                dag.add_global_phase(&Param::Float(total_phase))?;

                for node in cancel_set {
                    dag.remove_op_node(*node);
                }
            }
        }
    }

    Ok(())
}

pub fn commutation_cancellation_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(cancel_commutations))?;
    Ok(())
}
