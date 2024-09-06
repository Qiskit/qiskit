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

use crate::commutation_analysis::analyze_commutations_inner;
use crate::commutation_checker::CommutationChecker;
use crate::target_transpiler::Target;
use hashbrown::{HashMap, HashSet};
use pyo3::prelude::*;
use pyo3::{pyfunction, pymodule, wrap_pyfunction, Bound, PyResult, Python};
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType, Wire};
use qiskit_circuit::operations::StandardGate::{PhaseGate, RXGate, RZGate, U1Gate};
use qiskit_circuit::operations::{Operation, Param, StandardGate};
use qiskit_circuit::Qubit;
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use std::f64::consts::PI;

const _CUTOFF_PRECISION: f64 = 1e-5;
static ROTATION_GATES: [&str; 4] = ["p", "u1", "rz", "rx"];
static HALF_TURNS: [&str; 2] = ["z", "x"];
static QUARTER_TURNS: [&str; 1] = ["s"];
static EIGHTH_TURNS: [&str; 1] = ["t"];

const Z_ROTATION: &str = "z_rotation";
const X_ROTATION: &str = "x_rotation";

#[pyfunction]
#[pyo3(signature = (dag, commutation_checker, basis_gates=None, target=None))]
pub(crate) fn cancel_commutations(
    py: Python,
    dag: &mut DAGCircuit,
    commutation_checker: &mut CommutationChecker,
    basis_gates: Option<HashSet<String>>,
    target: Option<&Target>,
) -> PyResult<()> {
    let basis: HashSet<String> = if let Some(tar) = target {
        HashSet::from_iter(tar.operation_names().map(String::from))
    } else if let Some(basis) = basis_gates {
        basis
    } else {
        HashSet::new()
    };

    let _var_z_map: HashMap<&str, StandardGate> =
        HashMap::from([("rz", RZGate), ("p", PhaseGate), ("u1", U1Gate)]);

    let _z_rotations: HashSet<&str> = HashSet::from(["p", "z", "u1", "rz", "t", "s"]);
    let _x_rotations: HashSet<&str> = HashSet::from(["x", "rx"]);
    let _gates: HashSet<&str> = HashSet::from(["cx", "cy", "cz", "h", "y"]);

    let z_var_gate = dag
        .op_names
        .keys()
        .find(|g| _var_z_map.contains_key(g.as_str()))
        // Fallback to the first matching key from basis if there is no match in dag.op_names
        .or_else(|| basis.iter().find(|g| _var_z_map.contains_key(g.as_str())))
        // get the StandardGate associated with that string
        .and_then(|key| _var_z_map.get(key.as_str()));

    // Gate sets to be cancelled
    /* Traverse each qubit to generate the cancel dictionaries
     Cancel dictionaries:
      - For 1-qubit gates the key is (gate_type, qubit_id, commutation_set_id),
        the value is the list of gates that share the same gate type, qubit, commutation set.
      - For 2qbit gates the key: (gate_type, first_qbit, sec_qbit, first commutation_set_id,
        sec_commutation_set_id), the value is the list gates that share the same gate type,
        qubits and commutation sets.
    */
    let (commutation_set, node_indices) = analyze_commutations_inner(py, dag, commutation_checker)?;
    let mut single_q_cancellation_sets: HashMap<(String, Qubit, usize), Vec<NodeIndex>> =
        HashMap::new();
    let mut two_q_cancellation_sets: HashMap<(String, Qubit, Qubit, usize, usize), Vec<NodeIndex>> =
        HashMap::new();

    (0..dag.num_qubits() as u32).for_each(|qubit| {
        let wire = Qubit(qubit);
        if let Some(wire_commutation_set) = commutation_set.get(&Wire::Qubit(wire)) {
            wire_commutation_set
                .iter()
                .enumerate()
                .for_each(|(com_set_idx, com_set)| {
                    // This ensures that we only have DAGOPNodes in the current com_set, yuck...
                    if let NodeType::Operation(_node0) = &dag.dag[*com_set.first().unwrap()] {
                        com_set.iter().for_each(|node| {
                            let op = match &dag.dag[*node] {
                                NodeType::Operation(instr) => instr,
                                _ => panic!("Unexpected type in commutation set."),
                            };
                            let num_qargs = dag.get_qargs(op.qubits).len();
                            // no support for cancellation of parameterized gates
                            if op
                                .params_view()
                                .iter()
                                .all(|p| !matches!(p, Param::ParameterExpression(_)))
                            {
                                let op_name = op.op.name().to_string();
                                if num_qargs == 1usize && _gates.contains(op_name.as_str()) {
                                    single_q_cancellation_sets
                                        .entry((op_name.clone(), wire, com_set_idx))
                                        .or_insert_with(Vec::new)
                                        .push(*node);
                                }

                                if num_qargs == 1usize && _z_rotations.contains(op_name.as_str()) {
                                    single_q_cancellation_sets
                                        .entry((Z_ROTATION.to_string(), wire, com_set_idx))
                                        .or_insert_with(Vec::new)
                                        .push(*node);
                                }
                                if num_qargs == 1usize && _x_rotations.contains(op_name.as_str()) {
                                    single_q_cancellation_sets
                                        .entry((X_ROTATION.to_string(), wire, com_set_idx))
                                        .or_insert_with(Vec::new)
                                        .push(*node);
                                }
                                // Don't deal with Y rotation, because Y rotation doesn't commute with
                                // CNOT, so it should be dealt with by optimized1qgate pass
                                if num_qargs == 2usize
                                    && dag.get_qargs(op.qubits).first().unwrap() == &wire
                                {
                                    let second_qarg = dag.get_qargs(op.qubits)[1];
                                    let q2_key = (
                                        op_name,
                                        wire,
                                        second_qarg,
                                        com_set_idx,
                                        *node_indices
                                            .get(&(*node, Wire::Qubit(second_qarg)))
                                            .unwrap(),
                                    );
                                    two_q_cancellation_sets
                                        .entry(q2_key)
                                        .or_insert_with(Vec::new)
                                        .push(*node);
                                }
                            }
                        })
                    }
                })
        }
    });

    for (cancel_key, cancel_set) in &two_q_cancellation_sets {
        if cancel_set.len() > 1 && _gates.contains(cancel_key.0.as_str()) {
            for &c_node in &cancel_set[0..(cancel_set.len() / 2) * 2] {
                dag.remove_op_node(c_node);
            }
        }
    }

    for (cancel_key, cancel_set) in &single_q_cancellation_sets {
        if cancel_key.0 == Z_ROTATION && z_var_gate.is_none() {
            continue;
        }
        if cancel_set.len() > 1 && _gates.contains(cancel_key.0.as_str()) {
            for &c_node in &cancel_set[0..(cancel_set.len() / 2) * 2] {
                dag.remove_op_node(c_node);
            }
        } else if cancel_set.len() > 1 && (cancel_key.0 == Z_ROTATION || cancel_key.0 == X_ROTATION)
        {
            let run_op = match &dag.dag[*cancel_set.first().unwrap()] {
                NodeType::Operation(instr) => instr,
                _ => panic!("Unexpected type in commutation set run."),
            };

            let run_qarg = dag.get_qargs(run_op.qubits).first().unwrap();
            let mut total_angle: f64 = 0.0f64;
            let mut total_phase: f64 = 0.0f64;
            for current_node in cancel_set {
                let node_op = match &dag.dag[*current_node] {
                    NodeType::Operation(instr) => instr,
                    _ => panic!("Unexpected type in commutation set run."),
                };
                let node_op_name = node_op.op.name();

                let node_qargs = dag.get_qargs(node_op.qubits);
                if node_op
                    .extra_attrs
                    .as_deref()
                    .is_some_and(|attr| attr.condition.is_some())
                    || node_qargs.len() > 1
                    || &node_qargs[0] != run_qarg
                {
                    panic!("internal error");
                }

                let node_angle = if ROTATION_GATES.contains(&node_op_name) {
                    match node_op.params_view().first() {
                        Some(Param::Float(f)) => *f,
                        _ => panic!(
                            "Rotational gate with parameter expression encoutned in cancellation"
                        ),
                    }
                } else if HALF_TURNS.contains(&node_op_name) {
                    PI
                } else if QUARTER_TURNS.contains(&node_op_name) {
                    PI / 2.0
                } else if EIGHTH_TURNS.contains(&node_op_name) {
                    PI / 4.0
                } else {
                    panic!("Angle for operation {node_op_name} is not defined")
                };
                total_angle += node_angle;

                if let Some(definition) = node_op.op.definition(node_op.params_view()) {
                    //TODO check for PyNone global phase?
                    //total_phase += match definition.global_phase() {Param::Float(f) => f, Param::Obj(pyop) => , Param::ParameterExpression(_) => panic!("PackedInstruction with definition has global phase set as parameter expression")};
                    total_phase += match definition.global_phase() {Param::Float(f) => f, _ => panic!("PackedInstruction with definition has no global phase set as floating point number")};
                }
            }

            let new_op = if cancel_key.0 == Z_ROTATION {
                z_var_gate.unwrap()
            } else if cancel_key.0 == X_ROTATION {
                &RXGate
            } else {
                panic!("impossible case!");
            };

            let gate_angle = mod_2pi(total_angle, 0.);

            let new_op_phase: f64 = if gate_angle.abs() > _CUTOFF_PRECISION {
                let new_index = dag.insert_1q_on_incoming_qubit(
                    (*new_op, &[total_angle]),
                    *cancel_set.first().unwrap(),
                );
                let new_node = match &dag.dag[new_index] {
                    NodeType::Operation(instr) => instr,
                    _ => panic!("Unexpected type in commutation set run."),
                };

                if let Some(definition) = new_node.op.definition(new_node.params_view()) {
                    //TODO check for PyNone global phase?
                    match definition.global_phase() {Param::Float(f) => *f, _ => panic!("PackedInstruction with definition has no global phase set as floating point number")}
                } else {
                    0.0
                }
            } else {
                0.0
            };

            dag.add_global_phase(py, &Param::Float(total_phase - new_op_phase))?;

            for node in cancel_set {
                dag.remove_op_node(*node);
            }

            //TODO do we need this due to numerical instability?
            /*
                if np.mod(total_angle, (2 * np.pi)) < _CUTOFF_PRECISION:
                    dag.remove_op_node(run[0])
            */
        }
    }

    Ok(())
}

/// Wrap angle into interval [-π,π). If within atol of the endpoint, clamp to -π
#[inline]
fn mod_2pi(angle: f64, atol: f64) -> f64 {
    // f64::rem_euclid() isn't exactly the same as Python's % operator, but because
    // the RHS here is a constant and positive it is effectively equivalent for
    // this case
    let wrapped = (angle + PI).rem_euclid(2. * PI) - PI;
    if (wrapped - PI).abs() < atol {
        -PI
    } else {
        wrapped
    }
}

#[pymodule]
pub fn commutation_cancellation(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(cancel_commutations))?;
    Ok(())
}
