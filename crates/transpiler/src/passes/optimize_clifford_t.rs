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

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::operations::{OperationRef, StandardGate};
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use crate::TranspilerError;

// List of 1-qubit Clifford+T gate names.
const CLIFFORD_T_GATE_NAMES: &[&str; 18] = &[
    "id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "cx", "cz", "cy", "swap", "iswap", "ecr",
    "dcx", "t", "tdg",
];

#[pyfunction]
#[pyo3(name = "optimize_clifford_t")]
pub fn run_optimize_clifford_t(dag: &mut DAGCircuit) -> PyResult<()> {
    let op_counts = dag.get_op_counts();

    // Skip the pass if there are unsupported gates.
    if !op_counts
        .keys()
        .all(|k| CLIFFORD_T_GATE_NAMES.contains(&k.as_str()))
    {
        let unsupported: Vec<_> = op_counts
            .keys()
            .filter(|k| !CLIFFORD_T_GATE_NAMES.contains(&k.as_str()))
            .collect();

        return Err(TranspilerError::new_err(format!(
            "Unable to run Litinski tranformation as the circuit contains gates not supported by the pass: {:?}",
            unsupported
        )));
    }

    let runs: Vec<Vec<NodeIndex>> = dag.collect_1q_runs().unwrap().collect();

    for raw_run in runs {
        let num_nodes = raw_run.len();

        let mut optimized_sequence = Vec::<StandardGate>::with_capacity(num_nodes);

        let mut idx = 0;
        while idx + 1 < num_nodes {
            let cur_node = &dag[raw_run[idx]];
            let nxt_node = &dag[raw_run[idx + 1]];

            let cur_gate = if let NodeType::Operation(inst) = cur_node {
                if let OperationRef::StandardGate(gate) = inst.op.view() {
                    gate
                } else {
                    unreachable!("Can only have Clifford+T gates at this point");
                }
            } else {
                unreachable!("Can only have op nodes here")
            };

            let nxt_gate = if let NodeType::Operation(inst) = nxt_node {
                if let OperationRef::StandardGate(gate) = inst.op.view() {
                    gate
                } else {
                    unreachable!("Can only have Clifford+T gates at this point");
                }
            } else {
                unreachable!("Can only have op nodes here")
            };

            if cur_gate == StandardGate::T && nxt_gate == StandardGate::T {
                optimized_sequence.push(StandardGate::S);
                idx += 2;
            } else if cur_gate == StandardGate::Tdg && nxt_gate == StandardGate::Tdg {
                optimized_sequence.push(StandardGate::Sdg);
                idx += 2;
            } else {
                optimized_sequence.push(cur_gate);
                idx += 1;
            }
        }

        // Handle the last element (if any)
        if idx + 1 == num_nodes {
            let cur_node = &dag[raw_run[idx]];
            let cur_gate = if let NodeType::Operation(inst) = cur_node {
                if let OperationRef::StandardGate(gate) = inst.op.view() {
                    gate
                } else {
                    unreachable!("Should only have Clifford+T gates at this point");
                }
            } else {
                unreachable!("Can only have op nodes here")
            };
            optimized_sequence.push(cur_gate);
        }

        if optimized_sequence.len() < raw_run.len() {
            for gate in optimized_sequence {
                dag.insert_1q_on_incoming_qubit((gate, &[]), raw_run[0]);
            }

            // dag.add_global_phase(&Param::Float(sequence.global_phase))?;
            dag.remove_1q_sequence(&raw_run);
        }
    }
    Ok(())
}

pub fn optimize_clifford_t_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_optimize_clifford_t))?;
    Ok(())
}
