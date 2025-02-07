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

use hashbrown::HashSet;
use pyo3::prelude::*;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use qiskit_circuit::circuit_instruction::ExtraInstructionAttributes;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::operations::{Operation, StandardInstruction};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::Qubit;

static FINAL_OP_NAMES: [&str; 2] = ["measure", "barrier"];

#[pyfunction]
#[pyo3(signature=(dag, label=None))]
pub fn barrier_before_final_measurements(
    py: Python,
    dag: &mut DAGCircuit,
    label: Option<String>,
) -> PyResult<()> {
    let is_exactly_final = |inst: &PackedInstruction| FINAL_OP_NAMES.contains(&inst.op.name());
    let final_ops: HashSet<NodeIndex> = dag
        .op_nodes(true)
        .filter_map(|(node, inst)| {
            if !is_exactly_final(inst) {
                return None;
            }
            dag.bfs_successors(node)
                .all(|(_, child_successors)| {
                    child_successors.iter().all(|suc| match dag[*suc] {
                        NodeType::Operation(ref suc_inst) => is_exactly_final(suc_inst),
                        _ => true,
                    })
                })
                .then_some(node)
        })
        .collect();
    if final_ops.is_empty() {
        return Ok(());
    }
    let ordered_node_indices: Vec<NodeIndex> = dag
        .topological_op_nodes()?
        .filter(|node| final_ops.contains(node))
        .collect();
    let final_packed_ops: Vec<PackedInstruction> = ordered_node_indices
        .into_iter()
        .map(|node| {
            let NodeType::Operation(ref inst) = dag[node] else {
                unreachable!()
            };
            let res = inst.clone();
            dag.remove_op_node(node);
            res
        })
        .collect();
    let qargs: Vec<Qubit> = (0..dag.num_qubits() as u32).map(Qubit).collect();
    dag.apply_operation_back(
        py,
        PackedOperation::from_standard_instruction(StandardInstruction::Barrier(
            dag.num_qubits() as u32
        )),
        qargs.as_slice(),
        &[],
        None,
        ExtraInstructionAttributes::new(label, None, None, None),
        #[cfg(feature = "cache_pygates")]
        None,
    )?;
    for inst in final_packed_ops {
        dag.push_back(py, inst)?;
    }
    Ok(())
}

pub fn barrier_before_final_measurements_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(barrier_before_final_measurements))?;
    Ok(())
}
