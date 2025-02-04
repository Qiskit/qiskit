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

use pyo3::prelude::*;
use rayon::prelude::*;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use qiskit_circuit::circuit_instruction::ExtraInstructionAttributes;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::imports::BARRIER;
use qiskit_circuit::operations::{Operation, PyInstruction};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::Qubit;

const PARALLEL_THRESHOLD: usize = 150;

#[pyfunction]
#[pyo3(signature=(dag, label=None))]
pub fn barrier_before_final_measurements(
    py: Python,
    dag: &mut DAGCircuit,
    label: Option<String>,
) -> PyResult<()> {
    let find_final_nodes = |[_in_index, out_index]: &[NodeIndex; 2]| -> Vec<NodeIndex> {
        let mut next_nodes: Vec<NodeIndex> = dag
            .quantum_predecessors(*out_index)
            .filter(|index| {
                let node = &dag[*index];
                match node {
                    NodeType::Operation(inst) => {
                        if inst.op.name() == "measure" || inst.op.name() == "barrier" {
                            dag.bfs_successors(*index).all(|(_, child_successors)| {
                                child_successors.iter().all(|suc| match &dag[*suc] {
                                    NodeType::Operation(suc_inst) => {
                                        suc_inst.op.name() == "measure"
                                            || suc_inst.op.name() == "barrier"
                                    }
                                    _ => true,
                                })
                            })
                        } else {
                            false
                        }
                    }
                    _ => false,
                }
            })
            .collect();
        let mut nodes: Vec<NodeIndex> = Vec::new();
        while let Some(node_index) = next_nodes.pop() {
            if node_index != *out_index
                && dag.bfs_successors(node_index).all(|(_, child_successors)| {
                    child_successors.iter().all(|suc| match &dag[*suc] {
                        NodeType::Operation(suc_inst) => {
                            suc_inst.op.name() == "measure" || suc_inst.op.name() == "barrier"
                        }
                        _ => true,
                    })
                })
            {
                nodes.push(node_index);
            }
            for pred in dag.quantum_predecessors(node_index) {
                match &dag[pred] {
                    NodeType::Operation(inst) => {
                        if inst.op.name() == "measure" || inst.op.name() == "barrier" {
                            next_nodes.push(pred)
                        }
                    }
                    _ => continue,
                }
            }
        }
        nodes.reverse();
        nodes
    };

    let final_ops: Vec<NodeIndex> =
        if dag.num_qubits() >= PARALLEL_THRESHOLD && crate::getenv_use_multiple_threads() {
            dag.qubit_io_map()
                .par_iter()
                .flat_map(find_final_nodes)
                .collect()
        } else {
            dag.qubit_io_map()
                .iter()
                .flat_map(find_final_nodes)
                .collect()
        };

    if final_ops.is_empty() {
        return Ok(());
    }
    let final_packed_ops: Vec<PackedInstruction> = final_ops
        .into_iter()
        .filter_map(|node| match dag.dag().node_weight(node) {
            Some(weight) => {
                let NodeType::Operation(_) = weight else {
                    return None;
                };
                let res = dag.remove_op_node(node);
                Some(res)
            }
            None => None,
        })
        .collect();
    let new_barrier = BARRIER
        .get_bound(py)
        .call1((dag.num_qubits(), label.as_deref()))?;

    let new_barrier_py_inst = PyInstruction {
        qubits: dag.num_qubits() as u32,
        clbits: 0,
        params: 0,
        op_name: "barrier".to_string(),
        control_flow: false,
        #[cfg(feature = "cache_pygates")]
        instruction: new_barrier.clone().unbind(),
        #[cfg(not(feature = "cache_pygates"))]
        instruction: new_barrier.unbind(),
    };
    let qargs: Vec<Qubit> = (0..dag.num_qubits() as u32).map(Qubit).collect();
    #[cfg(feature = "cache_pygates")]
    {
        dag.apply_operation_back(
            py,
            PackedOperation::from_instruction(Box::new(new_barrier_py_inst)),
            qargs.as_slice(),
            &[],
            None,
            ExtraInstructionAttributes::new(label, None, None, None),
            Some(new_barrier.unbind()),
        )?;
    }
    #[cfg(not(feature = "cache_pygates"))]
    {
        dag.apply_operation_back(
            py,
            PackedOperation::from_instruction(Box::new(new_barrier_py_inst)),
            qargs.as_slice(),
            &[],
            None,
            ExtraInstructionAttributes::new(label, None, None, None),
        )?;
    }
    for inst in final_packed_ops {
        dag.push_back(py, inst)?;
    }
    Ok(())
}

pub fn barrier_before_final_measurements_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(barrier_before_final_measurements))?;
    Ok(())
}
