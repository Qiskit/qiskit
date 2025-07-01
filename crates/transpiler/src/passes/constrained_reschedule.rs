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

use hashbrown::{HashMap, HashSet};
use crate::TranspilerError;
use pyo3::prelude::*;
use pyo3::{pyfunction, wrap_pyfunction, Bound, PyResult, Python};
use pyo3::types::PyDict;
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use qiskit_circuit::dag_circuit::{DAGCircuit,NodeType, Wire};
use qiskit_circuit::dag_node::DAGOpNode;
use qiskit_circuit::operations::{Operation, OperationRef, StandardInstruction};
use qiskit_circuit::Qubit;

fn push_node_back(py: Python, dag: &DAGCircuit, node_index: NodeIndex, node_start_time: &Bound<PyDict>, clbit_write_latency: u64, pulse_align: u64, acquire_align: u64){
    let op = match dag.dag().node_weight(node_index) {
             Some(NodeType::Operation(op)) => op,
             _ => panic!("topological_op_nodes() should only return instances of DagOpNode."),
         };
    
    let op_view=op.op.view();
    let alignment = match op_view {
        OperationRef::Gate(_) | OperationRef::StandardGate(_) => Some(pulse_align),
        OperationRef::StandardInstruction(StandardInstruction::Reset) |
        OperationRef::StandardInstruction(StandardInstruction::Measure) => Some(acquire_align),
        OperationRef::StandardInstruction(StandardInstruction::Delay(_)) | _ => None,
    };
    let obj = node_start_time.get_item(node_index.index()).unwrap();
    let mut this_t0: f64 = obj
        .as_ref()
        .expect("Expected value in node_start_time for node_index")
        .extract()
        .unwrap();

    if let Some(alignment) = alignment {
        let misalignment = this_t0 % alignment as f64;
        let shift = if misalignment != 0.0 {
        (alignment as f64 - misalignment).max(0.0)
        } else {
        0.0
        };
        this_t0 += shift;
        node_start_time.set_item(node_index.index(), this_t0).unwrap();
    }

    //need to continue from here
    


    
}

#[pyfunction]
#[pyo3(name="constrained_reschedule", signature=(dag, node_start_time, clbit_write_latency, acquire_align, pulse_align))]
pub fn run_constrained_reschedule(
    py: Python,
    dag: &DAGCircuit,
    node_start_time: &Bound<PyDict>,
    clbit_write_latency: u64,
    acquire_align: u64,
    pulse_align: u64
) -> PyResult<Py<PyDict>> {
    for node_index in dag.topological_op_nodes()? {
        // Use node_index.index() (usize) as the key for PyDict lookup
        let start_time= node_start_time.get_item(node_index.index());
        match start_time {
            Ok(Some(obj)) => {
                let val: f64 = obj.extract().map_err(|e| {
                    TranspilerError::new_err(format!(
                        "Failed to extract start time for node index {}: {}", node_index.index(), e
                    ))
                })?;
                if val == 0.0 {
                    continue;
                }
                val
            },
            Ok(None) => {
                return Err(TranspilerError::new_err(format!(
                    "Start time of node at node index {} is not found. This node is likely added after this circuit is scheduled. Run scheduler again.",
                    node_index.index()
                )));
            },
            Err(e) => {
                return Err(TranspilerError::new_err(format!(
                    "PyDict get_item error for node index {}: {}",
                    node_index.index(), e
                )));
            }
        };

        push_node_back(py, dag, node_index, node_start_time, clbit_write_latency, acquire_align, pulse_align);


    }
    // Return an empty dict for now (replace with actual result as needed)
    Ok(PyDict::new(py).into())
}

pub fn constrained_reschedule_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_constrained_reschedule))?;
    Ok(())
}