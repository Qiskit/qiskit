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

use crate::TranspilerError;
use hashbrown::HashMap;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType, Wire};
use qiskit_circuit::operations::{OperationRef, StandardInstruction};
use qiskit_circuit::{Clbit, Qubit};
use rustworkx_core::petgraph::prelude::NodeIndex;

pub fn run_alap_schedule_analysis(
    dag: &DAGCircuit,
    clbit_write_latency: u64,
    node_durations: HashMap<NodeIndex, f64>,
) -> PyResult<HashMap<NodeIndex, u64>> {
    if dag.qregs().len() != 1 || !dag.qregs_data().contains_key("q") {
        return Err(TranspilerError::new_err(
            "ALAP schedule runs on physical circuits only",
        ));
    }

    let mut node_start_time: HashMap<NodeIndex, f64> = HashMap::new();
    let mut idle_before: HashMap<Wire, f64> = HashMap::new();

    for index in 0..dag.qubits().len() {
        idle_before.insert(Wire::Qubit(Qubit::new(index)), 0.0);
    }

    for index in 0..dag.clbits().len() {
        idle_before.insert(Wire::Clbit(Clbit::new(index)), 0.0);
    }

    // Since this is alap scheduling, node is scheduled in reversed topological ordering
    // and nodes are packed from the very end of the circuit.
    // The physical meaning of t0 and t1 is flipped here.

    for node_index in dag
        .topological_op_nodes()?
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
    {
        let op = match dag.dag().node_weight(node_index) {
            Some(NodeType::Operation(op)) => op,
            _ => panic!("topological_op_nodes() should only return instances of DagOpNode."),
        };

        let qargs: Vec<Wire> = dag
            .qargs_interner()
            .get(op.qubits)
            .iter()
            .map(|&q| Wire::Qubit(q))
            .collect();
        let cargs: Vec<Wire> = dag
            .cargs_interner()
            .get(op.clbits)
            .iter()
            .map(|&c| Wire::Clbit(c))
            .collect();

        let op_duration = *node_durations.get(&node_index).ok_or_else(|| {
            TranspilerError::new_err(format!(
                "No duration for node index {} in node_durations",
                node_index.index()
            ))
        })?;

        let op_view = op.op.view();
        let is_gate_or_delay = matches!(
            op_view,
            OperationRef::Gate(_)
                | OperationRef::StandardGate(_)
                | OperationRef::StandardInstruction(StandardInstruction::Delay(_))
        );

        // compute t0, t1: instruction interval, note that
        // t0: start time of instruction
        // t1: end time of instruction

        let t1 = if is_gate_or_delay {
            // Gate or Delay operation
            let t0 = qargs
                .iter()
                .map(|q| *idle_before.get(q).unwrap_or(&0.0))
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);
            t0 + op_duration
        } else if matches!(
            op_view,
            OperationRef::StandardInstruction(StandardInstruction::Measure)
        ) {
            // Measure operation
            let t0 = qargs
                .iter()
                .chain(cargs.iter())
                .map(|bit| *idle_before.get(bit).unwrap_or(&0.0))
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);

            //           |t1 = t0 + duration
            //    Q ░░░░░▒▒▒▒▒▒▒▒▒▒▒
            //    C ░░░░░░░░░▒▒▒▒▒▒▒
            //               |t0 + (duration - clbit_write_latency)

            let t1 = t0 + op_duration;
            for clbit in cargs.iter() {
                idle_before.insert(*clbit, t0 + (op_duration - clbit_write_latency as f64));
            }
            t1
        } else {
            // Directives (like Barrier)
            let t0 = qargs
                .iter()
                .chain(cargs.iter())
                .map(|bit| *idle_before.get(bit).unwrap_or(&0.0))
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);
            t0 + op_duration
        };

        for qubit in qargs {
            idle_before.insert(qubit, t1);
        }

        node_start_time.insert(node_index, t1);
    }

    // Compute maximum instruction available time
    let circuit_duration = *idle_before
        .values()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(&0.0);

    let mut final_start_times: HashMap<NodeIndex, u64> = HashMap::new();

    // Note that ALAP pass is inversely scheduled, thus
    // t0 is computed by subtracting t1 from the entire circuit duration.
    for (idx, t1) in node_start_time {
        let time = circuit_duration - t1;
        if time.fract() == 0.0 {
            final_start_times.insert(idx, time as u64);
        }
    }

    Ok(final_start_times)
}

#[pyfunction]
/// Runs the ALAPSchedule analysis pass on dag.
///
/// Args:
///     dag (DAGCircuit): DAG to schedule.
///     clbit_write_latency (u64): The latency to write classical bits.
///     node_durations (HashMap<usize, f64>): Mapping from node indices to operation durations.
///
/// Returns:
///     PyDict: A dictionary mapping each DAGOpNode to its scheduled start time.
///
#[pyo3(name = "alap_schedule_analysis", signature= (dag, clbit_write_latency, node_durations))]
pub fn py_run_alap_schedule_analysis(
    py: Python,
    dag: &DAGCircuit,
    clbit_write_latency: u64,
    node_durations: &Bound<PyDict>,
) -> PyResult<Py<PyDict>> {
    // Extract indices and durations from PyDict
    let mut op_durations: HashMap<NodeIndex, f64> = HashMap::new();
    for node_index in dag
        .topological_op_nodes()?
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
    {
        let node = dag.get_node(py, node_index)?;
        let op_duration = match node_durations.get_item(node) {
            Ok(Some(val)) => val.extract::<f64>()?,
            Ok(None) => return Err(TranspilerError::new_err("Node duration not found for node")),
            Err(e) => {
                return Err(TranspilerError::new_err(format!(
                    "PyDict get_item error: {}",
                    e
                )))
            }
        };
        op_durations.insert(node_index, op_duration);
    }

    let final_start_times = run_alap_schedule_analysis(dag, clbit_write_latency, op_durations)?;

    let py_dict = PyDict::new(py);
    for (node_idx, time) in final_start_times {
        let node = dag.get_node(py, node_idx)?;
        py_dict.set_item(node, time)?;
    }

    Ok(py_dict.into())
}

pub fn alap_schedule_analysis_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_alap_schedule_analysis))?;
    Ok(())
}
