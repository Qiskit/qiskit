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
use qiskit_circuit::dag_circuit::{DAGCircuit, Wire};
use qiskit_circuit::dag_node::{DAGNode, DAGOpNode};
use qiskit_circuit::operations::{OperationRef, StandardInstruction};
use qiskit_circuit::{Clbit, Qubit};
use rustworkx_core::petgraph::prelude::NodeIndex;
use std::ops::{Add, Sub};

pub trait TimeOps: Copy + PartialOrd + Add<Output = Self> + Sub<Output = Self> {
    fn zero() -> Self;
    fn max<'a>(a: &'a Self, b: &'a Self) -> &'a Self;
}

impl TimeOps for u64 {
    fn zero() -> Self {
        0
    }
    fn max<'a>(a: &'a Self, b: &'a Self) -> &'a Self {
        if a >= b { a } else { b }
    }
}

impl TimeOps for f64 {
    fn zero() -> Self {
        0.0
    }
    fn max<'a>(a: &'a Self, b: &'a Self) -> &'a Self {
        if a >= b { a } else { b }
    }
}

pub fn run_alap_schedule_analysis<T: TimeOps>(
    dag: &DAGCircuit,
    clbit_write_latency: T,
    node_durations: HashMap<NodeIndex, T>,
) -> PyResult<HashMap<NodeIndex, T>> {
    if dag.qregs().len() != 1 || !dag.qregs_data().contains_key("q") {
        return Err(TranspilerError::new_err(
            "ALAP schedule runs on physical circuits only",
        ));
    }

    let mut node_start_time: HashMap<NodeIndex, T> = HashMap::new();
    let mut idle_before: HashMap<Wire, T> = HashMap::new();

    let zero = T::zero();

    for index in 0..dag.qubits().len() {
        idle_before.insert(Wire::Qubit(Qubit::new(index)), zero);
    }

    for index in 0..dag.clbits().len() {
        idle_before.insert(Wire::Clbit(Clbit::new(index)), zero);
    }

    // Since this is alap scheduling, node is scheduled in reversed topological ordering
    // and nodes are packed from the very end of the circuit.
    // The physical meaning of t0 and t1 is flipped here.

    for node_index in dag
        .topological_op_nodes(false)?
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
    {
        let op = dag[node_index].unwrap_operation();

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

        let &op_duration = node_durations
            .get(&node_index)
            .ok_or_else(|| TranspilerError::new_err("No duration for node"))?;

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
            let &t0 = qargs
                .iter()
                .map(|q| idle_before.get(q).unwrap_or(&zero))
                .fold(&zero, |acc, x| T::max(acc, x));
            t0 + op_duration
        } else if matches!(
            op_view,
            OperationRef::StandardInstruction(StandardInstruction::Measure)
        ) {
            // clbit time is always right (alap) justified
            let &t0 = qargs
                .iter()
                .chain(cargs.iter())
                .map(|bit| idle_before.get(bit).unwrap_or(&zero))
                .fold(&zero, |acc, x| T::max(acc, x));

            let t1 = t0 + op_duration;
            //
            //        |t1 = t0 + duration
            // Q ░░░░░▒▒▒▒▒▒▒▒▒▒▒
            // C ░░░░░░░░░▒▒▒▒▒▒▒
            //            |t0 + (duration - clbit_write_latency)

            for clbit in cargs.iter() {
                idle_before.insert(*clbit, t0 + op_duration - clbit_write_latency);
            }
            t1
        } else {
            // Directives (like Barrier)
            let &t0 = qargs
                .iter()
                .chain(cargs.iter())
                .map(|bit| idle_before.get(bit).unwrap_or(&zero))
                .fold(&zero, |acc, x| T::max(acc, x));
            t0 + op_duration
        };

        for qubit in qargs {
            idle_before.insert(qubit, t1);
        }

        node_start_time.insert(node_index, t1);
    }

    // Compute maximum instruction available time
    let circuit_duration = idle_before.values().fold(&zero, |acc, x| T::max(acc, x));

    // Note that ALAP pass is inversely scheduled, thus
    // t0 is computed by subtracting t1 from the entire circuit duration.
    let mut result: HashMap<NodeIndex, T> = HashMap::new();
    for (node_idx, t1) in node_start_time {
        let final_time = *circuit_duration - t1;
        result.insert(node_idx, final_time);
    }

    Ok(result)
}

#[pyfunction]
/// Runs the ALAPSchedule analysis pass on dag.
///
/// Args:
///     dag (DAGCircuit): DAG to schedule.
///     clbit_write_latency (u64): The latency to write classical bits.
///     node_durations (PyDict): Mapping from node indices to operation durations.
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
    // Get the first duration type
    let mut iter = node_durations.iter();
    let py_dict = PyDict::new(py);
    let Some((_, first_duration)) = iter.next() else {
        // Empty circuit.
        return Ok(py_dict.into());
    };
    if first_duration.extract::<u64>().is_ok() {
        // All durations are of type u64
        let mut op_durations = HashMap::new();
        for (py_node, py_duration) in node_durations.iter() {
            let node_idx = py_node
                .cast_into::<DAGOpNode>()?
                .extract::<DAGNode>()?
                .node
                .expect("Node index not found.");
            let val = py_duration.extract::<u64>()?;
            op_durations.insert(node_idx, val);
        }
        let node_start_time =
            run_alap_schedule_analysis::<u64>(dag, clbit_write_latency, op_durations)?;
        for (node_idx, t1) in node_start_time {
            let node = dag.get_node(py, node_idx)?;
            py_dict.set_item(node, t1)?;
        }
    } else if first_duration.extract::<f64>().is_ok() {
        // All durations are of type f64
        let mut op_durations = HashMap::new();
        for (py_node, py_duration) in node_durations.iter() {
            let node_idx = py_node
                .cast_into::<DAGOpNode>()?
                .extract::<DAGNode>()?
                .node
                .expect("Node index not found.");
            let val = py_duration.extract::<f64>()?;
            op_durations.insert(node_idx, val);
        }
        let node_start_time =
            run_alap_schedule_analysis::<f64>(dag, clbit_write_latency as f64, op_durations)?;
        for (node_idx, t1) in node_start_time {
            let node = dag.get_node(py, node_idx)?;
            py_dict.set_item(node, t1)?;
        }
    } else {
        return Err(TranspilerError::new_err("Duration must be int or float"));
    }
    Ok(py_dict.into())
}

pub fn alap_schedule_analysis_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_run_alap_schedule_analysis))?;
    Ok(())
}
