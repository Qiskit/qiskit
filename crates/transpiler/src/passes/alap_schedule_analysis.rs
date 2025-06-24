use crate::TranspilerError;
use hashbrown::HashMap;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType, Wire};
use qiskit_circuit::operations::{Operation, OperationRef};
use qiskit_circuit::{Clbit, Qubit};
use rustworkx_core::petgraph::prelude::NodeIndex;

#[pyfunction]
#[pyo3(name= "alap_schedule_analysis", signature=(dag, clbit_write_latency, node_durations))]
pub fn run_alap_schedule_analysis(
    py: Python,
    dag: &DAGCircuit,
    clbit_write_latency: u64,
    node_durations: &Bound<PyDict>,
) -> PyResult<Py<PyDict>> {
    if dag.qregs().len() != 1 || !dag.qregs().iter().any(|reg| reg.name() == "q") {
        return Err(TranspilerError::new_err(
            "ALAP schedule runs on physical circuits only",
        ));
    }

    let mut node_start_time: HashMap<NodeIndex, f64> = HashMap::new();
    let mut idle_before: HashMap<Wire, f64> = HashMap::new();

    for (i, _) in dag.qubits().objects().iter().enumerate() {
        idle_before.insert(Wire::Qubit(Qubit::new(i)), 0.0);
    }

    for (i, _) in dag.clbits().objects().iter().enumerate() {
        idle_before.insert(Wire::Clbit(Clbit::new(i)), 0.0);
    }

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

        let op = match dag.dag().node_weight(node_index) {
            Some(NodeType::Operation(op)) => op,
            _ => continue,
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

        /*
         compute t0, t1: instruction interval, note that
         t0: start time of instruction
         t1: end time of instruction

         since this is alap scheduling, node is scheduled in reversed topological ordering
         and nodes are packed from the very end of the circuit.
         the physical meaning of t0 and t1 is flipped here.
        */

        // Get operation type
        let op_name = op.op.name();
        let op_view = op.op.view();
        let (is_gate, is_delay) = match op_view {
            OperationRef::Gate(_) => (true, false),
            OperationRef::StandardGate(_) => (true, false),
            _ => (false, op_name == "delay"),
        };

        let t1 = if is_gate || is_delay {
            // Gate or Delay operation
            let t0 = qargs
                .iter()
                .map(|q| *idle_before.get(q).unwrap_or(&0.0))
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);
            t0 + op_duration
        } else if op_name == "measure" {
            // Measure operation
            // clbit time is always right (alap) justified
            let t0 = qargs
                .iter()
                .chain(cargs.iter())
                .map(|bit| *idle_before.get(bit).unwrap_or(&0.0))
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);
            /*
                       |t1 = t0 + duration
                Q ░░░░░▒▒▒▒▒▒▒▒▒▒▒
                C ░░░░░░░░░▒▒▒▒▒▒▒
                           |t0 + (duration - clbit_write_latency)
            */
            let t1 = t0 + op_duration;
            for clbit in cargs.iter() {
                idle_before.insert(*clbit, t0 + (op_duration - clbit_write_latency as f64));
            }
            t1
        } else {
            // Directive (like barrier)
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
    // Note that ALAP pass is inversely schedule, thus
    // t0 is computed by subtracting entire circuit duration from t1.
    let node_start_time_final: HashMap<NodeIndex, f64> = node_start_time
        .into_iter()
        .map(|(n, t1)| (n, circuit_duration - t1))
        .collect();

    // Create a Python dictionary to return
    let py_dict = PyDict::new(py);
    for (node_idx, time) in node_start_time_final {
        let node = dag.get_node(py, node_idx)?;
        if time.fract() == 0.0 {
            py_dict.set_item(node, time as u64)?;
        } else {
            py_dict.set_item(node, time)?;
        }
    }

    Ok(py_dict.into())
}

pub fn alap_schedule_analysis_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_alap_schedule_analysis))?;
    Ok(())
}
