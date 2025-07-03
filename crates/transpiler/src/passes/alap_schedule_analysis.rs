use crate::TranspilerError;
use hashbrown::HashMap;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType, Wire};
use qiskit_circuit::operations::{Operation, OperationRef, StandardInstruction};
use qiskit_circuit::{Clbit, Qubit};
use rustworkx_core::petgraph::prelude::NodeIndex;

#[pyfunction]
#[pyo3(name = "alap_schedule_analysis")]
pub fn run_alap_schedule_analysis(
    py: Python,
    dag: &DAGCircuit,
    clbit_write_latency: u64,
    durations: HashMap<(String, Option<Vec<usize>>), (f64, String)>,
) -> PyResult<Py<PyDict>> {
    if dag.qregs().len() != 1 || !dag.qregs_data().contains_key("q") {
        return Err(TranspilerError::new_err(
            "ALAP schedule runs on physical circuits only",
        ));
    }

    let mut node_start_time: HashMap<NodeIndex, (f64, String)> = HashMap::new();
    let mut idle_before: HashMap<Wire, f64> = HashMap::new();

    for index in 0..dag.qubits().len() {
        idle_before.insert(Wire::Qubit(Qubit::new(index)), 0.0);
    }
    for index in 0..dag.clbits().len() {
        idle_before.insert(Wire::Clbit(Clbit::new(index)), 0.0);
    }

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

        let op_name = op.op.name().to_string();
        let qubit_indices: Vec<usize> = qargs
            .iter()
            .filter_map(|w| match w {
                Wire::Qubit(q) => Some(q.index()),
                _ => None,
            })
            .collect();
        let (op_duration, unit) = match op_name.as_str() {
            "barrier" => (0.0, "dt".to_string()),
            _ => {
                if let Some(&(dur, ref unit)) =
                    durations.get(&(op_name.clone(), Some(qubit_indices.clone())))
                {
                    (dur, unit.clone())
                } else if let Some(&(dur, ref unit)) = durations.get(&(op_name.clone(), None)) {
                    (dur, unit.clone())
                } else {
                    return Err(TranspilerError::new_err(format!(
                        "No duration for operation {} on qubits {:?} in durations",
                        op_name, qubit_indices
                    )));
                }
            }
        };

        let is_gate_or_delay = matches!(
            op.op.view(),
            OperationRef::Gate(_)
                | OperationRef::StandardGate(_)
                | OperationRef::StandardInstruction(StandardInstruction::Delay(_))
        );

        let t1 = if is_gate_or_delay {
            let t0 = qargs
                .iter()
                .map(|q| *idle_before.get(q).unwrap_or(&0.0))
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);
            t0 + op_duration
        } else if op_name == "measure" {
            let t0 = qargs
                .iter()
                .chain(cargs.iter())
                .map(|bit| *idle_before.get(bit).unwrap_or(&0.0))
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0);
            let t1 = t0 + op_duration;
            for clbit in cargs.iter() {
                idle_before.insert(*clbit, t0 + (op_duration - clbit_write_latency as f64));
            }
            t1
        } else {
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

        node_start_time.insert(node_index, (t1, unit));
    }

    let circuit_duration = *idle_before
        .values()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(&0.0);

    let py_dict = PyDict::new(py);
    for (node_idx, (t1, unit)) in node_start_time {
        let node = dag.get_node(py, node_idx)?;
        let time = circuit_duration - t1;
        // Cast to integer if unit is "dt"
        if unit == "dt" {
            // Always round, cast to i64, and use .into() to ensure Python sees a true int
            py_dict.set_item(node, time.round() as i64)?;
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
