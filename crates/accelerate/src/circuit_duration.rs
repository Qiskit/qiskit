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

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType, Wire};
use qiskit_circuit::operations::{Operation, OperationRef, Param};

use crate::nlayout::PhysicalQubit;
use crate::target_transpiler::Target;
use crate::QiskitError;
use rustworkx_core::dag_algo::longest_path;
use rustworkx_core::petgraph::stable_graph::StableDiGraph;
use rustworkx_core::petgraph::visit::{EdgeRef, IntoEdgeReferences};

/// Estimate the duration of a scheduled circuit in seconds
#[pyfunction]
pub(crate) fn compute_estimated_duration(dag: &DAGCircuit, target: &Target) -> PyResult<f64> {
    let dt = target.dt;

    let get_duration =
        |edge: <&StableDiGraph<NodeType, Wire> as IntoEdgeReferences>::EdgeRef| -> PyResult<f64> {
            let node_weight = &dag[edge.target()];
            match node_weight {
                NodeType::Operation(inst) => {
                    let name = inst.op.name();
                    let qubits = dag.get_qargs(inst.qubits);
                    let physical_qubits: Vec<PhysicalQubit> =
                        qubits.iter().map(|x| PhysicalQubit::new(x.0)).collect();

                    if name == "delay" {
                        let dur = &inst.params.as_ref().unwrap()[0];
                        if let OperationRef::Instruction(op) = inst.op.view() {
                            return Python::with_gil(|py| {
                                let unit: String = op
                                    .instruction
                                    .getattr(py, intern!(py, "unit"))?
                                    .extract(py)?;
                                if unit == "dt" {
                                    if let Some(dt) = dt {
                                        match dur {
                                        Param::Float(val) => Ok(val / dt),
                                        Param::Obj(val) => {
                                            let dur_float: f64 = val.extract(py)?;
                                            Ok(dur_float / dt)
                                        },
                                        Param::ParameterExpression(_) => Err(QiskitError::new_err("Circuit contains parameterized delays, can't compute a duration estimate with this circuit")),
                                    }
                                    } else {
                                        Err(QiskitError::new_err("Circuit contains delays in dt but the target doesn't specify dt"))
                                    }
                                } else if unit == "s" {
                                    match dur {
                                        Param::Float(val) => Ok(*val),
                                        _ => Err(QiskitError::new_err(
                                            "Invalid type for parameter value for delay in circuit",
                                        )),
                                    }
                                } else {
                                    Err(QiskitError::new_err("Circuit contains delays in units other then seconds or dt, the circuit is not scheduled."))
                                }
                            });
                        } else {
                            unreachable!("Invalid type for delay instruction");
                        }
                    }

                    match target.get_duration(name, &physical_qubits) {
                        Some(dur) => Ok(dur),
                        None => Err(QiskitError::new_err(format!(
                            "Duration not found for {} on qubits: {:?}",
                            name, qubits
                        ))),
                    }
                }
                NodeType::QubitOut(_) | NodeType::ClbitOut(_) => Ok(0.),
                NodeType::ClbitIn(_) | NodeType::QubitIn(_) => {
                    Err(QiskitError::new_err("Invalid circuit provided"))
                }
                _ => Err(QiskitError::new_err(
                    "Circuit contains Vars, duration can't be calculated with classical variables",
                )),
            }
        };
    match longest_path(dag.dag(), get_duration)? {
        Some((_, weight)) => Ok(weight),
        None => Err(QiskitError::new_err("Invalid circuit provided")),
    }
}

pub fn compute_duration(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(compute_estimated_duration))?;
    Ok(())
}
