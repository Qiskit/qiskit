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
use num_complex::Complex64;
use num_complex::ComplexFloat;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use crate::gate_metrics::rotation_trace_and_dim;
use crate::target::Target;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::getenv_use_multiple_threads;
use qiskit_circuit::operations::Operation;
use qiskit_circuit::operations::OperationRef;
use qiskit_circuit::operations::Param;
use qiskit_circuit::operations::StandardGate;
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::PhysicalQubit;

const MINIMUM_TOL: f64 = 1e-12;

#[pyfunction]
#[pyo3(name = "remove_identity_equiv", signature=(dag, approx_degree=Some(1.0), target=None))]
pub fn py_remove_identity_equiv(
    py: Python,
    dag: &mut DAGCircuit,
    approx_degree: Option<f64>,
    target: Option<&Target>,
) {
    // Explicitly release GIL because threads may call Python to get a
    // the matrix for a PyGate
    py.allow_threads(|| run_remove_identity_equiv(dag, approx_degree, target))
}

pub fn run_remove_identity_equiv(
    dag: &mut DAGCircuit,
    approx_degree: Option<f64>,
    target: Option<&Target>,
) {
    // Minimum threshold to compare average gate fidelity to 1. This is chosen to account
    // for roundoff errors and to be consistent with other places.
    let get_error_cutoff = |inst: &PackedInstruction| -> f64 {
        match approx_degree {
            Some(degree) => {
                if degree == 1.0 {
                    MINIMUM_TOL
                } else {
                    match target {
                        Some(target) => {
                            let qargs: Vec<PhysicalQubit> = dag
                                .get_qargs(inst.qubits)
                                .iter()
                                .map(|x| PhysicalQubit::new(x.0))
                                .collect();
                            let error_rate = target.get_error(inst.op.name(), &qargs);
                            match error_rate {
                                Some(err) => err * degree,
                                None => MINIMUM_TOL.max(1. - degree),
                            }
                        }
                        None => MINIMUM_TOL.max(1. - degree),
                    }
                }
            }
            None => match target {
                Some(target) => {
                    let qargs: Vec<PhysicalQubit> = dag
                        .get_qargs(inst.qubits)
                        .iter()
                        .map(|x| PhysicalQubit::new(x.0))
                        .collect();
                    let error_rate = target.get_error(inst.op.name(), &qargs);
                    match error_rate {
                        Some(err) => err,
                        None => MINIMUM_TOL,
                    }
                }
                None => MINIMUM_TOL,
            },
        }
    };

    let process_node = |op_node: NodeIndex, inst: &PackedInstruction| {
        if inst.is_parameterized() {
            // Skip parameterized gates
            return None;
        }
        let view = inst.op.view();
        match view {
            OperationRef::StandardGate(gate) => {
                let (tr_over_dim, dim) = match gate {
                    StandardGate::RX
                    | StandardGate::RY
                    | StandardGate::RZ
                    | StandardGate::Phase
                    | StandardGate::RXX
                    | StandardGate::RYY
                    | StandardGate::RZX
                    | StandardGate::RZZ
                    | StandardGate::CRX
                    | StandardGate::CRY
                    | StandardGate::CRZ
                    | StandardGate::CPhase => {
                        if let Param::Float(angle) = inst.params_view()[0] {
                            let (tr_over_dim, dim) =
                                rotation_trace_and_dim(gate, angle).expect("Since only supported rotation gates are given, the result is not None");
                            (tr_over_dim, dim)
                        } else {
                            return None;
                        }
                    }
                    _ => {
                        if let Some(matrix) = gate.matrix(inst.params_view()) {
                            let dim = matrix.shape()[0] as f64;
                            let tr_over_dim = matrix.diag().iter().sum::<Complex64>() / dim;
                            (tr_over_dim, dim)
                        } else {
                            return None;
                        }
                    }
                };
                let error = get_error_cutoff(inst);
                let f_pro = tr_over_dim.abs().powi(2);
                let gate_fidelity = (dim * f_pro + 1.) / (dim + 1.);
                if (1. - gate_fidelity).abs() < error {
                    return Some((op_node, tr_over_dim.arg()));
                }
            }
            _ => {
                let matrix = view.matrix(inst.params_view());
                // If view.matrix() returns None, then there is no matrix and we skip the operation.
                if let Some(matrix) = matrix {
                    let error = get_error_cutoff(inst);
                    let dim = matrix.shape()[0] as f64;
                    let tr_over_dim = matrix.diag().iter().sum::<Complex64>() / dim;
                    let f_pro = tr_over_dim.abs().powi(2);
                    let gate_fidelity = (dim * f_pro + 1.) / (dim + 1.);
                    if (1. - gate_fidelity).abs() < error {
                        return Some((op_node, tr_over_dim.arg()));
                    }
                }
            }
        }
        None
    };
    let run_in_parallel = getenv_use_multiple_threads();
    let remove_list: Vec<(NodeIndex, f64)> = if dag.num_ops() >= 5e4 as usize && run_in_parallel {
        let node_indices = dag.dag().node_indices().collect::<Vec<_>>();
        node_indices
            .into_par_iter()
            .filter_map(|index| {
                if let NodeType::Operation(ref inst) = dag.dag()[index] {
                    process_node(index, inst)
                } else {
                    None
                }
            })
            .collect()
    } else {
        dag.op_nodes(false)
            .filter_map(|x| process_node(x.0, x.1))
            .collect()
    };

    for (node, phase_update) in remove_list {
        dag.remove_op_node(node);
        dag.add_global_phase(&Param::Float(phase_update))
            .expect("The global phase is guaranteed to be a float");
    }
}

pub fn remove_identity_equiv_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_remove_identity_equiv))?;
    Ok(())
}
