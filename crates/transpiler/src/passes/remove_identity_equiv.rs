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
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use crate::target::Target;
use qiskit_accelerate::gate_metrics::rotation_trace_and_dim;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::operations::Operation;
use qiskit_circuit::operations::OperationRef;
use qiskit_circuit::operations::Param;
use qiskit_circuit::operations::StandardGate;
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::PhysicalQubit;

const MINIMUM_TOL: f64 = 1e-12;

#[pyfunction]
#[pyo3(name = "remove_identity_equiv", signature=(dag, approx_degree=Some(1.0), target=None))]
pub fn run_remove_identity_equiv(
    dag: &mut DAGCircuit,
    approx_degree: Option<f64>,
    target: Option<&Target>,
) {
    let mut remove_list: Vec<NodeIndex> = Vec::new();
    let mut global_phase_update: f64 = 0.;
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

    for (op_node, inst) in dag.op_nodes(false) {
        if inst.is_parameterized() {
            // Skip parameterized gates
            continue;
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
                            continue;
                        }
                    }
                    _ => {
                        if let Some(matrix) = gate.matrix(inst.params_view()) {
                            let dim = matrix.shape()[0] as f64;
                            let tr_over_dim = matrix.diag().iter().sum::<Complex64>() / dim;
                            (tr_over_dim, dim)
                        } else {
                            continue;
                        }
                    }
                };
                let error = get_error_cutoff(inst);
                let f_pro = tr_over_dim.abs().powi(2);
                let gate_fidelity = (dim * f_pro + 1.) / (dim + 1.);
                if (1. - gate_fidelity).abs() < error {
                    remove_list.push(op_node);
                    global_phase_update += tr_over_dim.arg();
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
                        remove_list.push(op_node);
                        global_phase_update += tr_over_dim.arg();
                    }
                }
            }
        }
    }
    for node in remove_list {
        dag.remove_op_node(node);
    }

    if global_phase_update != 0. {
        dag.add_global_phase(&Param::Float(global_phase_update))
            .expect("The global phase is guaranteed to be a float");
    }
}

pub fn remove_identity_equiv_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_remove_identity_equiv))?;
    Ok(())
}
