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

use crate::commutation_checker::get_matrix;
use crate::gate_metrics::rotation_trace_and_dim;
use crate::target::Target;
use qiskit_circuit::PhysicalQubit;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::imports;
use qiskit_circuit::operations::Operation;
use qiskit_circuit::operations::OperationRef;
use qiskit_circuit::operations::Param;
use qiskit_circuit::operations::StandardGate;
use qiskit_circuit::packed_instruction::PackedInstruction;

const MINIMUM_TOL: f64 = 1e-12;

/// Fidelity-based computation to check whether an operation `G` is equivalent
/// to identity up to a global phase.
///
/// # Arguments
///
/// * `tr_over_dim`: `|Tr(G)| / dim(G)`.
/// * `dim`: `dim(G)`.
/// * `error`: tolerance.
///
/// # Returns
///
/// * `(true, update to the global phase)`` if the operation can be removed.
/// * `(false, 0.)` if the operation cannot be removed.
pub fn can_remove(tr_over_dim: Complex64, dim: f64, error: f64) -> (bool, f64) {
    let f_pro = tr_over_dim.abs().powi(2);
    let gate_fidelity = (dim * f_pro + 1.) / (dim + 1.);
    if (1. - gate_fidelity).abs() < error {
        (true, tr_over_dim.arg())
    } else {
        (false, 0.)
    }
}

/// Fidelity-based computation to check whether an operation `inst` is equivalent
/// to identity up to a global phase.
///
/// # Arguments
///
/// * `inst`: the packed instruction
/// * `matrix_max_num_qubits`: maximum number of qubits allowed for matrix-based checks.
/// * `matrix_use_view_only`: if `true`, the matrix of the operation is retrieved from its view;
///   if `false` also calls the Python-space `Operator` class when the above is not sufficient.
/// * `error_cutoff_fn`: function to compute the allowed error tolerance.
///
/// # Returns
///
/// * `(true, update to the global phase)`` if the operation can be removed.
/// * `(false, 0.)` if the operation cannot be removed.
pub fn is_identity_equiv<F>(
    inst: &PackedInstruction,
    matrix_max_num_qubits: Option<u32>,
    matrix_use_view_only: bool,
    error_cutoff_fn: F,
) -> PyResult<(bool, f64)>
where
    F: Fn(&PackedInstruction) -> f64,
{
    if inst.is_parameterized() {
        // Skip parameterized gates
        return Ok((false, 0.));
    }

    let view = inst.op.view();

    if let OperationRef::StandardGate(gate) = view {
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
                    let (tr_over_dim, dim) = rotation_trace_and_dim(gate, angle).expect(
                        "Since only supported rotation gates are given, the result is not None",
                    );
                    (tr_over_dim, dim)
                } else {
                    return Ok((false, 0.));
                }
            }
            _ => {
                // matrix for standard gates. Julien's comment: we can hardcode but need to support other standard rotation gates in trace computations.
                if let Some(matrix) = gate.matrix(inst.params_view()) {
                    let dim = matrix.shape()[0] as f64;
                    let tr_over_dim = matrix.diag().iter().sum::<Complex64>() / dim;
                    (tr_over_dim, dim)
                } else {
                    return Ok((false, 0.));
                }
            }
        };

        return Ok(can_remove(tr_over_dim, dim, error_cutoff_fn(inst)));
    }

    // Special handling for large pauli rotation gates.
    if let OperationRef::Gate(py_gate) = view {
        let result = Python::attach(|py| -> PyResult<Option<(Complex64, usize)>> {
            let result = imports::PAULI_ROTATION_TRACE_AND_DIM
                .get_bound(py)
                .call1((py_gate.gate.clone_ref(py),))?
                .extract()?;
            Ok(result)
        })?;

        if let Some((tr_over_dim, dim)) = result {
            return Ok(can_remove(tr_over_dim, dim as f64, error_cutoff_fn(inst)));
        }
    }

    // If view.matrix() returns None, then there is no matrix and we skip the operation.
    if let Some(matrix) = get_matrix(
        &view,
        inst.params_view(),
        matrix_max_num_qubits,
        matrix_use_view_only,
    ) {
        let dim = matrix.shape()[0] as f64;
        let tr_over_dim = matrix.diag().iter().sum::<Complex64>() / dim;
        return Ok(can_remove(tr_over_dim, dim, error_cutoff_fn(inst)));
    }

    Ok((false, 0.))
}

#[pyfunction]
#[pyo3(name = "remove_identity_equiv", signature=(dag, approx_degree=Some(1.0), target=None))]
pub fn run_remove_identity_equiv(
    dag: &mut DAGCircuit,
    approx_degree: Option<f64>,
    target: Option<&Target>,
) -> PyResult<()> {
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
        let (can_be_removed, phase_update) = is_identity_equiv(inst, None, true, get_error_cutoff)?;
        if can_be_removed {
            remove_list.push(op_node);
            global_phase_update += phase_update;
        }
    }
    for node in remove_list {
        dag.remove_op_node(node);
    }

    if global_phase_update != 0. {
        dag.add_global_phase(&Param::Float(global_phase_update))
            .expect("The global phase is guaranteed to be a float");
    }

    Ok(())
}

pub fn remove_identity_equiv_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run_remove_identity_equiv))?;
    Ok(())
}
