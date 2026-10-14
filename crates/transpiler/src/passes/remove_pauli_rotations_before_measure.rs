// This code is part of Qiskit.
//
// (C) Copyright IBM 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

/// Remove redundant Pauli product rotations before a Pauli product measurement.
use pyo3::prelude::*;
use qiskit_circuit::Qubit;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType, PyDAGCircuit};
use qiskit_circuit::operations::{OperationRef, PauliProductMeasurement, PauliProductRotation};

/// Run the RemovePauliRotationsBeforeMeasure pass on `dag`.
/// Args:
///     dag (DAGCircuit): the DAG to be optimized.
/// Returns:
///     DAGCircuit: the optimized DAG.
#[pyfunction]
#[pyo3(name = "remove_pauli_rotations_before_measure")]
pub fn py_run_remove_pauli_rotations_before_measure(dag: &mut PyDAGCircuit) -> PyResult<()> {
    run_remove_pauli_rotations_before_measure(dag.try_write()?);
    Ok(())
}

/// Checks whether the Pauli components of a PPR and a PPM match on the same physical qubits.
#[inline]
fn pauli_products_match(
    ppr_qargs: &[Qubit],
    ppr: &PauliProductRotation,
    ppm_qargs: &[Qubit],
    ppm: &PauliProductMeasurement,
) -> bool {
    if ppr_qargs.len() != ppm_qargs.len()
        || ppr_qargs.len() != ppr.z.len()
        || ppm_qargs.len() != ppm.z.len()
    {
        return false;
    }

    let mut ppr_terms: Vec<(Qubit, bool, bool)> = ppr_qargs
        .iter()
        .zip(ppr.z.iter())
        .zip(ppr.x.iter())
        .map(|((&q, &z), &x)| (q, z, x))
        .collect();

    let mut ppm_terms: Vec<(Qubit, bool, bool)> = ppm_qargs
        .iter()
        .zip(ppm.z.iter())
        .zip(ppm.x.iter())
        .map(|((&q, &z), &x)| (q, z, x))
        .collect();

    ppr_terms.sort_unstable_by_key(|(q, _, _)| q.0);
    ppm_terms.sort_unstable_by_key(|(q, _, _)| q.0);

    ppr_terms == ppm_terms
}

/// Run the RemovePauliRotationsBeforeMeasure pass on `dag`.
/// # Arguments:
/// * `dag` - the DAG to be optimized.
///
/// # Returns:
/// The optimized DAG.
pub fn run_remove_pauli_rotations_before_measure(dag: &mut DAGCircuit) {
    let mut nodes_to_remove = Vec::new();

    for (meas_index, inst) in dag.op_nodes(true) {
        let OperationRef::PauliProductMeasurement(ppm) = inst.op.view() else {
            continue;
        };

        let ppm_qargs = dag.get_qargs(inst.qubits);

        let mut current_target = meas_index;
        loop {
            let mut preds = dag.quantum_predecessors(current_target);
            let Some(first_pred) = preds.next() else {
                break;
            };

            // All quantum incoming wires of current_target must originate from the same predecessor
            if !preds.all(|p| p == first_pred) {
                break;
            }

            let NodeType::Operation(pred_inst) = &dag[first_pred] else {
                break;
            };

            let OperationRef::PauliProductRotation(ppr) = pred_inst.op.view() else {
                break;
            };

            // All quantum successors of first_pred must be current_target
            let mut succs = dag.quantum_successors(first_pred);
            if !succs.all(|s| s == current_target) {
                break;
            }

            let ppr_qargs = dag.get_qargs(pred_inst.qubits);
            if pauli_products_match(ppr_qargs, ppr, ppm_qargs, ppm) {
                nodes_to_remove.push(first_pred);
                current_target = first_pred;
            } else {
                break;
            }
        }
    }

    for node_to_remove in nodes_to_remove {
        if dag.dag().node_weight(node_to_remove).is_some() {
            dag.remove_op_node(node_to_remove);
        }
    }
}

pub fn remove_pauli_rotations_before_measure_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(
        py_run_remove_pauli_rotations_before_measure
    ))?;
    Ok(())
}
