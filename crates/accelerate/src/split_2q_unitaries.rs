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

use pyo3::prelude::*;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType, Wire};
use qiskit_circuit::imports::UNITARY_GATE;
use qiskit_circuit::operations::{Operation, Param};

use crate::two_qubit_decompose::{Specialization, TwoQubitWeylDecomposition};

#[pyfunction]
pub fn split_2q_unitaries(
    py: Python,
    dag: &mut DAGCircuit,
    requested_fidelity: f64,
) -> PyResult<()> {
    let nodes: Vec<NodeIndex> = dag.topological_op_nodes()?.collect();
    for node in nodes {
        if let NodeType::Operation(inst) = &dag.dag[node] {
            let qubits = dag.get_qargs(inst.qubits).to_vec();
            let matrix = inst.op.matrix(inst.params_view());
            if !dag.get_cargs(inst.clbits).is_empty()
                || qubits.len() != 2
                || matrix.is_none()
                || inst.is_parameterized()
                || inst.condition().is_some()
            {
                continue;
            }
            let decomp = TwoQubitWeylDecomposition::new_inner(
                matrix.unwrap().view(),
                Some(requested_fidelity),
                None,
            )?;
            if matches!(decomp.specialization, Specialization::IdEquiv) {
                let k1r_arr = decomp.K1r(py);
                let k1l_arr = decomp.K1l(py);
                let k1r_gate = UNITARY_GATE.get_bound(py).call1((k1r_arr,))?;
                let k1l_gate = UNITARY_GATE.get_bound(py).call1((k1l_arr,))?;
                let insert_fn = |edge: &Wire| -> PyResult<OperationFromPython> {
                    if let Wire::Qubit(qubit) = edge {
                        if *qubit == qubits[0] {
                            k1r_gate.extract()
                        } else {
                            k1l_gate.extract()
                        }
                    } else {
                        unreachable!("This will only be called on ops with no classical wires.");
                    }
                };
                dag.replace_on_incoming_qubits(py, node, insert_fn)?;
                dag.add_global_phase(py, &Param::Float(decomp.global_phase))?;
            }
        }
    }
    Ok(())
}

pub fn split_2q_unitaries_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(split_2q_unitaries))?;
    Ok(())
}
