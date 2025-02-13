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

use nalgebra::Matrix2;
use num_complex::Complex64;
use pyo3::prelude::*;
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use smallvec::{smallvec, SmallVec};

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType, Wire};
use qiskit_circuit::operations::{ArrayType, Operation, OperationRef, Param, UnitaryGate};
use qiskit_circuit::packed_instruction::PackedOperation;

use crate::two_qubit_decompose::{Specialization, TwoQubitWeylDecomposition};

#[pyfunction]
pub fn split_2q_unitaries(
    py: Python,
    dag: &mut DAGCircuit,
    requested_fidelity: f64,
) -> PyResult<()> {
    if !dag.get_op_counts().contains_key("unitary") {
        return Ok(());
    }
    let nodes: Vec<NodeIndex> = dag.op_node_indices(false).collect();

    for node in nodes {
        if let NodeType::Operation(inst) = &dag[node] {
            let qubits = dag.get_qargs(inst.qubits).to_vec();
            // We only attempt to split UnitaryGate objects, but this could be extended in future
            // -- however we need to ensure that we can compile the resulting single-qubit unitaries
            // to the supported basis gate set.
            if qubits.len() != 2 || !matches!(inst.op.view(), OperationRef::Unitary(_)) {
                continue;
            }
            let matrix = inst
                .op
                .matrix(inst.params_view())
                .expect("'unitary' gates should always have a matrix form");
            let decomp = TwoQubitWeylDecomposition::new_inner(
                matrix.view(),
                Some(requested_fidelity),
                None,
            )?;
            if matches!(decomp.specialization, Specialization::IdEquiv) {
                let k1r_arr = decomp.k1r_view();
                let k1l_arr = decomp.k1l_view();

                let insert_fn = |edge: &Wire| -> (PackedOperation, SmallVec<[Param; 3]>) {
                    if let Wire::Qubit(qubit) = edge {
                        if *qubit == qubits[0] {
                            let mat: Matrix2<Complex64> = [
                                [k1r_arr[[0, 0]], k1r_arr[[0, 1]]],
                                [k1r_arr[[1, 0]], k1r_arr[[1, 1]]],
                            ]
                            .into();
                            let k1r_gate = Box::new(UnitaryGate {
                                array: ArrayType::OneQ(mat),
                            });
                            (PackedOperation::from_unitary(k1r_gate), smallvec![])
                        } else {
                            let mat: Matrix2<Complex64> = [
                                [k1l_arr[[0, 0]], k1l_arr[[0, 1]]],
                                [k1l_arr[[1, 0]], k1l_arr[[1, 1]]],
                            ]
                            .into();

                            let k1l_gate = Box::new(UnitaryGate {
                                array: ArrayType::OneQ(mat),
                            });

                            (PackedOperation::from_unitary(k1l_gate), smallvec![])
                        }
                    } else {
                        unreachable!("This will only be called on ops with no classical wires.");
                    }
                };
                dag.replace_node_with_1q_ops(py, node, insert_fn)?;
                dag.add_global_phase(py, &Param::Float(decomp.global_phase))?;
            }
            // TODO: also look into splitting on Specialization::Swap and just
            // swap the virtual qubits. Doing this we will need to update the
            // permutation like in ElidePermutations
        }
    }
    Ok(())
}

pub fn split_2q_unitaries_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(split_2q_unitaries))?;
    Ok(())
}
