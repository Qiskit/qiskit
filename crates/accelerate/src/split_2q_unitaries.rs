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
use std::f64::consts::PI;
const PI4: f64 = PI / 4.;

use nalgebra::Matrix2;
use num_complex::Complex64;
use pyo3::prelude::*;
use rustworkx_core::petgraph::stable_graph::NodeIndex;
use smallvec::{smallvec, SmallVec};

use qiskit_circuit::circuit_instruction::ExtraInstructionAttributes;
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType, Wire};
use qiskit_circuit::operations::{ArrayType, Operation, OperationRef, Param, UnitaryGate};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::Qubit;

use crate::two_qubit_decompose::{Specialization, TwoQubitWeylDecomposition};

#[pyfunction]
pub fn split_2q_unitaries(
    py: Python,
    dag: &mut DAGCircuit,
    requested_fidelity: f64,
    split_swaps: bool,
) -> PyResult<Option<(DAGCircuit, Vec<usize>)>> {
    if !dag.get_op_counts().contains_key("unitary") {
        return Ok(None);
    }
    let nodes: Vec<NodeIndex> = dag.op_node_indices(false).collect();
    let mut has_swaps = false;
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
            if matches!(decomp.specialization, Specialization::SWAPEquiv) {
                has_swaps = true;
            }
            if matches!(decomp.specialization, Specialization::IdEquiv) {
                let k1r_arr = decomp.k1r_view();
                let k1l_arr = decomp.k1l_view();

                let insert_fn = |edge: &Wire| -> (PackedOperation, SmallVec<[Param; 3]>) {
                    if let Wire::Qubit(qubit) = edge {
                        if *qubit == qubits[0] {
                            let mat: Matrix2<Complex64> = [
                                [k1r_arr[[0, 0]], k1r_arr[[1, 0]]],
                                [k1r_arr[[0, 1]], k1r_arr[[1, 1]]],
                            ]
                            .into();
                            let k1r_gate = Box::new(UnitaryGate {
                                array: ArrayType::OneQ(mat),
                            });
                            (PackedOperation::from_unitary(k1r_gate), smallvec![])
                        } else {
                            let mat: Matrix2<Complex64> = [
                                [k1l_arr[[0, 0]], k1l_arr[[1, 0]]],
                                [k1l_arr[[0, 1]], k1l_arr[[1, 1]]],
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
        }
    }
    if !split_swaps || !has_swaps {
        return Ok(None);
    }
    // We have swap-like unitaries, so we create a new DAG in a manner similar to
    // The Elide Permutations pass, while also splitting the unitaries to 1-qubit gates
    let mut mapping: Vec<usize> = (0..dag.num_qubits()).collect();
    let mut new_dag = dag.copy_empty_like(py, "alike")?;
    for node in dag.topological_op_nodes()? {
        if let NodeType::Operation(inst) = &dag.dag()[node] {
            let qubits = dag.get_qargs(inst.qubits).to_vec();
            if qubits.len() == 2 && inst.op.name() == "unitary" {
                let matrix = inst
                    .op
                    .matrix(inst.params_view())
                    .expect("'unitary' gates should always have a matrix form");
                let decomp = TwoQubitWeylDecomposition::new_inner(
                    matrix.view(),
                    Some(requested_fidelity),
                    None,
                )?;
                if matches!(decomp.specialization, Specialization::SWAPEquiv) {
                    let k1r_arr = decomp.k1r_view();
                    let k1r_mat: Matrix2<Complex64> = [
                        [k1r_arr[[0, 0]], k1r_arr[[1, 0]]],
                        [k1r_arr[[0, 1]], k1r_arr[[1, 1]]],
                    ]
                    .into();
                    let k1r_gate = Box::new(UnitaryGate {
                        array: ArrayType::OneQ(k1r_mat),
                    });
                    let k1l_arr = decomp.k1l_view();
                    let k1l_mat: Matrix2<Complex64> = [
                        [k1l_arr[[0, 0]], k1l_arr[[1, 0]]],
                        [k1l_arr[[0, 1]], k1l_arr[[1, 1]]],
                    ]
                    .into();
                    let k1l_gate = Box::new(UnitaryGate {
                        array: ArrayType::OneQ(k1l_mat),
                    });
                    // perform the virtual swap
                    let qargs = dag.get_qargs(inst.qubits);
                    let index0 = qargs[0].index();
                    let index1 = qargs[1].index();
                    mapping.swap(index0, index1);
                    // now add the two 1-qubit gates
                    new_dag.apply_operation_back(
                        py,
                        PackedOperation::from_unitary(k1r_gate),
                        &[Qubit::new(mapping[index0])],
                        &[],
                        None,
                        ExtraInstructionAttributes::default(),
                        #[cfg(feature = "cache_pygates")]
                        None,
                    )?;
                    new_dag.apply_operation_back(
                        py,
                        PackedOperation::from_unitary(k1l_gate),
                        &[Qubit::new(mapping[index1])],
                        &[],
                        None,
                        ExtraInstructionAttributes::default(),
                        #[cfg(feature = "cache_pygates")]
                        None,
                    )?;
                    new_dag.add_global_phase(py, &Param::Float(decomp.global_phase + PI4))?;
                    continue; // skip the general instruction handling code
                }
            }
            // General instruction
            let qargs = dag.get_qargs(inst.qubits);
            let cargs = dag.get_cargs(inst.clbits);
            let mapped_qargs: Vec<Qubit> = qargs
                .iter()
                .map(|q| Qubit::new(mapping[q.index()]))
                .collect();

            new_dag.apply_operation_back(
                py,
                inst.op.clone(),
                &mapped_qargs,
                cargs,
                inst.params.as_deref().cloned(),
                inst.extra_attrs.clone(),
                #[cfg(feature = "cache_pygates")]
                inst.py_op.get().map(|x| x.clone_ref(py)),
            )?;
        }
    }
    Ok(Some((new_dag, mapping)))
}

pub fn split_2q_unitaries_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(split_2q_unitaries))?;
    Ok(())
}
