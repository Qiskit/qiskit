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

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType};
use qiskit_circuit::operations::{Operation, Param};
use qiskit_circuit::Qubit;

/// Run the ElidePermutations pass on `dag`.
/// Args:
///     dag (DAGCircuit): the DAG to be optimized.
/// Returns:
///     DAGCircuit: the optimized DAG.
#[pyfunction]
fn run(py: Python, dag: &mut DAGCircuit) -> PyResult<Option<(DAGCircuit, Vec<usize>)>> {
    let permutation_gate_names = ["swap".to_string(), "permutation".to_string()];
    let op_counts = dag.count_ops(py, false)?;
    if !permutation_gate_names
        .iter()
        .any(|name| op_counts.contains_key(name))
    {
        return Ok(None);
    }
    let mut mapping: Vec<usize> = (0..dag.num_qubits()).collect();

    // note that DAGCircuit::copy_empty_like clones the interners
    let mut new_dag = dag.copy_empty_like(py, "alike")?;
    for node_index in dag.topological_op_nodes()? {
        if let NodeType::Operation(inst) = &dag.dag[node_index] {
            match (inst.op.name(), inst.condition()) {
                ("swap", None) => {
                    let qargs = dag.get_qargs(inst.qubits);
                    let index0 = qargs[0].0 as usize;
                    let index1 = qargs[1].0 as usize;
                    let prev0 = mapping[index0];
                    let prev1 = mapping[index1];
                    mapping[index0] = prev1;
                    mapping[index1] = prev0;
                }
                ("permutation", None) => {
                    if let Param::Obj(ref pyobj) = inst.params.as_ref().unwrap()[0] {
                        let pyarray: PyReadonlyArray1<i32> = pyobj.extract(py)?;
                        let pattern = pyarray.as_array();

                        let qindices: Vec<usize> = dag
                            .get_qargs(inst.qubits)
                            .iter()
                            .map(|q| q.0 as usize)
                            .collect();

                        let remapped_qindices: Vec<usize> = (0..qindices.len())
                            .map(|i| pattern[i])
                            .map(|i| qindices[i as usize])
                            .collect();

                        qindices
                            .iter()
                            .zip(remapped_qindices.iter())
                            .for_each(|(old, new)| {
                                mapping[*old] = *new;
                            });
                    } else {
                        unreachable!();
                    }
                }
                _ => {
                    // General instruction
                    let mut mapped_inst = inst.clone();
                    let qargs = dag.get_qargs(inst.qubits);
                    let mapped_qargs: Vec<Qubit> = qargs
                        .iter()
                        .map(|q| q.0 as usize)
                        .map(|q| mapping[q])
                        .map(|q| Qubit(q.try_into().unwrap()))
                        .collect();
                    let mapped_qubits = new_dag.set_qargs(&mapped_qargs);
                    mapped_inst.qubits = mapped_qubits;
                    new_dag.push_back(py, mapped_inst)?;
                }
            }
        } else {
            unreachable!("Not an op node")
        }
    }
    Ok(Some((new_dag, mapping)))
}

pub fn elide_permutations(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(run))?;
    Ok(())
}
