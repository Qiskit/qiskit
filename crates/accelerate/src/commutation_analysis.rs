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

use pyo3::prelude::PyModule;
use pyo3::{pyfunction, pymodule, wrap_pyfunction, Bound, PyResult, Python};
use qiskit_circuit::operations::Param;
use qiskit_circuit::Qubit;
use smallvec::{smallvec, SmallVec};
use std::hash::BuildHasherDefault;

use crate::commutation_checker::CommutationChecker;
use ahash::AHasher;
use hashbrown::HashMap;
use indexmap::IndexSet;
use pyo3::prelude::*;

use pyo3::types::{PyDict, PyList};
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType, Wire};
use rustworkx_core::petgraph::stable_graph::NodeIndex;

type AIndexSet<T> = IndexSet<T, BuildHasherDefault<AHasher>>;
#[derive(Clone, Debug)]
pub enum CommutationSetEntry {
    Index(usize),
    SetExists(Vec<AIndexSet<NodeIndex>>),
}

fn analyze_commutations_inner(
    py: Python,
    dag: &mut DAGCircuit,
    commutation_checker: &mut CommutationChecker,
) -> HashMap<(Option<NodeIndex>, Wire), CommutationSetEntry> {
    let mut commutation_set: HashMap<(Option<NodeIndex>, Wire), CommutationSetEntry> =
        HashMap::new();
    let max_num_qubits = 3;

    (0..dag.num_qubits()).for_each(|qubit| {
        let wire = Wire::Qubit(Qubit(qubit as u32));
        dag.nodes_on_wire(py, &wire, false)
            .iter()
            .for_each(|current_gate_idx| {
                if let CommutationSetEntry::SetExists(ref mut commutation_entry) = commutation_set
                    .entry((None, wire.clone()))
                    .or_insert_with(|| {
                        CommutationSetEntry::SetExists(vec![AIndexSet::from_iter([
                            *current_gate_idx,
                        ])])
                    })
                {
                    let last = commutation_entry.last_mut().unwrap();

                    if !last.contains(current_gate_idx) {
                        if last.iter().all(|prev_gate_idx| {
                            if let (
                                NodeType::Operation(packed_inst0),
                                NodeType::Operation(packed_inst1),
                            ) = (&dag.dag[*current_gate_idx], &dag.dag[*prev_gate_idx])
                            {
                                let empty_params: Box<SmallVec<[Param; 3]>> = Box::new(smallvec![]);
                                let op1 = packed_inst0.op.view();
                                let op2 = packed_inst1.op.view();
                                let params1 = match packed_inst0.params.as_ref() {
                                    Some(params) => params,
                                    None => &empty_params,
                                };
                                let params2 = match packed_inst1.params.as_ref() {
                                    Some(params) => params,
                                    None => &empty_params,
                                };
                                let qargs1 = dag.qargs_interner.get(packed_inst0.qubits);
                                let qargs2 = dag.qargs_interner.get(packed_inst1.qubits);
                                let cargs1 = dag.cargs_interner.get(packed_inst0.clbits);
                                let cargs2 = dag.cargs_interner.get(packed_inst1.clbits);

                                // TODO preliminary interface, change this when dagcircuit merges
                                commutation_checker
                                    .commute_inner(
                                        py,
                                        &op1,
                                        params1,
                                        packed_inst0.extra_attrs.as_deref(),
                                        qargs1,
                                        cargs1,
                                        &op2,
                                        params2,
                                        packed_inst1.extra_attrs.as_deref(),
                                        qargs2,
                                        cargs2,
                                        max_num_qubits,
                                    )
                                    .unwrap()
                            } else {
                                false
                            }
                        }) {
                            // all commute, add to current list
                            last.insert(*current_gate_idx);
                        } else {
                            // does not commute, create new list
                            commutation_entry.push(AIndexSet::from_iter([*current_gate_idx]))
                        }
                    }
                } else {
                    panic!("Wrong type in dictionary!");
                }
                if let CommutationSetEntry::SetExists(last_entry) =
                    commutation_set.get(&(None, wire.clone())).unwrap()
                {
                    commutation_set.insert(
                        (Some(*current_gate_idx), wire.clone()),
                        CommutationSetEntry::Index(last_entry.len() - 1),
                    );
                }
            })
    });
    commutation_set
}

#[pyfunction]
#[pyo3(signature = (dag, commutation_checker))]
pub(crate) fn analyze_commutations(
    py: Python,
    dag: &mut DAGCircuit,
    commutation_checker: &mut CommutationChecker,
) -> PyResult<Py<PyDict>> {
    let commutations = analyze_commutations_inner(py, dag, commutation_checker);
    let out_dict = PyDict::new_bound(py);
    for (k, comms) in commutations {
        let nidx = k.0;
        let wire = match k.1 {
            Wire::Qubit(q) => dag.qubits.get(q).unwrap().to_object(py),
            Wire::Clbit(c) => dag.clbits.get(c).unwrap().to_object(py),
            Wire::Var(v) => v,
        };

        if nidx.is_some() {
            match comms {
                CommutationSetEntry::Index(idx) => {
                    out_dict.set_item((dag.get_node(py, nidx.unwrap())?, wire), idx)?
                }
                _ => panic!("Wrong format in commutation analysis"),
            };
        } else {
            match comms {
                CommutationSetEntry::SetExists(comm_set) => out_dict.set_item(
                    wire,
                    PyList::new_bound(
                        py,
                        comm_set.iter().map(|inner| {
                            PyList::new_bound(
                                py,
                                inner
                                    .into_iter()
                                    .map(|ndidx| dag.get_node(py, *ndidx).unwrap()),
                            )
                        }),
                    ),
                )?,
                _ => panic!("Wrong format in commutation analysis"),
            }
        }
    }
    Ok(out_dict.unbind())
}

#[pymodule]
pub fn commutation_analysis(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(analyze_commutations))?;
    Ok(())
}
