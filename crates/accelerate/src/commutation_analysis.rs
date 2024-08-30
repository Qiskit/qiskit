use pyo3::prelude::PyModule;
use pyo3::{pyfunction, pymodule, wrap_pyfunction, Bound, PyResult, Python};
use std::hash::BuildHasherDefault;

use crate::commutation_checker::CommutationChecker;
use ahash::AHasher;
use hashbrown::HashMap;
use indexmap::IndexSet;
use pyo3::prelude::*;

use pyo3::types::{PyDict, PyList, PyTuple};
use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType, Wire};
use qiskit_circuit::dag_node::DAGOpNode;
use rustworkx_core::petgraph::stable_graph::NodeIndex;

use qiskit_circuit::circuit_instruction::CircuitInstruction;
use qiskit_circuit::packed_instruction::PackedInstruction;

fn op_node_from_packed(py: Python, dag: &DAGCircuit, packed: &PackedInstruction) -> DAGOpNode {
    let qubits = dag.get_qubits(packed.qubits);
    let clbits = dag.get_clbits(packed.clbits);
    DAGOpNode {
        instruction: CircuitInstruction {
            operation: packed.op.clone(),
            qubits: PyTuple::new_bound(py, dag.qubits.map_indices(qubits)).unbind(),
            clbits: PyTuple::new_bound(py, dag.clbits.map_indices(clbits)).unbind(),
            params: packed.params_view().iter().cloned().collect(),
            extra_attrs: packed.extra_attrs.clone(),
            py_op: packed.py_op.clone(),
        },
        sort_key: format!("{:?}", -1).into_py(py),
    }
}

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

    dag.qubit_input_map.keys().for_each(|qubit| {
        let wire = Wire::Qubit(*qubit);
        dag.nodes_on_wire(&wire, false)
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
                            //check if both are op nodes, then run commute
                            if let (NodeType::Operation(packed0), NodeType::Operation(packed1)) =
                                (&dag.dag[*current_gate_idx], &dag.dag[*prev_gate_idx])
                            {
                                //TODO preliminary interface, change this when dagcircuit merges
                                commutation_checker
                                    .commute_nodes(
                                        py,
                                        &op_node_from_packed(py, dag, packed0),
                                        &op_node_from_packed(py, dag, packed1),
                                        3,
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
