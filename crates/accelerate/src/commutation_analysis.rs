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
use pyo3::{pyfunction, wrap_pyfunction, Bound, PyResult, Python};
use qiskit_circuit::Qubit;

use crate::commutation_checker::CommutationChecker;
use hashbrown::HashMap;
use pyo3::prelude::*;

use pyo3::types::{PyDict, PyList};
use qiskit_circuit::dag_circuit::{DAGCircuit, OperationIndex, Wire};

// Custom types to store the commutation sets and node indices,
// see the docstring below for more information.
type CommutationSet = HashMap<Qubit, Vec<Vec<OperationIndex>>>;
type NodeIndices = HashMap<(OperationIndex, Qubit), usize>;

// the maximum number of qubits we check commutativity for
const MAX_NUM_QUBITS: u32 = 3;

/// Compute the commutation sets for a given DAG.
///
/// We return two HashMaps:
///  * {qubit: commutation_sets}: For each qubit, we keep a vector of index sets, where each index
///     set contains mutually commuting nodes.
///  * {(node, qubit): index}: For each (node, qubit) pair we store the index indicating in which
///     commutation set the node appears on a given wire.
///
/// For example, if we have a circuit
///
///     |0> -- X -- SX -- Z (out)
///      0     2    3     4   1   <-- node indices including input (0) and output (1) nodes
///
/// Then we would have
///
///     commutation_set = {0: [[2, 3], [4]]}
///     node_indices = {(2, 0): 0, (3, 0): 0, (4, 0): 1}
///
pub(crate) fn analyze_commutations_inner(
    py: Python,
    dag: &mut DAGCircuit,
    commutation_checker: &mut CommutationChecker,
) -> PyResult<(CommutationSet, NodeIndices)> {
    let mut commutation_set: CommutationSet = HashMap::new();
    let mut node_indices: NodeIndices = HashMap::new();

    for qubit in 0..dag.num_qubits() {
        let qubit = Qubit(qubit as u32);

        for current_gate_idx in dag.op_nodes_on_wire(&Wire::Qubit(qubit)) {
            // get the commutation set associated with the current wire, or create a new
            // index set containing the current gate
            let commutation_entry = commutation_set
                .entry(qubit)
                .or_insert_with(|| vec![vec![current_gate_idx]]);

            // we can unwrap as we know the commutation entry has at least one element
            let last = commutation_entry.last_mut().unwrap();

            // if the current gate index is not in the set, check whether it commutes with
            // the previous nodes -- if yes, add it to the commutation set
            if !last.contains(&current_gate_idx) {
                let mut all_commute = true;

                for prev_gate_idx in last.iter() {
                    let cur_inst = &dag[current_gate_idx];
                    let prev_inst = &dag[*prev_gate_idx];
                    all_commute = commutation_checker.commute_inner(
                        py,
                        &cur_inst.op.view(),
                        cur_inst.params_view(),
                        &cur_inst.extra_attrs,
                        dag.get_qargs(cur_inst.qubits),
                        dag.get_cargs(cur_inst.clbits),
                        &prev_inst.op.view(),
                        prev_inst.params_view(),
                        &prev_inst.extra_attrs,
                        dag.get_qargs(prev_inst.qubits),
                        dag.get_cargs(prev_inst.clbits),
                        MAX_NUM_QUBITS,
                    )?;
                    if !all_commute {
                        break;
                    }
                }

                if all_commute {
                    // all commute, add to current list
                    last.push(current_gate_idx);
                } else {
                    // does not commute, create new list
                    commutation_entry.push(vec![current_gate_idx]);
                }
            }

            node_indices.insert((current_gate_idx, qubit), commutation_entry.len() - 1);
        }
    }

    Ok((commutation_set, node_indices))
}

#[pyfunction]
#[pyo3(signature = (dag, commutation_checker))]
pub(crate) fn analyze_commutations(
    py: Python,
    dag: &mut DAGCircuit,
    commutation_checker: &mut CommutationChecker,
) -> PyResult<Py<PyDict>> {
    // This returns two HashMaps:
    //   * The commuting nodes per wire: {wire: [commuting_nodes_1, commuting_nodes_2, ...]}
    //   * The index in which commutation set a given node is located on a wire: {(node, wire): index}
    //
    // The historical Python versions of this function stored those two maps together in a single
    // `dict`, so we do the same here.
    //
    // For other historical reasons, Python space expects the `DAGInNode` and the `DAGOutNode`
    // objects to be present for each qubit, and list zero commutation relations.
    let (commutation_set, node_indices) = analyze_commutations_inner(py, dag, commutation_checker)?;

    let out_dict = PyDict::new_bound(py);

    // First set the {qubit: [commuting_nodes_1, ...]} bit
    for (qubit, commutations) in commutation_set {
        let py_wire = dag.qubits().get(qubit).unwrap().to_object(py);
        let outer_list = PyList::empty_bound(py);
        let [in_node, out_node] = dag
            .wire_io_nodes(Wire::Qubit(qubit))
            .expect("the commutation analysis only considers qubits that exist");
        let num_commutations = commutations.len();
        outer_list.append(PyList::new_bound(py, [dag.get_node(py, in_node)?]))?;
        for set in commutations {
            outer_list.append(PyList::new_bound(
                py,
                set.iter().map(|index| dag.get_node(py, *index).unwrap()),
            ))?;
        }
        outer_list.append(PyList::new_bound(py, [dag.get_node(py, out_node)?]))?;
        out_dict.set_item(py_wire.clone_ref(py), outer_list)?;

        // Also add the `qubit: in_node_index` and `qubit: out_node_index`  entries now.
        out_dict.set_item((dag.get_node(py, in_node)?, py_wire.clone_ref(py)), 0)?;
        out_dict.set_item((dag.get_node(py, out_node)?, py_wire), num_commutations + 1)?;
    }

    // Then we add the {(node, qubit): index} dictionary.
    //
    // Note that since we prepended the `[in_node]` list, everything is offset by one.
    for ((node_index, qubit), index) in node_indices {
        let py_wire = dag.qubits().get(qubit).unwrap().to_object(py);
        out_dict.set_item((dag.get_node(py, node_index.node())?, py_wire), index + 1)?;
    }

    Ok(out_dict.unbind())
}

pub fn commutation_analysis(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(analyze_commutations))?;
    Ok(())
}
