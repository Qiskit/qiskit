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

use std::cell::RefCell;

use crate::equivalence::{CircuitRep, EdgeData, Equivalence, EquivalenceLibrary, Key, NodeData};
use ahash::RandomState;
use indexmap::{IndexMap, IndexSet};
use pyo3::prelude::*;
use pyo3::types::PySet;
use qiskit_circuit::operations::{Operation, Param};
use rustworkx_core::petgraph::stable_graph::{EdgeReference, NodeIndex, StableDiGraph};
use rustworkx_core::petgraph::visit::Control;
use rustworkx_core::traversal::{dijkstra_search, DijkstraEvent};
use smallvec::SmallVec;

#[pyfunction]
#[pyo3(name = "basis_search")]
pub(crate) fn py_basis_search(
    equiv_lib: &mut EquivalenceLibrary,
    source_basis: &Bound<PySet>,
    target_basis: &Bound<PySet>,
) -> PyResult<Option<Vec<(String, u32, SmallVec<[Param; 3]>, CircuitRep)>>> {
    let source_basis: PyResult<IndexSet<(String, u32), RandomState>> =
        source_basis.iter().map(|item| item.extract()).collect();
    let target_basis: PyResult<IndexSet<String, RandomState>> =
        target_basis.iter().map(|item| item.extract()).collect();
    Ok(basis_search(equiv_lib, source_basis?, target_basis?))
}

/// Search for a set of transformations from source_basis to target_basis.
/// Args:
///     equiv_lib (EquivalenceLibrary): Source of valid translations
///     source_basis (Set[Tuple[gate_name: str, gate_num_qubits: int]]): Starting basis.
///     target_basis (Set[gate_name: str]): Target basis.
///
/// Returns:
///     Optional[List[Tuple[gate, equiv_params, equiv_circuit]]]: List of (gate,
///         equiv_params, equiv_circuit) tuples tuples which, if applied in order
///         will map from source_basis to target_basis. Returns None if no path
///         was found.
pub(crate) fn basis_search(
    equiv_lib: &mut EquivalenceLibrary,
    source_basis: IndexSet<(String, u32), RandomState>,
    target_basis: IndexSet<String, RandomState>,
) -> Option<Vec<(String, u32, SmallVec<[Param; 3]>, CircuitRep)>> {
    // Build the visitor attributes:
    let mut num_gates_remaining_for_rule: IndexMap<usize, usize, RandomState> = IndexMap::default();
    let predecessors: IndexMap<(&str, u32), Equivalence, RandomState> = IndexMap::default();
    let predecessors_cell: RefCell<IndexMap<(&str, u32), Equivalence, RandomState>> =
        RefCell::new(predecessors);
    let opt_cost_map: IndexMap<(&str, u32), u32, RandomState> = IndexMap::default();
    let opt_cost_map_cell: RefCell<IndexMap<(&str, u32), u32, RandomState>> =
        RefCell::new(opt_cost_map);
    let mut basis_transforms: Vec<(String, u32, SmallVec<[Param; 3]>, CircuitRep)> = vec![];

    // Initialize visitor attributes:
    initialize_num_gates_remain_for_rule(&equiv_lib.graph, &mut num_gates_remaining_for_rule);
    // println!("{:#?}", num_gates_remaining_for_rule);

    // TODO: Logs
    let mut source_basis_remain: IndexSet<Key> = source_basis
        .iter()
        .filter_map(|(gate_name, gate_num_qubits)| {
            if !target_basis.contains(gate_name) {
                Some(Key {
                    name: gate_name.to_string(),
                    num_qubits: *gate_num_qubits,
                })
            } else {
                None
            }
        })
        .collect();

    // If source_basis is empty, no work needs to be done.
    if source_basis_remain.is_empty() {
        return Some(vec![]);
    }

    // This is only necessary since gates in target basis are currently reported by
    // their names and we need to have in addition the number of qubits they act on.
    let target_basis_keys: Vec<Key> = equiv_lib
        .keys()
        .cloned()
        .filter(|key| target_basis.contains(&key.name))
        .collect();
    // println!("Target basis keys {:#?}", target_basis_keys);
    let dummy: NodeIndex = equiv_lib.graph.add_node(NodeData {
        equivs: vec![],
        key: Key {
            name: "key".to_string(),
            num_qubits: 0,
        },
    });

    let target_basis_indices: Vec<NodeIndex> = target_basis_keys
        .iter()
        .map(|key| equiv_lib.node_index(key))
        .collect();

    target_basis_indices.iter().for_each(|node| {
        equiv_lib.graph.add_edge(dummy, *node, None);
    });

    // Build visitor methods

    let edge_weight =
        |edge: EdgeReference<Option<EdgeData>>| -> Result<u32, ()> {
            if edge.weight().is_none() {
                return Ok(1);
            }
            let edge_data = edge.weight().as_ref().unwrap();
            let mut cost_tot = 0;
            let borrowed_cost = opt_cost_map_cell.borrow();
            for instruction in edge_data.rule.circuit.0.iter() {
                cost_tot += borrowed_cost[&(instruction.op.name(), instruction.op.num_qubits())];
            }
            Ok(cost_tot
                - borrowed_cost[&(edge_data.source.name.as_str(), edge_data.source.num_qubits)])
        };

    dijkstra_search(
        &equiv_lib.graph,
        [dummy],
        edge_weight,
        |event: DijkstraEvent<NodeIndex, &Option<EdgeData>, u32>| {
            match event {
                DijkstraEvent::Discover(n, score) => {
                    let gate_key = &equiv_lib.graph[n].key;
                    let gate = &(gate_key.name.as_str(), gate_key.num_qubits);
                    source_basis_remain.swap_remove(gate_key);
                    let mut borrowed_cost_map = opt_cost_map_cell.borrow_mut();
                    borrowed_cost_map
                        .entry(*gate)
                        .and_modify(|cost_ref| *cost_ref = score)
                        .or_insert(score);
                    if let Some(rule) = predecessors_cell.borrow().get(gate) {
                        // TODO: Logger
                        basis_transforms.push((
                            gate_key.name.to_string(),
                            gate_key.num_qubits,
                            rule.params.clone(),
                            rule.circuit.clone(),
                        ));
                    }

                    if source_basis_remain.is_empty() {
                        basis_transforms.reverse();
                        return Control::Break(());
                    }
                }
                DijkstraEvent::EdgeRelaxed(_, target, edata) => {
                    if let Some(edata) = edata {
                        let gate = &equiv_lib.graph[target].key;
                        predecessors_cell
                            .borrow_mut()
                            .entry((gate.name.as_str(), gate.num_qubits))
                            .and_modify(|value| *value = edata.rule.clone())
                            .or_insert(edata.rule.clone());
                    }
                }
                DijkstraEvent::ExamineEdge(_, target, edata) => {
                    if edata.is_some() {
                        let edata = edata.as_ref().unwrap();
                        num_gates_remaining_for_rule
                            .entry(edata.index)
                            .and_modify(|val| *val -= 1)
                            .or_insert(0);
                        let target = &equiv_lib.graph[target].key;

                        if num_gates_remaining_for_rule[&edata.index] > 0
                            || target_basis_keys.contains(target)
                        {
                            return Control::Prune;
                        }
                    }
                }
                _ => {}
            };
            Control::Continue
        },
    )
    .unwrap();
    // Values will have to be cloned in order for the dummy node to be removed.
    // Will be revised
    drop(opt_cost_map_cell);
    drop(predecessors_cell);
    equiv_lib.graph.remove_node(dummy);
    Some(basis_transforms)
}

fn initialize_num_gates_remain_for_rule(
    graph: &StableDiGraph<NodeData, Option<EdgeData>>,
    source: &mut IndexMap<usize, usize, RandomState>,
) {
    let mut save_index = usize::MAX;
    for edge_data in graph.edge_weights().flatten() {
        if save_index == edge_data.index {
            continue;
        }
        source.insert(edge_data.index, edge_data.num_gates);
        save_index = edge_data.index;
    }
}
