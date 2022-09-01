// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#![allow(clippy::too_many_arguments)]

pub mod neighbor_table;
pub mod sabre_dag;
pub mod swap_map;

use std::cmp::Ordering;

use hashbrown::{HashMap, HashSet};
use ndarray::prelude::*;
use numpy::IntoPyArray;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;
use rand::prelude::SliceRandom;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use retworkx_core::dictmap::*;
use retworkx_core::petgraph::prelude::*;
use retworkx_core::petgraph::visit::EdgeRef;
use retworkx_core::shortest_path::dijkstra;

use crate::getenv_use_multiple_threads;
use crate::nlayout::NLayout;

use neighbor_table::NeighborTable;
use sabre_dag::SabreDAG;
use swap_map::SwapMap;

const EXTENDED_SET_SIZE: usize = 20; // Size of lookahead window.
const DECAY_RATE: f64 = 0.001; // Decay coefficient for penalizing serial swaps.
const DECAY_RESET_INTERVAL: u8 = 5; // How often to reset all decay rates to 1.
const EXTENDED_SET_WEIGHT: f64 = 0.5; // Weight of lookahead window compared to front_layer.

#[pyclass]
pub enum Heuristic {
    Basic,
    Lookahead,
    Decay,
}

struct TrialResult {
    out_map: HashMap<usize, Vec<[usize; 2]>>,
    gate_order: Vec<usize>,
    layout: NLayout,
}

/// Return a set of candidate swaps that affect qubits in front_layer.
///
/// For each virtual qubit in front_layer, find its current location
/// on hardware and the physical qubits in that neighborhood. Every SWAP
/// on virtual qubits that corresponds to one of those physical couplings
/// is a candidate SWAP.
///
/// Candidate swaps are sorted so SWAP(i,j) and SWAP(j,i) are not duplicated.
fn obtain_swaps(
    front_layer: &[[usize; 2]],
    neighbors: &NeighborTable,
    layout: &NLayout,
) -> HashSet<[usize; 2]> {
    // This will likely under allocate as it's a function of the number of
    // neighbors for the qubits in the layer too, but this is basically a
    // minimum allocation assuming each qubit has only 1 unique neighbor
    let mut candidate_swaps: HashSet<[usize; 2]> = HashSet::with_capacity(2 * front_layer.len());
    for node in front_layer {
        for v in node {
            let physical = layout.logic_to_phys[*v];
            for neighbor in &neighbors.neighbors[physical] {
                let virtual_neighbor = layout.phys_to_logic[*neighbor];
                let swap: [usize; 2] = if &virtual_neighbor > v {
                    [*v, virtual_neighbor]
                } else {
                    [virtual_neighbor, *v]
                };
                candidate_swaps.insert(swap);
            }
        }
    }
    candidate_swaps
}

fn obtain_extended_set(
    dag: &SabreDAG,
    front_layer: &[NodeIndex],
    required_predecessors: &mut [u32],
) -> Vec<[usize; 2]> {
    let mut extended_set: Vec<[usize; 2]> = Vec::new();
    let mut decremented: Vec<usize> = Vec::new();
    let mut tmp_front_layer: Vec<NodeIndex> = front_layer.to_vec();
    let mut done: bool = false;
    while !tmp_front_layer.is_empty() && !done {
        let mut new_tmp_front_layer = Vec::new();
        for node in tmp_front_layer {
            for edge in dag.dag.edges(node) {
                let successor_index = edge.target();
                let successor = successor_index.index();
                decremented.push(successor);
                required_predecessors[successor] -= 1;
                if required_predecessors[successor] == 0 {
                    new_tmp_front_layer.push(successor_index);
                    let node_weight = dag.dag.node_weight(successor_index).unwrap();
                    let qargs = &node_weight.1;
                    if qargs.len() == 2 {
                        let extended_set_edges: [usize; 2] = [qargs[0], qargs[1]];
                        extended_set.push(extended_set_edges);
                    }
                }
            }
            if extended_set.len() >= EXTENDED_SET_SIZE {
                done = true;
                break;
            }
        }
        tmp_front_layer = new_tmp_front_layer;
    }
    for node in decremented {
        required_predecessors[node] += 1;
    }
    extended_set
}

fn cmap_from_neighor_table(neighbor_table: &NeighborTable) -> DiGraph<(), ()> {
    DiGraph::<(), ()>::from_edges(neighbor_table.neighbors.iter().enumerate().flat_map(
        |(u, targets)| {
            targets
                .iter()
                .map(move |v| (NodeIndex::new(u), NodeIndex::new(*v)))
        },
    ))
}

/// Run sabre swap on a circuit
///
/// Returns:
///     (SwapMap, gate_order): A tuple where the first element is a mapping of
///     DAGCircuit node ids to a list of virtual qubit swaps that should be
///     added before that operation. The second element is a numpy array of
///     node ids that represents the traversal order used by sabre.
#[pyfunction]
pub fn build_swap_map(
    py: Python,
    num_qubits: usize,
    dag: &SabreDAG,
    neighbor_table: &NeighborTable,
    distance_matrix: PyReadonlyArray2<f64>,
    heuristic: &Heuristic,
    seed: u64,
    layout: &mut NLayout,
    num_trials: usize,
) -> (SwapMap, PyObject) {
    let run_in_parallel = getenv_use_multiple_threads();
    let dist = distance_matrix.as_array();
    let coupling_graph: DiGraph<(), ()> = cmap_from_neighor_table(neighbor_table);
    let outer_rng = Pcg64Mcg::seed_from_u64(seed);
    let seed_vec: Vec<u64> = outer_rng
        .sample_iter(&rand::distributions::Standard)
        .take(num_trials)
        .collect();
    let result = if run_in_parallel {
        seed_vec
            .into_par_iter()
            .enumerate()
            .map(|(index, seed_trial)| {
                (
                    index,
                    swap_map_trial(
                        num_qubits,
                        dag,
                        neighbor_table,
                        &dist,
                        &coupling_graph,
                        heuristic,
                        seed_trial,
                        layout.clone(),
                    ),
                )
            })
            .min_by_key(|(index, result)| {
                [
                    result.out_map.values().map(|x| x.len()).sum::<usize>(),
                    *index,
                ]
            })
            .unwrap()
            .1
    } else {
        seed_vec
            .into_iter()
            .map(|seed_trial| {
                swap_map_trial(
                    num_qubits,
                    dag,
                    neighbor_table,
                    &dist,
                    &coupling_graph,
                    heuristic,
                    seed_trial,
                    layout.clone(),
                )
            })
            .min_by_key(|result| result.out_map.values().map(|x| x.len()).sum::<usize>())
            .unwrap()
    };
    *layout = result.layout;
    (
        SwapMap {
            map: result.out_map,
        },
        result.gate_order.into_pyarray(py).into(),
    )
}

fn swap_map_trial(
    num_qubits: usize,
    dag: &SabreDAG,
    neighbor_table: &NeighborTable,
    dist: &ArrayView2<f64>,
    coupling_graph: &DiGraph<(), ()>,
    heuristic: &Heuristic,
    seed: u64,
    mut layout: NLayout,
) -> TrialResult {
    let max_iterations_without_progress = 10 * neighbor_table.neighbors.len();
    let mut gate_order: Vec<usize> = Vec::with_capacity(dag.dag.node_count());
    let mut ops_since_progress: Vec<[usize; 2]> = Vec::new();
    let mut out_map: HashMap<usize, Vec<[usize; 2]>> = HashMap::new();
    let mut front_layer: Vec<NodeIndex> = dag.first_layer.clone();
    let mut required_predecessors: Vec<u32> = vec![0; dag.dag.node_count()];
    let mut extended_set: Option<Vec<[usize; 2]>> = None;
    let mut num_search_steps: u8 = 0;
    let mut qubits_decay: Vec<f64> = vec![1.; num_qubits];
    let mut rng = Pcg64Mcg::seed_from_u64(seed);

    for node in dag.dag.node_indices() {
        for edge in dag.dag.edges(node) {
            required_predecessors[edge.target().index()] += 1;
        }
    }
    while !front_layer.is_empty() {
        let mut execute_gate_list: Vec<NodeIndex> = Vec::new();
        // Remove as many immediately applicable gates as possible
        let mut new_front_layer: Vec<NodeIndex> = Vec::new();
        for node in front_layer {
            let node_weight = dag.dag.node_weight(node).unwrap();
            let qargs = &node_weight.1;
            if qargs.len() == 2 {
                let physical_qargs: [usize; 2] = [
                    layout.logic_to_phys[qargs[0]],
                    layout.logic_to_phys[qargs[1]],
                ];
                if coupling_graph
                    .find_edge(
                        NodeIndex::new(physical_qargs[0]),
                        NodeIndex::new(physical_qargs[1]),
                    )
                    .is_none()
                {
                    new_front_layer.push(node);
                } else {
                    execute_gate_list.push(node);
                }
            } else {
                execute_gate_list.push(node);
            }
        }
        front_layer = new_front_layer.clone();

        // Backtrack to the last time we made progress, then greedily insert swaps to route
        // the gate with the smallest distance between its arguments.  This is f block a release
        // valve for the algorithm to avoid infinite loops only, and should generally not
        // come into play for most circuits.
        if execute_gate_list.is_empty()
            && ops_since_progress.len() > max_iterations_without_progress
        {
            // If we're stuck in a loop without making progress first undo swaps:
            ops_since_progress
                .drain(..)
                .rev()
                .for_each(|swap| layout.swap_logical(swap[0], swap[1]));
            // Then pick the  closest pair in the current layer
            let target_qubits = front_layer
                .iter()
                .map(|n| {
                    let node_weight = dag.dag.node_weight(*n).unwrap();
                    let qargs = &node_weight.1;
                    [qargs[0], qargs[1]]
                })
                .min_by(|qargs_a, qargs_b| {
                    let dist_a = dist[[
                        layout.logic_to_phys[qargs_a[0]],
                        layout.logic_to_phys[qargs_a[1]],
                    ]];
                    let dist_b = dist[[
                        layout.logic_to_phys[qargs_b[0]],
                        layout.logic_to_phys[qargs_b[1]],
                    ]];
                    dist_a.partial_cmp(&dist_b).unwrap_or(Ordering::Equal)
                })
                .unwrap();
            // find Shortest path between target qubits
            let mut shortest_paths: DictMap<NodeIndex, Vec<NodeIndex>> = DictMap::new();
            let u = layout.logic_to_phys[target_qubits[0]];
            let v = layout.logic_to_phys[target_qubits[1]];
            (dijkstra(
                &coupling_graph,
                NodeIndex::<u32>::new(u),
                Some(NodeIndex::<u32>::new(v)),
                |_| Ok(1.),
                Some(&mut shortest_paths),
            ) as PyResult<Vec<Option<f64>>>)
                .unwrap();
            let shortest_path: Vec<usize> = shortest_paths
                .get(&NodeIndex::new(v))
                .unwrap()
                .iter()
                .map(|n| n.index())
                .collect();
            // Insert greedy swaps along that shortest path
            let split: usize = shortest_path.len() / 2;
            let forwards = &shortest_path[1..split];
            let backwards = &shortest_path[split..shortest_path.len() - 1];
            let mut greedy_swaps: Vec<[usize; 2]> = Vec::with_capacity(split);
            for swap in forwards {
                let logical_swap_bit = layout.phys_to_logic[*swap];
                greedy_swaps.push([target_qubits[0], logical_swap_bit]);
                layout.swap_logical(target_qubits[0], logical_swap_bit);
            }
            backwards.iter().rev().for_each(|swap| {
                let logical_swap_bit = layout.phys_to_logic[*swap];
                greedy_swaps.push([target_qubits[1], logical_swap_bit]);
                layout.swap_logical(target_qubits[1], logical_swap_bit);
            });
            ops_since_progress = greedy_swaps;
            continue;
        }
        if !execute_gate_list.is_empty() {
            for node in execute_gate_list {
                let node_weight = dag.dag.node_weight(node).unwrap();
                gate_order.push(node_weight.0);
                let out_swaps: Vec<[usize; 2]> = ops_since_progress.drain(..).collect();
                if !out_swaps.is_empty() {
                    out_map.insert(dag.dag.node_weight(node).unwrap().0, out_swaps);
                }
                for edge in dag.dag.edges(node) {
                    let successor = edge.target().index();
                    required_predecessors[successor] -= 1;
                    if required_predecessors[successor] == 0 {
                        front_layer.push(edge.target());
                    }
                }
            }
            qubits_decay.fill_with(|| 1.);
            extended_set = None;
            continue;
        }
        let first_layer: Vec<[usize; 2]> = front_layer
            .iter()
            .map(|n| {
                let node_weight = dag.dag.node_weight(*n).unwrap();
                let qargs = &node_weight.1;
                [qargs[0], qargs[1]]
            })
            .collect();
        if extended_set.is_none() {
            extended_set = Some(obtain_extended_set(
                dag,
                &front_layer,
                &mut required_predecessors,
            ));
        }

        let best_swap = sabre_score_heuristic(
            &first_layer,
            &mut layout,
            neighbor_table,
            extended_set.as_ref().unwrap(),
            dist,
            &qubits_decay,
            heuristic,
            &mut rng,
        );
        num_search_steps += 1;
        if num_search_steps >= DECAY_RESET_INTERVAL {
            qubits_decay.fill_with(|| 1.);
            num_search_steps = 0;
        } else {
            qubits_decay[best_swap[0]] += DECAY_RATE;
            qubits_decay[best_swap[1]] += DECAY_RATE;
        }
        ops_since_progress.push(best_swap);
    }
    TrialResult {
        out_map,
        gate_order,
        layout,
    }
}

fn sabre_score_heuristic(
    layer: &[[usize; 2]],
    layout: &mut NLayout,
    neighbor_table: &NeighborTable,
    extended_set: &[[usize; 2]],
    dist: &ArrayView2<f64>,
    qubits_decay: &[f64],
    heuristic: &Heuristic,
    rng: &mut Pcg64Mcg,
) -> [usize; 2] {
    // Run in parallel only if we're not already in a multiprocessing context
    // unless force threads is set.
    let candidate_swaps = obtain_swaps(layer, neighbor_table, layout);
    let mut min_score = f64::MAX;
    let mut best_swaps: Vec<[usize; 2]> = Vec::new();
    for swap_qubits in candidate_swaps {
        layout.swap_logical(swap_qubits[0], swap_qubits[1]);
        let score = score_heuristic(
            heuristic,
            layer,
            extended_set,
            layout,
            &swap_qubits,
            dist,
            qubits_decay,
        );
        if score < min_score {
            min_score = score;
            best_swaps.clear();
            best_swaps.push(swap_qubits);
        } else if score == min_score {
            best_swaps.push(swap_qubits);
        }
        layout.swap_logical(swap_qubits[0], swap_qubits[1]);
    }
    best_swaps.sort_unstable();
    let best_swap = *best_swaps.choose(rng).unwrap();
    layout.swap_logical(best_swap[0], best_swap[1]);
    best_swap
}

#[inline]
fn compute_cost(layer: &[[usize; 2]], layout: &NLayout, dist: &ArrayView2<f64>) -> f64 {
    layer
        .iter()
        .map(|gate| dist[[layout.logic_to_phys[gate[0]], layout.logic_to_phys[gate[1]]]])
        .sum()
}

fn score_lookahead(
    layer: &[[usize; 2]],
    extended_set: &[[usize; 2]],
    layout: &NLayout,
    dist: &ArrayView2<f64>,
) -> f64 {
    let mut first_cost = compute_cost(layer, layout, dist);
    first_cost /= layer.len() as f64;
    let second_cost = if extended_set.is_empty() {
        0.
    } else {
        compute_cost(extended_set, layout, dist) / extended_set.len() as f64
    };
    first_cost + EXTENDED_SET_WEIGHT * second_cost
}

fn score_decay(
    layer: &[[usize; 2]],
    extended_set: &[[usize; 2]],
    layout: &NLayout,
    dist: &ArrayView2<f64>,
    swap_qubits: &[usize; 2],
    qubits_decay: &[f64],
) -> f64 {
    let total_cost = score_lookahead(layer, extended_set, layout, dist);
    qubits_decay[swap_qubits[0]].max(qubits_decay[swap_qubits[1]]) * total_cost
}

fn score_heuristic(
    heuristic: &Heuristic,
    layer: &[[usize; 2]],
    extended_set: &[[usize; 2]],
    layout: &NLayout,
    swap_qubits: &[usize; 2],
    dist: &ArrayView2<f64>,
    qubits_decay: &[f64],
) -> f64 {
    match heuristic {
        Heuristic::Basic => compute_cost(layer, layout, dist),
        Heuristic::Lookahead => score_lookahead(layer, extended_set, layout, dist),
        Heuristic::Decay => {
            score_decay(layer, extended_set, layout, dist, swap_qubits, qubits_decay)
        }
    }
}

#[pymodule]
pub fn sabre_swap(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(build_swap_map))?;
    m.add_class::<Heuristic>()?;
    m.add_class::<NeighborTable>()?;
    m.add_class::<SabreDAG>()?;
    m.add_class::<SwapMap>()?;
    Ok(())
}
