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

pub mod layer;
pub mod neighbor_table;
pub mod sabre_dag;
pub mod swap_map;

use hashbrown::HashMap;
use indexmap::IndexMap;
use ndarray::prelude::*;
use numpy::PyReadonlyArray2;
use numpy::{IntoPyArray, PyArray, ToPyArray};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;
use rand::prelude::SliceRandom;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;
use rustworkx_core::dictmap::*;
use rustworkx_core::petgraph::prelude::*;
use rustworkx_core::petgraph::visit::EdgeRef;
use rustworkx_core::shortest_path::dijkstra;
use rustworkx_core::token_swapper::token_swapper;
use std::cmp::Ordering;

use crate::getenv_use_multiple_threads;
use crate::nlayout::{NLayout, PhysicalQubit};

use layer::{ExtendedSet, FrontLayer};
use neighbor_table::NeighborTable;
use sabre_dag::SabreDAG;
use swap_map::SwapMap;

const BEST_EPSILON: f64 = 1e-10; // Epsilon used in minimum-score calculations.

const EXTENDED_SET_SIZE: usize = 20; // Size of lookahead window.
const DECAY_RATE: f64 = 0.001; // Decay coefficient for penalizing serial swaps.
const DECAY_RESET_INTERVAL: u8 = 5; // How often to reset all decay rates to 1.
const EXTENDED_SET_WEIGHT: f64 = 0.5; // Weight of lookahead window compared to front_layer.
const SWAP_EPILOGUE_TRIALS: usize = 4; // Number of trials for control flow block swap epilogues.

#[pyclass]
pub enum Heuristic {
    Basic,
    Lookahead,
    Decay,
}

/// A container for Sabre mapping results.
#[pyclass(module = "qiskit._accelerate.sabre_swap")]
#[derive(Clone, Debug)]
pub struct SabreResult {
    #[pyo3(get)]
    pub map: SwapMap,
    pub node_order: Vec<usize>,
    #[pyo3(get)]
    pub node_block_results: NodeBlockResults,
}

#[pymethods]
impl SabreResult {
    #[getter]
    fn node_order(&self, py: Python) -> PyObject {
        self.node_order.to_pyarray(py).into()
    }
}

#[pyclass(mapping, module = "qiskit._accelerate.sabre_swap")]
#[derive(Clone, Debug)]
pub struct NodeBlockResults {
    pub results: HashMap<usize, Vec<BlockResult>>,
}

#[pymethods]
impl NodeBlockResults {
    // Mapping Protocol
    pub fn __len__(&self) -> usize {
        self.results.len()
    }

    pub fn __contains__(&self, object: usize) -> bool {
        self.results.contains_key(&object)
    }

    pub fn __getitem__(&self, py: Python, object: usize) -> PyResult<PyObject> {
        match self.results.get(&object) {
            Some(val) => Ok(val
                .iter()
                .map(|x| x.clone().into_py(py))
                .collect::<Vec<_>>()
                .into_pyarray(py)
                .into()),
            None => Err(PyIndexError::new_err(format!(
                "Node index {object} has no block results",
            ))),
        }
    }

    pub fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.results))
    }
}

#[pyclass(module = "qiskit._accelerate.sabre_swap")]
#[derive(Clone, Debug)]
pub struct BlockResult {
    #[pyo3(get)]
    pub result: SabreResult,
    pub swap_epilogue: Vec<[PhysicalQubit; 2]>,
}

#[pymethods]
impl BlockResult {
    #[getter]
    fn swap_epilogue(&self, py: Python) -> PyObject {
        self.swap_epilogue
            .iter()
            .map(|x| x.into_py(py))
            .collect::<Vec<_>>()
            .into_pyarray(py)
            .into()
    }
}

/// Return a set of candidate swaps that affect qubits in front_layer.
///
/// For each virtual qubit in front_layer, find its current location
/// on hardware and the physical qubits in that neighborhood. Every SWAP
/// on virtual qubits that corresponds to one of those physical couplings
/// is a candidate SWAP.
fn obtain_swaps<'a>(
    front_layer: &'a FrontLayer,
    neighbors: &'a NeighborTable,
) -> impl Iterator<Item = [PhysicalQubit; 2]> + 'a {
    front_layer.iter_active().flat_map(move |&p| {
        neighbors[p].iter().filter_map(move |&neighbor| {
            if neighbor > p || !front_layer.is_active(neighbor) {
                Some([p, neighbor])
            } else {
                None
            }
        })
    })
}

/// Fill the given `extended_set` with the next nodes that would be reachable after the front layer
/// (and themselves).  This uses `required_predecessors` as scratch space for efficiency, but
/// returns it to the same state as the input on return.
fn populate_extended_set(
    extended_set: &mut ExtendedSet,
    dag: &SabreDAG,
    front_layer: &FrontLayer,
    layout: &NLayout,
    required_predecessors: &mut [u32],
) {
    let mut to_visit = front_layer.iter_nodes().copied().collect::<Vec<_>>();
    let mut decremented: IndexMap<usize, u32, ahash::RandomState> =
        IndexMap::with_hasher(ahash::RandomState::default());
    let mut i = 0;
    let mut visit_now: Vec<NodeIndex> = Vec::new();
    while i < to_visit.len() && extended_set.len() < EXTENDED_SET_SIZE {
        // Visit runs of non-2Q gates fully before moving on to children
        // of 2Q gates. This way, traversal order is a BFS of 2Q gates rather
        // than of all gates.
        visit_now.push(to_visit[i]);
        let mut j = 0;
        while let Some(node) = visit_now.get(j) {
            for edge in dag.dag.edges_directed(*node, Direction::Outgoing) {
                let successor_node = edge.target();
                let successor_index = successor_node.index();
                *decremented.entry(successor_index).or_insert(0) += 1;
                required_predecessors[successor_index] -= 1;
                if required_predecessors[successor_index] == 0 {
                    if !dag.dag[successor_node].directive
                        && !dag.node_blocks.contains_key(&successor_index)
                    {
                        if let [a, b] = dag.dag[successor_node].qubits[..] {
                            extended_set.push([a.to_phys(layout), b.to_phys(layout)]);
                            to_visit.push(successor_node);
                            continue;
                        }
                    }
                    visit_now.push(successor_node);
                }
            }
            j += 1;
        }
        visit_now.clear();
        i += 1;
    }
    for (node, amount) in decremented.iter() {
        required_predecessors[*node] += *amount;
    }
}

/// Run sabre swap on a circuit
///
/// Returns:
///     (SwapMap, gate_order, node_block_results, final_permutation): A tuple where the first
///     element is a mapping of DAGCircuit node ids to a list of virtual qubit swaps that should be
///     added before that operation. The second element is a numpy array of node ids that
///     represents the traversal order used by sabre.  The third is inner results for the blocks of
///     control flow, and the fourth is a permutation, where `final_permution[i]` is the final
///     logical position of the qubit that began in position `i`.
#[pyfunction]
pub fn build_swap_map(
    py: Python,
    num_qubits: u32,
    dag: &SabreDAG,
    neighbor_table: &NeighborTable,
    distance_matrix: PyReadonlyArray2<f64>,
    heuristic: &Heuristic,
    initial_layout: &NLayout,
    num_trials: usize,
    seed: Option<u64>,
    run_in_parallel: Option<bool>,
) -> (SwapMap, PyObject, NodeBlockResults, PyObject) {
    let dist = distance_matrix.as_array();
    let (res, final_layout) = build_swap_map_inner(
        num_qubits,
        dag,
        neighbor_table,
        &dist,
        heuristic,
        seed,
        initial_layout,
        num_trials,
        run_in_parallel,
    );
    (
        res.map,
        res.node_order.into_pyarray(py).into(),
        res.node_block_results,
        PyArray::from_iter(
            py,
            (0..num_qubits).map(|phys| {
                PhysicalQubit::new(phys)
                    .to_virt(initial_layout)
                    .to_phys(&final_layout)
            }),
        )
        .into(),
    )
}

pub fn build_swap_map_inner(
    num_qubits: u32,
    dag: &SabreDAG,
    neighbor_table: &NeighborTable,
    dist: &ArrayView2<f64>,
    heuristic: &Heuristic,
    seed: Option<u64>,
    initial_layout: &NLayout,
    num_trials: usize,
    run_in_parallel: Option<bool>,
) -> (SabreResult, NLayout) {
    let run_in_parallel = match run_in_parallel {
        Some(run_in_parallel) => run_in_parallel,
        None => getenv_use_multiple_threads() && num_trials > 1,
    };
    let coupling_graph = neighbor_table.coupling_graph();
    let outer_rng = match seed {
        Some(seed) => Pcg64Mcg::seed_from_u64(seed),
        None => Pcg64Mcg::from_entropy(),
    };
    let seed_vec: Vec<u64> = outer_rng
        .sample_iter(&rand::distributions::Standard)
        .take(num_trials)
        .collect();
    if run_in_parallel {
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
                        dist,
                        &coupling_graph,
                        heuristic,
                        seed_trial,
                        initial_layout,
                    ),
                )
            })
            .min_by_key(|(index, (result, _))| {
                [
                    result.map.map.values().map(|x| x.len()).sum::<usize>(),
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
                    dist,
                    &coupling_graph,
                    heuristic,
                    seed_trial,
                    initial_layout,
                )
            })
            .min_by_key(|(result, _)| result.map.map.values().map(|x| x.len()).sum::<usize>())
            .unwrap()
    }
}

fn swap_map_trial(
    num_qubits: u32,
    dag: &SabreDAG,
    neighbor_table: &NeighborTable,
    dist: &ArrayView2<f64>,
    coupling_graph: &DiGraph<(), ()>,
    heuristic: &Heuristic,
    seed: u64,
    initial_layout: &NLayout,
) -> (SabreResult, NLayout) {
    let max_iterations_without_progress = 10 * num_qubits as usize;
    let mut out_map: HashMap<usize, Vec<[PhysicalQubit; 2]>> = HashMap::new();
    let mut gate_order = Vec::with_capacity(dag.dag.node_count());
    let mut front_layer = FrontLayer::new(num_qubits);
    let mut extended_set = ExtendedSet::new(num_qubits);
    let mut required_predecessors: Vec<u32> = vec![0; dag.dag.node_count()];
    let mut layout = initial_layout.clone();
    let mut num_search_steps: u8 = 0;
    let mut qubits_decay: Vec<f64> = vec![1.; num_qubits as usize];
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let mut node_block_results: HashMap<usize, Vec<BlockResult>> =
        HashMap::with_capacity(dag.node_blocks.len());

    for node in dag.dag.node_indices() {
        for edge in dag.dag.edges(node) {
            required_predecessors[edge.target().index()] += 1;
        }
    }

    // This closure is used to curry parameters so we can avoid passing
    // everything and the kitchen sink to update_routes and
    // route_reachable_nodes.
    let route_block_dag = |block_dag: &SabreDAG, current_layout: &NLayout| {
        swap_map_trial(
            num_qubits,
            block_dag,
            neighbor_table,
            dist,
            coupling_graph,
            heuristic,
            seed,
            current_layout,
        )
    };

    route_reachable_nodes(
        seed,
        &dag.first_layer,
        dag,
        &layout,
        coupling_graph,
        &mut gate_order,
        &mut front_layer,
        &mut required_predecessors,
        &mut node_block_results,
        &route_block_dag,
    );
    populate_extended_set(
        &mut extended_set,
        dag,
        &front_layer,
        &layout,
        &mut required_predecessors,
    );
    // Main logic loop; the front layer only becomes empty when all nodes have been routed.  At
    // each iteration of this loop, we route either one or two gates.
    let mut routable_nodes = Vec::<NodeIndex>::with_capacity(2);
    // Reusable allocated storage space for choosing the best swap.  This is owned outside of the
    // `choose_best_swap` function so that we don't need to reallocate and then re-grow the
    // collection on every entry.
    let mut swap_scratch = Vec::<[PhysicalQubit; 2]>::new();
    while !front_layer.is_empty() {
        let mut current_swaps: Vec<[PhysicalQubit; 2]> = Vec::new();
        // Swap-mapping loop.  This is the main part of the algorithm, which we repeat until we
        // either successfully route a node, or exceed the maximum number of attempts.
        while routable_nodes.is_empty() && current_swaps.len() <= max_iterations_without_progress {
            let best_swap = choose_best_swap(
                &front_layer,
                &extended_set,
                neighbor_table,
                dist,
                &qubits_decay,
                heuristic,
                &mut rng,
                &mut swap_scratch,
            );
            front_layer.routable_after(&mut routable_nodes, &best_swap, coupling_graph);
            front_layer.apply_swap(best_swap);
            extended_set.apply_swap(best_swap);
            layout.swap_physical(best_swap[0], best_swap[1]);
            current_swaps.push(best_swap);
            num_search_steps += 1;
            if num_search_steps >= DECAY_RESET_INTERVAL {
                qubits_decay.fill(1.);
                num_search_steps = 0;
            } else {
                qubits_decay[best_swap[0].index()] += DECAY_RATE;
                qubits_decay[best_swap[1].index()] += DECAY_RATE;
            }
        }
        // If we exceeded the number of allowed attempts without successfully routing a node, we
        // reset back to the state we were in last time we routed a node, then find the node in the
        // front layer whose qubits are the closest in the coupling map, and greedily insert swaps
        // to make the node routable.  We could clone the layout each time we route a gate, but
        // this path is only an escape mechansim for the algorithm getting stuck, so it should
        // ideally never be taken, and it doesn't matter if it's not the speediest---it's better to
        // keep the other path faster.
        if routable_nodes.is_empty() {
            undo_swaps(
                &mut current_swaps,
                &mut front_layer,
                &mut extended_set,
                &mut layout,
            );
            let (&node, &qubits) = closest_operation(&front_layer, dist);
            swaps_to_route(&mut current_swaps, &qubits, coupling_graph);
            for &[a, b] in current_swaps.iter() {
                front_layer.apply_swap([a, b]);
                extended_set.apply_swap([a, b]);
                layout.swap_physical(a, b);
            }
            routable_nodes.push(node);
        }
        update_route(
            seed,
            &routable_nodes,
            current_swaps,
            dag,
            &layout,
            coupling_graph,
            &mut gate_order,
            &mut out_map,
            &mut front_layer,
            &mut extended_set,
            &mut required_predecessors,
            &mut node_block_results,
            &route_block_dag,
        );
        qubits_decay.fill(1.);
        routable_nodes.clear();
    }
    (
        SabreResult {
            map: SwapMap { map: out_map },
            node_order: gate_order,
            node_block_results: NodeBlockResults {
                results: node_block_results,
            },
        },
        layout,
    )
}

/// Update the system state as the given `nodes` are added to the routing order, preceded by the
/// given `swaps`.  This involves updating the output values `gate_order` and `out_map`, but also
/// the tracking objects `front_layer`, `extended_set` and `required_predecessors` by removing the
/// routed nodes and adding any now-reachable ones.
fn update_route<F>(
    seed: u64,
    nodes: &[NodeIndex],
    swaps: Vec<[PhysicalQubit; 2]>,
    dag: &SabreDAG,
    layout: &NLayout,
    coupling: &DiGraph<(), ()>,
    gate_order: &mut Vec<usize>,
    out_map: &mut HashMap<usize, Vec<[PhysicalQubit; 2]>>,
    front_layer: &mut FrontLayer,
    extended_set: &mut ExtendedSet,
    required_predecessors: &mut [u32],
    node_block_results: &mut HashMap<usize, Vec<BlockResult>>,
    route_block_dag: &F,
) where
    F: Fn(&SabreDAG, &NLayout) -> (SabreResult, NLayout),
{
    // First node gets the swaps attached.  We don't add to the `gate_order` here because
    // `route_reachable_nodes` is responsible for that part.
    out_map.insert(dag.dag[nodes[0]].py_node_id, swaps);
    for node in nodes {
        front_layer.remove(node);
    }
    route_reachable_nodes(
        seed,
        nodes,
        dag,
        layout,
        coupling,
        gate_order,
        front_layer,
        required_predecessors,
        node_block_results,
        route_block_dag,
    );
    // Ideally we'd know how to mutate the extended set directly, but since its limited size ties
    // its construction strongly to the iteration order through the front layer, it's not easy to
    // do better than just emptying it and rebuilding.
    extended_set.clear();
    populate_extended_set(
        extended_set,
        dag,
        front_layer,
        layout,
        required_predecessors,
    );
}

fn gen_swap_epilogue(
    coupling: &DiGraph<(), ()>,
    mut from_layout: NLayout,
    to_layout: &NLayout,
    seed: u64,
) -> Vec<[PhysicalQubit; 2]> {
    // Map physical location in from_layout to physical location in to_layout
    let mapping: HashMap<NodeIndex, NodeIndex> = from_layout
        .iter_physical()
        .map(|(p, v)| {
            (
                NodeIndex::new(p.index()),
                NodeIndex::new(v.to_phys(to_layout).index()),
            )
        })
        .collect();

    let swaps = token_swapper(
        coupling,
        mapping,
        Some(SWAP_EPILOGUE_TRIALS),
        Some(seed),
        None,
    )
    .unwrap();

    // Convert physical swaps to virtual swaps
    swaps
        .into_iter()
        .map(|(l, r)| {
            let p_l = PhysicalQubit::new(l.index().try_into().unwrap());
            let p_r = PhysicalQubit::new(r.index().try_into().unwrap());
            from_layout.swap_physical(p_l, p_r);
            [p_l, p_r]
        })
        .collect()
}

/// Search forwards in the `dag` from all the nodes in `to_visit`, adding them to the `gate_order`
/// or the current `front_layer` as appropriate, and continue inspecting gates until there is
/// nothing further with no required predecessors.
///
/// The nodes in `to_visit` should all already have no further required predecessors.
fn route_reachable_nodes<F>(
    seed: u64,
    to_visit: &[NodeIndex],
    dag: &SabreDAG,
    layout: &NLayout,
    coupling: &DiGraph<(), ()>,
    gate_order: &mut Vec<usize>,
    front_layer: &mut FrontLayer,
    required_predecessors: &mut [u32],
    node_block_results: &mut HashMap<usize, Vec<BlockResult>>,
    route_block_dag: &F,
) where
    F: Fn(&SabreDAG, &NLayout) -> (SabreResult, NLayout),
{
    let mut to_visit = to_visit.to_vec();
    let mut i = 0;
    // Iterate through `to_visit`, except we often push new nodes onto the end of it.
    while i < to_visit.len() {
        let node_id = to_visit[i];
        let node = &dag.dag[node_id];
        i += 1;
        // If the node is a directive that means it can be placed anywhere
        if !node.directive {
            match dag.node_blocks.get(&node.py_node_id) {
                Some(blocks) => {
                    // Control flow op. Route all blocks for current layout.
                    let mut block_results: Vec<BlockResult> = Vec::with_capacity(blocks.len());
                    for inner_dag in blocks {
                        let (inner_dag_routed, inner_final_layout) =
                            route_block_dag(inner_dag, layout);
                        // For now, we always append a swap circuit that gets the inner block
                        // back to the parent's layout.
                        let swap_epilogue =
                            gen_swap_epilogue(coupling, inner_final_layout, layout, seed);
                        let block_result = BlockResult {
                            result: inner_dag_routed,
                            swap_epilogue,
                        };
                        block_results.push(block_result);
                    }
                    node_block_results.insert_unique_unchecked(node.py_node_id, block_results);
                }
                None => match node.qubits[..] {
                    // A gate op whose connectivity must match the device to be
                    // placed in the gate order.
                    [a, b]
                        if !coupling.contains_edge(
                            NodeIndex::new(a.to_phys(layout).index()),
                            NodeIndex::new(b.to_phys(layout).index()),
                        ) =>
                    {
                        // 2Q op that cannot be placed. Add it to the front layer
                        // and move on.
                        front_layer.insert(node_id, [a.to_phys(layout), b.to_phys(layout)]);
                        continue;
                    }
                    _ => {}
                },
            }
        }

        gate_order.push(node.py_node_id);
        for edge in dag.dag.edges_directed(node_id, Direction::Outgoing) {
            let successor_node = edge.target();
            let successor_index = successor_node.index();
            required_predecessors[successor_index] -= 1;
            if required_predecessors[successor_index] == 0 {
                to_visit.push(successor_node);
            }
        }
    }
}

/// Walk through the swaps in the given vector, undoing them on the layout and removing them.
fn undo_swaps(
    swaps: &mut Vec<[PhysicalQubit; 2]>,
    front_layer: &mut FrontLayer,
    extended_set: &mut ExtendedSet,
    layout: &mut NLayout,
) {
    swaps.drain(..).rev().for_each(|swap| {
        front_layer.apply_swap(swap);
        extended_set.apply_swap(swap);
        layout.swap_physical(swap[0], swap[1]);
    });
}

/// Find the node index and its associated virtual qubits that is currently the closest to being
/// routable in terms of number of swaps.
fn closest_operation<'a>(
    front_layer: &'a FrontLayer,
    dist: &'_ ArrayView2<f64>,
) -> (&'a NodeIndex, &'a [PhysicalQubit; 2]) {
    front_layer
        .iter()
        .min_by(|(_, qubits_a), (_, qubits_b)| {
            dist[[qubits_a[0].index(), qubits_a[1].index()]]
                .partial_cmp(&dist[[qubits_b[0].index(), qubits_b[1].index()]])
                .unwrap_or(Ordering::Equal)
        })
        .unwrap()
}

/// Add the minimal set of swaps to the `swaps` vector that bring the two `qubits` together so that
/// a 2q gate on them could be routed.
fn swaps_to_route(
    swaps: &mut Vec<[PhysicalQubit; 2]>,
    qubits: &[PhysicalQubit; 2],
    coupling_graph: &DiGraph<(), ()>,
) {
    let mut shortest_paths: DictMap<NodeIndex, Vec<NodeIndex>> = DictMap::new();
    (dijkstra(
        coupling_graph,
        NodeIndex::new(qubits[0].index()),
        Some(NodeIndex::new(qubits[1].index())),
        |_| Ok(1.),
        Some(&mut shortest_paths),
    ) as PyResult<Vec<Option<f64>>>)
        .unwrap();
    let shortest_path = shortest_paths
        .get(&NodeIndex::new(qubits[1].index()))
        .unwrap()
        .iter()
        .map(|n| PhysicalQubit::new(n.index() as u32))
        .collect::<Vec<_>>();
    // Insert greedy swaps along that shortest path, splitting them between moving the left side
    // and moving the right side to minimise the depth.  One side needs to move up to the split
    // point and the other can stop one short because the gate will be routable then.
    let split: usize = shortest_path.len() / 2;
    swaps.reserve(shortest_path.len() - 2);
    for i in 0..split {
        swaps.push([shortest_path[i], shortest_path[i + 1]]);
    }
    for i in 0..split - 1 {
        let end = shortest_path.len() - 1 - i;
        swaps.push([shortest_path[end], shortest_path[end - 1]]);
    }
}

/// Return the swap of two virtual qubits that produces the best score of all possible swaps.
fn choose_best_swap(
    layer: &FrontLayer,
    extended_set: &ExtendedSet,
    neighbor_table: &NeighborTable,
    dist: &ArrayView2<f64>,
    qubits_decay: &[f64],
    heuristic: &Heuristic,
    rng: &mut Pcg64Mcg,
    best_swaps: &mut Vec<[PhysicalQubit; 2]>,
) -> [PhysicalQubit; 2] {
    best_swaps.clear();
    let mut min_score = f64::MAX;
    // The decay heuristic is the only one that actually needs the absolute score.
    let absolute_score = match heuristic {
        Heuristic::Decay => {
            layer.total_score(dist) + EXTENDED_SET_WEIGHT * extended_set.total_score(dist)
        }
        _ => 0.0,
    };
    for swap in obtain_swaps(layer, neighbor_table) {
        let score = match heuristic {
            Heuristic::Basic => layer.score(swap, dist),
            Heuristic::Lookahead => {
                layer.score(swap, dist) + EXTENDED_SET_WEIGHT * extended_set.score(swap, dist)
            }
            Heuristic::Decay => {
                qubits_decay[swap[0].index()].max(qubits_decay[swap[1].index()])
                    * (absolute_score
                        + layer.score(swap, dist)
                        + EXTENDED_SET_WEIGHT * extended_set.score(swap, dist))
            }
        };
        if score < min_score - BEST_EPSILON {
            min_score = score;
            best_swaps.clear();
            best_swaps.push(swap);
        } else if (score - min_score).abs() < BEST_EPSILON {
            best_swaps.push(swap);
        }
    }
    *best_swaps.choose(rng).unwrap()
}

#[pymodule]
pub fn sabre_swap(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(build_swap_map))?;
    m.add_class::<Heuristic>()?;
    m.add_class::<NeighborTable>()?;
    m.add_class::<SabreDAG>()?;
    m.add_class::<SwapMap>()?;
    m.add_class::<BlockResult>()?;
    m.add_class::<NodeBlockResults>()?;
    m.add_class::<SabreResult>()?;
    Ok(())
}
