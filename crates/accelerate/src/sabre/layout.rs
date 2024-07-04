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

use pyo3::prelude::*;
use pyo3::Python;

use hashbrown::HashSet;
use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray, PyReadonlyArray2};
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;

use crate::getenv_use_multiple_threads;
use crate::nlayout::{NLayout, PhysicalQubit};

use super::neighbor_table::NeighborTable;
use super::route::{swap_map, swap_map_trial, RoutingTargetView};
use super::sabre_dag::SabreDAG;
use super::swap_map::SwapMap;
use super::{Heuristic, NodeBlockResults, SabreResult};

use crate::dense_layout::best_subset_inner;

#[pyfunction]
#[pyo3(signature = (dag, neighbor_table, distance_matrix, heuristic, max_iterations, num_swap_trials, num_random_trials, seed=None, partial_layouts=vec![]))]
pub fn sabre_layout_and_routing(
    py: Python,
    dag: &SabreDAG,
    neighbor_table: &NeighborTable,
    distance_matrix: PyReadonlyArray2<f64>,
    heuristic: Heuristic,
    max_iterations: usize,
    num_swap_trials: usize,
    num_random_trials: usize,
    seed: Option<u64>,
    mut partial_layouts: Vec<Vec<Option<u32>>>,
) -> (NLayout, PyObject, (SwapMap, PyObject, NodeBlockResults)) {
    let run_in_parallel = getenv_use_multiple_threads();
    let target = RoutingTargetView {
        neighbors: neighbor_table,
        coupling: &neighbor_table.coupling_graph(),
        distance: distance_matrix.as_array(),
    };
    let mut starting_layouts: Vec<Vec<Option<u32>>> =
        (0..num_random_trials).map(|_| vec![]).collect();
    starting_layouts.append(&mut partial_layouts);
    // Run a dense layout trial
    starting_layouts.push(compute_dense_starting_layout(
        dag.num_qubits,
        &target,
        run_in_parallel,
    ));
    let outer_rng = match seed {
        Some(seed) => Pcg64Mcg::seed_from_u64(seed),
        None => Pcg64Mcg::from_entropy(),
    };
    let seed_vec: Vec<u64> = outer_rng
        .sample_iter(&rand::distributions::Standard)
        .take(starting_layouts.len())
        .collect();
    let res = if run_in_parallel && starting_layouts.len() > 1 {
        seed_vec
            .into_par_iter()
            .enumerate()
            .map(|(index, seed_trial)| {
                (
                    index,
                    layout_trial(
                        &target,
                        dag,
                        heuristic,
                        seed_trial,
                        max_iterations,
                        num_swap_trials,
                        run_in_parallel,
                        &starting_layouts[index],
                    ),
                )
            })
            .min_by_key(|(index, (_, _, result))| {
                (
                    result.map.map.values().map(|x| x.len()).sum::<usize>(),
                    *index,
                )
            })
            .unwrap()
            .1
    } else {
        seed_vec
            .into_iter()
            .enumerate()
            .map(|(index, seed_trial)| {
                layout_trial(
                    &target,
                    dag,
                    heuristic,
                    seed_trial,
                    max_iterations,
                    num_swap_trials,
                    run_in_parallel,
                    &starting_layouts[index],
                )
            })
            .min_by_key(|(_, _, result)| result.map.map.values().map(|x| x.len()).sum::<usize>())
            .unwrap()
    };
    (
        res.0,
        PyArray::from_vec_bound(py, res.1).into(),
        (
            res.2.map,
            res.2.node_order.into_pyarray_bound(py).into(),
            res.2.node_block_results,
        ),
    )
}

fn layout_trial(
    target: &RoutingTargetView,
    dag: &SabreDAG,
    heuristic: Heuristic,
    seed: u64,
    max_iterations: usize,
    num_swap_trials: usize,
    run_swap_in_parallel: bool,
    starting_layout: &[Option<u32>],
) -> (NLayout, Vec<PhysicalQubit>, SabreResult) {
    let num_physical_qubits: u32 = target.neighbors.num_qubits().try_into().unwrap();
    let mut rng = Pcg64Mcg::seed_from_u64(seed);

    // This is purely for RNG compatibility during a refactor.
    let routing_seed = Pcg64Mcg::seed_from_u64(seed).next_u64();

    // Pick a random initial layout including a full ancilla allocation.
    let mut initial_layout = {
        let physical_qubits: Vec<PhysicalQubit> = if !starting_layout.is_empty() {
            let used_bits: HashSet<u32> = starting_layout
                .iter()
                .filter_map(|x| x.as_ref())
                .copied()
                .collect();
            let mut free_bits: Vec<u32> = (0..num_physical_qubits)
                .filter(|x| !used_bits.contains(x))
                .collect();
            free_bits.shuffle(&mut rng);
            (0..num_physical_qubits)
                .map(|x| {
                    let bit_index = match starting_layout.get(x as usize) {
                        Some(phys) => phys.unwrap_or_else(|| free_bits.pop().unwrap()),
                        None => free_bits.pop().unwrap(),
                    };
                    PhysicalQubit::new(bit_index)
                })
                .collect()
        } else {
            let mut physical_qubits: Vec<PhysicalQubit> =
                (0..num_physical_qubits).map(PhysicalQubit::new).collect();
            physical_qubits.shuffle(&mut rng);
            physical_qubits
        };
        NLayout::from_virtual_to_physical(physical_qubits).unwrap()
    };

    // Sabre routing currently enforces that control-flow blocks return to their starting layout,
    // which means they don't actually affect any heuristics that affect our layout choice.
    let dag_no_control_forward = SabreDAG {
        num_qubits: dag.num_qubits,
        num_clbits: dag.num_clbits,
        dag: dag.dag.clone(),
        nodes: dag.nodes.clone(),
        first_layer: dag.first_layer.clone(),
        node_blocks: dag
            .node_blocks
            .keys()
            .map(|index| (*index, Vec::new()))
            .collect(),
    };
    let dag_no_control_reverse = SabreDAG::new(
        dag_no_control_forward.num_qubits,
        dag_no_control_forward.num_clbits,
        dag_no_control_forward.nodes.iter().rev().cloned().collect(),
        dag_no_control_forward.node_blocks.clone(),
    )
    .unwrap();

    for _iter in 0..max_iterations {
        for dag in [&dag_no_control_forward, &dag_no_control_reverse] {
            let (_result, final_layout) =
                swap_map_trial(target, dag, heuristic, &initial_layout, routing_seed);
            initial_layout = final_layout;
        }
    }

    let (sabre_result, final_layout) = swap_map(
        target,
        dag,
        heuristic,
        &initial_layout,
        Some(seed),
        num_swap_trials,
        Some(run_swap_in_parallel),
    );
    let final_permutation = initial_layout
        .iter_physical()
        .map(|(_, virt)| virt.to_phys(&final_layout))
        .collect();
    (initial_layout, final_permutation, sabre_result)
}

fn compute_dense_starting_layout(
    num_qubits: usize,
    target: &RoutingTargetView,
    run_in_parallel: bool,
) -> Vec<Option<u32>> {
    let mut adj_matrix = target.distance.to_owned();
    if run_in_parallel {
        adj_matrix.par_mapv_inplace(|x| if x == 1. { 1. } else { 0. });
    } else {
        adj_matrix.mapv_inplace(|x| if x == 1. { 1. } else { 0. });
    }
    let [_rows, _cols, map] = best_subset_inner(
        num_qubits,
        adj_matrix.view(),
        0,
        0,
        false,
        true,
        aview2(&[[0.]]),
    );
    map.into_iter().map(|x| Some(x as u32)).collect()
}
