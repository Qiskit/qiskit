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

// Needed to pass shared state between functions
// closures don't work because of recurssion
#![allow(clippy::too_many_arguments)]

use std::sync::RwLock;

use hashbrown::HashSet;

use ndarray::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use rayon::prelude::*;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;

use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use rand_pcg::Pcg64Mcg;

use crate::edge_collections::EdgeCollection;
use crate::nlayout::NLayout;

#[inline]
fn compute_cost(
    dist: &ArrayView2<f64>,
    layout: &NLayout,
    gates: &[usize],
    num_gates: usize,
) -> f64 {
    (0..num_gates)
        .into_par_iter()
        .map(|kk| {
            let ii = layout.logic_to_phys[gates[2 * kk]];
            let jj = layout.logic_to_phys[gates[2 * kk + 1]];
            dist[[ii, jj]]
        })
        .sum()
}

/// Computes the symmetric random scaling (perturbation) matrix,
/// and places the values in the 'scale' array.
///
/// Args:
///     scale (ndarray): An array of doubles where the values are to be stored.
///     cdist2 (ndarray): Array representing the coupling map distance squared.
///     rand (double *): Array of rands of length num_qubits*(num_qubits+1)//2.
///     num_qubits (int): Number of physical qubits.
#[inline]
fn compute_random_scaling(
    scale: &mut Array2<f64>,
    cdist2: &ArrayView2<f64>,
    rand: &[f64],
    num_qubits: usize,
) {
    let mut idx: usize = 0;
    for ii in 0..num_qubits {
        for jj in 0..ii {
            scale[[ii, jj]] = rand[idx] * cdist2[[ii, jj]];
            scale[[jj, ii]] = scale[[ii, jj]];
            idx += 1
        }
    }
}

fn swap_trial(
    num_qubits: usize,
    int_layout: &NLayout,
    int_qubit_subset: &[usize],
    gates: &[usize],
    cdist: ArrayView2<f64>,
    cdist2: ArrayView2<f64>,
    edges: &[usize],
    seed: Option<u64>,
    trial_num: u64,
    locked_best_possible: &RwLock<&mut Option<(f64, EdgeCollection, NLayout, usize)>>,
) -> Option<(f64, EdgeCollection, NLayout, usize)> {
    if locked_best_possible.read().unwrap().is_some() {
        return None;
    }
    let mut opt_edges = EdgeCollection::new();
    let mut new_layout = int_layout.clone();
    let mut trial_layout = int_layout.clone();
    let mut optimal_layout = int_layout.clone();

    let num_gates: usize = gates.len() / 2;
    let num_edges: usize = edges.len() / 2;

    let mut need_copy;
    let mut cost_reduced;
    let mut depth_step: usize = 1;
    let depth_max: usize = 2 * num_qubits + 1;
    let mut min_cost: f64;
    let mut new_cost: f64;
    let mut dist: f64;

    let mut optimal_start: usize = std::usize::MAX;
    let mut optimal_end: usize = std::usize::MAX;
    let mut optimal_start_qubit = std::usize::MAX;
    let mut optimal_end_qubit = std::usize::MAX;

    let mut scale = Array2::zeros((num_qubits, num_qubits));

    let distribution = Normal::new(1.0, 1.0 / num_qubits as f64).unwrap();
    let mut rng: Pcg64Mcg = match seed {
        Some(seed) => Pcg64Mcg::seed_from_u64(seed + trial_num),
        None => Pcg64Mcg::from_entropy(),
    };
    let rand: Vec<f64> = distribution
        .sample_iter(&mut rng)
        .take(num_qubits * (num_qubits + 1) / 2)
        .collect();

    compute_random_scaling(&mut scale, &cdist2, &rand, num_qubits);

    let mut qubit_set: HashSet<usize>;
    let input_qubit_set: HashSet<usize> = int_qubit_subset.iter().copied().collect();

    while depth_step < depth_max {
        qubit_set = input_qubit_set.clone();
        while !qubit_set.is_empty() {
            min_cost = compute_cost(&scale.view(), &trial_layout, gates, num_gates);
            // Try to decrease the objective function
            cost_reduced = false;
            // Loop over edges of coupling graph
            need_copy = true;
            for idx in 0..num_edges {
                let start_edge = edges[2 * idx];
                let end_edge = edges[2 * idx + 1];
                let start_qubit = trial_layout.phys_to_logic[start_edge];
                let end_qubit = trial_layout.phys_to_logic[end_edge];
                if qubit_set.contains(&start_qubit) && qubit_set.contains(&end_qubit) {
                    // Try this edge to reduce cost
                    if need_copy {
                        new_layout = trial_layout.clone();
                        need_copy = false;
                    }
                    new_layout.swap(start_edge, end_edge);
                    // compute objective function
                    new_cost = compute_cost(&scale.view(), &new_layout, gates, num_gates);
                    // record progress if we succeed
                    if new_cost < min_cost {
                        cost_reduced = true;
                        min_cost = new_cost;
                        optimal_layout = new_layout.clone();
                        optimal_start = start_edge;
                        optimal_end = end_edge;
                        optimal_start_qubit = start_qubit;
                        optimal_end_qubit = end_qubit;
                        need_copy = true;
                    } else {
                        new_layout.swap(start_edge, end_edge);
                    }
                }
            }
            // After going over all edges
            // Were there any good swap choices?
            if cost_reduced {
                qubit_set.remove(&optimal_start_qubit);
                qubit_set.remove(&optimal_end_qubit);
                trial_layout = optimal_layout.clone();
                opt_edges.add(optimal_start, optimal_end);
            } else {
                break;
            }
        }
        // We have either run out of swap pairs to try or failed to improve
        // the cost

        // Compute the coupling graph distance
        dist = compute_cost(&cdist, &trial_layout, gates, num_gates);
        // If all gates can be applied now we're finished.
        // Otherwise we need to consider a deeper swap circuit
        if dist as usize == num_gates {
            break;
        }
        // increment the depth
        depth_step += 1;
    }
    // Either we have succeeded at some depth d < d_max or failed
    dist = compute_cost(&cdist, &trial_layout, gates, num_gates);
    if dist as usize == num_gates && depth_step == 1 {
        let mut best_possible = locked_best_possible.write().unwrap();
        **best_possible = Some((dist, opt_edges, trial_layout, depth_step));
        return None;
    }
    Some((dist, opt_edges, trial_layout, depth_step))
}

#[pyfunction]
#[pyo3(text_signature = "(graph, weight_fn, /)")]
pub fn swap_trials(
    num_trials: u64,
    num_qubits: usize,
    num_gates: usize,
    int_layout: &NLayout,
    int_qubit_subset: PyReadonlyArray1<usize>,
    int_gates: PyReadonlyArray1<usize>,
    cdist: PyReadonlyArray2<f64>,
    cdist2: PyReadonlyArray2<f64>,
    edges: PyReadonlyArray1<usize>,
    seed: Option<u64>,
) -> PyResult<(Option<EdgeCollection>, Option<NLayout>, usize)> {
    let int_qubit_subset_arr = int_qubit_subset.as_slice()?;
    let int_gates_arr = int_gates.as_slice()?;
    let cdist_arr = cdist.as_array();
    let cdist2_arr = cdist2.as_array();
    let edges_arr = edges.as_slice()?;
    let mut best_possible: Option<(f64, EdgeCollection, NLayout, usize)> = None;
    let locked_best_possible: RwLock<&mut Option<(f64, EdgeCollection, NLayout, usize)>> =
        RwLock::new(&mut best_possible);
    let result: Vec<Option<(f64, EdgeCollection, NLayout, usize)>> = (0..num_trials)
        .into_par_iter()
        .map(|trial_num| {
            swap_trial(
                num_qubits,
                int_layout,
                int_qubit_subset_arr,
                int_gates_arr,
                cdist_arr,
                cdist2_arr,
                edges_arr,
                seed,
                trial_num,
                &locked_best_possible,
            )
        })
        .collect();
    match best_possible {
        Some((_dist, edges, layout, depth)) => Ok((Some(edges), Some(layout), depth)),
        None => {
            let mut best_depth = std::usize::MAX;
            let mut best_edges: Option<EdgeCollection> = None;
            let mut best_layout: Option<NLayout> = None;
            for (dist, edges, layout, depth) in result.into_iter().flatten() {
                if dist as usize == num_gates && depth < best_depth {
                    best_edges = Some(edges);
                    best_layout = Some(layout);
                    best_depth = depth;
                }
            }
            Ok((best_edges, best_layout, best_depth))
        }
    }
}

#[pymodule]
pub fn stochastic_swap(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(swap_trials))?;
    m.add_class::<NLayout>()?;
    m.add_class::<EdgeCollection>()?;
    Ok(())
}
