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

pub mod edge_list;
pub mod neighbor_table;
pub mod qubits_decay;

use ndarray::prelude::*;
use numpy::IntoPyArray;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;

use hashbrown::HashSet;
use rayon::prelude::*;

use crate::nlayout::NLayout;

use edge_list::EdgeList;
use neighbor_table::NeighborTable;
use qubits_decay::QubitsDecay;

const EXTENDED_SET_WEIGHT: f64 = 0.5; // Weight of lookahead window compared to front_layer.

#[pyclass]
pub enum Heuristic {
    Basic,
    Lookahead,
    Decay,
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
    front_layer: &EdgeList,
    neighbors: &NeighborTable,
    layout: &NLayout,
) -> HashSet<[usize; 2]> {
    let mut candidate_swaps: HashSet<[usize; 2]> = HashSet::new();
    for node in &front_layer.edges {
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

/// Run the sabre heuristic scoring
///
/// Args:
///     layers (EdgeList): The input layer edge list to score and find the
///         best swaps
///     layout (NLayout): The current layout
///     neighbor_table (NeighborTable): The table of neighbors for each node
///         in the coupling graph
///     extended_set (EdgeList): The extended set
///     distance_matrix (ndarray): The 2D array distance matrix for the coupling
///         graph
///     qubits_decay (QubitsDecay): The current qubit decay factors for
///     heuristic (Heuristic): The chosen heuristic method to use
/// Returns:
///     ndarray: A 2d array of the best swap candidates all with the minimum score
#[pyfunction]
pub fn sabre_score_heuristic(
    py: Python,
    layer: EdgeList,
    layout: &NLayout,
    neighbor_table: &NeighborTable,
    extended_set: EdgeList,
    distance_matrix: PyReadonlyArray2<f64>,
    qubits_decay: QubitsDecay,
    heuristic: &Heuristic,
) -> PyObject {
    let dist = distance_matrix.as_array();
    let candidate_swaps = obtain_swaps(&layer, neighbor_table, layout);
    let mut min_score = f64::MAX;
    let mut best_swaps: Vec<[usize; 2]> = Vec::new();
    for swap_qubits in candidate_swaps {
        let mut trial_layout = layout.clone();
        trial_layout.swap_logical(swap_qubits[0], swap_qubits[1]);
        let score = score_heuristic(
            heuristic,
            &layer.edges,
            &extended_set.edges,
            &trial_layout,
            &swap_qubits,
            &dist,
            &qubits_decay.decay,
        );
        if score < min_score {
            min_score = score;
            best_swaps.clear();
            best_swaps.push(swap_qubits);
        } else if score == min_score {
            best_swaps.push(swap_qubits);
        }
    }
    best_swaps.par_sort();
    Array2::from(best_swaps).into_pyarray(py).into()
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
    m.add_wrapped(wrap_pyfunction!(sabre_score_heuristic))?;
    m.add_class::<Heuristic>()?;
    m.add_class::<EdgeList>()?;
    m.add_class::<QubitsDecay>()?;
    m.add_class::<NeighborTable>()?;
    Ok(())
}
