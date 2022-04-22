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

use ndarray::prelude::*;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;

use rayon::prelude::*;

use crate::edge_list::EdgeList;
use crate::nlayout::NLayout;
use crate::qubits_decay::QubitsDecay;
use crate::swap_scores::SwapScores;

const EXTENDED_SET_WEIGHT: f64 = 0.5;

#[pyclass]
pub enum Heuristic {
    Basic,
    Lookahead,
    Decay,
}

#[pyfunction]
pub fn sabre_score_heuristic(
    layer: EdgeList,
    layout: &NLayout,
    swap_scores: &mut SwapScores,
    extended_set: EdgeList,
    distance_matrix: PyReadonlyArray2<f64>,
    qubits_decay: QubitsDecay,
    heuristic: &Heuristic,
) -> Vec<[usize; 2]> {
    let dist = distance_matrix.as_array();
    swap_scores
        .scores
        .par_iter_mut()
        .for_each(|(swap_qubits, score)| {
            let mut trial_layout = layout.clone();
            trial_layout.swap_logic(swap_qubits[0], swap_qubits[1]);
            *score = score_heuristic(
                heuristic,
                &layer.edges,
                &extended_set.edges,
                &trial_layout,
                swap_qubits,
                &dist,
                &qubits_decay.decay,
            );
        });
    let min_score = swap_scores
        .scores
        .par_values()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let mut best_swaps: Vec<[usize; 2]> = swap_scores
        .scores
        .iter()
        .filter_map(|(k, v)| if v == min_score { Some(*k) } else { None })
        .collect();
    best_swaps.par_sort();
    best_swaps
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
    let second_cost = if !extended_set.is_empty() {
        compute_cost(extended_set, layout, dist)
    } else {
        0.
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
    m.add_class::<SwapScores>()?;
    m.add_class::<Heuristic>()?;
    m.add_class::<EdgeList>()?;
    m.add_class::<QubitsDecay>()?;
    Ok(())
}
