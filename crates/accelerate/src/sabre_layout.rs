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

use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;

use crate::getenv_use_multiple_threads;
use crate::nlayout::NLayout;
use crate::sabre_swap::neighbor_table::NeighborTable;
use crate::sabre_swap::sabre_dag::SabreDAG;
use crate::sabre_swap::swap_map::SwapMap;
use crate::sabre_swap::{build_swap_map_inner, Heuristic, NodeBlockResults, SabreResult};

#[pyfunction]
pub fn sabre_layout_and_routing(
    py: Python,
    dag: &SabreDAG,
    neighbor_table: &NeighborTable,
    distance_matrix: PyReadonlyArray2<f64>,
    heuristic: &Heuristic,
    max_iterations: usize,
    num_swap_trials: usize,
    num_layout_trials: usize,
    seed: Option<u64>,
) -> (NLayout, PyObject, (SwapMap, PyObject, NodeBlockResults)) {
    let run_in_parallel = getenv_use_multiple_threads();
    let outer_rng = match seed {
        Some(seed) => Pcg64Mcg::seed_from_u64(seed),
        None => Pcg64Mcg::from_entropy(),
    };
    let seed_vec: Vec<u64> = outer_rng
        .sample_iter(&rand::distributions::Standard)
        .take(num_layout_trials)
        .collect();
    let dist = distance_matrix.as_array();
    let res = if run_in_parallel && num_layout_trials > 1 {
        seed_vec
            .into_par_iter()
            .enumerate()
            .map(|(index, seed_trial)| {
                (
                    index,
                    layout_trial(
                        dag,
                        neighbor_table,
                        &dist,
                        heuristic,
                        seed_trial,
                        max_iterations,
                        num_swap_trials,
                        run_in_parallel,
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
            .map(|seed_trial| {
                layout_trial(
                    dag,
                    neighbor_table,
                    &dist,
                    heuristic,
                    seed_trial,
                    max_iterations,
                    num_swap_trials,
                    run_in_parallel,
                )
            })
            .min_by_key(|(_, _, result)| result.map.map.values().map(|x| x.len()).sum::<usize>())
            .unwrap()
    };
    (
        res.0,
        PyArray::from_vec(py, res.1).into(),
        (
            res.2.map,
            res.2.node_order.into_pyarray(py).into(),
            res.2.node_block_results,
        ),
    )
}

fn layout_trial(
    dag: &SabreDAG,
    neighbor_table: &NeighborTable,
    distance_matrix: &ArrayView2<f64>,
    heuristic: &Heuristic,
    seed: u64,
    max_iterations: usize,
    num_swap_trials: usize,
    run_swap_in_parallel: bool,
) -> (NLayout, Vec<usize>, SabreResult) {
    let num_physical_qubits = distance_matrix.shape()[0];
    let mut rng = Pcg64Mcg::seed_from_u64(seed);

    // Pick a random initial layout including a full ancilla allocation.
    let mut initial_layout = {
        let mut physical_qubits: Vec<usize> = (0..num_physical_qubits).collect();
        physical_qubits.shuffle(&mut rng);
        NLayout::from_logical_to_physical(physical_qubits)
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
    );

    for _iter in 0..max_iterations {
        for dag in [&dag_no_control_forward, &dag_no_control_reverse] {
            let (_result, final_layout) = build_swap_map_inner(
                num_physical_qubits,
                dag,
                neighbor_table,
                distance_matrix,
                heuristic,
                Some(seed),
                &initial_layout,
                num_swap_trials,
                Some(run_swap_in_parallel),
            );
            initial_layout = final_layout;
        }
    }

    let (sabre_result, final_layout) = build_swap_map_inner(
        num_physical_qubits,
        dag,
        neighbor_table,
        distance_matrix,
        heuristic,
        Some(seed),
        &initial_layout,
        num_swap_trials,
        Some(run_swap_in_parallel),
    );
    let final_permutation = initial_layout
        .phys_to_logic
        .iter()
        .map(|initial| final_layout.logic_to_phys[*initial])
        .collect();
    (initial_layout, final_permutation, sabre_result)
}

#[pymodule]
pub fn sabre_layout(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(sabre_layout_and_routing))?;
    Ok(())
}
