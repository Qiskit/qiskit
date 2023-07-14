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
use numpy::PyReadonlyArray2;
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
use crate::sabre_swap::{build_swap_map_inner, Heuristic, SabreResult};

#[pyfunction]
pub fn sabre_layout_and_routing(
    dag: &SabreDAG,
    neighbor_table: &NeighborTable,
    distance_matrix: PyReadonlyArray2<f64>,
    heuristic: &Heuristic,
    max_iterations: usize,
    num_swap_trials: usize,
    num_layout_trials: usize,
    seed: Option<u64>,
) -> ([NLayout; 2], SabreResult) {
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
    if run_in_parallel && num_layout_trials > 1 {
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
            .min_by_key(|(index, (_, result))| {
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
            .min_by_key(|(_, result)| result.map.map.values().map(|x| x.len()).sum::<usize>())
            .unwrap()
    }
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
) -> ([NLayout; 2], SabreResult) {
    // Pick a random initial layout and fully populate ancillas in that layout too
    let num_physical_qubits = distance_matrix.shape()[0];
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let mut physical_qubits: Vec<usize> = (0..num_physical_qubits).collect();
    physical_qubits.shuffle(&mut rng);
    let mut initial_layout = NLayout::from_logical_to_physical(physical_qubits);
    let new_dag_fn = |nodes| {
        // Because the current implementation of Sabre swap doesn't permute
        // the layout when placing control flow ops, there's no need to
        // recurse into blocks. We remove them here, but still map control
        // flow node IDs to an empty block list so Sabre treats these ops
        // as control flow nodes, but doesn't route their blocks.
        let node_blocks_empty = dag
            .node_blocks
            .iter()
            .map(|(node_index, _)| (*node_index, Vec::with_capacity(0)));
        SabreDAG::new(
            dag.num_qubits,
            dag.num_clbits,
            nodes,
            node_blocks_empty.collect(),
        )
        .unwrap()
    };

    // Create forward and reverse dags (without node blocks).
    // Once we've settled on a layout, we recursively apply it to the original
    // DAG and its node blocks.
    let mut dag_forward: SabreDAG = new_dag_fn(dag.nodes.clone());
    let mut dag_reverse: SabreDAG = new_dag_fn(dag.nodes.iter().rev().cloned().collect());
    for _iter in 0..max_iterations {
        // forward and reverse
        for _direction in 0..2 {
            let layout_dag = apply_layout(&dag_forward, &initial_layout);
            let mut pass_final_layout = NLayout::generate_trivial_layout(num_physical_qubits);
            build_swap_map_inner(
                num_physical_qubits,
                &layout_dag,
                neighbor_table,
                distance_matrix,
                heuristic,
                Some(seed),
                &mut pass_final_layout,
                num_swap_trials,
                Some(run_swap_in_parallel),
            );
            let final_layout = compose_layout(&initial_layout, &pass_final_layout);
            initial_layout = final_layout;
            std::mem::swap(&mut dag_forward, &mut dag_reverse);
        }
    }

    // Apply the layout to the original DAG.
    let layout_dag = apply_layout(dag, &initial_layout);

    let mut final_layout = NLayout::generate_trivial_layout(num_physical_qubits);
    let sabre_result = build_swap_map_inner(
        num_physical_qubits,
        &layout_dag,
        neighbor_table,
        distance_matrix,
        heuristic,
        Some(seed),
        &mut final_layout,
        num_swap_trials,
        Some(run_swap_in_parallel),
    );
    ([initial_layout, final_layout], sabre_result)
}

fn apply_layout(dag: &SabreDAG, layout: &NLayout) -> SabreDAG {
    let layout_nodes = dag.nodes.iter().map(|(node_index, qargs, cargs)| {
        let new_qargs: Vec<usize> = qargs.iter().map(|n| layout.logic_to_phys[*n]).collect();
        (*node_index, new_qargs, cargs.clone())
    });
    let node_blocks = dag.node_blocks.iter().map(|(node_index, blocks)| {
        (
            *node_index,
            blocks.iter().map(|d| apply_layout(d, layout)).collect(),
        )
    });
    SabreDAG::new(
        dag.num_qubits,
        dag.num_clbits,
        layout_nodes.collect(),
        node_blocks.collect(),
    )
    .unwrap()
}

fn compose_layout(initial_layout: &NLayout, final_layout: &NLayout) -> NLayout {
    let logic_to_phys = initial_layout
        .logic_to_phys
        .iter()
        .map(|n| final_layout.logic_to_phys[*n])
        .collect();
    let phys_to_logic = final_layout
        .phys_to_logic
        .iter()
        .map(|n| initial_layout.phys_to_logic[*n])
        .collect();

    NLayout {
        logic_to_phys,
        phys_to_logic,
    }
}

#[pymodule]
pub fn sabre_layout(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(sabre_layout_and_routing))?;
    Ok(())
}
