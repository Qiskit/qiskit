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

use indexmap::IndexMap;

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;

use crate::getenv_use_multiple_threads;
use crate::nlayout::NLayout;

const PARALLEL_THRESHOLD: usize = 50;

/// Score a given circuit with a layout applied
#[pyfunction]
#[pyo3(text_signature = "(data, num_qubits, z_mask, /)")]
pub fn score_layout(
    bit_list: PyReadonlyArray1<i32>,
    edge_list: IndexMap<[usize; 2], i32>,
    error_matrix: PyReadonlyArray2<f64>,
    layout: &NLayout,
    strict_direction: bool,
) -> PyResult<f64> {
    let bit_counts = bit_list.as_slice()?;
    let err_mat = error_matrix.as_array();
    // Run in parallel only if we're not already in a multiprocessing context
    // unless force threads is set.
    let run_in_parallel = getenv_use_multiple_threads();
    let mut fidelity = if edge_list.len() < PARALLEL_THRESHOLD || !run_in_parallel {
        edge_list
            .iter()
            .filter_map(|(index_arr, gate_count)| {
                let mut error = err_mat[[
                    layout.logic_to_phys[index_arr[0]],
                    layout.logic_to_phys[index_arr[1]],
                ]];
                if !strict_direction && error.is_nan() {
                    error = err_mat[[
                        layout.logic_to_phys[index_arr[1]],
                        layout.logic_to_phys[index_arr[0]],
                    ]];
                }
                if !error.is_nan() {
                    Some((1. - error).powi(*gate_count))
                } else {
                    None
                }
            })
            .reduce(|a, b| a * b)
            .unwrap_or(1.)
    } else {
        edge_list
            .par_iter()
            .filter_map(|(index_arr, gate_count)| {
                let mut error = err_mat[[
                    layout.logic_to_phys[index_arr[0]],
                    layout.logic_to_phys[index_arr[1]],
                ]];
                if !strict_direction && error.is_nan() {
                    error = err_mat[[
                        layout.logic_to_phys[index_arr[1]],
                        layout.logic_to_phys[index_arr[0]],
                    ]];
                }
                if !error.is_nan() {
                    Some((1. - error).powi(*gate_count))
                } else {
                    None
                }
            })
            .reduce(|| 1., |a, b| a * b)
    };
    fidelity *= if bit_list.len() < PARALLEL_THRESHOLD || !run_in_parallel {
        bit_counts
            .iter()
            .enumerate()
            .filter_map(|(index, gate_counts)| {
                let bit_index = layout.logic_to_phys[index];
                let error = err_mat[[bit_index, bit_index]];

                if !error.is_nan() {
                    Some((1. - error).powi(*gate_counts))
                } else {
                    None
                }
            })
            .reduce(|a, b| a * b)
            .unwrap_or(1.)
    } else {
        bit_counts
            .par_iter()
            .enumerate()
            .filter_map(|(index, gate_counts)| {
                let bit_index = layout.logic_to_phys[index];
                let error = err_mat[[bit_index, bit_index]];
                if error > 0. {
                    Some((1. - error).powi(*gate_counts))
                } else {
                    None
                }
            })
            .reduce(|| 1., |a, b| a * b)
    };
    Ok(1. - fidelity)
}

#[pymodule]
pub fn vf2_layout(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(score_layout))?;
    Ok(())
}
