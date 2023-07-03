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

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;

use crate::error_map::ErrorMap;
use crate::nlayout::NLayout;

const PARALLEL_THRESHOLD: usize = 50;

/// Score a given circuit with a layout applied
#[pyfunction]
#[pyo3(
    text_signature = "(bit_list, edge_list, error_matrix, layout, strict_direction, run_in_parallel, /)"
)]
pub fn score_layout(
    bit_list: PyReadonlyArray1<i32>,
    edge_list: IndexMap<[usize; 2], i32>,
    error_map: &ErrorMap,
    layout: &NLayout,
    strict_direction: bool,
    run_in_parallel: bool,
) -> PyResult<f64> {
    let bit_counts = bit_list.as_slice()?;
    let edge_filter_map = |(index_arr, gate_count): (&[usize; 2], &i32)| -> Option<f64> {
        let mut error = error_map.error_map.get(&[
            layout.logic_to_phys[index_arr[0]],
            layout.logic_to_phys[index_arr[1]],
        ]);
        if !strict_direction && error.is_none() {
            error = error_map.error_map.get(&[
                layout.logic_to_phys[index_arr[1]],
                layout.logic_to_phys[index_arr[0]],
            ]);
        }
        error.map(|error| {
            if !error.is_nan() {
                (1. - error).powi(*gate_count)
            } else {
                1.
            }
        })
    };
    let bit_filter_map = |(index, gate_counts): (usize, &i32)| -> Option<f64> {
        let bit_index = layout.logic_to_phys[index];
        let error = error_map.error_map.get(&[bit_index, bit_index]);

        error.map(|error| {
            if !error.is_nan() {
                (1. - error).powi(*gate_counts)
            } else {
                1.
            }
        })
    };

    let mut fidelity: f64 = if edge_list.len() < PARALLEL_THRESHOLD || !run_in_parallel {
        edge_list.iter().filter_map(edge_filter_map).product()
    } else {
        edge_list.par_iter().filter_map(edge_filter_map).product()
    };
    fidelity *= if bit_list.len() < PARALLEL_THRESHOLD || !run_in_parallel {
        bit_counts
            .iter()
            .enumerate()
            .filter_map(bit_filter_map)
            .product::<f64>()
    } else {
        bit_counts
            .par_iter()
            .enumerate()
            .filter_map(bit_filter_map)
            .product()
    };
    Ok(1. - fidelity)
}

#[pymodule]
pub fn vf2_layout(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(score_layout))?;
    Ok(())
}
