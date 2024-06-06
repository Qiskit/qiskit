// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use ndarray::{Array1, ArrayView1};
use numpy::PyArrayLike1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::vec::Vec;

fn validate_permutation(pattern: &ArrayView1<i64>) -> PyResult<()> {
    let n = pattern.len();
    let mut seen: Vec<bool> = vec![false; n];

    for &x in pattern {
        if x < 0 {
            return Err(PyValueError::new_err(
                "Invalid permutation: input contains a negative number.",
            ));
        }

        if x as usize >= n {
            return Err(PyValueError::new_err(format!(
                "Invalid permutation: input has length {} and contains {}.",
                n, x
            )));
        }

        if seen[x as usize] {
            return Err(PyValueError::new_err(format!(
                "Invalid permutation: input contains {} more than once.",
                x
            )));
        }

        seen[x as usize] = true;
    }

    Ok(())
}

fn invert(pattern: &ArrayView1<i64>) -> Array1<usize> {
    let mut inverse: Array1<usize> = Array1::zeros(pattern.len());
    pattern.iter().enumerate().for_each(|(ii, &jj)| {
        inverse[jj as usize] = ii;
    });
    inverse
}

fn get_ordered_swap(pattern: &ArrayView1<i64>) -> Vec<(i64, i64)> {
    let mut permutation: Vec<usize> = pattern.iter().map(|&x| x as usize).collect();
    let mut index_map = invert(pattern);

    let n = permutation.len();
    let mut swaps: Vec<(i64, i64)> = Vec::with_capacity(n);
    for ii in 0..n {
        let val = permutation[ii];
        if val == ii {
            continue;
        }
        let jj = index_map[ii];
        swaps.push((ii as i64, jj as i64));
        (permutation[ii], permutation[jj]) = (permutation[jj], permutation[ii]);
        index_map[val] = jj;
        index_map[ii] = ii;
    }

    swaps[..].reverse();
    swaps
}

/// Finds inverse of a permutation pattern.
#[pyfunction]
#[pyo3(signature = (pattern))]
fn _inverse_pattern(py: Python, pattern: PyArrayLike1<i64>) -> PyResult<PyObject> {
    let view = pattern.as_array();
    validate_permutation(&view)?;
    let inverse_i64: Vec<i64> = invert(&view).iter().map(|&x| x as i64).collect();
    Ok(inverse_i64.to_object(py))
}

/// Sorts the input permutation by iterating through the permutation list
/// and putting each element to its correct position via a SWAP (if it's not
/// at the correct position already). If ``n`` is the length of the input
/// permutation, this requires at most ``n`` SWAPs.
///
/// More precisely, if the input permutation is a cycle of length ``m``,
/// then this creates a quantum circuit with ``m-1`` SWAPs (and of depth ``m-1``);
/// if the input  permutation consists of several disjoint cycles, then each cycle
/// is essentially treated independently.
#[pyfunction]
#[pyo3(signature = (permutation_in))]
fn _get_ordered_swap(py: Python, permutation_in: PyArrayLike1<i64>) -> PyResult<PyObject> {
    let view = permutation_in.as_array();
    validate_permutation(&view)?;
    Ok(get_ordered_swap(&view).to_object(py))
}

#[pymodule]
pub fn permutation(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_inverse_pattern, m)?)?;
    m.add_function(wrap_pyfunction!(_get_ordered_swap, m)?)?;
    Ok(())
}
