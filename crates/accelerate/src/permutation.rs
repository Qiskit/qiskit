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

use numpy::{AllowTypeChange, PyArrayLike1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pymodule;
use pyo3::PyErr;
use std::vec::Vec;

fn validate_permutation(pattern: &PyReadonlyArray1<i64>) -> Result<(), PyErr> {
    let view = pattern.as_array();
    let n = view.len();
    let mut seen: Vec<bool> = vec![false; n];

    for &x in view {
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

fn invert(pattern: &PyReadonlyArray1<i64>) -> Vec<usize> {
    let view = pattern.as_array();
    let mut inverse: Vec<usize> = vec![0; view.len()];
    view.iter().enumerate().for_each(|(ii, &jj)| {
        inverse[jj as usize] = ii;
    });
    inverse
}

#[pyfunction]
#[pyo3(signature = (pattern))]
pub fn _inverse_pattern(
    py: Python,
    pattern: PyArrayLike1<i64, AllowTypeChange>,
) -> PyResult<PyObject> {
    validate_permutation(&pattern)?;
    Ok(invert(&pattern).to_object(py))
}

#[pyfunction]
#[pyo3(signature = (permutation_in))]
pub fn _get_ordered_swap(
    py: Python,
    permutation_in: PyArrayLike1<i64, AllowTypeChange>,
) -> PyResult<PyObject> {
    validate_permutation(&permutation_in)?;

    let mut permutation: Vec<usize> = permutation_in
        .as_array()
        .iter()
        .map(|&x| x as usize)
        .collect();
    let mut index_map = invert(&permutation_in);

    let s = permutation.len();
    let mut swaps: Vec<(i64, i64)> = Vec::with_capacity(s);
    for ii in 0..s {
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
    Ok(swaps.to_object(py))
}

#[pymodule]
pub fn permutation_utils(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_inverse_pattern, m)?)?;
    m.add_function(wrap_pyfunction!(_get_ordered_swap, m)?)?;
    Ok(())
}

#[pymodule]
pub fn permutation(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(permutation_utils))?;
    Ok(())
}
