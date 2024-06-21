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
use smallvec::smallvec;
use std::vec::Vec;

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{Param, StandardGate};
use qiskit_circuit::Qubit;

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

/// Sorts the input permutation by iterating through the permutation list
/// and putting each element to its correct position via a SWAP (if it's not
/// at the correct position already). If ``n`` is the length of the input
/// permutation, this requires at most ``n`` SWAPs.
///
/// More precisely, if the input permutation is a cycle of length ``m``,
/// then this creates a quantum circuit with ``m-1`` SWAPs (and of depth ``m-1``);
/// if the input  permutation consists of several disjoint cycles, then each cycle
/// is essentially treated independently.
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

/// Checks whether an array of size N is a permutation of 0, 1, ..., N - 1.
#[pyfunction]
#[pyo3(signature = (pattern))]
fn _validate_permutation(py: Python, pattern: PyArrayLike1<i64>) -> PyResult<PyObject> {
    let view = pattern.as_array();
    validate_permutation(&view)?;
    Ok(py.None())
}

/// Finds inverse of a permutation pattern.
#[pyfunction]
#[pyo3(signature = (pattern))]
fn _inverse_pattern(py: Python, pattern: PyArrayLike1<i64>) -> PyResult<PyObject> {
    let view = pattern.as_array();
    let inverse_i64: Vec<i64> = invert(&view).iter().map(|&x| x as i64).collect();
    Ok(inverse_i64.to_object(py))
}

#[pyfunction]
#[pyo3(signature = (pattern))]
fn _synth_permutation_basic(py: Python, pattern: PyArrayLike1<i64>) -> PyResult<CircuitData> {
    let view = pattern.as_array();
    let num_qubits = view.len();
    CircuitData::from_standard_gates(
        py,
        num_qubits as u32,
        get_ordered_swap(&view).iter().map(|(i, j)| {
            (
                StandardGate::SwapGate,
                smallvec![],
                smallvec![Qubit(*i as u32), Qubit(*j as u32)],
            )
        }),
        Param::Float(0.0),
    )
}

#[pymodule]
pub fn permutation(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_validate_permutation, m)?)?;
    m.add_function(wrap_pyfunction!(_inverse_pattern, m)?)?;
    m.add_function(wrap_pyfunction!(_synth_permutation_basic, m)?)?;
    Ok(())
}
