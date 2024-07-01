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

use numpy::PyArrayLike1;
use smallvec::smallvec;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{Param, StandardGate};
use qiskit_circuit::Qubit;

mod utils;

/// Checks whether an array of size N is a permutation of 0, 1, ..., N - 1.
#[pyfunction]
#[pyo3(signature = (pattern))]
pub fn _validate_permutation(py: Python, pattern: PyArrayLike1<i64>) -> PyResult<PyObject> {
    let view = pattern.as_array();
    utils::validate_permutation(&view)?;
    Ok(py.None())
}

/// Finds inverse of a permutation pattern.
#[pyfunction]
#[pyo3(signature = (pattern))]
pub fn _inverse_pattern(py: Python, pattern: PyArrayLike1<i64>) -> PyResult<PyObject> {
    let view = pattern.as_array();
    let inverse_i64: Vec<i64> = utils::invert(&view).iter().map(|&x| x as i64).collect();
    Ok(inverse_i64.to_object(py))
}

#[pyfunction]
#[pyo3(signature = (pattern))]
pub fn _synth_permutation_basic(py: Python, pattern: PyArrayLike1<i64>) -> PyResult<CircuitData> {
    let view = pattern.as_array();
    let num_qubits = view.len();
    CircuitData::from_standard_gates(
        py,
        num_qubits as u32,
        utils::get_ordered_swap(&view).iter().map(|(i, j)| {
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
