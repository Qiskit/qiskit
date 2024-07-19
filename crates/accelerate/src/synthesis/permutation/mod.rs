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

#[pyfunction]
#[pyo3(signature = (pattern))]
fn _synth_permutation_acg(py: Python, pattern: PyArrayLike1<i64>) -> PyResult<CircuitData> {
    let inverted = utils::invert(&pattern.as_array());
    let view = inverted.view();
    let num_qubits = view.len();
    let cycles = utils::pattern_to_cycles(&view);
    let swaps = utils::decompose_cycles(&cycles);

    CircuitData::from_standard_gates(
        py,
        num_qubits as u32,
        swaps.iter().map(|(i, j)| {
            (
                StandardGate::SwapGate,
                smallvec![],
                smallvec![Qubit(*i as u32), Qubit(*j as u32)],
            )
        }),
        Param::Float(0.0),
    )
}

/// Synthesize a permutation circuit for a linear nearest-neighbor
/// architecture using the Kutin, Moulton, Smithline method.
#[pyfunction]
#[pyo3(signature = (pattern))]
pub fn _synth_permutation_depth_lnn_kms(
    py: Python,
    pattern: PyArrayLike1<i64>,
) -> PyResult<CircuitData> {
    let mut inverted = utils::invert(&pattern.as_array());
    let mut view = inverted.view_mut();
    let num_qubits = view.len();
    let mut swap_layers: Vec<(usize, usize)> = Vec::new();

    for i in 0..num_qubits {
        let swap_layer: Vec<(usize, usize)> = utils::create_swap_layer(&mut view, i % 2);
        swap_layers.extend(swap_layer);
    }

    CircuitData::from_standard_gates(
        py,
        num_qubits as u32,
        swap_layers.iter().map(|(i, j)| {
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
    m.add_function(wrap_pyfunction!(_synth_permutation_acg, m)?)?;
    m.add_function(wrap_pyfunction!(_synth_permutation_depth_lnn_kms, m)?)?;
    Ok(())
}
