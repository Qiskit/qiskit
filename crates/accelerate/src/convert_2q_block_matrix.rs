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

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;

use num_complex::Complex64;
use numpy::ndarray::linalg::kron;
use numpy::ndarray::{array, Array, Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};

/// Return the matrix Operator resulting from a block of Instructions.
#[pyfunction]
#[pyo3(text_signature = "(op_list, /")]
pub fn blocks_to_matrix(
    py: Python,
    op_list: Vec<(PyReadonlyArray2<Complex64>, Vec<usize>)>,
) -> PyResult<Py<PyArray2<Complex64>>> {
    let mut matrix: Array2<Complex64> = Array::eye(4);
    let swap_gate = array![
        [
            Complex64::new(1., 0.),
            Complex64::new(0., 0.),
            Complex64::new(0., 0.),
            Complex64::new(0., 0.)
        ],
        [
            Complex64::new(0., 0.),
            Complex64::new(0., 0.),
            Complex64::new(1., 0.),
            Complex64::new(0., 0.)
        ],
        [
            Complex64::new(0., 0.),
            Complex64::new(1., 0.),
            Complex64::new(0., 0.),
            Complex64::new(0., 0.)
        ],
        [
            Complex64::new(0., 0.),
            Complex64::new(0., 0.),
            Complex64::new(0., 0.),
            Complex64::new(1., 0.)
        ]
    ];
    let identity: Array2<Complex64> = Array::eye(2);
    for (op_matrix, q_list) in op_list {
        let op_matrix = op_matrix.as_array();
        let result = calculate_matrix(op_matrix, &q_list, &swap_gate, &identity);
        matrix = match result {
            Some(result) => result.dot(&matrix),
            None => op_matrix.dot(&matrix),
        };
    }
    Ok(matrix.into_pyarray(py).to_owned())
}

/// Performs the matrix operations for an Instruction in a 2 qubit system
fn calculate_matrix(
    matrix: ArrayView2<Complex64>,
    q_list: &[usize],
    swap_gate: &Array2<Complex64>,
    identity: &Array2<Complex64>,
) -> Option<Array2<Complex64>> {
    match q_list {
        [0] => Some(kron(identity, &matrix)),
        [1] => Some(kron(&matrix, identity)),
        [1, 0] => Some(swap_gate.dot(&matrix).dot(swap_gate)),
        _ => None,
    }
}

#[pymodule]
pub fn convert_2q_block_matrix(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(blocks_to_matrix))?;
    Ok(())
}
