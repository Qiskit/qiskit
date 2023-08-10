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
use numpy::ndarray::{Array, Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};

/// Return the matrix Operator resulting from a block of Instructions.
#[pyfunction]
#[pyo3(text_signature = "(op_list, /")]
pub fn blocks_to_matrix(
    py: Python,
    op_list: Vec<(PyReadonlyArray2<Complex64>, Vec<usize>)>,
) -> PyResult<Py<PyArray2<Complex64>>> {
    let identity: Array2<Complex64> = Array::eye(2);
    let input_matrix = op_list[0].0.as_array();
    let mut matrix: Array2<Complex64> = match op_list[0].1.as_slice() {
        [0] => kron(&identity, &input_matrix),
        [1] => kron(&input_matrix, &identity),
        [0, 1] => input_matrix.to_owned(),
        [1, 0] => change_basis(input_matrix),
        [] => Array::eye(4),
        _ => unreachable!(),
    };
    for (op_matrix, q_list) in op_list.into_iter().skip(1) {
        let op_matrix = op_matrix.as_array();
        let result = match q_list.as_slice() {
            [0] => Some(kron(&identity, &op_matrix)),
            [1] => Some(kron(&op_matrix, &identity)),
            [1, 0] => Some(change_basis(op_matrix)),
            [] => Some(Array::eye(4)),
            _ => None,
        };
        matrix = match result {
            Some(result) => result.dot(&matrix),
            None => op_matrix.dot(&matrix),
        };
    }
    Ok(matrix.into_pyarray(py).to_owned())
}

/// Switches the order of qubits in a two qubit operation.
fn change_basis(matrix: ArrayView2<Complex64>) -> Array2<Complex64> {
    let mut trans_matrix: Array2<Complex64> = matrix.reversed_axes().to_owned();
    for index in 0..trans_matrix.ncols() {
        trans_matrix.swap([1, index], [2, index]);
    }
    trans_matrix = trans_matrix.reversed_axes();
    for index in 0..trans_matrix.ncols() {
        trans_matrix.swap([1, index], [2, index]);
    }
    trans_matrix
}

#[pymodule]
pub fn convert_2q_block_matrix(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(blocks_to_matrix))?;
    Ok(())
}
