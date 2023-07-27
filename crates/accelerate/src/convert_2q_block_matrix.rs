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
use numpy::ndarray::{s, Array, Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};

/// Return the matrix Operator resulting from a block of Instructions.
#[pyfunction]
#[pyo3(text_signature = "(op_list, /")]
pub fn blocks_to_matrix(
    py: Python,
    op_list: Vec<(PyReadonlyArray2<Complex64>, Vec<usize>)>,
) -> PyResult<Py<PyArray2<Complex64>>> {
    let mut matrix: Array2<Complex64> = Array::eye(4);
    let identity: Array2<Complex64> = Array::eye(2);
    for (op_matrix, q_list) in op_list {
        let op_matrix = op_matrix.as_array();
        let q_list = q_list.as_slice();
        let result = match q_list {
            [0] => Some(kron(&identity, &op_matrix)),
            [1] => Some(kron(&op_matrix, &identity)),
            [1, 0] => Some(change_basis(op_matrix)),
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
    let temp = trans_matrix.slice(s![2_usize, ..]).to_owned();
    for (index, value) in temp.into_iter().enumerate() {
        trans_matrix[[2, index]] = trans_matrix[[1, index]].to_owned();
        trans_matrix[[1, index]] = value;
    }
    trans_matrix = trans_matrix.reversed_axes();
    let temp = trans_matrix.slice(s![2_usize, ..]).to_owned();
    for (index, value) in temp.into_iter().enumerate() {
        trans_matrix[[2, index]] = trans_matrix[[1, index]];
        trans_matrix[[1, index]] = value;
    }
    trans_matrix
}

#[pymodule]
pub fn convert_2q_block_matrix(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(blocks_to_matrix))?;
    Ok(())
}
