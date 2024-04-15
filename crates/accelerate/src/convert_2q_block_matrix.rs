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
use numpy::ndarray::{aview2, Array2, Array, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use smallvec::SmallVec;

use faer::modules::core::mul::matmul;
use faer::modules::core::kron as kron;
use faer::{Mat, mat, Parallelism};
use faer_core::permutation::swap_rows;

static ONE_QUBIT_IDENTITY: [[Complex64; 2]; 2] = [
    [Complex64::new(1., 0.), Complex64::new(0., 0.)],
    [Complex64::new(0., 0.), Complex64::new(1., 0.)],
];

/// Return the matrix Operator resulting from a block of Instructions.
#[pyfunction]
#[pyo3(text_signature = "(op_list, /")]
pub fn blocks_to_matrix(
    py: Python,
    op_list: Vec<(PyReadonlyArray2<Complex64>, SmallVec<[u8; 2]>)>,
) -> PyResult<Py<PyArray2<Complex64>>> {
    let identity = Mat::from_fn(2, 2, |i, j| ONE_QUBIT_IDENTITY[i][j] as Complex64);
    let input_matrix = Mat::from_fn(2, 2, |i, j|  op_list[0].0.as_array()[(i,j)] as Complex64);
    let mut matrix: Mat::<Complex64>;
    let mut matrix_result: Mat::<Complex64>;

    match op_list[0].1.as_slice() {
        [0] => kron(matrix.as_mut(), identity.as_ref(), input_matrix.as_ref()),
        [1] => kron(matrix.as_mut(), input_matrix.as_ref(), identity.as_ref()),
        [0, 1] => matrix = input_matrix.to_owned(),
        [1, 0] => matrix = change_basis(input_matrix),
        [] => matrix = Mat::<Complex64>::identity(4, 4),
        _ => unreachable!(),
    };
    for (op_matrix, q_list) in op_list.into_iter().skip(1) {
        let op_matrix = Mat::from_fn(2, 2, |i, j| op_matrix.as_array()[(i,j)] as Complex64);

        let mut result: Mat::<Complex64>;
        match q_list.as_slice() {
            [0] => kron(result.as_mut(), identity.as_ref(), op_matrix.as_ref()),
            [1] => kron(result.as_mut(), op_matrix.as_ref(), identity.as_ref()),
            [1, 0] => result = change_basis(op_matrix),
            [] => result = Mat::<Complex64>::identity(4, 4),
            _ => result = op_matrix,
        };

        matmul(matrix_result.as_mut(), result.as_ref(), matrix.as_ref(), None, Complex64::new(1.,0.), Parallelism::None);
    }

    let mut array_result = Array2::from_shape_fn((4,4), |(i,j)| Complex64::new(*matrix_result.get(i, j).re, *matrix_result.get(i, j).im)); 
    Ok(array_result.into_pyarray_bound(py).unbind())
}

/// Switches the order of qubits in a two qubit operation.
#[inline]
pub fn change_basis(matrix: Mat::<Complex64>) -> Mat::<Complex64> {
    let mut trans_matrix: Mat::<Complex64> = matrix.transpose().to_owned();
    swap_rows(trans_matrix.as_mut(), 1, 2);
    trans_matrix = trans_matrix.transpose().to_owned();
    swap_rows(trans_matrix.as_mut(), 1, 2);
    trans_matrix
}

#[pymodule]
pub fn convert_2q_block_matrix(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(blocks_to_matrix))?;
    Ok(())
}
