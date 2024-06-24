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

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyReadwriteArray2};
use pyo3::prelude::*;

mod utils;

#[pyfunction]
#[pyo3(signature = (mat, ncols=None, full_elim=false))]
/// Gauss elimination of a matrix mat with m rows and n columns.
/// If full_elim = True, it allows full elimination of mat[:, 0 : ncols]
/// Returns the matrix mat, and the permutation perm that was done on the rows during the process.
/// perm[0 : rank] represents the indices of linearly independent rows in the original matrix.
fn _gauss_elimination_with_perm(
    py: Python,
    mut mat: PyReadwriteArray2<bool>,
    ncols: Option<usize>,
    full_elim: Option<bool>,
) -> PyResult<PyObject> {
    let matmut = mat.as_array_mut();
    let perm = utils::gauss_elimination_with_perm(matmut, ncols, full_elim);
    Ok(perm.to_object(py))
}

#[pyfunction]
#[pyo3(signature = (mat, ncols=None, full_elim=false))]
/// Gauss elimination of a matrix mat with m rows and n columns.
/// If full_elim = True, it allows full elimination of mat[:, 0 : ncols]
/// Returns the updated matrix mat.
fn _gauss_elimination(
    mut mat: PyReadwriteArray2<bool>,
    ncols: Option<usize>,
    full_elim: Option<bool>,
) {
    let matmut = mat.as_array_mut();
    let _perm = utils::gauss_elimination_with_perm(matmut, ncols, full_elim);
}

#[pyfunction]
#[pyo3(signature = (mat))]
/// Given a boolean matrix A after Gaussian elimination, computes its rank
/// (i.e. simply the number of nonzero rows)
fn _compute_rank_after_gauss_elim(py: Python, mat: PyReadonlyArray2<bool>) -> PyResult<PyObject> {
    let view = mat.as_array();
    let rank = utils::compute_rank_after_gauss_elim(view);
    Ok(rank.to_object(py))
}

#[pyfunction]
#[pyo3(signature = (mat, verify=false))]
/// Given a boolean matrix mat, tries to calculate its inverse matrix
/// Args:
///    mat: a boolean square matrix.
///    verify: if True asserts that the multiplication of mat and its inverse is the identity matrix.
/// Returns:
///   the inverse matrix.
/// Raises:
///  QiskitError: if the matrix is not square.
///  QiskitError: if the matrix is not invertible.
pub fn calc_inverse_matrix(
    py: Python,
    mat: PyReadonlyArray2<bool>,
    verify: Option<bool>,
) -> PyResult<Py<PyArray2<bool>>> {
    let view = mat.as_array();
    let invmat = utils::calc_inverse_matrix_inner(view, verify.is_some());
    Ok(invmat.into_pyarray_bound(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (mat1, mat2))]
// Binary matrix multiplication
pub fn _binary_matmul(
    py: Python,
    mat1: PyReadonlyArray2<bool>,
    mat2: PyReadonlyArray2<bool>,
) -> PyResult<Py<PyArray2<bool>>> {
    let view1 = mat1.as_array();
    let view2 = mat2.as_array();
    let result = utils::binary_matmul(view1, view2);
    Ok(result.into_pyarray_bound(py).unbind())
}

#[pyfunction]
#[pyo3(signature = (mat, ctrl, trgt))]
// Perform ROW operation on a matrix mat
fn _row_op(mut mat: PyReadwriteArray2<bool>, ctrl: usize, trgt: usize) {
    let matmut = mat.as_array_mut();
    utils::_add_row_or_col(matmut, &false, ctrl, trgt)
}

#[pyfunction]
#[pyo3(signature = (mat, ctrl, trgt))]
// Perform COL operation on a matrix mat (in the inverse direction)
fn _col_op(mut mat: PyReadwriteArray2<bool>, ctrl: usize, trgt: usize) {
    let matmut = mat.as_array_mut();
    utils::_add_row_or_col(matmut, &true, trgt, ctrl)
}

#[pymodule]
pub fn linear(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(_gauss_elimination_with_perm))?;
    m.add_wrapped(wrap_pyfunction!(_gauss_elimination))?;
    m.add_wrapped(wrap_pyfunction!(_compute_rank_after_gauss_elim))?;
    m.add_wrapped(wrap_pyfunction!(calc_inverse_matrix))?;
    m.add_wrapped(wrap_pyfunction!(_row_op))?;
    m.add_wrapped(wrap_pyfunction!(_col_op))?;
    m.add_wrapped(wrap_pyfunction!(_binary_matmul))?;
    Ok(())
}
