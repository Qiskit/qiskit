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

use ndarray::{concatenate, s, Array2, ArrayView2, ArrayViewMut2, Axis};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyReadwriteArray2};
use pyo3::prelude::*;

// Perform ROW operation on a matrix mat
fn _row_op(mat: &mut ArrayViewMut2<bool>, ctrl: usize, trgt: usize) {
    let row0 = mat.row(ctrl).to_owned();
    let mut row1 = mat.row_mut(trgt);
    row1.zip_mut_with(&row0, |x, &y| *x ^= y);
}

// Perform COL operation on a matrix mat
fn _col_op(mat: &mut ArrayViewMut2<bool>, ctrl: usize, trgt: usize) {
    let col0 = mat.column(ctrl).to_owned();
    let mut col1 = mat.column_mut(trgt);
    col1.zip_mut_with(&col0, |x, &y| *x ^= y);
}

// Gauss elimination of a matrix mat with m rows and n columns.
// If full_elim = True, it allows full elimination of mat[:, 0 : ncols]
// Returns the matrix mat, and the permutation perm that was done on the rows during the process.
// perm[0 : rank] represents the indices of linearly independent rows in the original matrix.
fn gauss_elimination_with_perm_inner(
    mut mat: ArrayViewMut2<bool>,
    ncols: Option<usize>,
    full_elim: Option<bool>,
) -> Vec<usize> {
    let (m, mut n) = (mat.nrows(), mat.ncols()); // no. of rows and columns
    if let Some(ncols_val) = ncols {
        n = usize::min(n, ncols_val); // no. of active columns
    }
    let mut perm: Vec<usize> = Vec::from_iter(0..m);

    let mut r = 0; // current rank
    let k = 0; // current pivot column
    let mut new_k = 0;
    while (r < m) && (k < n) {
        let mut is_non_zero = false;
        let mut new_r = r;
        for j in k..n {
            new_k = k;
            for i in r..m {
                if mat[(i, j)] {
                    is_non_zero = true;
                    new_k = j;
                    new_r = i;
                    break;
                }
            }
            if is_non_zero {
                break;
            }
        }
        if !is_non_zero {
            return perm; // A is in the canonical form
        }

        if new_r != r {
            let temp_r = mat.slice_mut(s![r, ..]).to_owned();
            let temp_new_r = mat.slice_mut(s![new_r, ..]).to_owned();
            mat.slice_mut(s![r, ..]).assign(&temp_new_r);
            mat.slice_mut(s![new_r, ..]).assign(&temp_r);
            perm.swap(r, new_r);
        }

        // Copy source row to avoid trying multiple borrows at once
        let row0 = mat.row(r).to_owned();
        mat.axis_iter_mut(Axis(0))
            .enumerate()
            .filter(|(i, row)| {
                (full_elim == Some(true) && (*i < r) && row[new_k])
                    || (*i > r && *i < m && row[new_k])
            })
            .for_each(|(_i, mut row)| {
                row.zip_mut_with(&row0, |x, &y| *x ^= y);
            });

        r += 1;
    }
    perm
}

/// computes an inverse of a binary invertible matrix
/// todo: error handling, for now just panic
pub fn calc_inverse_matrix_inner(mat: ArrayView2<bool>) -> Array2<bool> {
    if mat.shape()[0] != mat.shape()[1] {
        panic!("Matrix to invert is a non-square matrix.");
    }
    let n = mat.shape()[0];

    // concatenate the matrix and identity
    let identity_matrix: Array2<bool> = Array2::from_shape_fn((n, n), |(i, j)| i == j);
    let mut mat1 = concatenate(Axis(1), &[mat.view(), identity_matrix.view()]).unwrap();

    gauss_elimination_with_perm_inner(mat1.view_mut(), None, Some(true));

    let invmat = mat1.slice(s![.., n..2 * n]).to_owned();

    invmat
}

#[pyfunction]
#[pyo3(signature = (mat, ncols=None, full_elim=false))]
fn gauss_elimination_with_perm(
    py: Python,
    mut mat: PyReadwriteArray2<bool>,
    ncols: Option<usize>,
    full_elim: Option<bool>,
) -> PyResult<PyObject> {
    let view = mat.as_array_mut();
    let perm = gauss_elimination_with_perm_inner(view, ncols, full_elim);
    Ok(perm.to_object(py))
}

#[pyfunction]
#[pyo3(signature = (mat, ncols=None, full_elim=false))]
// Gauss elimination of a matrix mat with m rows and n columns.
// If full_elim = True, it allows full elimination of mat[:, 0 : ncols]
// Returns the updated matrix mat.
fn gauss_elimination(
    mut mat: PyReadwriteArray2<bool>,
    ncols: Option<usize>,
    full_elim: Option<bool>,
) {
    let view = mat.as_array_mut();
    let _perm = gauss_elimination_with_perm_inner(view, ncols, full_elim);
}

#[pyfunction]
#[pyo3(signature = (mat))]
// Given a boolean matrix A after Gaussian elimination, computes its rank
// (i.e. simply the number of nonzero rows)"""
fn compute_rank_after_gauss_elim(py: Python, mat: PyReadonlyArray2<bool>) -> PyResult<PyObject> {
    let view = mat.as_array();
    let rank: usize = view
        .axis_iter(Axis(0))
        .map(|row| row.fold(false, |out, val| out | *val) as usize)
        .sum();
    Ok(rank.to_object(py))
}

#[pyfunction]
#[pyo3(signature = (mat))]
fn calc_inverse_matrix(py: Python, mat: PyReadonlyArray2<bool>) -> PyResult<Py<PyArray2<bool>>> {
    println!("calling calc_inverse_matrix");
    let view = mat.as_array();
    let invmat = calc_inverse_matrix_inner(view);
    Ok(invmat.into_pyarray_bound(py).unbind())
}

#[pymodule]
pub fn linear_matrix(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(gauss_elimination_with_perm))?;
    m.add_wrapped(wrap_pyfunction!(gauss_elimination))?;
    m.add_wrapped(wrap_pyfunction!(compute_rank_after_gauss_elim))?;
    m.add_wrapped(wrap_pyfunction!(calc_inverse_matrix))?;
    Ok(())
}