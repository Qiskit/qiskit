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

use ndarray::{s, ArrayViewMut2, Axis};
use numpy::{PyReadonlyArray2, PyReadwriteArray2};
use pyo3::prelude::*;

// Gauss elimination of a matrix mat with m rows and n columns.
// If full_elim = True, it allows full elimination of mat[:, 0 : ncols]
// Returns the matrix mat, and the permutation perm that was done on the rows during the process.
// perm[0 : rank] represents the indices of linearly independent rows in the original matrix.
fn gauss_elimination_with_perm(
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

#[pyfunction]
#[pyo3(signature = (mat, ncols=None, full_elim=false))]
fn _gauss_elimination_with_perm(
    py: Python,
    mut mat: PyReadwriteArray2<bool>,
    ncols: Option<usize>,
    full_elim: Option<bool>,
) -> PyResult<PyObject> {
    let matmut = mat.as_array_mut();
    let perm = gauss_elimination_with_perm(matmut, ncols, full_elim);
    Ok(perm.to_object(py))
}

#[pyfunction]
#[pyo3(signature = (mat, ncols=None, full_elim=false))]
// Gauss elimination of a matrix mat with m rows and n columns.
// If full_elim = True, it allows full elimination of mat[:, 0 : ncols]
// Returns the updated matrix mat.
fn _gauss_elimination(
    mut mat: PyReadwriteArray2<bool>,
    ncols: Option<usize>,
    full_elim: Option<bool>,
) {
    let matmut = mat.as_array_mut();
    let _perm = gauss_elimination_with_perm(matmut, ncols, full_elim);
}

#[pyfunction]
#[pyo3(signature = (mat))]
// Given a boolean matrix A after Gaussian elimination, computes its rank
// (i.e. simply the number of nonzero rows)"""
fn _compute_rank_after_gauss_elim(py: Python, mat: PyReadonlyArray2<bool>) -> PyResult<PyObject> {
    let view = mat.as_array();
    let rank: usize = view
        .axis_iter(Axis(0))
        .map(|row| row.fold(false, |out, val| out | *val) as usize)
        .sum();
    Ok(rank.to_object(py))
}

#[pyfunction]
#[pyo3(signature = (mat, ctrl, trgt))]
// Perform ROW operation on a matrix mat
fn _row_op(mut mat: PyReadwriteArray2<bool>, ctrl: usize, trgt: usize) {
    let mut matmut = mat.as_array_mut();
    let (row0, mut row1) = matmut.multi_slice_mut((s![ctrl, ..], s![trgt, ..]));
    row1.zip_mut_with(&row0, |x, &y| *x ^= y);
}

#[pyfunction]
#[pyo3(signature = (mat, ctrl, trgt))]
// Perform COL operation on a matrix mat
fn _col_op(mut mat: PyReadwriteArray2<bool>, ctrl: usize, trgt: usize) {
    let mut matmut = mat.as_array_mut();
    let (col0, mut col1) = matmut.multi_slice_mut((s![.., trgt], s![.., ctrl]));
    col1.zip_mut_with(&col0, |x, &y| *x ^= y);
}

#[pymodule]
pub fn linear_matrix(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(_gauss_elimination_with_perm))?;
    m.add_wrapped(wrap_pyfunction!(_gauss_elimination))?;
    m.add_wrapped(wrap_pyfunction!(_compute_rank_after_gauss_elim))?;
    m.add_wrapped(wrap_pyfunction!(_row_op))?;
    m.add_wrapped(wrap_pyfunction!(_col_op))?;
    Ok(())
}
