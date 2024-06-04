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

use ndarray::{s, Array2, Axis};
use numpy::{AllowTypeChange, IntoPyArray, PyArray2, PyArrayLike2};
use pyo3::prelude::*;
use rayon::prelude::*;

// Perform ROW operation on a matrix mat
fn _row_op(mat: &mut Array2<i8>, ctrl: usize, trgt: usize) {
    let row0 = mat.row(ctrl).to_owned();
    let mut row1 = mat.row_mut(trgt);
    row1.zip_mut_with(&row0, |x, &y| *x ^= y);
}

// Perform COL operation on a matrix mat
fn _col_op(mat: &mut Array2<i8>, ctrl: usize, trgt: usize) {
    let col0 = mat.column(ctrl).to_owned();
    let mut col1 = mat.column_mut(trgt);
    col1.zip_mut_with(&col0, |x, &y| *x ^= y);
}

// Gauss elimination of a matrix mat with m rows and n columns.
// If full_elim = True, it allows full elimination of mat[:, 0 : ncols]
// Returns the matrix mat.
fn gauss_elimination(
    mut mat: Array2<i8>,
    ncols: Option<usize>,
    full_elim: Option<bool>,
) -> Array2<i8> {
    let (m, mut n) = (mat.nrows(), mat.ncols()); // no. of rows and columns
    if let Some(ncols_val) = ncols {
        n = usize::min(n, ncols_val); // no. of active columns
    }

    let mut r = 0; // current rank
    let k = 0; // current pivot column
    let mut new_k = 0;
    while (r < m) && (k < n) {
        let mut is_non_zero = false;
        let mut new_r = r;
        for j in k..n {
            new_k = k;
            for i in r..m {
                if mat[(i, j)] == 1 {
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
            return mat; // A is in the canonical form
        }

        if new_r != r {
            let temp_r = mat.slice_mut(s![r, ..]).to_owned();
            let temp_new_r = mat.slice_mut(s![new_r, ..]).to_owned();
            mat.slice_mut(s![r, ..]).assign(&temp_new_r);
            mat.slice_mut(s![new_r, ..]).assign(&temp_r);
        }

        // Copy source row to avoid trying multiple borrows at once
        let row0 = mat.row(r).to_owned();
        if full_elim == Some(true) {
            mat.axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .filter(|(i, row)| (*i < r) && row[new_k] == 1)
                .for_each(|(_i, mut row)| {
                    row.zip_mut_with(&row0, |x, &y| *x ^= y);
                });
        }

        mat.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .filter(|(i, row)| (*i > r) && (*i < m) && row[new_k] == 1)
            .for_each(|(_i, mut row)| {
                row.zip_mut_with(&row0, |x, &y| *x ^= y);
            });

        r += 1;
    }
    mat
}

#[pyfunction]
#[pyo3(signature = (mat, ncols=None, full_elim=false))]
fn _gauss_elimination(
    py: Python,
    mat: PyArrayLike2<i8, AllowTypeChange>,
    ncols: Option<usize>,
    full_elim: Option<bool>,
) -> PyResult<Py<PyArray2<i8>>> {
    let view = mat.as_array().to_owned();
    Ok(gauss_elimination(view, ncols, full_elim)
        .into_pyarray_bound(py)
        .unbind())
}

#[pymodule]
pub fn linear_matrix(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(_gauss_elimination))?;
    Ok(())
}
