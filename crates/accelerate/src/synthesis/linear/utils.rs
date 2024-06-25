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
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;

/// Binary matrix multiplication
pub fn binary_matmul_inner(
    mat1: ArrayView2<bool>,
    mat2: ArrayView2<bool>,
) -> Result<Array2<bool>, String> {
    let n1_rows = mat1.nrows();
    let n1_cols = mat1.ncols();
    let n2_rows = mat2.nrows();
    let n2_cols = mat2.ncols();
    if n1_cols != n2_rows {
        return Err(format!(
            "Cannot multiply matrices with inappropriate dimensions {}, {}",
            n1_cols, n2_rows
        ));
    }

    Ok(Array2::from_shape_fn((n1_rows, n2_cols), |(i, j)| {
        (0..n2_rows)
            .map(|k| mat1[[i, k]] & mat2[[k, j]])
            .fold(false, |acc, v| acc ^ v)
    }))
}

/// Gauss elimination of a matrix mat with m rows and n columns.
/// If full_elim = True, it allows full elimination of mat[:, 0 : ncols]
/// Returns the matrix mat, and the permutation perm that was done on the rows during the process.
/// perm[0 : rank] represents the indices of linearly independent rows in the original matrix.
pub fn gauss_elimination_with_perm_inner(
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

/// Given a boolean matrix A after Gaussian elimination, computes its rank
/// (i.e. simply the number of nonzero rows)
pub fn compute_rank_after_gauss_elim_inner(mat: ArrayView2<bool>) -> usize {
    let rank: usize = mat
        .axis_iter(Axis(0))
        .map(|row| row.fold(false, |out, val| out | *val) as usize)
        .sum();
    rank
}

/// Given a boolean matrix mat computes its rank
pub fn compute_rank_inner(mat: ArrayView2<bool>) -> usize {
    let mut temp_mat = mat.to_owned();
    gauss_elimination_with_perm_inner(temp_mat.view_mut(), None, Some(false));
    let rank = compute_rank_after_gauss_elim_inner(temp_mat.view());
    rank
}

/// Given a square boolean matrix mat, tries to compute its inverse.
pub fn calc_inverse_matrix_inner(
    mat: ArrayView2<bool>,
    verify: bool,
) -> Result<Array2<bool>, String> {
    if mat.shape()[0] != mat.shape()[1] {
        return Err("Matrix to invert is a non-square matrix.".to_string());
    }
    let n = mat.shape()[0];

    // concatenate the matrix and identity
    let identity_matrix: Array2<bool> = Array2::from_shape_fn((n, n), |(i, j)| i == j);
    let mut mat1 = concatenate(Axis(1), &[mat.view(), identity_matrix.view()]).unwrap();

    gauss_elimination_with_perm_inner(mat1.view_mut(), None, Some(true));

    let r = compute_rank_after_gauss_elim_inner(mat1.slice(s![.., 0..n]));
    if r < n {
        return Err("The matrix is not invertible.".to_string());
    }

    let invmat = mat1.slice(s![.., n..2 * n]).to_owned();

    if verify {
        let mat2 = binary_matmul_inner(mat, (&invmat).into())?;
        let identity_matrix: Array2<bool> = Array2::from_shape_fn((n, n), |(i, j)| i == j);
        if mat2.ne(&identity_matrix) {
            return Err("The inverse matrix is not correct.".to_string());
        }
    }

    Ok(invmat)
}

/// Mutate a matrix inplace by adding the value of the ``ctrl`` row to the
/// ``target`` row. If ``add_cols`` is true, add columns instead of rows.
pub fn _add_row_or_col(mut mat: ArrayViewMut2<bool>, add_cols: &bool, ctrl: usize, trgt: usize) {
    // get the two rows (or columns)
    let info = if *add_cols {
        (s![.., ctrl], s![.., trgt])
    } else {
        (s![ctrl, ..], s![trgt, ..])
    };
    let (row0, mut row1) = mat.multi_slice_mut(info);

    // add them inplace
    row1.zip_mut_with(&row0, |x, &y| *x ^= y);
}

/// Generate a random invertible n x n binary matrix.
pub fn random_invertible_binary_matrix_inner(num_qubits: usize, seed: Option<u64>) -> Array2<bool> {
    let mut rng = match seed {
        Some(seed) => Pcg64Mcg::seed_from_u64(seed),
        None => Pcg64Mcg::from_entropy(),
    };

    let mut matrix = Array2::from_elem((num_qubits, num_qubits), false);

    loop {
        for value in matrix.iter_mut() {
            *value = rng.gen_bool(0.5);
        }

        let rank = compute_rank_inner(matrix.view());
        if rank == num_qubits {
            break;
        }
    }
    matrix
}

/// Check that a binary matrix is invertible.
pub fn check_invertible_binary_matrix_inner(mat: ArrayView2<bool>) -> bool {
    if mat.nrows() != mat.ncols() {
        return false;
    }
    let rank = compute_rank_inner(mat);
    rank == mat.nrows()
}
