// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use approx::{abs_diff_eq, relative_ne};
use faer::MatRef;
use faer_ext::{IntoFaer, IntoNalgebra};
use nalgebra::{DMatrix, DMatrixView};
use num_complex::Complex64;

pub mod cos_sin_decomp;

const ATOL_DEFAULT: f64 = 1e-8;
const RTOL_DEFAULT: f64 = 1e-5;

/// Check whether the given matrix is hermitian by comparing it (up to tolerance) with its hermitian adjoint.
pub fn is_hermitian_matrix(mat: DMatrixView<Complex64>) -> bool {
    let shape = mat.shape();
    let adjoint = mat.adjoint();
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            if relative_ne!(
                mat[(i, j)],
                adjoint[(i, j)],
                epsilon = ATOL_DEFAULT,
                max_relative = RTOL_DEFAULT
            ) {
                return false;
            }
        }
    }
    true
}

/// Verify SVD decomposition gives the same unitary
fn verify_svd_decomp(
    mat: DMatrixView<Complex64>,
    v: DMatrixView<Complex64>,
    s: DMatrixView<Complex64>,
    w: DMatrixView<Complex64>,
) -> bool {
    let mat_check = v * s * w;
    abs_diff_eq!(mat, mat_check.as_view(), epsilon = 1e-7)
}

/// Verifies the given matrix U is unitary by comparing U*U to the identity matrix
pub fn verify_unitary(u: &DMatrix<Complex64>) -> bool {
    let n = u.shape().0;

    let id_mat = DMatrix::identity(n, n);
    let uu = u.adjoint() * u;

    abs_diff_eq!(uu, id_mat, epsilon = 1e-7)
}

/// Given a matrix that is "close" to unitary, returns the closest
/// unitary matrix.
/// See https://michaelgoerz.net/notes/finding-the-closest-unitary-for-a-given-matrix/,
pub fn closest_unitary(mat: DMatrixView<Complex64>) -> DMatrix<Complex64> {
    let (u, _sigma, v_t) = svd_decomposition(mat);
    &u * &v_t
}

/// Calculate the condition number of a matrix w.r.t the L2 norm
/// using SVD
pub fn condition_number(mat: DMatrix<Complex64>) -> Option<f64> {
    let svd = mat.svd(false, false);
    let singular_values = svd.singular_values;

    if singular_values.is_empty() {
        return None;
    }

    let max_sv = singular_values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_sv = singular_values
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);

    if min_sv == 0.0 {
        return None; // Singular matrix
    }

    Some(max_sv / min_sv)
}

/// Returns the SVD decomposition of the given matrix M as three matrices A,S,B such that
/// M=ASB (In the usual notations, M=USV*, and this function returns A=U and B=V*)
pub fn svd_decomposition(
    mat: DMatrixView<Complex64>,
) -> (DMatrix<Complex64>, DMatrix<Complex64>, DMatrix<Complex64>) {
    svd_decomposition_using_faer(mat)
}

fn svd_decomposition_using_faer(
    mat: DMatrixView<Complex64>,
) -> (DMatrix<Complex64>, DMatrix<Complex64>, DMatrix<Complex64>) {
    let mat_view: DMatrixView<Complex64> = mat.as_view();
    let faer_mat: MatRef<Complex64> = mat_view.into_faer();
    let faer_svd = faer_mat.svd().unwrap();

    let u_faer = faer_svd.U();
    let s_faer = faer_svd.S();
    let v_faer = faer_svd.V();

    let s_na = DMatrix::from_fn(u_faer.ncols(), v_faer.nrows(), |i, j| {
        if i == j {
            s_faer[i]
        } else {
            Complex64::new(0.0, 0.0)
        }
    });

    let u_na = u_faer.into_nalgebra();
    let v_na = v_faer.into_nalgebra().adjoint();

    debug_assert!(verify_svd_decomp(
        mat_view,
        u_na.as_view(),
        s_na.as_view(),
        v_na.as_view()
    ));

    (u_na.into(), s_na, v_na)
}
