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

use approx::relative_ne;
use nalgebra::DMatrix;
use num_complex::Complex64;

pub mod cos_sin_decomp;

const ATOL_DEFAULT: f64 = 1e-8;
const RTOL_DEFAULT: f64 = 1e-5;

pub fn is_hermitian_matrix(mat: &DMatrix<Complex64>) -> bool {
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

/// Given a matrix that is "close" to unitary, returns the closest
/// unitary matrix.
/// See https://michaelgoerz.net/notes/finding-the-closest-unitary-for-a-given-matrix/,
pub fn closest_unitary(mat: DMatrix<Complex64>) -> DMatrix<Complex64> {
    // This implementation consumes the original mat but avoids calling
    // an unnecessary clone.
    let svd = mat.try_svd(true, true, 1e-12, 0).unwrap();
    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();
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
