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
use faer_ext::IntoFaer;
use nalgebra::{DMatrix, DMatrixView, Dim, Dyn, MatrixView, ViewStorage};
use ndarray::ArrayView2;
use ndarray::ShapeBuilder;
use num_complex::Complex64;

pub mod cos_sin_decomp;

const ATOL_DEFAULT: f64 = 1e-8;
const RTOL_DEFAULT: f64 = 1e-5;

fn nalgebra_to_faer<R: Dim, C: Dim, RStride: Dim, CStride: Dim>(
    mat: MatrixView<'_, Complex64, R, C, RStride, CStride>,
) -> MatRef<'_, Complex64> {
    let dim = ::ndarray::Dim(mat.shape());
    let strides = ::ndarray::Dim(mat.strides());

    // SAFETY: We know the array is a 2d array from nalgebra and we get the pointer and memory layout
    // description from nalgebra and can be assumed to be valid since the constraints on
    // `ArrayView2::from_shape_ptr()`
    // (https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html#method.from_shape_ptr)
    // should be be met for a valid nalgebra matrix.
    let array = unsafe { ArrayView2::from_shape_ptr(dim.strides(strides), mat.get_unchecked(0)) };
    array.into_faer()
}

/// Convert a faer MatRef into a nalgebra MatrixView without copying
///
/// This function has some potential sharp edeges around converting strided
/// views. If you are using a strided `MatRef` you'll typically want to ensure
/// the underlying matrix the view is over is contiguous. Specifically if it's
/// not there is a potential path to undefined behavior when using nalgebra
/// methods like `MatrixView::into_slice()` which doesn't understand
/// striding and will access the memory as if the array was a contiguous R x C
/// matrix. If using striding here it's best to not ever call `into_slice()`. There
/// might also be similar sharp edges with strided matrices.
fn faer_to_nalgebra(mat: MatRef<'_, Complex64>) -> MatrixView<'_, Complex64, Dyn, Dyn, Dyn, Dyn> {
    // This function's code is based on faer-ext's IntoNalgebra::into_nalgebra implementation at:
    // https://codeberg.org/sarah-quinones/faer-ext/src/commit/0f055b39529c94d1a000982df745cb9ce170f994/src/lib.rs#L77-L96

    let nrows = mat.nrows();
    let ncols = mat.ncols();
    let row_stride = mat.row_stride();
    let col_stride = mat.col_stride();

    let ptr = mat.as_ptr();
    // SAFETY: Pointer came from a faer MatRef as does the description of the memory layout
    // so creating a view of the data from the from the pointer is safe unless the faer object
    // is already corrupt. nalgebra doesn't support negative striding so that panics and we
    // only work with positive strides
    unsafe {
        MatrixView::<'_, Complex64, Dyn, Dyn, Dyn, Dyn>::from_data(ViewStorage::<
            '_,
            Complex64,
            Dyn,
            Dyn,
            Dyn,
            Dyn,
        >::from_raw_parts(
            ptr,
            (Dyn(nrows), Dyn(ncols)),
            (
                Dyn(row_stride
                    .try_into()
                    .expect("only works for positive strides")),
                Dyn(col_stride
                    .try_into()
                    .expect("only works for positive strides")),
            ),
        ))
    }
}

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
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_sv = singular_values
        .iter()
        .copied()
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
    let mat_view: DMatrixView<Complex64> = mat.as_view();
    let faer_mat: MatRef<Complex64> = nalgebra_to_faer(mat_view);
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

    let u_na = faer_to_nalgebra(u_faer);
    let v_na = faer_to_nalgebra(v_faer).adjoint();

    debug_assert!(verify_svd_decomp(
        mat_view,
        u_na.as_view(),
        s_na.as_view(),
        v_na.as_view()
    ));

    (u_na.into(), s_na, v_na)
}

#[cfg(test)]
mod test {
    use super::*;
    use faer::prelude::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_basic_faer_to_nalgebra_conversion() {
        let matrix = Mat::from_fn(10, 10, |i, j| {
            Complex64::new(i as f64, 0.0) + Complex64::new(j as f64 * 10., 0.)
        });
        let mat_view = faer_to_nalgebra(matrix.as_ref());
        let expected = DMatrix::from_fn(10, 10, |i, j| {
            Complex64::new(i as f64, 0.0) + Complex64::new(j as f64 * 10., 0.)
        });
        assert_eq!(mat_view, expected);
    }

    #[cfg(not(miri))] // TODO: Remove this after dimforge/nalgebra#1562 is released
    #[test]
    fn test_transpose_faer_to_nalgebra_conversion() {
        let matrix = Mat::from_fn(10, 10, |i, j| {
            Complex64::new(i as f64, 0.0) + Complex64::new(j as f64 * 10., 0.)
        });
        let mat_view = faer_to_nalgebra(matrix.transpose());
        let expected = DMatrix::from_fn(10, 10, |i, j| {
            Complex64::new(j as f64, 0.0) + Complex64::new(i as f64 * 10., 0.)
        });
        assert_eq!(mat_view, expected);
    }

    #[test]
    #[should_panic]
    fn test_negative_strided_view_faer_to_nalgebra_conversion() {
        let matrix = Mat::identity(10, 10);
        faer_to_nalgebra(matrix.reverse_rows());
    }
}
