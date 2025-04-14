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

// Based on the Cosine Sine decomposition TKET uses for unitary matrices:
//
// https://github.com/CQCL/tket/blob/a61bbac1c07487521acd8da463023f3c85c799e8/tket/src/Utils/CosSinDecomposition.cpp

use numpy::PyReadonlyArray2;
use numpy::ToPyArray;
use pyo3::prelude::*;

use nalgebra::{DMatrix, DVector};
use num_complex::{Complex64, ComplexFloat};

const EPS: f64 = 1e-11;

type CosSinDecompReturn = (
    DMatrix<Complex64>,
    DMatrix<Complex64>,
    DMatrix<Complex64>,
    DMatrix<Complex64>,
    DMatrix<f64>,
    DMatrix<f64>,
);

/// Comute the cosine-sin decomposition of a Unitary matrix
///
/// # Args
///
/// - `u`: The input unitary to decompose
///
/// # Returns
///
/// The tuple (l0, l1, r0, r1, c, s) which represents the decomposition
///
/// [l0   ] [c -s] [r0   ]
/// [   l1] [s  c] [   r1]
///
/// where l0, l1, r0 and r1 are unitaries of equal size, c and s are diagonal
/// matrices with non-negative entries, the diagonal entries of c are in
/// non-decreasimg order, and
///
/// c^2 + s^2 = I
pub fn cos_sin_decomposition(u: DMatrix<Complex64>) -> CosSinDecompReturn {
    let shape = u.shape();
    let n = shape.0 / 2;
    // Upper left corner
    let u00 = u.view((0, 0), (n, n));
    // Upper right corner
    let u01 = u.view((0, n), (n, n));
    // Lower left corner
    let u10 = u.view((n, 0), (n, n)).to_owned();
    // Lower right corner
    let u11 = u.view((n, n), (n, n));

    let svd = u00.svd(true, true);
    let mut l0 = svd.u.unwrap();
    // Row-wise reverse u
    let mut start = 0;
    let mut end = n - 1;
    while start < end {
        l0.swap_rows(start, end);
        start += 1;
        end -= 1;
    }
    let mut r0_dag = svd.v_t.unwrap();
    // Row-wise reverse v
    start = 0;
    end = 0;
    while start < end {
        r0_dag.swap_rows(start, end);
        start += 1;
        end -= 1;
    }
    let singular_values = svd.singular_values;
    let values_diag: DVector<f64> =
        DVector::from_iterator(singular_values.len(), singular_values.iter().rev().copied());
    let c = DMatrix::from_diagonal(&values_diag);
    let r0 = r0_dag.adjoint();

    // Now u00 = l0 c r0; l0 and r0 are unitary, and c is diagonal with positive
    // non-decreasing entries. Because u00 is a submatrix of a unitary matrix, its
    // singular values (the entries of c) are all <= 1.

    let u10_r0_dag = u10 * r0_dag;
    let qr = u10_r0_dag.qr();
    let mut l1 = qr.q();
    let mut s = qr.unpack_r();
    // Now u10 r0* = l1 S; l1 is unitary, and S is upper triangular.
    //
    // Claim: S is diagonal.
    // Proof: Since u is unitary, we have
    //     I = u00* u00 + u10* u10
    //       = (l0 c r0)* (l0 c r0) + (l1 S r0)* (l1 S r0)
    //       = r0* c l0* l0 c r0 + r0* S* l1* l1 S r0
    //       = r0* c^2 r0 + r0* S* S r0
    //       = r0* (c^2 + S* S) r0
    // So I = c^2 + (S* S), so (S* S) = I - c^2 is a diagonal matrix with non-
    // increasing entries in the range [0,1). As S is upper triangular, this
    // implies that S must be diagonal. (Proof by induction over the dimension n
    // of S: consider the two cases S_00 = 0 and S_00 != 0 and reduce to the n-1
    // case.)
    //
    // We want S to be real. This is not guaranteed, though since it is diagonal
    // it can be made so by adjusting l1.
    if s.iter().any(|x| x.im != 0.) {
        for j in 0..n {
            let z = s[(j, j)];
            let r = z.abs();
            if r > EPS {
                let w = z.conj() / r;
                s[(j, j)] *= w;
                l1.column_mut(j).iter_mut().for_each(|x| *x /= w);
            }
        }
    }

    // Now s is real and diagonal, and c^2 + S^2 = I
    let mut s_real: DMatrix<f64> = s.map(|x| x.re);
    // Make all entries in s_real non-negative.
    for j in 0..n {
        let val = s_real[(j, j)];
        if val < 0. {
            s_real[(j, j)] = -val;
            l1.column_mut(j).iter_mut().for_each(|x| *x = -(*x));
        }
    }
    // Finally compute r1, being careful not to divide by small things.
    let mut r1: DMatrix<Complex64> = DMatrix::zeros(n, n);
    let l0_adjoint = l0.adjoint();
    let l1_adjoint = l1.adjoint();
    for i in 0..n {
        let mut row = r1.row_mut(i);
        if s_real[(i, i)] > c[(i, i)] {
            let mut new_mat = &l0_adjoint * u01;
            let mut new_row = new_mat.row_mut(i);
            new_row.iter_mut().for_each(|x| *x = -(*x) / s[(i, i)]);
            row.iter_mut()
                .zip(new_row.into_iter())
                .for_each(|(x, y)| *x = *y);
        } else {
            let mut new_mat = &l1_adjoint * u11;
            let mut new_row = new_mat.row_mut(i);
            new_row.iter_mut().for_each(|x| *x /= c[(i, i)]);
            row.iter_mut()
                .zip(new_row.into_iter())
                .for_each(|(x, y)| *x = *y);
        }
    }

    (l0.to_owned(), l1, r0.to_owned(), r1, c, s_real)
}

// TODO: Remove this function and all the python interface when QSD is ported to rust
#[pyfunction]
pub fn cossin<'py>(py: Python<'py>, u: PyReadonlyArray2<Complex64>) -> PyResult<Bound<'py, PyAny>> {
    let array = u.as_array();
    let shape = array.shape();
    let mat: DMatrix<Complex64> = DMatrix::from_fn(shape[0], shape[1], |i, j| array[[i, j]]);
    let res = cos_sin_decomposition(mat);
    Ok((
        (res.0.to_pyarray(py), res.1.to_pyarray(py)),
        res.5.to_pyarray(py),
        (res.2.to_pyarray(py), res.3.to_pyarray(py)),
    )
        .into_pyobject(py)?
        .into_any())
}

pub fn cos_sin_decomp(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(cossin))?;
    Ok(())
}
