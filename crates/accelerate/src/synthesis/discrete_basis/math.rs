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

use nalgebra::{Complex, Matrix2, Matrix3, Matrix3x1};

/// A bisection search on [f64] functions.
fn bisect(f: impl Fn(f64) -> f64, a: f64, b: f64, tol: f64, maxiter: usize) -> Option<f64> {
    // sort: we always want a < b, as we assume the interval is [a, b]
    let (mut a, mut b) = if a > b { (b, a) } else { (a, b) };

    // check trivial cases
    if f(a).abs() < tol {
        return Some(a);
    }
    if f(b).abs() < tol {
        return Some(b);
    }
    if f(a) * f(b) > 0. {
        return None; // invalid boundaries
    }

    let half = 0.5;
    let mut midpoint = half * (a + b);
    for _ in 0..maxiter {
        // solution is in [a, midpoint]
        if f(a) * f(midpoint) < 0. {
            b = midpoint;
        } else {
            a = midpoint;
        }
        midpoint = half * (a + b);

        if (b - a) < tol {
            break; // we're done!
        }
    }
    Some(midpoint)
}

pub(crate) fn solve_decomposition_angle(matrix: &Matrix3<f64>) -> f64 {
    let trace = matrix.trace().min(3.0); // avoid roundoff errors
    let angle = ((trace - 1.) / 2.).acos();
    let lhs = (angle / 2.).sin();

    // we use a bisect search to solve the angle equation (Eq. 10 in the paper)
    let min_angle = 0.;
    let max_angle = 2. * (1. / 2_f64.powf(0.25)).asin();

    let f = move |phi: f64| -> f64 {
        let sin_sq = (phi / 2.).sin().powi(2);
        2. * sin_sq * (1. - sin_sq.powi(2)).sqrt() - lhs
    };

    bisect(f, min_angle, max_angle, 1e-20, 70).expect("Who coded this cannot math")
}

/// Add coeff * V to the out matrix, where V is the skew-symmetric representation of the
/// cross product with the vector v.
fn add_cross_prod(coeff: f64, v: &Matrix3x1<f64>, out: &mut Matrix3<f64>) {
    out[(0, 1)] -= coeff * v[2];
    out[(0, 2)] += coeff * v[1];
    out[(1, 0)] += coeff * v[2];
    out[(1, 2)] -= coeff * v[0];
    out[(2, 0)] -= coeff * v[1];
    out[(2, 1)] += coeff * v[0];
}

/// Add coeff * v.dot(v.T) to the output matrix.
fn add_outer_prod(coeff: f64, v: &Matrix3x1<f64>, out: &mut Matrix3<f64>) {
    out[(0, 0)] += coeff * v[0] * v[0];
    out[(0, 1)] += coeff * v[0] * v[1];
    out[(0, 2)] += coeff * v[0] * v[2];
    out[(1, 0)] += coeff * v[1] * v[0];
    out[(1, 1)] += coeff * v[1] * v[1];
    out[(1, 2)] += coeff * v[1] * v[2];
    out[(2, 0)] += coeff * v[2] * v[0];
    out[(2, 1)] += coeff * v[2] * v[1];
    out[(2, 2)] += coeff * v[2] * v[2];
}

/// Compute the group commutator UVU^{-1}V^{-1}. This code leverages the fact that
/// the matrices as SO(3) and U^{-1} = U^T.
fn group_commutator(left: &Matrix3<f64>, right: &Matrix3<f64>) -> Matrix3<f64> {
    left * right * left.transpose() * right.transpose()
}

/// Compute the rotation axis of a SO(3) matrix.
fn rotation_axis_from_so3(matrix: &Matrix3<f64>, tol: f64) -> Matrix3x1<f64> {
    let trace = matrix.trace();

    // Try to obtain the matrix via Rodrigues formula. If the matrix is diagonal (or the Rodrigues
    // approach fails, likely because the matrix was diagonal up to numerical error), determine
    // the axis from the diagonal.
    if trace >= 3. - tol {
        // matrix is the identity, ie. no rotation
        return Matrix3x1::identity();
    }

    if trace >= tol - 1. {
        // try skew symmetric case
        let theta = ((trace - 1.) / 2.).acos();
        if theta.sin() > tol {
            let coeff = 1. / 2. / theta.sin();
            let axis = Matrix3x1::new(
                coeff * (matrix[(2, 1)] - matrix[(1, 2)]),
                coeff * (matrix[(0, 2)] - matrix[(2, 0)]),
                coeff * (matrix[(1, 0)] - matrix[(0, 1)]),
            );

            // this might fail due to numerical error, in that case go to diagonal case
            if !axis.iter().any(|el| el.is_nan()) {
                return axis / vector_norm(&axis);
            }
        }
    }

    // this is a 180 degree rotation about any of X, Y, or Z axis (then the trace is -1)
    let index = matrix
        .diagonal()
        .iter()
        .enumerate()
        .find(|(_index, el)| el.is_sign_positive())
        .expect("At least one diagonal element must be 1")
        .0;
    let mut axis = Matrix3x1::zeros();
    axis[index] = 1.;
    axis
}

/// Compute the SO(3) matrix that rotates from ``from`` to ``to``.
fn rotation_matrix(from: &Matrix3x1<f64>, to: &Matrix3x1<f64>, do_checks: bool) -> Matrix3<f64> {
    let from = from / vector_norm(from);
    let to = to / vector_norm(to);
    let dot = from.dot(&to);

    let mut cross = Matrix3::zeros();
    add_cross_prod(1., &from.cross(&to), &mut cross);

    let out = Matrix3::identity() + cross + cross * cross / (1. + dot);
    if do_checks {
        assert_so3("rotation matrix", &out);
    }
    out
}

/// Compute the SO(3) matrix implementing rotation about ``axis`` for an ``angle``.
fn so3_from_angle_axis(angle: f64, axis: &Matrix3x1<f64>) -> Matrix3<f64> {
    let mut out = Matrix3::<f64>::identity() * angle.cos();
    add_cross_prod(angle.sin(), axis, &mut out);
    add_outer_prod(1. - angle.cos(), axis, &mut out);

    out
}

/// Decompose the SO(3) input matrix M into a balanced group commutator, that is
///
///     M = V W V^T W^T
///
/// for two SO(3) matrices V and W. See section 4.1 of https://arxiv.org/abs/quant-ph/0505030.
pub fn group_commutator_decomposition(
    matrix_so3: &Matrix3<f64>,
    do_checks: bool,
) -> (Matrix3<f64>, Matrix3<f64>) {
    if do_checks {
        assert_so3("GC input", matrix_so3);
    }

    let angle = solve_decomposition_angle(matrix_so3);

    let e1 = Matrix3x1::new(1., 0., 0.);
    let e2 = Matrix3x1::new(0., 1., 0.);

    let vx = so3_from_angle_axis(angle, &e1);
    let wy = so3_from_angle_axis(angle, &e2);

    let tol = 1e-15;
    let group_comm = group_commutator(&vx, &wy);
    let group_comm_axis = rotation_axis_from_so3(&group_comm, tol);
    let matrix_axis = rotation_axis_from_so3(matrix_so3, tol);

    let sim_matrix = rotation_matrix(&group_comm_axis, &matrix_axis, do_checks);
    let sim_matrix_t = sim_matrix.transpose();

    let v = sim_matrix * vx * sim_matrix_t;
    let w = sim_matrix * wy * sim_matrix_t;

    if do_checks {
        assert_so3("group commutator", &group_comm);
        assert_so3("vx", &vx);
        assert_so3("wy", &wy);
        assert_so3("sim_matrix", &sim_matrix);
        assert_so3("v", &v);
        assert_so3("w", &w);
    }

    (v, w)
}

/// Compute the L2 norm of a 3-element vector.
fn vector_norm(mat: &Matrix3x1<f64>) -> f64 {
    let summed = mat.iter().map(|el| el.powi(2)).sum::<f64>();
    summed.sqrt()
}

pub(crate) fn matmul_bigcomplex(
    a: &Matrix2<Complex<f64>>,
    b: &Matrix2<Complex<f64>>,
) -> Matrix2<Complex<f64>> {
    Matrix2::new(
        a[(0, 0)] * b[(0, 0)] + a[(0, 1)] * b[(1, 0)],
        a[(0, 0)] * b[(0, 1)] + a[(0, 1)] * b[(1, 1)],
        a[(1, 0)] * b[(0, 0)] + a[(1, 1)] * b[(1, 0)],
        a[(1, 0)] * b[(0, 1)] + a[(1, 1)] * b[(1, 1)],
    )
}

/// Compute the determinant of a 3x3 matrix.
fn determinant(matrix: &Matrix3<f64>) -> f64 {
    matrix[(0, 0)] * matrix[(1, 1)] * matrix[(2, 2)]
        + matrix[(0, 1)] * matrix[(1, 2)] * matrix[(2, 0)]
        + matrix[(0, 2)] * matrix[(1, 0)] * matrix[(2, 1)]
        - matrix[(0, 2)] * matrix[(1, 1)] * matrix[(2, 0)]
        - matrix[(0, 1)] * matrix[(1, 0)] * matrix[(2, 2)]
        - matrix[(0, 0)] * matrix[(1, 2)] * matrix[(2, 1)]
}

pub(super) fn assert_so3(name: &str, matrix: &Matrix3<f64>) {
    if matrix.iter().any(|el| el.is_nan()) {
        panic!("{} has NaN value.", name);
    }
    if (1. - determinant(matrix)) > (1e-5) {
        panic!(
            "{} is not SO(3): Determinant is {}, not 1.",
            name,
            determinant(matrix)
        );
    }
    let diff = matrix * matrix.transpose() - Matrix3::<f64>::identity();
    let tol = 1e-5;
    if diff.iter().any(|el| el.abs() > tol) {
        panic!("{} is not SO(3): Matrix is not orthogonal.", name)
    }
}
