// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::f64::consts::FRAC_1_SQRT_2;

use ndarray::{Array2, ArrayView2, arr1, array, aview2};
use num_complex::{Complex64, ComplexFloat};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::linalg::ndarray_to_faer;
use qiskit_circuit::util::{C_M_ONE, C_ONE, C_ZERO, GateArray1Q, GateArray2Q, IM, M_IM, c64};

pub(super) const DEFAULT_FIDELITY: f64 = 1.0 - 1.0e-9;

pub trait TraceToFidelity {
    /// Average gate fidelity is :math:`Fbar = (d + |Tr (Utarget \\cdot U^dag)|^2) / d(d+1)`
    /// M. Horodecki, P. Horodecki and R. Horodecki, PRA 60, 1888 (1999)
    fn trace_to_fid(self) -> f64;
}

impl TraceToFidelity for Complex64 {
    fn trace_to_fid(self) -> f64 {
        (4.0 + self.abs().powi(2)) / 20.0
    }
}

#[pyfunction]
#[pyo3(name = "trace_to_fid")]
/// Average gate fidelity is :math:`Fbar = (d + |Tr (Utarget \\cdot U^dag)|^2) / d(d+1)`
/// M. Horodecki, P. Horodecki and R. Horodecki, PRA 60, 1888 (1999)
pub fn py_trace_to_fid(trace: Complex64) -> PyResult<f64> {
    let fid = trace.trace_to_fid();
    Ok(fid)
}

static MAGIC: GateArray2Q = [
    [
        c64(FRAC_1_SQRT_2, 0.),
        C_ZERO,
        C_ZERO,
        c64(0., FRAC_1_SQRT_2),
    ],
    [
        C_ZERO,
        c64(0., FRAC_1_SQRT_2),
        c64(FRAC_1_SQRT_2, 0.),
        C_ZERO,
    ],
    [
        C_ZERO,
        c64(0., FRAC_1_SQRT_2),
        c64(-FRAC_1_SQRT_2, 0.),
        C_ZERO,
    ],
    [
        c64(FRAC_1_SQRT_2, 0.),
        C_ZERO,
        C_ZERO,
        c64(0., -FRAC_1_SQRT_2),
    ],
];

static MAGIC_DAGGER: GateArray2Q = [
    [
        c64(FRAC_1_SQRT_2, 0.),
        C_ZERO,
        C_ZERO,
        c64(FRAC_1_SQRT_2, 0.),
    ],
    [
        C_ZERO,
        c64(0., -FRAC_1_SQRT_2),
        c64(0., -FRAC_1_SQRT_2),
        C_ZERO,
    ],
    [
        C_ZERO,
        c64(FRAC_1_SQRT_2, 0.),
        c64(-FRAC_1_SQRT_2, 0.),
        C_ZERO,
    ],
    [
        c64(0., -FRAC_1_SQRT_2),
        C_ZERO,
        C_ZERO,
        c64(0., FRAC_1_SQRT_2),
    ],
];

pub static K12R_ARR: GateArray1Q = [
    [c64(0., FRAC_1_SQRT_2), c64(FRAC_1_SQRT_2, 0.)],
    [c64(-FRAC_1_SQRT_2, 0.), c64(0., -FRAC_1_SQRT_2)],
];

pub static K12L_ARR: GateArray1Q = [
    [c64(0.5, 0.5), c64(0.5, 0.5)],
    [c64(-0.5, 0.5), c64(0.5, -0.5)],
];

pub static B_NON_NORMALIZED: GateArray2Q = [
    [C_ONE, IM, C_ZERO, C_ZERO],
    [C_ZERO, C_ZERO, IM, C_ONE],
    [C_ZERO, C_ZERO, IM, C_M_ONE],
    [C_ONE, M_IM, C_ZERO, C_ZERO],
];

pub static B_NON_NORMALIZED_DAGGER: GateArray2Q = [
    [c64(0.5, 0.), C_ZERO, C_ZERO, c64(0.5, 0.)],
    [c64(0., -0.5), C_ZERO, C_ZERO, c64(0., 0.5)],
    [C_ZERO, c64(0., -0.5), c64(0., -0.5), C_ZERO],
    [C_ZERO, c64(0.5, 0.), c64(-0.5, 0.), C_ZERO],
];

pub static IPZ: GateArray1Q = [[IM, C_ZERO], [C_ZERO, M_IM]];
pub static IPY: GateArray1Q = [[C_ZERO, C_ONE], [C_M_ONE, C_ZERO]];
pub static IPX: GateArray1Q = [[C_ZERO, IM], [IM, C_ZERO]];

#[inline(always)]
pub(crate) fn transpose_conjugate(mat: ArrayView2<Complex64>) -> Array2<Complex64> {
    mat.t().mapv(|x| x.conj())
}

/// A good approximation to the best value x to get the minimum
/// trace distance for :math:`U_d(x, x, x)` from :math:`U_d(a, b, c)`.
pub(crate) fn closest_partial_swap(a: f64, b: f64, c: f64) -> f64 {
    let m = (a + b + c) / 3.;
    let [am, bm, cm] = [a - m, b - m, c - m];
    let [ab, bc, ca] = [a - b, b - c, c - a];
    m + am * bm * cm * (6. + ab * ab + bc * bc + ca * ca) / 18.
}

pub(crate) fn rx_matrix(theta: f64) -> Array2<Complex64> {
    let half_theta = theta / 2.;
    let cos = c64(half_theta.cos(), 0.);
    let isin = c64(0., -half_theta.sin());
    array![[cos, isin], [isin, cos]]
}

pub(crate) fn ry_matrix(theta: f64) -> Array2<Complex64> {
    let half_theta = theta / 2.;
    let cos = c64(half_theta.cos(), 0.);
    let sin = c64(half_theta.sin(), 0.);
    array![[cos, -sin], [sin, cos]]
}

pub(crate) fn rz_matrix(theta: f64) -> Array2<Complex64> {
    let ilam2 = c64(0., 0.5 * theta);
    array![[(-ilam2).exp(), C_ZERO], [C_ZERO, ilam2.exp()]]
}

/// Generates the array :math:`e^{(i a XX + i b YY + i c ZZ)}`
pub(crate) fn ud(a: f64, b: f64, c: f64) -> Array2<Complex64> {
    array![
        [
            (IM * c).exp() * (a - b).cos(),
            C_ZERO,
            C_ZERO,
            IM * (IM * c).exp() * (a - b).sin()
        ],
        [
            C_ZERO,
            (M_IM * c).exp() * (a + b).cos(),
            IM * (M_IM * c).exp() * (a + b).sin(),
            C_ZERO
        ],
        [
            C_ZERO,
            IM * (M_IM * c).exp() * (a + b).sin(),
            (M_IM * c).exp() * (a + b).cos(),
            C_ZERO
        ],
        [
            IM * (IM * c).exp() * (a - b).sin(),
            C_ZERO,
            C_ZERO,
            (IM * c).exp() * (a - b).cos()
        ]
    ]
}

#[pyfunction]
#[pyo3(name = "Ud")]
pub fn py_ud(py: Python, a: f64, b: f64, c: f64) -> Py<PyArray2<Complex64>> {
    let ud_mat = ud(a, b, c);
    ud_mat.into_pyarray(py).unbind()
}

/// Computes the local invariants for a two-qubit unitary.
///
/// Based on:
///
/// Y. Makhlin, Quant. Info. Proc. 1, 243-252 (2002).
///
/// Zhang et al., Phys Rev A. 67, 042313 (2003).
#[pyfunction]
pub fn two_qubit_local_invariants(unitary: PyReadonlyArray2<Complex64>) -> [f64; 3] {
    let mat = unitary.as_array();
    // Transform to bell basis
    let bell_basis_unitary = aview2(&MAGIC_DAGGER).dot(&mat.dot(&aview2(&MAGIC)));
    // Get determinate since +- one is allowed.
    let det_bell_basis = ndarray_to_faer(bell_basis_unitary.view()).determinant();
    let m = bell_basis_unitary.t().dot(&bell_basis_unitary);
    let mut m_tr2 = m.diag().sum();
    m_tr2 *= m_tr2;
    // Table II of Ref. 1 or Eq. 28 of Ref. 2.
    let g1 = m_tr2 / (16. * det_bell_basis);
    let g2 = (m_tr2 - m.dot(&m).diag().sum()) / (4. * det_bell_basis);

    // Here we split the real and imag pieces of G1 into two so as
    // to better equate to the Weyl chamber coordinates (c0,c1,c2)
    // and explore the parameter space.
    // Also do a FP trick -0.0 + 0.0 = 0.0
    [g1.re + 0., g1.im + 0., g2.re + 0.]
}

#[pyfunction]
pub fn local_equivalence(weyl: PyReadonlyArray1<f64>) -> PyResult<[f64; 3]> {
    let weyl = weyl.as_array();
    let weyl_2_cos_squared_product: f64 = weyl.iter().map(|x| (x * 2.).cos().powi(2)).product();
    let weyl_2_sin_squared_product: f64 = weyl.iter().map(|x| (x * 2.).sin().powi(2)).product();
    let g0_equiv = weyl_2_cos_squared_product - weyl_2_sin_squared_product;
    let g1_equiv = weyl.iter().map(|x| (x * 4.).sin()).product::<f64>() / 4.;
    let g2_equiv = 4. * weyl_2_cos_squared_product
        - 4. * weyl_2_sin_squared_product
        - weyl.iter().map(|x| (4. * x).cos()).product::<f64>();
    Ok([g0_equiv + 0., g1_equiv + 0., g2_equiv + 0.])
}

/// Convert a 4x4 unitary matrix into a unitary matrix with determinant 1
pub fn u4_to_su4(u4: ArrayView2<Complex64>) -> (Array2<Complex64>, f64) {
    let det_u = ndarray_to_faer(u4).determinant();
    let phase_factor = det_u.powf(-0.25).conj();
    let su4 = u4.mapv(|x| x / phase_factor);
    (su4, phase_factor.arg())
}

pub fn real_trace_transform(mat: ArrayView2<Complex64>) -> Array2<Complex64> {
    let a1 = -mat[[1, 3]] * mat[[2, 0]] + mat[[1, 2]] * mat[[2, 1]] + mat[[1, 1]] * mat[[2, 2]]
        - mat[[1, 0]] * mat[[2, 3]];
    let a2 = mat[[0, 3]] * mat[[3, 0]] - mat[[0, 2]] * mat[[3, 1]] - mat[[0, 1]] * mat[[3, 2]]
        + mat[[0, 0]] * mat[[3, 3]];
    let theta = 0.; // Arbitrary!
    let phi = 0.; // This is extra arbitrary!
    let psi = f64::atan2(a1.im + a2.im, a1.re - a2.re) - phi;
    let im = Complex64::new(0., -1.);
    let temp = [
        (theta * im).exp(),
        (phi * im).exp(),
        (psi * im).exp(),
        (-(theta + phi + psi) * im).exp(),
    ];
    Array2::from_diag(&arr1(&temp))
}
