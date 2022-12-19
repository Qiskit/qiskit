// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use num_complex::{Complex64, ComplexFloat};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;
use std::f64::consts::PI;

use ndarray::prelude::*;
use numpy::PyReadonlyArray2;

#[inline]
fn det_one_qubit(mat: ArrayView2<Complex64>) -> Complex64 {
    mat[[0, 0]] * mat[[1, 1]] - mat[[0, 1]] * mat[[1, 0]]
}

#[inline]
fn complex_phase(x: Complex64) -> f64 {
    x.im.atan2(x.re)
}

#[inline]
fn mod_2pi(angle: f64) -> f64 {
    (angle + PI) % (2. * PI) - PI
}

fn params_zyz_inner(mat: ArrayView2<Complex64>) -> [f64; 4] {
    let coeff: Complex64 = 1. / det_one_qubit(mat).sqrt();
    let phase = -complex_phase(coeff);
    let tmp_1_0 = (coeff * mat[[1, 0]]).abs();
    let tmp_0_0 = (coeff * mat[[0, 0]]).abs();
    let theta = 2. * tmp_1_0.atan2(tmp_0_0);
    let phiplambda2 = complex_phase(coeff * mat[[1, 1]]);
    let phimlambda2 = complex_phase(coeff * mat[[1, 0]]);
    let phi = phiplambda2 + phimlambda2;
    let lam = phiplambda2 - phimlambda2;
    [theta, phi, lam, phase]
}

fn params_zxz_inner(mat: ArrayView2<Complex64>) -> [f64; 4] {
    let [theta, phi, lam, phase] = params_zyz_inner(mat);
    [theta, phi + PI / 2., lam - PI / 2., phase]
}

#[pyfunction]
pub fn params_zyz(unitary: PyReadonlyArray2<Complex64>) -> [f64; 4] {
    let mat = unitary.as_array();
    params_zyz_inner(mat)
}

#[pyfunction]
pub fn params_xyx(unitary: PyReadonlyArray2<Complex64>) -> [f64; 4] {
    let mat = unitary.as_array();
    let mat_zyz = arr2(&[
        [
            0.5 * (mat[[0, 0]] + mat[[0, 1]] + mat[[1, 0]] + mat[[1, 1]]),
            0.5 * (mat[[0, 0]] - mat[[0, 1]] + mat[[1, 0]] - mat[[1, 1]]),
        ],
        [
            0.5 * (mat[[0, 0]] + mat[[0, 1]] - mat[[1, 0]] - mat[[1, 1]]),
            0.5 * (mat[[0, 0]] - mat[[0, 1]] - mat[[1, 0]] + mat[[1, 1]]),
        ],
    ]);
    let [theta, phi, lam, phase] = params_zyz_inner(mat_zyz.view());
    let new_phi = mod_2pi(phi + PI);
    let new_lam = mod_2pi(lam + PI);
    [
        theta,
        new_phi,
        new_lam,
        phase + (new_phi + new_lam - phi - lam) / 2.,
    ]
}

#[pyfunction]
pub fn params_xzx(unitary: PyReadonlyArray2<Complex64>) -> [f64; 4] {
    let umat = unitary.as_array();
    let det = det_one_qubit(umat);
    let phase = (Complex64::new(0., -1.) * det.ln()).re / 2.;
    let sqrt_det = det.sqrt();
    let mat_zyz = arr2(&[
        [
            Complex64::new((umat[[0, 0]] / sqrt_det).re, (umat[[1, 0]] / sqrt_det).im),
            Complex64::new((umat[[1, 0]] / sqrt_det).re, (umat[[0, 0]] / sqrt_det).im),
        ],
        [
            Complex64::new(-(umat[[1, 0]] / sqrt_det).re, (umat[[0, 0]] / sqrt_det).im),
            Complex64::new((umat[[0, 0]] / sqrt_det).re, -(umat[[1, 0]] / sqrt_det).im),
        ],
    ]);
    let [theta, phi, lam, phase_zxz] = params_zxz_inner(mat_zyz.view());
    [theta, phi, lam, phase + phase_zxz]
}

#[pyfunction]
pub fn params_zxz(unitary: PyReadonlyArray2<Complex64>) -> [f64; 4] {
    let mat = unitary.as_array();
    params_zxz_inner(mat)
}

#[pymodule]
pub fn euler_one_qubit_decomposer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(params_zyz))?;
    m.add_wrapped(wrap_pyfunction!(params_xyx))?;
    m.add_wrapped(wrap_pyfunction!(params_xzx))?;
    m.add_wrapped(wrap_pyfunction!(params_zxz))?;
    Ok(())
}
