// This code is part of Qiskit.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// In numpy matrices real and imaginary components are adjacent:
//   np.array([1,2,3], dtype='complex').view('float64')
//   array([1., 0., 2., 0., 3., 0.])
// The matrix faer::Mat<c64> has this layout.
// faer::Mat<num_complex::Complex<f64>> instead stores a matrix
// of real components and one of imaginary components.
// In order to avoid copying we want to use `MatRef<c64>` or `MatMut<c64>`.

use num_complex::Complex;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;
use std::f64::consts::PI;

use faer::IntoFaerComplex;
use faer::{mat, prelude::*, scale, Mat, MatRef};
use faer_core::{c64, ComplexField};
use numpy::PyReadonlyArray2;

use crate::utils;

const PI2: f64 = PI / 2.0;
const PI4: f64 = PI / 4.0;
const PI32: f64 = 3.0 * PI2;
// FIXME: zero and one exist but I cant find the right incantation
const C0: c64 = c64 { re: 0.0, im: 0.0 };
const C1: c64 = c64 { re: 1.0, im: 0.0 };
const C1IM: c64 = c64 { re: 0.0, im: 1.0 };

// FIXME: Find a way to compute these matrices at compile time.
fn transform_from_magic_basis(unitary: Mat<c64>) -> Mat<c64> {
    let _b_nonnormalized: Mat<c64> = mat![
        [C1, C1IM, C0, C0],
        [C0, C0, C1IM, C1],
        [C0, C0, C1IM, -C1],
        [C1, -C1IM, C0, C0]
    ];
    let _b_nonnormalized_dagger = scale(c64 { re: 0.5, im: 0.0 }) * _b_nonnormalized.adjoint();
    _b_nonnormalized_dagger * unitary * _b_nonnormalized
}

// faer::c64 and num_complex::Complex<f64> are both structs
// holding two f64's. But several functions are not defined for
// c64. So we implement them here. These things should be contribute
// upstream.

pub trait PowF {
    fn powf(self, pow: f64) -> c64;
}

impl PowF for c64 {
    fn powf(self, pow: f64) -> c64 {
        c64::from(self.to_num_complex().powf(pow))
    }
}

pub trait Arg {
    fn arg(self) -> f64;
}

impl Arg for c64 {
    fn arg(self) -> f64 {
        self.to_num_complex().arg()
    }
}

fn __weyl_coordinates(unitary: MatRef<c64>) -> [f64; 3] {
    let uscaled = scale(C1 / unitary.determinant().powf(0.25)) * unitary;
    let uup = transform_from_magic_basis(uscaled);
    let mut darg: Vec<_> = (uup.transpose() * &uup)
        .complex_eigenvalues()
        .into_iter()
        .map(|x: c64| -x.arg() / 2.0)
        .collect();
    darg[3] = -darg[0] - darg[1] - darg[2];
    let mut cs: Vec<_> = (0..3)
        .map(|i| ((darg[i] + darg[3]) / 2.0).rem_euclid(2.0 * PI))
        .collect();
    let cstemp: Vec<f64> = cs
        .iter()
        .map(|x| x.rem_euclid(PI2))
        .map(|x| x.min(PI2 - x))
        .collect();
    let mut order = utils::arg_sort(&cstemp);
    (order[0], order[1], order[2]) = (order[1], order[2], order[0]);
    (cs[0], cs[1], cs[2]) = (cs[order[0]], cs[order[1]], cs[order[2]]);

    // Flip into Weyl chamber
    if cs[0] > PI2 {
        cs[0] -= PI32;
    }
    if cs[1] > PI2 {
        cs[1] -= PI32;
    }
    let mut conjs = 0;
    if cs[0] > PI4 {
        cs[0] = PI2 - cs[0];
        conjs += 1;
    }
    if cs[1] > PI4 {
        cs[1] = PI2 - cs[1];
        conjs += 1;
    }
    if cs[2] > PI2 {
        cs[2] -= PI32;
    }
    if conjs == 1 {
        cs[2] = PI2 - cs[2];
    }
    if cs[2] > PI4 {
        cs[2] -= PI2;
    }
    [cs[1], cs[0], cs[2]]
}

#[pyfunction]
#[pyo3(text_signature = "(basis_b, basis_fidelity, unitary, /")]
pub fn _num_basis_gates(
    basis_b: f64,
    basis_fidelity: f64,
    unitary: PyReadonlyArray2<Complex<f64>>,
) -> usize {
    let u = unitary.as_array().into_faer_complex();
    __num_basis_gates(basis_b, basis_fidelity, u)
}

fn __num_basis_gates(basis_b: f64, basis_fidelity: f64, unitary: MatRef<c64>) -> usize {
    let [a, b, c] = __weyl_coordinates(unitary);
    let traces = [
        c64::new(
            4.0 * (a.cos() * b.cos() * c.cos()),
            4.0 * (a.sin() * b.sin() * c.sin()),
        ),
        c64::new(
            4.0 * (PI4 - a).cos() * (basis_b - b).cos() * c.cos(),
            4.0 * (PI4 - a).sin() * (basis_b - b).sin() * c.sin(),
        ),
        c64::new(4.0 * c.cos(), 0.0),
        c64::new(4.0, 0.0),
    ];
    // The originial Python had `np.argmax`, which returns the lowest index in case two or more
    // values have a common maximum value.
    // `max_by` and `min_by` return the highest and lowest indices respectively, in case of ties.
    // So to reproduce `np.argmax`, we use `min_by` and switch the order of the
    // arguments in the comparison.
    traces
        .into_iter()
        .enumerate()
        .map(|(idx, trace)| (idx, trace_to_fid(&trace) * basis_fidelity.powi(idx as i32)))
        .min_by(|(_idx1, fid1), (_idx2, fid2)| fid2.partial_cmp(fid1).unwrap())
        .unwrap()
        .0
}

/// Average gate fidelity is :math:`Fbar = (d + |Tr (Utarget \\cdot U^dag)|^2) / d(d+1)`
/// M. Horodecki, P. Horodecki and R. Horodecki, PRA 60, 1888 (1999)
fn trace_to_fid(trace: &c64) -> f64 {
    (4.0 + trace.faer_abs2()) / 20.0
}

#[pymodule]
pub fn two_qubit_decompose(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(_num_basis_gates))?;
    Ok(())
}
