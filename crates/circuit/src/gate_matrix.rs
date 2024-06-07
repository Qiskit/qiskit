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
use num_complex::Complex64;
use std::f64::consts::FRAC_1_SQRT_2;

// `c64` is a function for constructing complex numbers.
use crate::util::{
    c64, GateArray0Q, GateArray1Q, GateArray2Q, GateArray3Q, IM, M_IM, M_ONE, ONE, ZERO,
};
// Import the c64! macro for constructing complex numbers.
use crate::c64;

pub static ONE_QUBIT_IDENTITY: GateArray1Q = [[ONE, ZERO], [ZERO, ONE]];

#[inline]
pub fn rx_gate(theta: f64) -> GateArray1Q {
    let half_theta = theta / 2.;
    let cos = c64(half_theta.cos(), 0);
    let isin = c64(0., -half_theta.sin());
    [[cos, isin], [isin, cos]]
}

#[inline]
pub fn ry_gate(theta: f64) -> GateArray1Q {
    let half_theta = theta / 2.;
    let cos = c64(half_theta.cos(), 0);
    let sin = c64(half_theta.sin(), 0);
    [[cos, -sin], [sin, cos]]
}

#[inline]
pub fn r_gate(theta: f64, phi: f64) -> GateArray1Q {
    let half_theta = theta / 2.;
    let cost = c64(half_theta.cos(), 0);
    let sint = half_theta.sin();
    let cosphi = phi.cos();
    let sinphi = phi.sin();
    [
        [cost, c64(sint * sinphi, -sint * cosphi)],
        [c64(-sint * sinphi, -sint * cosphi), cost],
    ]
}

#[inline]
pub fn rz_gate(theta: f64) -> GateArray1Q {
    let ilam2 = c64(0, 0.5 * theta);
    [[(-ilam2).exp(), ZERO], [ZERO, ilam2.exp()]]
}

pub static HGATE: GateArray1Q = [
    [c64!(FRAC_1_SQRT_2, 0), c64!(FRAC_1_SQRT_2, 0)],
    [c64!(FRAC_1_SQRT_2, 0), c64!(-FRAC_1_SQRT_2, 0)],
];

pub static CXGATE: GateArray2Q = [
    [ONE, ZERO, ZERO, ZERO],
    [ZERO, ZERO, ZERO, ONE],
    [ZERO, ZERO, ONE, ZERO],
    [ZERO, ONE, ZERO, ZERO],
];

pub static SXGATE: GateArray1Q = [
    [c64!(0.5, 0.5), c64!(0.5, -0.5)],
    [c64!(0.5, -0.5), c64!(0.5, 0.5)],
];

pub static XGATE: GateArray1Q = [[ZERO, ONE], [ONE, ZERO]];

pub static ZGATE: GateArray1Q = [[ONE, ZERO], [ZERO, M_ONE]];

pub static YGATE: GateArray1Q = [[ZERO, M_IM], [IM, ZERO]];

pub static CZGATE: GateArray2Q = [
    [ONE, ZERO, ZERO, ZERO],
    [ZERO, ONE, ZERO, ZERO],
    [ZERO, ZERO, ONE, ZERO],
    [ZERO, ZERO, ZERO, M_ONE],
];

pub static CYGATE: GateArray2Q = [
    [ONE, ZERO, ZERO, ZERO],
    [ZERO, ZERO, ZERO, M_IM],
    [ZERO, ZERO, ONE, ZERO],
    [ZERO, IM, ZERO, ZERO],
];

pub static CCXGATE: GateArray3Q = [
    [ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO],
    [ZERO, ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO],
    [ZERO, ZERO, ONE, ZERO, ZERO, ZERO, ZERO, ZERO],
    [ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE],
    [ZERO, ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ZERO],
    [ZERO, ZERO, ZERO, ZERO, ZERO, ONE, ZERO, ZERO],
    [ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE, ZERO],
    [ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ZERO, ZERO],
];

pub static ECRGATE: GateArray2Q = [
    [ZERO, c64!(FRAC_1_SQRT_2, 0), ZERO, c64!(0, FRAC_1_SQRT_2)],
    [c64!(FRAC_1_SQRT_2, 0), ZERO, c64!(0, -FRAC_1_SQRT_2), ZERO],
    [ZERO, c64!(0, FRAC_1_SQRT_2), ZERO, c64!(FRAC_1_SQRT_2, 0)],
    [c64!(0, -FRAC_1_SQRT_2), ZERO, c64!(FRAC_1_SQRT_2, 0), ZERO],
];

pub static SWAPGATE: GateArray2Q = [
    [ONE, ZERO, ZERO, ZERO],
    [ZERO, ZERO, ONE, ZERO],
    [ZERO, ONE, ZERO, ZERO],
    [ZERO, ZERO, ZERO, ONE],
];

#[inline]
pub fn global_phase_gate(theta: f64) -> GateArray0Q {
    [[c64(0, theta).exp()]]
}

#[inline]
pub fn phase_gate(lam: f64) -> GateArray1Q {
    [[ONE, ZERO], [ZERO, c64(0, lam).exp()]]
}

#[inline]
pub fn u_gate(theta: f64, phi: f64, lam: f64) -> GateArray1Q {
    let cos = (theta / 2.).cos();
    let sin = (theta / 2.).sin();
    [
        [c64(cos, 0), -sin * c64(0, lam).exp()],
        [sin * c64(0, phi).exp(), cos * c64(0, phi + lam).exp()],
    ]
}
