// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use num_complex::Complex64;

/// A short-hand constructor for Complex64 used in constant and static definitions.
#[inline(always)]
pub const fn c64(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

pub type GateArray0Q = [[Complex64; 1]; 1];
pub type GateArray1Q = [[Complex64; 2]; 2];
pub type GateArray2Q = [[Complex64; 4]; 4];
pub type GateArray3Q = [[Complex64; 8]; 8];
pub type GateArray4Q = [[Complex64; 16]; 16];

// Use prefix `C_` to distinguish from real, for example
pub const C_ZERO: Complex64 = c64(0., 0.);
pub const C_ONE: Complex64 = c64(1., 0.);
pub const C_M_ONE: Complex64 = c64(-1., 0.);
pub const IM: Complex64 = c64(0., 1.);
pub const M_IM: Complex64 = c64(0., -1.);

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, FRAC_PI_8, SQRT_2};
pub const C_FRAC_PI_2: Complex64 = c64(FRAC_PI_2, 0.0);
pub const C_FRAC_PI_4: Complex64 = c64(FRAC_PI_4, 0.0);
pub const C_FRAC_PI_8: Complex64 = c64(FRAC_PI_8, 0.0);
pub const C_FRAC_PI_2_SQRT_2: Complex64 = c64(FRAC_PI_2 / SQRT_2, 0.0);
pub const C_M_FRAC_PI_4: Complex64 = c64(-FRAC_PI_4, 0.0);
pub const C_M_FRAC_PI_8: Complex64 = c64(-FRAC_PI_8, 0.0);
pub const C_M_FRAC_PI_2_SQRT_2: Complex64 = c64(-FRAC_PI_2 / SQRT_2, 0.0);
