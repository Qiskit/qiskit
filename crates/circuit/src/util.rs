// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use num_complex::Complex64;

// This is a very conservative version of an abbreviation for constructing new Complex64.
// A couple of alternatives to this function are
// `c64<T: Into<f64>, V: Into<f64>>(re: T, im: V) -> Complex64`
// Disadvantages are:
//  1. Some people don't like that this allows things like `c64(1, 0)`. Presumably,
//     they prefer a more explicit construction.
//  2. This will not work in `const` and `static` constructs.
// Another alternative is
//   macro_rules! c64 {
//       ($re: expr, $im: expr $(,)*) => {
//           Complex64::new($re as f64, $im as f64)
//       };
// Advantages: This allows things like `c64!(1, 2.0)`, including in
// `static` and `const` constructs.
//  Disadvantages:
//  1. Three characters `c64!` rather than two `c64`.
//  2. Some people prefer the opposite of the advantages, i.e. more explicitness.
/// Create a new [`Complex<f64>`]
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
