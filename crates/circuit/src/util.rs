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

use num_complex::{Complex, Complex64};

// This is almost the same as the function that became available in
// num-complex 0.4.6. The difference is that two generic parameters are
// used here rather than one. This allows call like `c64(half_theta.cos(), 0);`
// that mix f64 and integer arguments.
/// Create a new [`Complex<f64>`] with arguments that can convert [`Into<f64>`].
///
/// ```
/// use num_complex::{c64, Complex64};
/// assert_eq!(c64(1, 2), Complex64::new(1.0, 2.0));
/// ```
#[inline]
pub fn c64<T: Into<f64>, V: Into<f64>>(re: T, im: V) -> Complex64 {
    Complex::new(re.into(), im.into())
}

/// Create a new [`Complex<f64>`] with arguments that can be converted to `f64`
/// via `as`.
/// This macro may be used in `static` and `const` statements. That is, the replacement
/// by the constructor occurs at compile time.
#[macro_export]
macro_rules! c64 {
    ($re: expr, $im: expr $(,)*) => {
        Complex64::new($re as f64, $im as f64)
    };
}

pub type GateArray0Q = [[Complex64; 1]; 1];
pub type GateArray1Q = [[Complex64; 2]; 2];
pub type GateArray2Q = [[Complex64; 4]; 4];
pub type GateArray3Q = [[Complex64; 8]; 8];

pub const ZERO: Complex64 = c64!(0, 0);
pub const ONE: Complex64 = c64!(1, 0);
pub const M_ONE: Complex64 = c64!(-1, 0);
pub const IM: Complex64 = c64!(0, 1);
pub const M_IM: Complex64 = c64!(0, -1);
