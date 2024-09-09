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

//use num_complex::{Complex64, ComplexFloat};
use num_complex::Complex64;
use ndarray::prelude::*;
// For Complex64::zero()
// use num_traits::Zero;
use qiskit_circuit::util::{c64, C_ZERO};

pub(crate) fn rz_matrix(theta: f64) -> Array2<Complex64> {
    let ilam2 = c64(0., 0.5 * theta);
    array![[(-ilam2).exp(), C_ZERO], [C_ZERO, ilam2.exp()]]
}

pub(crate) fn rxx_matrix(theta: f64) -> Array2<Complex64> {
    let theta2 = theta / 2.0;
//    let cos = c64::new(theta2.cos(), 0.0);
    let cos = c64(theta2.cos(), 0.0);    
    let isin = c64(0.0, theta2.sin());
    let cz = C_ZERO;
    array![
        [cos, cz, cz, -isin],
        [cz, cos, -isin, cz],
        [cz, -isin, cos, cz],
        [-isin, cz, cz, cos],
        ]
}

pub(crate) fn ryy_matrix(theta: f64) -> Array2<Complex64> {
    let theta2 = theta / 2.0;
    let cos = Complex64::new(theta2.cos(), 0.0);
    let isin = Complex64::new(0.0, theta2.sin());
    let cz = C_ZERO;
    array![
        [cos, cz, cz, isin],
        [cz, cos, -isin, cz],
        [cz, -isin, cos, cz],
        [isin, cz, cz, cos],
        ]
}
