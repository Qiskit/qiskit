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

use num_complex::Complex64;
use std::f64::consts::FRAC_1_SQRT_2;

pub static ONE_QUBIT_IDENTITY: [[Complex64; 2]; 2] = [
    [Complex64::new(1., 0.), Complex64::new(0., 0.)],
    [Complex64::new(0., 0.), Complex64::new(1., 0.)],
];

#[inline]
pub fn rx_gate(theta: f64) -> [[Complex64; 2]; 2] {
    let half_theta = theta / 2.;
    let cos = Complex64::new(half_theta.cos(), 0.);
    let isin = Complex64::new(0., -half_theta.sin());
    [[cos, isin], [isin, cos]]
}

#[inline]
pub fn ry_gate(theta: f64) -> [[Complex64; 2]; 2] {
    let half_theta = theta / 2.;
    let cos = Complex64::new(half_theta.cos(), 0.);
    let sin = Complex64::new(half_theta.sin(), 0.);
    [[cos, -sin], [sin, cos]]
}

#[inline]
pub fn rz_gate(theta: f64) -> [[Complex64; 2]; 2] {
    let ilam2 = Complex64::new(0., 0.5 * theta);
    [
        [(-ilam2).exp(), Complex64::new(0., 0.)],
        [Complex64::new(0., 0.), ilam2.exp()],
    ]
}

pub static HGATE: [[Complex64; 2]; 2] = [
    [
        Complex64::new(FRAC_1_SQRT_2, 0.),
        Complex64::new(FRAC_1_SQRT_2, 0.),
    ],
    [
        Complex64::new(FRAC_1_SQRT_2, 0.),
        Complex64::new(-FRAC_1_SQRT_2, 0.),
    ],
];

pub static CXGATE: [[Complex64; 4]; 4] = [
    [
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(1., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
    ],
];

pub static SXGATE: [[Complex64; 2]; 2] = [
    [Complex64::new(0.5, 0.5), Complex64::new(0.5, -0.5)],
    [Complex64::new(0.5, -0.5), Complex64::new(0.5, 0.5)],
];

pub static XGATE: [[Complex64; 2]; 2] = [
    [Complex64::new(0., 0.), Complex64::new(1., 0.)],
    [Complex64::new(1., 0.), Complex64::new(0., 0.)],
];

pub static ZGATE: [[Complex64; 2]; 2] = [
    [Complex64::new(1., 0.), Complex64::new(0., 0.)],
    [Complex64::new(0., 0.), Complex64::new(-1., 0.)],
];

pub static YGATE: [[Complex64; 2]; 2] = [
    [Complex64::new(0., 0.), Complex64::new(0., -1.)],
    [Complex64::new(0., 1.), Complex64::new(0., 0.)],
];

pub static CZGATE: [[Complex64; 4]; 4] = [
    [
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(-1., 0.),
    ],
];

pub static CYGATE: [[Complex64; 4]; 4] = [
    [
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., -1.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0., 1.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
    ],
];

pub static CCXGATE: [[Complex64; 8]; 8] = [
    [
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(1., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
    ],
];

pub static ECRGATE: [[Complex64; 4]; 4] = [
    [
        Complex64::new(0., 0.),
        Complex64::new(FRAC_1_SQRT_2, 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., FRAC_1_SQRT_2),
    ],
    [
        Complex64::new(FRAC_1_SQRT_2, 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., -FRAC_1_SQRT_2),
        Complex64::new(0., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0., FRAC_1_SQRT_2),
        Complex64::new(0., 0.),
        Complex64::new(FRAC_1_SQRT_2, 0.),
    ],
    [
        Complex64::new(0., -FRAC_1_SQRT_2),
        Complex64::new(0., 0.),
        Complex64::new(FRAC_1_SQRT_2, 0.),
        Complex64::new(0., 0.),
    ],
];

pub static SWAPGATE: [[Complex64; 4]; 4] = [
    [
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
    ],
    [
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(1., 0.),
    ],
];

#[inline]
pub fn global_phase_gate(theta: f64) -> [[Complex64; 1]; 1] {
    [[Complex64::new(0., theta).exp()]]
}

#[inline]
pub fn phase_gate(lam: f64) -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(1., 0.), Complex64::new(0., 0.)],
        [Complex64::new(0., 0.), Complex64::new(0., lam).exp()],
    ]
}

#[inline]
pub fn u_gate(theta: f64, phi: f64, lam: f64) -> [[Complex64; 2]; 2] {
    let cos = (theta / 2.).cos();
    let sin = (theta / 2.).sin();
    [
        [
            Complex64::new(cos, 0.),
            (-Complex64::new(0., lam).exp()) * sin,
        ],
        [
            Complex64::new(0., phi).exp() * sin,
            Complex64::new(0., phi + lam).exp() * cos,
        ],
    ]
}
