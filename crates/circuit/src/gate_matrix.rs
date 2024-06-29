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
use numpy::npyffi::npy_half;
use std::f64::consts::FRAC_1_SQRT_2;

// num-complex exposes an equivalent function but it's not a const function
// so it's not compatible with static definitions. This is a const func and
// just reduces the amount of typing we need.
#[inline(always)]
const fn c64(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

pub static ONE_QUBIT_IDENTITY: [[Complex64; 2]; 2] =
    [[c64(1., 0.), c64(0., 0.)], [c64(0., 0.), c64(1., 0.)]];

#[inline]
pub fn rx_gate(theta: f64) -> [[Complex64; 2]; 2] {
    let half_theta = theta / 2.;
    let cos = c64(half_theta.cos(), 0.);
    let isin = c64(0., -half_theta.sin());
    [[cos, isin], [isin, cos]]
}

#[inline]
pub fn ry_gate(theta: f64) -> [[Complex64; 2]; 2] {
    let half_theta = theta / 2.;
    let cos = c64(half_theta.cos(), 0.);
    let sin = c64(half_theta.sin(), 0.);
    [[cos, -sin], [sin, cos]]
}

#[inline]
pub fn rz_gate(theta: f64) -> [[Complex64; 2]; 2] {
    let ilam2 = c64(0., 0.5 * theta);
    [[(-ilam2).exp(), c64(0., 0.)], [c64(0., 0.), ilam2.exp()]]
}

pub static H_GATE: [[Complex64; 2]; 2] = [
    [c64(FRAC_1_SQRT_2, 0.), c64(FRAC_1_SQRT_2, 0.)],
    [c64(FRAC_1_SQRT_2, 0.), c64(-FRAC_1_SQRT_2, 0.)],
];

pub static CX_GATE: [[Complex64; 4]; 4] = [
    [c64(1., 0.), c64(0., 0.), c64(0., 0.), c64(0., 0.)],
    [c64(0., 0.), c64(0., 0.), c64(0., 0.), c64(1., 0.)],
    [c64(0., 0.), c64(0., 0.), c64(1., 0.), c64(0., 0.)],
    [c64(0., 0.), c64(1., 0.), c64(0., 0.), c64(0., 0.)],
];

pub static SX_GATE: [[Complex64; 2]; 2] = [
    [c64(0.5, 0.5), c64(0.5, -0.5)],
    [c64(0.5, -0.5), c64(0.5, 0.5)],
];

pub static SXDG_GATE: [[Complex64; 2]; 2] = [
    [c64(0.5, -0.5), c64(0.5, 0.5)],
    [c64(0.5, 0.5), c64(0.5, -0.5)],
];

pub static X_GATE: [[Complex64; 2]; 2] = [[c64(0., 0.), c64(1., 0.)], [c64(1., 0.), c64(0., 0.)]];

pub static Z_GATE: [[Complex64; 2]; 2] = [[c64(1., 0.), c64(0., 0.)], [c64(0., 0.), c64(-1., 0.)]];

pub static Y_GATE: [[Complex64; 2]; 2] = [[c64(0., 0.), c64(0., -1.)], [c64(0., 1.), c64(0., 0.)]];

pub static CZ_GATE: [[Complex64; 4]; 4] = [
    [c64(1., 0.), c64(0., 0.), c64(0., 0.), c64(0., 0.)],
    [c64(0., 0.), c64(1., 0.), c64(0., 0.), c64(0., 0.)],
    [c64(0., 0.), c64(0., 0.), c64(1., 0.), c64(0., 0.)],
    [c64(0., 0.), c64(0., 0.), c64(0., 0.), c64(-1., 0.)],
];

pub static CY_GATE: [[Complex64; 4]; 4] = [
    [c64(1., 0.), c64(0., 0.), c64(0., 0.), c64(0., 0.)],
    [c64(0., 0.), c64(0., 0.), c64(0., 0.), c64(0., -1.)],
    [c64(0., 0.), c64(0., 0.), c64(1., 0.), c64(0., 0.)],
    [c64(0., 0.), c64(0., 1.), c64(0., 0.), c64(0., 0.)],
];

pub static CCX_GATE: [[Complex64; 8]; 8] = [
    [
        c64(1., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
    ],
    [
        c64(0., 0.),
        c64(1., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
    ],
    [
        c64(0., 0.),
        c64(0., 0.),
        c64(1., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
    ],
    [
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(1., 0.),
    ],
    [
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(1., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
    ],
    [
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(1., 0.),
        c64(0., 0.),
        c64(0., 0.),
    ],
    [
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(1., 0.),
        c64(0., 0.),
    ],
    [
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(1., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
        c64(0., 0.),
    ],
];

pub static ECR_GATE: [[Complex64; 4]; 4] = [
    [
        c64(0., 0.),
        c64(FRAC_1_SQRT_2, 0.),
        c64(0., 0.),
        c64(0., FRAC_1_SQRT_2),
    ],
    [
        c64(FRAC_1_SQRT_2, 0.),
        c64(0., 0.),
        c64(0., -FRAC_1_SQRT_2),
        c64(0., 0.),
    ],
    [
        c64(0., 0.),
        c64(0., FRAC_1_SQRT_2),
        c64(0., 0.),
        c64(FRAC_1_SQRT_2, 0.),
    ],
    [
        c64(0., -FRAC_1_SQRT_2),
        c64(0., 0.),
        c64(FRAC_1_SQRT_2, 0.),
        c64(0., 0.),
    ],
];

pub static SWAP_GATE: [[Complex64; 4]; 4] = [
    [c64(1., 0.), c64(0., 0.), c64(0., 0.), c64(0., 0.)],
    [c64(0., 0.), c64(0., 0.), c64(1., 0.), c64(0., 0.)],
    [c64(0., 0.), c64(1., 0.), c64(0., 0.), c64(0., 0.)],
    [c64(0., 0.), c64(0., 0.), c64(0., 0.), c64(1., 0.)],
];
pub static ISWAP_GATE: [[Complex64; 4]; 4] = [
    [c64(1., 0.), c64(0., 0.), c64(0., 0.), c64(0., 0.)],
    [c64(0., 0.), c64(0., 0.), c64(0., 1.), c64(0., 0.)],
    [c64(0., 0.), c64(0., 1.), c64(0., 0.), c64(0., 0.)],
    [c64(0., 0.), c64(0., 0.), c64(0., 0.), c64(1., 0.)],
];

pub static S_GATE: [[Complex64; 2]; 2] = [[c64(1., 0.), c64(0., 0.)], [c64(0., 0.), c64(0., 1.)]];

pub static SDG_GATE: [[Complex64; 2]; 2] =
    [[c64(1., 0.), c64(0., 0.)], [c64(0., 0.), c64(0., -1.)]];

pub static T_GATE: [[Complex64; 2]; 2] = [
    [c64(1., 0.), c64(0., 0.)],
    [c64(0., 0.), c64(FRAC_1_SQRT_2, FRAC_1_SQRT_2)],
];

pub static TDG_GATE: [[Complex64; 2]; 2] = [
    [c64(1., 0.), c64(0., 0.)],
    [c64(0., 0.), c64(FRAC_1_SQRT_2, -FRAC_1_SQRT_2)],
];

pub static DCX_GATE: [[Complex64; 4]; 4] = [
    [c64(1., 0.), c64(0., 0.), c64(0., 0.), c64(0., 0.)],
    [c64(0., 0.), c64(0., 0.), c64(0., 0.), c64(1., 0.)],
    [c64(0., 0.), c64(1., 0.), c64(0., 0.), c64(0., 0.)],
    [c64(0., 0.), c64(0., 0.), c64(1., 0.), c64(0., 0.)],
];

#[inline]
pub fn global_phase_gate(theta: f64) -> [[Complex64; 1]; 1] {
    [[c64(0., theta).exp()]]
}

#[inline]
pub fn phase_gate(lam: f64) -> [[Complex64; 2]; 2] {
    [
        [c64(1., 0.), c64(0., 0.)],
        [c64(0., 0.), c64(0., lam).exp()],
    ]
}

#[inline]
pub fn u_gate(theta: f64, phi: f64, lam: f64) -> [[Complex64; 2]; 2] {
    let cos = (theta / 2.).cos();
    let sin = (theta / 2.).sin();
    [
        [c64(cos, 0.), (-c64(0., lam).exp()) * sin],
        [c64(0., phi).exp() * sin, c64(0., phi + lam).exp() * cos],
    ]
}

#[inline]
pub fn xx_minus_yy_gate(theta: f64, beta: f64) -> [[Complex64; 4]; 4] {
    let cos = (theta / 2.).cos();
    let sin = (theta / 2.).sin();
    [
        [
            c64(cos, 0.),
            c64(0., 0.),
            c64(0., 0.),
            c64(0., -sin) * c64(0., -beta).exp(),
        ],
        [c64(0., 0.), c64(1., 0.), c64(0., 0.), c64(0., 0.)],
        [c64(0., 0.), c64(0., 0.), c64(1., 0.), c64(0., 0.)],
        [
            c64(0., -sin) * c64(0., beta).exp(),
            c64(0., 0.),
            c64(0., 0.),
            c64(cos, 0.),
        ],
    ]
}

#[inline]
pub fn u1_gate(lam: f64) -> [[Complex64; 2]; 2] {
    [
        [c64(1., 0.), c64(0., 0.)],
        [c64(0., 0.), c64(0., lam).exp()],
    ]
}

#[inline]
pub fn u2_gate(phi: f64, lam: f64) -> [[Complex64; 2]; 2] {
    [
        [
            c64(FRAC_1_SQRT_2, 0.),
            (-c64(0., lam).exp()) * FRAC_1_SQRT_2,
        ],
        [
            c64(0., phi).exp() * FRAC_1_SQRT_2,
            c64(0., phi + lam).exp() * FRAC_1_SQRT_2,
        ],
    ]
}

#[inline]
pub fn u3_gate(theta: f64, phi: f64, lam: f64) -> [[Complex64; 2]; 2] {
    let cos = (theta / 2.).cos();
    let sin = (theta / 2.).sin();
    [
        [c64(cos, 0.), -(c64(0., lam).exp()) * sin],
        [c64(0., phi).exp() * sin, c64(0., phi + lam).exp() * cos],
    ]
}

#[inline]
pub fn xx_plus_yy_gate(theta: f64, beta: f64) -> [[Complex64; 4]; 4] {
    let cos = (theta / 2.).cos();
    let sin = (theta / 2.).sin();
    [
        [c64(1., 0.), c64(0., 0.), c64(0., 0.), c64(0., 0.)],
        [
            c64(0., 0.),
            c64(cos, 0.),
            c64(0., -sin) * c64(0., -beta).exp(),
            c64(0., 0.),
        ],
        [
            c64(0., 0.),
            c64(0., -sin) * c64(0., beta).exp(),
            c64(cos, 0.),
            c64(0., 0.),
        ],
        [c64(0., 0.), c64(0., 0.), c64(0., 0.), c64(1., 0.)],
    ]
}

#[inline]
pub fn rxx_gate(theta: f64) -> [[Complex64; 4]; 4] {
    let (sint, cost) = (theta / 2.0).sin_cos();
    let ccos = c64(cost, 0.);
    let csinm = c64(0., -sint);
    let c0 = c64(0., 0.);

    [
        [ccos, c0, c0, csinm],
        [c0, ccos, csinm, c0],
        [c0, csinm, ccos, c0],
        [csinm, c0, c0, ccos],
    ]
}

#[inline]
pub fn ryy_gate(theta: f64) -> [[Complex64; 4]; 4] {
    let (sint, cost) = (theta / 2.0).sin_cos();
    let ccos = c64(cost, 0.);
    let csin = c64(0., sint);
    let c0 = c64(0., 0.);

    [
        [ccos, c0, c0, csin],
        [c0, ccos, -csin, c0],
        [c0, -csin, ccos, c0],
        [csin, c0, c0, ccos],
    ]
}

#[inline]
pub fn rzz_gate(theta: f64) -> [[Complex64; 4]; 4] {
    let (sint, cost) = (theta / 2.0).sin_cos();
    let c0 = c64(0., 0.);
    let exp_it2 = c64(cost, sint);
    let exp_mit2 = c64(cost, -sint);

    [
        [exp_mit2, c0, c0, c0],
        [c0, exp_it2, c0, c0],
        [c0, c0, exp_it2, c0],
        [c0, c0, c0, exp_mit2],
    ]
}

#[inline]
pub fn rzx_gate(theta: f64) -> [[Complex64; 4]; 4] {
    let (sint, cost) = (theta / 2.0).sin_cos();
    let ccos = c64(cost, 0.);
    let csin = c64(0., sint);
    let c0 = c64(0., 0.);

    [
        [ccos, c0, -csin, c0],
        [c0, ccos, c0, csin],
        [-csin, c0, ccos, c0],
        [c0, csin, c0, ccos],
    ]
}
