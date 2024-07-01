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

use std::f64::consts::FRAC_1_SQRT_2;

use crate::util::{
    c64, GateArray0Q, GateArray1Q, GateArray2Q, GateArray3Q, C_M_ONE, C_ONE, C_ZERO, IM, M_IM,
};

pub static ONE_QUBIT_IDENTITY: GateArray1Q = [[C_ONE, C_ZERO], [C_ZERO, C_ONE]];

#[inline]
pub fn r_gate(theta: f64, phi: f64) -> GateArray1Q {
    let half_theta = theta / 2.;
    let cost = c64(half_theta.cos(), 0.);
    let sint = half_theta.sin();
    let cosphi = phi.cos();
    let sinphi = phi.sin();
    [
        [cost, c64(-sint * sinphi, -sint * cosphi)],
        [c64(sint * sinphi, -sint * cosphi), cost],
    ]
}

#[inline]
pub fn rx_gate(theta: f64) -> GateArray1Q {
    let half_theta = theta / 2.;
    let cos = c64(half_theta.cos(), 0.);
    let isin = c64(0., -half_theta.sin());
    [[cos, isin], [isin, cos]]
}

#[inline]
pub fn ry_gate(theta: f64) -> GateArray1Q {
    let half_theta = theta / 2.;
    let cos = c64(half_theta.cos(), 0.);
    let sin = c64(half_theta.sin(), 0.);
    [[cos, -sin], [sin, cos]]
}

#[inline]
pub fn rz_gate(theta: f64) -> GateArray1Q {
    let ilam2 = c64(0., 0.5 * theta);
    [[(-ilam2).exp(), C_ZERO], [C_ZERO, ilam2.exp()]]
}

#[inline]
pub fn crx_gate(theta: f64) -> GateArray2Q {
    let half_theta = theta / 2.;
    let cos = c64(half_theta.cos(), 0.);
    let isin = c64(0., half_theta.sin());
    [
        [C_ONE, C_ZERO, C_ZERO, C_ZERO],
        [C_ZERO, cos, C_ZERO, -isin],
        [C_ZERO, C_ZERO, C_ONE, C_ZERO],
        [C_ZERO, -isin, C_ZERO, cos],
    ]
}

#[inline]
pub fn cry_gate(theta: f64) -> GateArray2Q {
    let half_theta = theta / 2.;
    let cos = c64(half_theta.cos(), 0.);
    let sin = c64(half_theta.sin(), 0.);
    [
        [C_ONE, C_ZERO, C_ZERO, C_ZERO],
        [C_ZERO, cos, C_ZERO, -sin],
        [C_ZERO, C_ZERO, C_ONE, C_ZERO],
        [C_ZERO, sin, C_ZERO, cos],
    ]
}

#[inline]
pub fn crz_gate(theta: f64) -> GateArray2Q {
    let i_half_theta = c64(0., theta / 2.);
    [
        [C_ONE, C_ZERO, C_ZERO, C_ZERO],
        [C_ZERO, (-i_half_theta).exp(), C_ZERO, C_ZERO],
        [C_ZERO, C_ZERO, C_ONE, C_ZERO],
        [C_ZERO, C_ZERO, C_ZERO, i_half_theta.exp()],
    ]
}

pub static H_GATE: GateArray1Q = [
    [c64(FRAC_1_SQRT_2, 0.), c64(FRAC_1_SQRT_2, 0.)],
    [c64(FRAC_1_SQRT_2, 0.), c64(-FRAC_1_SQRT_2, 0.)],
];

pub static CX_GATE: GateArray2Q = [
    [C_ONE, C_ZERO, C_ZERO, C_ZERO],
    [C_ZERO, C_ZERO, C_ZERO, C_ONE],
    [C_ZERO, C_ZERO, C_ONE, C_ZERO],
    [C_ZERO, C_ONE, C_ZERO, C_ZERO],
];

pub static SX_GATE: GateArray1Q = [
    [c64(0.5, 0.5), c64(0.5, -0.5)],
    [c64(0.5, -0.5), c64(0.5, 0.5)],
];

pub static SXDG_GATE: GateArray1Q = [
    [c64(0.5, -0.5), c64(0.5, 0.5)],
    [c64(0.5, 0.5), c64(0.5, -0.5)],
];

pub static X_GATE: GateArray1Q = [[C_ZERO, C_ONE], [C_ONE, C_ZERO]];

pub static Z_GATE: GateArray1Q = [[C_ONE, C_ZERO], [C_ZERO, C_M_ONE]];

pub static Y_GATE: GateArray1Q = [[C_ZERO, M_IM], [IM, C_ZERO]];

pub static CZ_GATE: GateArray2Q = [
    [C_ONE, C_ZERO, C_ZERO, C_ZERO],
    [C_ZERO, C_ONE, C_ZERO, C_ZERO],
    [C_ZERO, C_ZERO, C_ONE, C_ZERO],
    [C_ZERO, C_ZERO, C_ZERO, C_M_ONE],
];

pub static CY_GATE: GateArray2Q = [
    [C_ONE, C_ZERO, C_ZERO, C_ZERO],
    [C_ZERO, C_ZERO, C_ZERO, M_IM],
    [C_ZERO, C_ZERO, C_ONE, C_ZERO],
    [C_ZERO, IM, C_ZERO, C_ZERO],
];

pub static CCX_GATE: GateArray3Q = [
    [
        C_ONE, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ONE, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ONE, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ONE,
    ],
    [
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ONE, C_ZERO, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ONE, C_ZERO, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ZERO, C_ONE, C_ZERO,
    ],
    [
        C_ZERO, C_ZERO, C_ZERO, C_ONE, C_ZERO, C_ZERO, C_ZERO, C_ZERO,
    ],
];

pub static ECR_GATE: GateArray2Q = [
    [
        C_ZERO,
        c64(FRAC_1_SQRT_2, 0.),
        C_ZERO,
        c64(0., FRAC_1_SQRT_2),
    ],
    [
        c64(FRAC_1_SQRT_2, 0.),
        C_ZERO,
        c64(0., -FRAC_1_SQRT_2),
        C_ZERO,
    ],
    [
        C_ZERO,
        c64(0., FRAC_1_SQRT_2),
        C_ZERO,
        c64(FRAC_1_SQRT_2, 0.),
    ],
    [
        c64(0., -FRAC_1_SQRT_2),
        C_ZERO,
        c64(FRAC_1_SQRT_2, 0.),
        C_ZERO,
    ],
];

pub static SWAP_GATE: GateArray2Q = [
    [C_ONE, C_ZERO, C_ZERO, C_ZERO],
    [C_ZERO, C_ZERO, C_ONE, C_ZERO],
    [C_ZERO, C_ONE, C_ZERO, C_ZERO],
    [C_ZERO, C_ZERO, C_ZERO, C_ONE],
];
pub static ISWAP_GATE: GateArray2Q = [
    [C_ONE, C_ZERO, C_ZERO, C_ZERO],
    [C_ZERO, C_ZERO, IM, C_ZERO],
    [C_ZERO, IM, C_ZERO, C_ZERO],
    [C_ZERO, C_ZERO, C_ZERO, C_ONE],
];

pub static S_GATE: GateArray1Q = [[C_ONE, C_ZERO], [C_ZERO, IM]];

pub static SDG_GATE: GateArray1Q = [[C_ONE, C_ZERO], [C_ZERO, M_IM]];

pub static T_GATE: GateArray1Q = [[C_ONE, C_ZERO], [C_ZERO, c64(FRAC_1_SQRT_2, FRAC_1_SQRT_2)]];

pub static TDG_GATE: GateArray1Q = [
    [C_ONE, C_ZERO],
    [C_ZERO, c64(FRAC_1_SQRT_2, -FRAC_1_SQRT_2)],
];

pub static DCX_GATE: GateArray2Q = [
    [C_ONE, C_ZERO, C_ZERO, C_ZERO],
    [C_ZERO, C_ZERO, C_ZERO, C_ONE],
    [C_ZERO, C_ONE, C_ZERO, C_ZERO],
    [C_ZERO, C_ZERO, C_ONE, C_ZERO],
];

#[inline]
pub fn global_phase_gate(theta: f64) -> GateArray0Q {
    [[c64(0., theta).exp()]]
}

#[inline]
pub fn phase_gate(lam: f64) -> GateArray1Q {
    [[C_ONE, C_ZERO], [C_ZERO, c64(0., lam).exp()]]
}

#[inline]
pub fn u_gate(theta: f64, phi: f64, lam: f64) -> GateArray1Q {
    let cos = (theta / 2.).cos();
    let sin = (theta / 2.).sin();
    [
        [c64(cos, 0.), (-c64(0., lam).exp()) * sin],
        [c64(0., phi).exp() * sin, c64(0., phi + lam).exp() * cos],
    ]
}

#[inline]
pub fn xx_minus_yy_gate(theta: f64, beta: f64) -> GateArray2Q {
    let cos = (theta / 2.).cos();
    let sin = (theta / 2.).sin();
    [
        [
            c64(cos, 0.),
            C_ZERO,
            C_ZERO,
            c64(0., -sin) * c64(0., -beta).exp(),
        ],
        [C_ZERO, C_ONE, C_ZERO, C_ZERO],
        [C_ZERO, C_ZERO, C_ONE, C_ZERO],
        [
            c64(0., -sin) * c64(0., beta).exp(),
            C_ZERO,
            C_ZERO,
            c64(cos, 0.),
        ],
    ]
}

#[inline]
pub fn u1_gate(lam: f64) -> GateArray1Q {
    [[C_ONE, C_ZERO], [C_ZERO, c64(0., lam).exp()]]
}

#[inline]
pub fn u2_gate(phi: f64, lam: f64) -> GateArray1Q {
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
pub fn u3_gate(theta: f64, phi: f64, lam: f64) -> GateArray1Q {
    let cos = (theta / 2.).cos();
    let sin = (theta / 2.).sin();
    [
        [c64(cos, 0.), -(c64(0., lam).exp()) * sin],
        [c64(0., phi).exp() * sin, c64(0., phi + lam).exp() * cos],
    ]
}

#[inline]
pub fn xx_plus_yy_gate(theta: f64, beta: f64) -> GateArray2Q {
    let cos = (theta / 2.).cos();
    let sin = (theta / 2.).sin();
    [
        [C_ONE, C_ZERO, C_ZERO, C_ZERO],
        [
            C_ZERO,
            c64(cos, 0.),
            c64(0., -sin) * c64(0., -beta).exp(),
            C_ZERO,
        ],
        [
            C_ZERO,
            c64(0., -sin) * c64(0., beta).exp(),
            c64(cos, 0.),
            C_ZERO,
        ],
        [C_ZERO, C_ZERO, C_ZERO, C_ONE],
    ]
}

#[inline]
pub fn rxx_gate(theta: f64) -> GateArray2Q {
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
pub fn ryy_gate(theta: f64) -> GateArray2Q {
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
pub fn rzz_gate(theta: f64) -> GateArray2Q {
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
pub fn rzx_gate(theta: f64) -> GateArray2Q {
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
