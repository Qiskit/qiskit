use std::f64::consts::PI;
use ndarray::prelude::*;
use ndarray::linalg::kron;
use crate::xx_decompose::utilities::{safe_acos, Square};
use crate::gates::{rz_matrix, rxx_matrix, ryy_matrix};

const PI2 : f64 = PI / 2.0;

fn decompose_xxyy_into_xxyy_xx(
    a_target: f64,
    b_target: f64,
    a_source: f64,
    b_source: f64,
    interaction: f64,
) -> [f64; 6] {
    let cplus = (a_source + b_source).cos();
    let cminus = (a_source - b_source).cos();
    let splus = (a_source + b_source).sin();
    let sminus = (a_source - b_source).sin();
    let ca = interaction.cos();
    let sa = interaction.sin();

    let uplusv =
        1. / 2. *
        safe_acos(cminus.sq() * ca.sq() + sminus.sq() * sa.sq() - (a_target - b_target).cos().sq(),
                  2. * cminus * ca * sminus * sa);

    let uminusv =
        1. / 2.
        * safe_acos(
            cplus.sq() * ca.sq() + splus.sq() * sa.sq() - (a_target + b_target).cos().sq(),
            2. * cplus * ca * splus * sa,
        );

    let (u, v) = ((uplusv + uminusv) / 2., (uplusv - uminusv) / 2.);

    let middle_matrix = rxx_matrix(2. * a_source)
        .dot(&ryy_matrix(2. * b_source))
        .dot(&kron(&rz_matrix(2. * u), &rz_matrix(2. * v)))
        .dot(&rxx_matrix(2. * interaction));

    let phase_solver = {
        let q = 1. / 4.;
        let mq = - 1. / 4.;
        array![
            [q, q, q, q],
            [q, mq, mq, q],
            [q, q, mq, mq],
            [q, mq, q, mq],
        ]
    };
    let inner_phases = array![
        middle_matrix[[0, 0]].arg(), middle_matrix[[1, 1]].arg(),
        middle_matrix[[1, 2]].arg() + PI2,
        middle_matrix[[0, 3]].arg() + PI2,
    ];
    let [mut r, mut s, mut x, mut y] = {
        let p = phase_solver.dot(&inner_phases);
        [p[0],p[1],p[2],p[3]]
    };

    let generated_matrix =
        kron(&rz_matrix(2. * r), &rz_matrix(2. * s))
        .dot(&middle_matrix)
        .dot(&kron(&rz_matrix(2. * x), &rz_matrix(2. * y)));

    // If there's a phase discrepancy, need to conjugate by an extra Z/2 (x) Z/2.
    if ((generated_matrix[[3, 0]].arg().abs() - PI2) < 0.01
        && a_target > b_target)
        || ((generated_matrix[[3, 0]].arg().abs() + PI2) < 0.01
            && a_target < b_target) {
            x += PI / 4.;
            y += PI / 4.;
            r -= PI / 4.;
            s -= PI / 4.;
        }
    [r, s, u, v, x, y]
}
