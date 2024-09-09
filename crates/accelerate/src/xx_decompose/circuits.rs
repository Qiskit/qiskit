use std::f64::consts::PI;
use ndarray::prelude::*;
use ndarray::linalg::kron;
use qiskit_circuit::circuit_data::CircuitData;
use crate::xx_decompose::utilities::{safe_acos, Square, EPSILON};
use crate::xx_decompose::weyl;
use crate::gates::{rz_matrix, rxx_matrix, ryy_matrix};
use crate::xx_decompose::types::{GateData, Coordinate};

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

// Builds a single step in an XX-based circuit.
//
// `source` and `target` are positive canonical coordinates; `strength` is the interaction strength
// at this step in the circuit as a canonical coordinate (so that CX = RZX(pi/2) corresponds to
// pi/4); and `embodiment` is a Qiskit circuit which enacts the canonical gate of the prescribed
// interaction `strength`.
fn xx_circuit_step(source: &Coordinate, strength: f64, target: &Coordinate,
                   embodiment: CircuitData) -> Result<(), String> {

    let mut permute_source_for_overlap: Option<Vec<GateData>> = None;
    let mut permute_target_for_overlap: Option<Vec<GateData>> = None;
    
    for reflection_name in &weyl::REFLECTION_NAMES {
        let (reflected_source_coord, source_reflection, reflection_phase_shift) = weyl::apply_reflection(*reflection_name, source);
        for source_shift_name in &weyl::SHIFT_NAMES {
            let (shifted_source_coord, source_shift, shift_phase_shift) = weyl::apply_shift(
                *source_shift_name, &reflected_source_coord);

            // check for overlap, back out permutation
            let (mut source_shared, mut target_shared) = (None, None);
            for (i, j) in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)] {
                if shifted_source_coord.distance(target, i, j) < EPSILON ||
                    shifted_source_coord.distance(target, j, i) < EPSILON
                {
                    (source_shared, target_shared) = (Some(i), Some(j));
                    break;
                }
            }
            if source_shared.is_none() {
                continue;
            }
            // Return [0, 1, 2] with `iex` deleted.
            fn exclude_one(iex: i32) -> [usize; 2] {
                let mut pair = [-1, -1];
                let mut j = 0;
                for i in 0..3 {
                    if i != iex {
                        pair[j] = i;
                        j += 1;
                    }
                }
                [pair[0] as usize, pair[1] as usize]
            } // exclude_one

            let [source_first, source_second] = exclude_one(source_shared.unwrap());
            let [target_first, target_second] = exclude_one(target_shared.unwrap());

            let decomposed_coords = decompose_xxyy_into_xxyy_xx(
                target[target_first],
                target[target_second],
                shifted_source_coord[source_first],
                shifted_source_coord[source_second],
                strength,
            );

            if decomposed_coords.iter().any(|val| val.is_nan()) {
                continue;
            }

            let [r, s, u, v, x, y] = decomposed_coords;
            // OK: this combination of things works.
            // save the permutation which rotates the shared coordinate into ZZ.
            permute_source_for_overlap = weyl::canonical_rotation_circuit(source_first, source_second);
            permute_target_for_overlap = weyl::canonical_rotation_circuit(target_first, target_second);
            break;
        } // for source_shift_name

        if permute_source_for_overlap.is_some() {
            break;
        }
    } // for reflection_name

    if permute_source_for_overlap.is_none() {
        // TODO: Decide which error to return.
        return Err(format!("Error during RZX decomposition: Could not find a suitable Weyl reflection to match {:?} to {:?} along {:?}.",
                    source, target, strength
        ));
    }

// the basic formula we're trying to work with is:
// target^p_t_f_o =
//     rs * (source^s_reflection * s_shift)^p_s_f_o * uv * operation * xy
// but we're rearranging it into the form
//   target = affix source prefix
// and computing just the prefix / affix circuits.

    
    return Ok(())
}

// fn canonical_xx_circuit(target, strength_sequence, basis_embodiments):
