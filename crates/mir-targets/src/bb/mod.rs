// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Bivariate-bicycle code targets.

mod bit;
mod gf2;
pub mod measurement_table;
pub mod pauli;

pub use measurement_table::MeasurementTable;

use pauli::Pauli;

// The bit patterns "read" right-to-left, bottom-to-top compared to the TdG paper because we
// store col 0 in the Lsb of each byte, and row 0 in the least-significant byte.  The two
// most-significant bits in each byte are 0 to pad the matrix.
pub const AX_G: gf2::Mat<6> =
    gf2::Mat::from_u64(0b00101111_00010010_00011011_00001100_00100010_00001010);
pub const AY_G: gf2::Mat<6> =
    gf2::Mat::from_u64(0b00101100_00100110_00000010_00010000_00100111_00100001);
pub const AX_2G: gf2::Mat<6> =
    gf2::Mat::from_u64(0b00011001_00111110_00111101_00010101_00110101_00101110);
pub const AY_2G: gf2::Mat<6> =
    gf2::Mat::from_u64(0b00100001_00111001_00010001_00000110_00111011_00011111);

/// Get all relevant powers of a single-shift automorphism matrix.
const fn powers(m: gf2::Mat<6>) -> [gf2::Mat<6>; 6] {
    let mut out = [gf2::Mat::ID; 6];
    let mut i = 0;
    while i < 5 {
        out[i + 1] = m.matmul(out[i]);
        i += 1;
    }
    out
}

/// Get the full binary matrix and its inverse for each shift automorphism.
///
/// Includes the identity matrix as the first element.
const fn shift_automorphisms(ax: gf2::Mat<6>, ay: gf2::Mat<6>) -> [[gf2::Mat<6>; 2]; 36] {
    let ax_power = powers(ax);
    let ay_power = powers(ay);
    let mut out = [[gf2::Mat::ID; 2]; 36];
    let mut x = 0;
    while x < 6 {
        let x_inv = if x == 0 { 0 } else { 6 - x };
        let mut y = 0;
        while y < 6 {
            let y_inv = if y == 0 { 0 } else { 6 - y };
            out[x * 6 + y] = [
                ax_power[x].matmul(ay_power[y]),
                ax_power[x_inv].matmul(ay_power[y_inv]),
            ];
            y += 1;
        }
        x += 1;
    }
    out
}

/// The shift automorphisms of the gross code, as binary matrices and their inverses.
pub static SHIFT_AUTOMORPHISMS_G: [[gf2::Mat<6>; 2]; 36] = shift_automorphisms(AX_G, AY_G);
/// The shift automorphisms of the two-gross code, as binary matrices and their inverses.
pub static SHIFT_AUTOMORPHISMS_2G: [[gf2::Mat<6>; 2]; 36] = shift_automorphisms(AX_2G, AY_2G);

/// The built-in hardware measurements as pairs of (low index), (high index) Paulis.
pub static BUILTIN_MEASURES: [[Pauli<6>; 2]; 15] = [
    [Pauli::I, Pauli::x(0)],
    [Pauli::I, Pauli::y(0)],
    [Pauli::I, Pauli::z(0)],
    [Pauli::x(0), Pauli::I],
    [Pauli::y(0), Pauli::I],
    [Pauli::z(0), Pauli::I],
    [Pauli::x(0), Pauli::x(0)],
    [Pauli::x(0), Pauli::y(0)],
    [Pauli::x(0), Pauli::z(0)],
    [Pauli::y(0), Pauli::x(0)],
    [Pauli::y(0), Pauli::y(0)],
    [Pauli::y(0), Pauli::z(0)],
    [Pauli::z(0), Pauli::x(0)],
    [Pauli::z(0), Pauli::y(0)],
    [Pauli::z(0), Pauli::z(0)],
];

/// Iterate over all native measurements of a bicycle code.
pub fn native_measures(automorphisms: &[[gf2::Mat<6>; 2]]) -> impl Iterator<Item = [Pauli<6>; 2]> {
    automorphisms.iter().flat_map(|&[a, a_inv]| {
        let [a_t, a_inv_t] = [a.transpose(), a_inv.transpose()];
        BUILTIN_MEASURES
            .into_iter()
            .map(move |[low, high]| [low.evolve_css(a, a_inv), high.evolve_css(a_t, a_inv_t)])
    })
}

/// Iterate over all native rotations of a bicycle code.
///
/// Each element of the output is guaranteed to have Pauli support on qubit index 0; this
/// indicates the pivot measurement.  There will be no full-Pauli-string duplicates in the
/// output, but there may be multiple instances of the same native rotation on the data qubits,
/// if multiple pivot single Paulis are valid for that rotation.
pub fn native_rotations(automorphisms: &[[gf2::Mat<6>; 2]]) -> impl Iterator<Item = [Pauli<6>; 2]> {
    native_measures(automorphisms).filter_map(|[low, high]| {
        let data = low.mask_support(0xFE);
        (low != data && [data, high] != [Pauli::I; 2]).then_some([low, high])
    })
}
