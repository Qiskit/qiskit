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

use qiskit_circuit::operations::StandardGate;

// In what follows, Pauli labels should be read left-to-right, thus XY
// means X on the first qubit and Y on the second qubits.

// A 2-qubit Pauli can be indexed by a number in 0..16.
// Explicitly, we compute the index as (X[0] << 3) | (Z[0] << 2) | (X[1] << 1) | Z[1].
// For instance, XY (read left-to-right) corresponds to 8 + 0 + 2 + 1 = 11.
// In this way, 2-qubit Paulis are given by
// [II, IZ, IX, IY, ZI, ZZ, ZX, ZY, XI, XZ, XX, XY, YI, YZ, YX, YY].

// For efficiency, the following table stores 2-qubit Pauli support sizes.
// For example, for 2-qubit Pauli ZI with index 4, its support size
// is given by PAULI_SUPPORT_SIZES[4] = 1.
pub static PAULI_SUPPORT_SIZES: [usize; 16] = [0, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2];

// A "chunk" is a small 2-qubit circuit constisting of some
// single-qubit gates followed either by CX(0, 1) or CX(1, 0).
// There are 18 chunks of interest, numbered 0..18.
// These are taken for the Rustiq paper.
pub const ALL_CHUNKS: [&[(StandardGate, &[usize])]; 18] = [
    &[(StandardGate::CX, &[0, 1])],
    &[(StandardGate::CX, &[1, 0])],
    &[(StandardGate::H, &[1]), (StandardGate::CX, &[0, 1])],
    &[(StandardGate::H, &[0]), (StandardGate::CX, &[1, 0])],
    &[(StandardGate::S, &[1]), (StandardGate::CX, &[0, 1])],
    &[(StandardGate::S, &[0]), (StandardGate::CX, &[1, 0])],
    &[(StandardGate::H, &[0]), (StandardGate::CX, &[0, 1])],
    &[(StandardGate::H, &[1]), (StandardGate::CX, &[1, 0])],
    &[
        (StandardGate::H, &[0]),
        (StandardGate::H, &[1]),
        (StandardGate::CX, &[0, 1]),
    ],
    &[
        (StandardGate::H, &[1]),
        (StandardGate::H, &[0]),
        (StandardGate::CX, &[1, 0]),
    ],
    &[
        (StandardGate::H, &[0]),
        (StandardGate::S, &[1]),
        (StandardGate::CX, &[0, 1]),
    ],
    &[
        (StandardGate::H, &[1]),
        (StandardGate::S, &[0]),
        (StandardGate::CX, &[1, 0]),
    ],
    &[(StandardGate::SX, &[0]), (StandardGate::CX, &[0, 1])],
    &[(StandardGate::SX, &[1]), (StandardGate::CX, &[1, 0])],
    &[
        (StandardGate::SX, &[0]),
        (StandardGate::H, &[1]),
        (StandardGate::CX, &[0, 1]),
    ],
    &[
        (StandardGate::SX, &[1]),
        (StandardGate::H, &[0]),
        (StandardGate::CX, &[1, 0]),
    ],
    &[
        (StandardGate::SX, &[0]),
        (StandardGate::S, &[1]),
        (StandardGate::CX, &[0, 1]),
    ],
    &[
        (StandardGate::SX, &[1]),
        (StandardGate::S, &[0]),
        (StandardGate::CX, &[1, 0]),
    ],
];

// Given a 2-qubit Pauli, we want to precompute its conjugation
// by various chunks according to the Schrödinger picture:
// C P C^\dagger.
//
// The following table is a visual representation of this data.
//
//      II IZ IX IY ZI ZZ ZX ZY XI XZ XX XY YI YZ YX YY
//   0: II ZZ IX ZY ZI IZ ZX IY XX YY XI YZ YX XY YI XZ
//   1: II IZ XX XY ZZ ZI YY YX XI XZ IX IY YZ YI ZY ZX
//   2: II IX ZZ ZY ZI ZX IZ IY XX XI YY YZ YX YI XY XZ
//   3: II IZ XX XY XI XZ IX IY ZZ ZI YY YX YZ YI ZY ZX
//   4: II ZZ ZY IX ZI IZ IY ZX XX YY YZ XI YX XY XZ YI
//   5: II IZ XX XY ZZ ZI YY YX YZ YI ZY ZX XI XZ IX IY
//   6: II ZZ IX ZY XX YY XI YZ ZI IZ ZX IY YX XY YI XZ
//   7: II XX IZ XY ZZ YY ZI YX XI IX XZ IY YZ ZY YI ZX
//   8: II IX ZZ ZY XX XI YY YZ ZI ZX IZ IY YX YI XY XZ
//   9: II XX IZ XY XI IX XZ IY ZZ YY ZI YX YZ ZY YI ZX
//  10: II ZZ ZY IX XX YY YZ XI ZI IZ IY ZX YX XY XZ YI
//  11: II XX IZ XY ZZ YY ZI YX YZ ZY YI ZX XI IX XZ IY
//  12: II ZZ IX ZY YX XY YI XZ XX YY XI YZ ZI IZ ZX IY
//  13: II XY XX IZ ZZ YX YY ZI XI IY IX XZ YZ ZX ZY YI
//  14: II IX ZZ ZY YX YI XY XZ XX XI YY YZ ZI ZX IZ IY
//  15: II XY XX IZ XI IY IX XZ ZZ YX YY ZI YZ ZX ZY YI
//  16: II ZZ ZY IX YX XY XZ YI XX YY YZ XI ZI IZ IY ZX
//  17: II XY XX IZ ZZ YX YY ZI YZ ZX ZY YI XI IY IX XZ
//
// This is the corresponding table using 2-qubit Pauli indices.
// CHUNK_CONJUGATION_TABLE[chunk_idx][pauli_pair_idx] represents
// the index of the 2-qubit Pauli we obtain by conjugation.
pub static CHUNK_CONJUGATION_TABLE: [[usize; 16]; 18] = [
    [0, 5, 2, 7, 4, 1, 6, 3, 10, 15, 8, 13, 14, 11, 12, 9],
    [0, 1, 10, 11, 5, 4, 15, 14, 8, 9, 2, 3, 13, 12, 7, 6],
    [0, 2, 5, 7, 4, 6, 1, 3, 10, 8, 15, 13, 14, 12, 11, 9],
    [0, 1, 10, 11, 8, 9, 2, 3, 5, 4, 15, 14, 13, 12, 7, 6],
    [0, 5, 7, 2, 4, 1, 3, 6, 10, 15, 13, 8, 14, 11, 9, 12],
    [0, 1, 10, 11, 5, 4, 15, 14, 13, 12, 7, 6, 8, 9, 2, 3],
    [0, 5, 2, 7, 10, 15, 8, 13, 4, 1, 6, 3, 14, 11, 12, 9],
    [0, 10, 1, 11, 5, 15, 4, 14, 8, 2, 9, 3, 13, 7, 12, 6],
    [0, 2, 5, 7, 10, 8, 15, 13, 4, 6, 1, 3, 14, 12, 11, 9],
    [0, 10, 1, 11, 8, 2, 9, 3, 5, 15, 4, 14, 13, 7, 12, 6],
    [0, 5, 7, 2, 10, 15, 13, 8, 4, 1, 3, 6, 14, 11, 9, 12],
    [0, 10, 1, 11, 5, 15, 4, 14, 13, 7, 12, 6, 8, 2, 9, 3],
    [0, 5, 2, 7, 14, 11, 12, 9, 10, 15, 8, 13, 4, 1, 6, 3],
    [0, 11, 10, 1, 5, 14, 15, 4, 8, 3, 2, 9, 13, 6, 7, 12],
    [0, 2, 5, 7, 14, 12, 11, 9, 10, 8, 15, 13, 4, 6, 1, 3],
    [0, 11, 10, 1, 8, 3, 2, 9, 5, 14, 15, 4, 13, 6, 7, 12],
    [0, 5, 7, 2, 14, 11, 9, 12, 10, 15, 13, 8, 4, 1, 3, 6],
    [0, 11, 10, 1, 5, 14, 15, 4, 13, 6, 7, 12, 8, 3, 2, 9],
];

// Precomputed change in support size for every (chunk, 2-qubit Pauli) pair
// (a negative value means the conjugation reduces the support).
pub static SUPPORT_DELTA: [[i8; 16]; 18] = build_support_delta();

const fn build_support_delta() -> [[i8; 16]; 18] {
    let mut table = [[0i8; 16]; 18];
    let mut chunk_idx = 0;
    while chunk_idx < 18 {
        let mut pair_idx = 0;
        while pair_idx < 16 {
            let conjugated_pair_idx = CHUNK_CONJUGATION_TABLE[chunk_idx][pair_idx];
            table[chunk_idx][pair_idx] = PAULI_SUPPORT_SIZES[conjugated_pair_idx] as i8
                - PAULI_SUPPORT_SIZES[pair_idx] as i8;
            pair_idx += 1;
        }
        chunk_idx += 1;
    }
    table
}

// For efficiency, we also precompute which conjugations reduce
// the size of the support set of a given 2-qubit Pauli.
pub static REDUCING_CHUNKS: [&[usize]; 16] = [
    &[],
    &[],
    &[],
    &[],
    &[],
    &[0, 1, 4, 5, 8, 9, 14, 15],
    &[2, 3, 4, 6, 7, 11, 12, 15],
    &[0, 2, 3, 9, 10, 13, 16, 17],
    &[],
    &[2, 3, 5, 6, 7, 10, 13, 14],
    &[0, 1, 8, 9, 10, 11, 12, 13],
    &[1, 4, 6, 7, 8, 15, 16, 17],
    &[],
    &[1, 2, 3, 8, 11, 12, 16, 17],
    &[0, 5, 6, 7, 9, 14, 16, 17],
    &[4, 5, 10, 11, 12, 13, 14, 15],
];
