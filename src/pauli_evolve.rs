// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use num_complex::Complex64;
use std::cmp;

use numpy::PyArray1;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyOverflowError;
use pyo3::prelude::*;
use rayon::prelude::*;

// constant values
const BITS: [usize; 64] = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
    2097152,
    4194304,
    8388608,
    16777216,
    33554432,
    67108864,
    134217728,
    268435456,
    536870912,
    1073741824,
    2147483648,
    4294967296,
    8589934592,
    17179869184,
    34359738368,
    68719476736,
    137438953472,
    274877906944,
    549755813888,
    1099511627776,
    2199023255552,
    4398046511104,
    8796093022208,
    17592186044416,
    35184372088832,
    70368744177664,
    140737488355328,
    281474976710656,
    562949953421312,
    1125899906842624,
    2251799813685248,
    4503599627370496,
    9007199254740992,
    18014398509481984,
    36028797018963968,
    72057594037927936,
    144115188075855872,
    288230376151711744,
    576460752303423488,
    1152921504606846976,
    2305843009213693952,
    4611686018427387904,
    9223372036854775808,
];
const MASKS: [usize; 64] = [
    0,
    1,
    3,
    7,
    15,
    31,
    63,
    127,
    255,
    511,
    1023,
    2047,
    4095,
    8191,
    16383,
    32767,
    65535,
    131071,
    262143,
    524287,
    1048575,
    2097151,
    4194303,
    8388607,
    16777215,
    33554431,
    67108863,
    134217727,
    268435455,
    536870911,
    1073741823,
    2147483647,
    4294967295,
    8589934591,
    17179869183,
    34359738367,
    68719476735,
    137438953471,
    274877906943,
    549755813887,
    1099511627775,
    2199023255551,
    4398046511103,
    8796093022207,
    17592186044415,
    35184372088831,
    70368744177663,
    140737488355327,
    281474976710655,
    562949953421311,
    1125899906842623,
    2251799813685247,
    4503599627370495,
    9007199254740991,
    18014398509481983,
    36028797018963967,
    72057594037927935,
    144115188075855871,
    288230376151711743,
    576460752303423487,
    1152921504606846975,
    2305843009213693951,
    4611686018427387903,
    9223372036854775807,
];

#[inline]
pub fn pauli_masks_and_phase(
    qubits: &[usize],
    pauli_x: &[bool],
    pauli_z: &[bool],
) -> (usize, usize, usize) {
    let mut x_mask: usize = 0;
    let mut z_mask: usize = 0;
    let mut x_max: usize = 0;
    for (idx, idx_q) in qubits.iter().enumerate() {
        let bit = BITS[*idx_q];
        match (pauli_x[idx], pauli_z[idx]) {
            // case for I
            (false, false) => (),
            // case for X
            (true, false) => {
                x_mask += bit;
                x_max = cmp::max(x_max, *idx_q);
            }
            // case for Z
            (false, true) => {
                z_mask += bit;
            }
            // case for Y
            (true, true) => {
                x_mask += bit;
                x_max = cmp::max(x_max, *idx_q);
                z_mask += bit;
                // num_y += 1;
            }
        }
    }
    (x_mask, z_mask, x_max)
}

#[inline]
pub fn add_y_phase(y_phase: &usize) -> Complex64 {
    // Add overall phase to the input coefficient

    // Compute the overall phase of the operator.
    // This is (-1j) ** number of Y terms modulo 4
    let mut coeff_ = Complex64::new(1.0, 0.0);

    match y_phase & 3 {
        0 => (),
        1 => {
            coeff_ = Complex64::new(0., -1.);
        }
        2 => {
            coeff_ = Complex64::new(-1., 0.);
        }
        3 => {
            coeff_ = Complex64::new(0., 1.);
        }
        _ => (),
    }
    coeff_
}

#[pyfunction]
#[pyo3(text_signature = "(data, qubits, pauli, /)")]
pub fn apply_pauli(
    _py: Python,
    data: &PyArray1<Complex64>,
    qubits: Vec<usize>,
    pauli_x: PyReadonlyArray1<bool>,
    pauli_z: PyReadonlyArray1<bool>,
    y_phase: usize,
) -> PyResult<()> {
    let num_qubits = qubits.len();
    if num_qubits > usize::BITS as usize {
        return Err(PyOverflowError::new_err(format!(
            "The value for num_qubits, {}, is too large and would overflow",
            num_qubits
        )));
    }
    let mut data_arr = unsafe { data.as_array_mut() };
    let n = data_arr.len();
    let res = pauli_masks_and_phase(&qubits, pauli_x.as_slice()?, pauli_z.as_slice()?);
    let x_mask: usize = res.0;
    let z_mask: usize = res.1;
    let x_max: usize = res.2;
    let phase = add_y_phase(&y_phase);

    // Special case for only I Paulis
    if x_mask + z_mask == 0 {
        if y_phase & 3 == 0 {
            return Ok(());
        }
        let mut lambda = |i: usize| {
            data_arr[i] *= phase;
        };
        // It would be preferable if it could be parallelized as follows
        // (0..n).into_par_iter().map(lambda);
        // (0..n).par_iter_mut().map(lambda);
        for i in 0..n {
            lambda(i);
        }
        return Ok(());
    }

    // specialize x_max == 0
    if x_mask == 0 {
        let mut lambda = |i: usize| {
            if (z_mask != 0) && (((i & z_mask).count_ones() & 1) != 0) {
                data_arr[i] *= -1.;
            }
            data_arr[i] *= phase;
        };

        // It would be preferable if it could be parallelized as follows
        // (0..n).into_par_iter().map(lambda);
        for i in 0..n {
            lambda(i);
        }
        return Ok(());
    }

    let mask_u = !MASKS[x_max + 1];
    let mask_l = MASKS[x_max];

    let mut lambda = |i: usize| {
        let mut idxs: [usize; 2] = [0; 2];
        idxs[0] = ((i << 1) & mask_u) | (i & mask_l);
        idxs[1] = idxs[0] ^ x_mask;
        data_arr.swap(idxs[0], idxs[1]);
        for j in 0..2 {
            if (z_mask != 0) && (((idxs[j] & z_mask).count_ones() & 1) != 0) {
                data_arr[idxs[j]] *= -1.;
            }
            data_arr[idxs[j]] *= phase;
        }
    };
    // It would be preferable if it could be parallelized as follows
    // (0..n).into_par_iter().map(lambda);

    for i in 0..(n >> 1) {
        lambda(i);
    }
    Ok(())
}

#[pymodule]
pub fn pauli_evolve(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_pauli, m)?)?;

    Ok(())
}
