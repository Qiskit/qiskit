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

use ndarray::{Array2, ArrayView2};
use ndarray_einsum_beta::*;
use num_complex::Complex64;

static UPPERCASE: [u8; 26] = [
    b'A', b'B', b'C', b'D', b'E', b'F', b'G', b'H', b'I', b'J', b'K', b'L', b'M', b'N', b'O', b'P',
    b'Q', b'R', b'S', b'T', b'U', b'V', b'W', b'X', b'Y', b'Z',
];
static LOWERCASE: [u8; 26] = [
    b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i', b'j', b'k', b'l', b'm', b'n', b'o', b'p',
    b'q', b'r', b's', b't', b'u', b'v', b'w', b'x', b'y', b'z',
];

fn einsum_matmul_helper(qubits: &[usize], num_qubits: usize) -> [String; 4] {
    let tens_in: Vec<u8> = LOWERCASE[..num_qubits].iter().copied().collect();
    let mut tens_out: Vec<u8> = tens_in.clone();
    let mut mat_l: Vec<u8> = Vec::with_capacity(num_qubits);
    let mut mat_r: Vec<u8> = Vec::with_capacity(num_qubits);
    qubits.iter().rev().enumerate().for_each(|(pos, idx)| {
        mat_r.push(tens_in[num_qubits - 1 - pos]);
        mat_l.push(LOWERCASE[25 - pos]);
        tens_out[num_qubits - 1 - idx] = LOWERCASE[25 - pos];
    });
    unsafe {
        [
            String::from_utf8_unchecked(mat_l),
            String::from_utf8_unchecked(mat_r),
            String::from_utf8_unchecked(tens_in),
            String::from_utf8_unchecked(tens_out),
        ]
    }
}

fn einsum_matmul_index(qubits: &[usize], num_qubits: usize) -> String {
    assert!(num_qubits > 26, "Can't compute unitary of > 26 qubits");
    let tens_r: String = unsafe {
        String::from_utf8_unchecked(UPPERCASE[..num_qubits].iter().copied().collect::<Vec<u8>>())
    };
    let [mat_l, mat_r, tens_lin, tens_lout] = einsum_matmul_helper(qubits, num_qubits);
    format!(
        "{}{}, {}{}->{}{}",
        mat_l, mat_r, tens_lin, tens_r, tens_lout, tens_r
    )
}

///
pub fn compose_unitary(
    gate_unitary: ArrayView2<Complex64>,
    overall_unitary: ArrayView2<Complex64>,
    qubits: &[usize],
) -> Array2<Complex64> {
    let num_qubits = qubits.len();
    let total_qubits = overall_unitary.shape()[0].ilog2() as usize;
    let indices = einsum_matmul_index(qubits, total_qubits);
    let gate_tensor = gate_unitary
        .into_shape(
            (0..num_qubits)
                .map(|_| [2, 2])
                .flatten()
                .collect::<Vec<usize>>(),
        )
        .unwrap();

    einsum(indices.as_str(), &[&gate_tensor, &overall_unitary])
        .unwrap()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap()
}
