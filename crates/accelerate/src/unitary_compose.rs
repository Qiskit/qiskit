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

use std::os::unix::raw::pthread_t;
use faer::sparse::linalg::amd::order;
use ndarray::{Array2, ArrayBase, ArrayView2, CowRepr, Dim, Dimension, Ix};
use ndarray_einsum_beta::*;
use num_complex::Complex64;

static FIRSTHALF: [u8; 13] = [
    b'n', b'o', b'p', b'q', b'r', b's', b't', b'u', b'v', b'w', b'x', b'y', b'z',
];
static SECONDHALF: [u8; 13] = [
    b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i', b'j', b'k', b'l', b'm'
];

fn einsum_matmul_helper(qubits: &[usize], num_qubits: usize) -> [String; 4] {
    let tens_in: Vec<u8> = SECONDHALF[..num_qubits].iter().copied().collect();
    let mut tens_out: Vec<u8> = tens_in.clone();
    let mut mat_l: Vec<u8> = Vec::with_capacity(num_qubits);
    let mut mat_r: Vec<u8> = Vec::with_capacity(num_qubits);
    qubits.iter().rev().enumerate().for_each(|(pos, idx)| {
        mat_r.push(tens_in[num_qubits - 1 - pos]);
        mat_l.push(SECONDHALF[12 - pos]);
        tens_out[num_qubits - 1 - idx] = SECONDHALF[12 - pos];
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
    assert!(num_qubits < 13, "Can't compute unitary of > 13 qubits");
    let tens_r: String = unsafe {
        String::from_utf8_unchecked(FIRSTHALF[..num_qubits].iter().copied().collect::<Vec<u8>>())
    };
    let [mat_l, mat_r, tens_lin, tens_lout] = einsum_matmul_helper(qubits, num_qubits);
    format!(
        "{}{},{}{}->{}{}",
        mat_l, mat_r, tens_lin, tens_r, tens_lout, tens_r
    )
}


pub fn compose_unitary<'a>(
    gate_unitary: ArrayView2<'a, Complex64>,
    overall_unitary: ArrayView2<'a, Complex64>,
    qubits: &'a [usize],
) -> ArrayView2<'a, Complex64> {
    let num_qubits = qubits.len();
    let total_qubits = overall_unitary.shape()[0].ilog2() as usize;
    let gate_qubits = gate_unitary.shape()[0].ilog2() as usize;
    println!("qubits: {:?}", qubits);
    println!("total qubits: {:?}", total_qubits);
    println!("gate qubits: {:?}", gate_qubits);
    let indices = einsum_matmul_index(qubits, total_qubits);
    let gate_shape =             (0..gate_qubits)
        .map(|_| [2, 2])
        .flatten()
        .collect::<Vec<usize>>();
    println!("gate_unitary {:?}", gate_unitary);
    println!("gateshaope {:?}", gate_shape);
    let gate_tensor = gate_unitary
        .into_shape(
            gate_shape
        )
        .unwrap();
    println!("huh");
    let overall_shape = (0..total_qubits)
        .map(|_| [2, 2])
        .flatten()
        .collect::<Vec<usize>>();
    println!("overall_unitary {:?}", overall_unitary);
    println!("overall shaope {:?}", overall_shape);
    let overall_tensor = overall_unitary
        .into_shape(
            overall_shape
        )
        .unwrap();
    let num_rows = usize::pow(2, total_qubits as u32);
    println!("numrows: {}", num_rows);
    //TODO einsum currently does not support uppercase characters, hence we are limited to 13 qubits
    let eisum = einsum(indices.as_str(), &[&gate_tensor, &overall_tensor])
        .unwrap();
    println!("eisum {:?}", eisum);
    println!("dyn eis {:?}", eisum.ndim());
    println!("dyn eis {:?}", eisum.dim());
    println!("conti?{:?}", eisum.is_standard_layout());
    println!("conti {:?}", eisum.as_standard_layout());
    /*
    let eisum3 = match  eisum.dim()
    {
        IxDynImlpl =>  eisum.into_dimensionality::<ndarray::Ix2>().unwrap(),
        _ => eisum.into_shape((num_rows, num_rows)).unwrap()
    };

     */

    //let eisum3 = eisum.as_standard_layout().into_shape((num_rows, num_rows)).unwrap();
    let eisum3 = eisum.as_standard_layout().into_dimensionality::<ndarray::Ix2>().unwrap();
    println!("arff: {:?}", eisum3);
    gate_unitary
}
