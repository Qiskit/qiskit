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
use ndarray::{Array, Array2, ArrayBase, ArrayView2, CowRepr, Dim, Dimension, Ix, Ix2, IxDyn};
use ndarray_einsum_beta::*;
use num_complex::{Complex, Complex64};

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


pub fn compose(gate_unitary: Array<Complex<f64>, Ix2>, overall_unitary: Array<Complex<f64>, Ix2>, qubits: &[usize], front: bool) -> Array2<Complex<f64>> {
    let gate_qubits = gate_unitary.shape()[0].ilog2() as usize;
    let overall_qubits = overall_unitary.shape()[0].ilog2() as usize;


    // Full composition of operators
    if qubits.len() == 0 {
        if front {
            return gate_unitary.dot(&overall_unitary);
        }
        else {
            return overall_unitary.dot(&gate_unitary);
        }
    }
    // Compose with other on subsystem
    let num_indices = gate_qubits;
    let shift = if front {gate_qubits} else {0usize};
    let right_mul = front;


    //Reshape current matrix
    //Note that we must reverse the subsystem dimension order as
    //qubit 0 corresponds to the right-most position in the tensor
    //product, which is the last tensor wire index.

    let tensor = per_qubit_shaped(gate_unitary.clone());
    let mat = per_qubit_shaped(overall_unitary.clone());
    let indices = qubits.iter().map(|q| num_indices-1-q).collect::<Vec<usize>>();
    let num_rows = usize::pow(2, num_indices as u32);

    _einsum_matmul(tensor, mat, indices, shift, right_mul).as_standard_layout().
        into_shape((num_rows, num_rows)).unwrap().
        into_dimensionality::<ndarray::Ix2>().unwrap().to_owned()
}
static LOWERCASE: [u8; 26] = [
    b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h', b'i', b'j', b'k', b'l', b'm', b'n', b'o', b'p',
    b'q', b'r', b's', b't', b'u', b'v', b'w', b'x', b'y', b'z',
];

fn _einsum_matmul(tensor: Array<Complex64,IxDyn>, mat: Array<Complex64, IxDyn>, indices: Vec<usize>,
                  shift: usize, right_mul: bool) -> Array<Complex64, IxDyn> {
    let rank = tensor.ndim();
    let rank_mat = mat.ndim();
    if rank_mat % 2 != 0 {
        panic!("Contracted matrix must have an even number of indices.");
    }
    //println!("rank:{:?}, rank_mat:{:?}", rank, rank_mat);
    // Get einsum indices for tensor
    let mut indices_tensor = (0..rank).collect::<Vec<usize>>();
    //println!("indices_tensor1:{:?}", indices_tensor);
    for (j, index) in indices.iter().enumerate() {
        //for j in 0..indices.len(){
        indices_tensor[index + shift] = rank + j;
    }
    //einstr [5, 4, 2, 3] [0, 1, 5, 4]
    //einstr: [5, 4, 2, 3], [0, 1, 5, 4] // false
    //einstr: [0, 1, 5, 4], [5, 4, 2, 3] // true
    //einsumstr zyab, abAB->zyAB
    //println!("indices_tensor2:{:?}", indices_tensor);

    // Get einsum indices for mat
    let mat_contract = (rank..rank+indices.len()).rev().collect::<Vec<usize>>();
    //println!("mat_contract:{:?}", mat_contract);
    let mat_free = indices.iter().rev().map(|index| index+shift).collect::<Vec<usize>>();
    //println!("mat_free:{:?}", mat_free);
    let indices_mat = if right_mul {[mat_contract, mat_free].concat() } else {[mat_free, mat_contract].concat()};
    //println!("indices_mat:{:?}", indices_mat);
    //println!("einstr: {:?}, {:?}", indices_tensor, indices_mat);
    //einstr: [7, 8, 6, 3, 4, 5] [1, 0, 2, 8, 7, 6] false
    //einsumstr zyxbca, abcABC->xzyABC false
    //einstr: [0, 1, 2, 7, 8, 6], [8, 7, 6, 4, 3, 5] true
    let tensor_str :String = unsafe {String::from_utf8_unchecked(indices_tensor.iter().map(|c| LOWERCASE[*c]).collect())};
    let mat_str :String = unsafe {String::from_utf8_unchecked(indices_mat.iter().map(|c| LOWERCASE[*c]).collect())};
    //println!("mastr: {:?},{:?}", tensor_str, mat_str);
    let emat = einsum(format!("{},{}", tensor_str, mat_str).as_str(), &[&tensor, &mat]).unwrap();
    //println!("emat: {:?}", emat);
    emat
    //return np.einsum(tensor, indices_tensor, mat, indices_mat)
    /*
        mat_l, mat_r, tens_lin, tens_lout = _einsum_matmul_index_helper(gate_indices, number_of_qubits)

        # Right indices for the N-qubit input and output tensor
        tens_r = ascii_uppercase[:number_of_qubits]

        # Combine indices into matrix multiplication string format
        # for numpy.einsum function
        return f"{mat_l}{mat_r}, {tens_lin}{tens_r}->{tens_lout}{tens_r}"
    */

    /* z.0 cx.0,1

    True
    einsumstr zyab, abAB->zyAB


    */
}
/*
    ##        return np.einsum(tensor, indices_tensor, mat, indices_mat)
    ##                        idZ,  5423, cx, 0154 ;;;;  zyab, abAB->zyAB
*/


fn per_qubit_shaped(array: Array<Complex<f64>, Ix2>) -> Array<Complex64, IxDyn> {
    let overall_shape = (0..array.shape()[0].ilog2() as usize)
        .map(|_| [2, 2])
        .flatten()
        .collect::<Vec<usize>>();
    array
        .into_shape(
            overall_shape
        )
        .unwrap().into_owned()
}
