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

use ndarray::{azip, s, ArrayView1, ArrayViewMut2, Zip};
use pyo3::prelude::*;
use std::ffi::c_long;
use std::collections::HashSet;
use itertools::min;

use numpy::prelude::*;

use numpy::{PyArray2, PyReadonlyArray2};

use crate::QiskitError;
use numpy::ndarray::{aview2, Array2, ArrayView2};
use pyo3::callback::IntoPyCallbackOutput;
use smallvec::SmallVec;

// /// Symplectic tableau
// #[pyclass(module = "qiskit._accelerate.clifford")]
// pub struct Tableau {
//     pub tableau: Array2<bool>,
// }
//
// #[pymethods]
// impl Tableau {
//     fn __new__(tableau: &Bound<PyReadonlyArray2<bool>>) -> Self {
//         Self {
//             tableau: tableau.asarray(),
//         }
//     }
//
//     ///
//     fn get_tableau(&self, py: Python) -> PyObject {
//         self.tableau.to_object(py)
//     }
//
// }

// the idea is to create a function to decompose cliffords

// Global arrays of the 16 pairs of Pauli operators
// divided into 5 equivalence classes under the action of single-qubit Cliffords.

// ToDo: replace this crap by a HashMap !!!!

// Class A - canonical representative is 'XZ'
static A_CLASS: [[[bool; 2]; 2]; 6] = [
    [[false, true], [true, true]],  // 'XY'
    [[false, true], [true, false]], // 'XZ'
    [[true, true], [false, true]],  // 'YX'
    [[true, true], [true, false]],  // 'YZ'
    [[true, false], [false, true]], // 'ZX'
    [[true, false], [true, true]],  // 'ZY'
];

// Class B - canonical representative is 'XX'
static B_CLASS: [[[bool; 2]; 2]; 3] = [
    [[true, false], [true, false]], // 'ZZ'
    [[false, true], [false, true]], // 'XX'
    [[true, true], [true, true]],   // 'YY'
];

// Class C - canonical representative is 'XI'
static C_CLASS: [[[bool; 2]; 2]; 3] = [
    [[true, false], [false, false]], // 'ZI'
    [[false, true], [false, false]], // 'XI'
    [[true, true], [false, false]],  // 'YI'
];

// Class D - canonical representative is 'IZ'
static D_CLASS: [[[bool; 2]; 2]; 3] = [
    [[false, false], [false, true]], // 'IX'
    [[false, false], [true, false]], // 'IZ'
    [[false, false], [true, true]],  // 'IY'
];

// Class E - only 'II'
static E_CLASS: [[[bool; 2]; 2]; 1] = [
    [[false, false], [false, false]], // 'II'
];

fn compute_greedy_cost(pairs: &Vec<[[bool; 2]; 2]>) -> PyResult<usize> {
    let mut a_num = 0;
    let mut b_num = 0;
    let mut c_num = 0;
    let mut d_num = 0;

    for pair in pairs {
        if A_CLASS.contains(&pair) {
            a_num += 1;
        } else if B_CLASS.contains(&pair) {
            b_num += 1;
        } else if C_CLASS.contains(&pair) {
            c_num += 1;
        } else if D_CLASS.contains(&pair) {
            d_num += 1;
        }
    }

    if a_num % 2 == 0 {
        return Err(QiskitError::new_err(
            "Symplectic Gaussian elimination fails",
        ));
    }

    let mut cnot_cost: usize =
        3 * (a_num - 1) / 2 + (b_num + 1) * ((b_num > 0) as usize) + c_num + d_num;

    if !A_CLASS.contains(&pairs[0]) {
        cnot_cost += 3;
    }
    Ok(cnot_cost)
}

// hack: for single-qubit gates second arg is always 0
type CliffordSequenceVec = Vec<(String, u8, u8)>;

fn from_pair_paulis_to_type(
    pauli_x: ArrayView1<bool>,
    pauli_z: ArrayView1<bool>,
    qubit: usize,
) -> [[bool; 2]; 2] {
    let num_qubits: usize = pauli_x.len() / 2;
    [
        [pauli_x[qubit], pauli_x[num_qubits + qubit]],
        [pauli_z[qubit], pauli_z[num_qubits + qubit]],
    ]
}

fn adjoint_no_phase(symp_mat: &mut Array2<bool>, num_qubits: usize)  {
    println!("Original:");
    println!("{:?}", symp_mat);

    let transposed = symp_mat.t().to_owned();
    *symp_mat = transposed;

    let (mut a, mut b, mut c, mut d) = symp_mat.multi_slice_mut((
        s![0..num_qubits, 0..num_qubits],
        s![0..num_qubits, num_qubits..2 * num_qubits],
        s![num_qubits..2 * num_qubits, 0..num_qubits],
        s![num_qubits..2 * num_qubits, num_qubits..2 * num_qubits],
    ));

    azip!((mut a in &mut a, mut d in &mut d)  (*a, *d) = (*d, *a));
    azip!((mut b in &mut b, mut c in &mut c)  (*b, *c) = (*c, *b));
}



// for now modify clifford in-place
fn synth_clifford_greedy_inner(clifford: &Array2<bool>) -> PyResult<()> {
   println!("I AM IN SYNTH_CLIFFORD_INNER!");

    let mut clifford_gates = CliffordSequenceVec::new();

    let num_qubits = clifford.shape()[0] / 2;
    println!("num_qubits = {:?}", num_qubits);

    let clifford_slice = clifford.slice(s![.., 0 .. 2 * num_qubits]);
    println!("clifford slice = {:?}", clifford_slice);

    let mut symplectic_mat: Array2<bool> = clifford_slice.to_owned();
    println!("symplectic_mat = {:?}", symplectic_mat);


    // ToDo: this is a vector for now to be compatible with the python
    // implementation, but we should really turn it into a set
    let mut qubit_list: Vec<usize> = (0..num_qubits).collect();

    while qubit_list.len() > 0 {
        let mut list_greedy_cost = Vec::<(usize, usize)>::new();

        for qubit in &qubit_list {
            let pauli_x = symplectic_mat.column(*qubit + num_qubits);
            let pauli_z = symplectic_mat.column(*qubit);

            println!("HERE: qubit = {}, pauli_x = {:?}, pauli_z = {:?}", qubit, pauli_x, pauli_z);

            let list_pairs: Vec<[[bool; 2]; 2]> = qubit_list
                .iter()
                .map(|i| from_pair_paulis_to_type(pauli_x, pauli_z, *i))
                .collect();

            println!("{:?}", list_pairs);

            let cost = compute_greedy_cost(&list_pairs)?;
            println!("{}", cost);
            list_greedy_cost.push((cost, *qubit));
        }
        println!("list_greedy_cost = {:?}", list_greedy_cost);
        let min_qubit = list_greedy_cost
            .iter()
            .min_by_key(|(cost, qubit)| cost)
            .unwrap()
            .1;
        println!("min_qubit = {:?}", min_qubit);
        // clifford[0 : self.num_qubits, :]

        let pauli_x = symplectic_mat.column(min_qubit + num_qubits);
        let pauli_z = symplectic_mat.column(min_qubit);

        let mut decouple_cliff = calc_decoupling(&mut clifford_gates, pauli_x, pauli_z, &qubit_list, min_qubit, num_qubits);

        println!("Decoupling cliff");
        println!("{:?}", decouple_cliff);
        adjoint_no_phase(&mut decouple_cliff, num_qubits);
        println!("After adjoint:");
        println!("{:?}", decouple_cliff);
        let composed_cliff = compose_ignore_phase(num_qubits, symplectic_mat.view(), decouple_cliff.view());
        println!("Composed:");
        println!("{:?}", composed_cliff);

        symplectic_mat = composed_cliff;

        println!("====================================================");
        println!("CLIFFORD CURRENT:");
        println!("{:?}", symplectic_mat);

        // qubit_list.remove(&min_qubit);
        qubit_list.retain(|&x| x != min_qubit);

    }

    // the symplectic matrix should be ok, but the phase is not
    // todo: fix final phase
    println!("====================================================");
    println!("SMAT END:");
    println!("{:?}", symplectic_mat);
    println!("{:?}", clifford_gates);
    println!("====================================================");



    Ok(())
}

// Zip::from(clifford.column_mut(2 * num_qubits))
//     .and(&x)
//     .and(&z)
//     .for_each(|phase, &x, &z| *phase ^= x & z);
// Zip::from(clifford.column_mut(num_qubits + qubit))
//     .and(&x)
//     .for_each(|z, x| *z ^= x );

fn append_s(mut clifford: &mut Array2<bool>, qubit: usize, num_qubits: usize) {
    let (x, mut z, mut p) = clifford.multi_slice_mut((
        s![.., qubit],
        s![.., num_qubits + qubit],
        s![.., 2 * num_qubits],
    ));

    azip!((mut p in &mut p, &x in &x, &z in &z)  *p ^= x & z);
    azip!((mut z in &mut z, &x in &x) *z ^= x);
}

fn append_h(mut clifford: &mut Array2<bool>, qubit: usize, num_qubits: usize) {
    let (mut x, mut z, mut p) = clifford.multi_slice_mut((
        s![.., qubit],
        s![.., num_qubits + qubit],
        s![.., 2 * num_qubits],
    ));

    azip!((mut p in &mut p, &x in &x, &z in &z)  *p ^= x & z);
    azip!((mut x in &mut x, mut z in &mut z)  (*x, *z) = (*z, *x));
}

/*
    clifford.x[:, [qubit0, qubit1]] = clifford.x[:, [qubit1, qubit0]]
    clifford.z[:, [qubit0, qubit1]] = clifford.z[:, [qubit1, qubit0]]
    return clifford

 */
fn append_swap(mut clifford: &mut Array2<bool>, qubit0: usize, qubit1:usize, num_qubits:usize) {
    let (mut x0, mut z0, mut x1, mut z1) = clifford.multi_slice_mut((
        s![.., qubit0],
        s![.., num_qubits + qubit0],
        s![.., qubit1],
        s![.., num_qubits + qubit1],
    ));
    azip!((mut x0 in &mut x0, mut x1 in &mut x1)  (*x0, *x1) = (*x1, *x0));
    azip!((mut z0 in &mut z0, mut z1 in &mut z1)  (*z0, *z1) = (*z1, *z0));
}

/*
    clifford.phase ^= (x1 ^ z0 ^ True) & z1 & x0
    x1 ^= x0
    z0 ^= z1

 */
// x0, z0 - control, x1, z1 - target
fn append_cx(mut clifford: &mut Array2<bool>, qubit0: usize, qubit1:usize, num_qubits:usize) {
    let (mut x0, mut z0, mut x1, mut z1, mut p) = clifford.multi_slice_mut((
        s![.., qubit0],
        s![.., num_qubits + qubit0],
        s![.., qubit1],
        s![.., num_qubits + qubit1],
        s![.., 2 * num_qubits],
    ));
    azip!((mut p in &mut p, &x0 in &x0, &z0 in &z0, &x1 in &x1, &z1 in &z1) *p ^= (x1 ^ z0 ^ true) & z1 & x0);
    azip!((mut x1 in &mut x1, &x0 in &x0) *x1 ^= x0);
    azip!((mut z0 in &mut z0, &z1 in &z1) *z0 ^= z1);
}









fn append_s_no_phase(mut clifford: &mut Array2<bool>, qubit: usize, num_qubits: usize) {
    let (x, mut z) = clifford.multi_slice_mut((
        s![.., qubit],
        s![.., num_qubits + qubit],
    ));

    azip!((mut z in &mut z, &x in &x) *z ^= x);
}

fn append_h_no_phase(mut clifford: &mut Array2<bool>, qubit: usize, num_qubits: usize) {
    let (mut x, mut z, ) = clifford.multi_slice_mut((
        s![.., qubit],
        s![.., num_qubits + qubit],
    ));

    azip!((mut x in &mut x, mut z in &mut z)  (*x, *z) = (*z, *x));
    println!("OK");
}

fn append_swap_no_phase(mut clifford: &mut Array2<bool>, qubit0: usize, qubit1:usize, num_qubits:usize) {
    let (mut x0, mut z0, mut x1, mut z1) = clifford.multi_slice_mut((
        s![.., qubit0],
        s![.., num_qubits + qubit0],
        s![.., qubit1],
        s![.., num_qubits + qubit1],
    ));
    azip!((mut x0 in &mut x0, mut x1 in &mut x1)  (*x0, *x1) = (*x1, *x0));
    azip!((mut z0 in &mut z0, mut z1 in &mut z1)  (*z0, *z1) = (*z1, *z0));
}

fn append_cx_no_phase(mut clifford: &mut Array2<bool>, qubit0: usize, qubit1:usize, num_qubits:usize) {
    let (mut x0, mut z0, mut x1, mut z1, ) = clifford.multi_slice_mut((
        s![.., qubit0],
        s![.., num_qubits + qubit0],
        s![.., qubit1],
        s![.., num_qubits + qubit1],
    ));
    azip!((mut x1 in &mut x1, &x0 in &x0) *x1 ^= x0);
    azip!((mut z0 in &mut z0, &z1 in &z1) *z0 ^= z1);
}








//
// fn append_s(mut clifford: &mut Array2<bool>, qubit: usize, num_qubits: usize) {
//     let x = clifford.column(qubit).to_owned();
//     let z = clifford.column(num_qubits + qubit).to_owned();
//
//     azip!((phase in clifford.column_mut(2 * num_qubits), &x in &x, &z in &z)  *phase ^= x & z);
//     azip!((z in clifford.column_mut(num_qubits + qubit), &x in &x) *z ^= x);
// }

// fn append_h(mut clifford: &mut Array2<bool>, qubit: usize, num_qubits: usize) {
//     let x = clifford.column(qubit).to_owned();
//     let z = clifford.column(num_qubits + qubit).to_owned();
//     azip!((phase in clifford.column_mut(2 * num_qubits), &x in &x, &z in &z)  *phase ^= x & z);
//     //
//     // tmp = x.copy()
//     // x[:] = z
//     // z[:] = tmp
//
// }

//
fn compose_ignore_phase(nq: usize, cliff1: ArrayView2<bool>, cliff2: ArrayView2<bool>) -> Array2<bool> {
    let arr1 = cliff1.slice(s![.., 0..2*nq]).map(|v| *v as u8);
    let arr2 = cliff2.slice(s![.., 0..2*nq]).map(|v| *v as u8);
    let arr3 = arr2.dot(&arr1).map(|v| (*v % 2) == 1);
    arr3
}



/// Calculate a decoupling operator D:
/// D^{-1} * Ox * D = x1
/// D^{-1} * Oz * D = z1
/// and reduce the clifford such that it will act trivially on min_qubit
fn calc_decoupling(
                   gate_seq: &mut CliffordSequenceVec,
                   pauli_x: ArrayView1<bool>,
                   pauli_z: ArrayView1<bool>,
                   qubit_list: &Vec<usize>,
                   min_qubit: usize,
                   num_qubits: usize) -> Array2<bool>
{
    println!("---------------------------------------------------------------------");
    let mut decouple_mat: Array2<bool> = Array2::<usize>::eye(2 * num_qubits).map(|v| v % 2 == 1);
    println!("pauli_x = {:?}", pauli_x);
    println!("pauli_z = {:?}", pauli_z);

    for qubit in qubit_list {
        let typeq = from_pair_paulis_to_type(pauli_x, pauli_z, *qubit);
        println!("qubit ={}, typeq = {:?}", qubit, typeq);

        if typeq == [[true, true], [false, false]] || typeq == [[true, true], [true, true]] || typeq == [[true, true], [true, false]] {
            println!("=> S GATE on qubit {}", qubit);
            gate_seq.push(("s".to_string(), *qubit as u8, 0));
            append_s_no_phase(&mut decouple_mat, *qubit, num_qubits);
        }
        else if
            typeq == [[true, false], [false, false]] ||
            typeq == [[true, false], [true, false]] ||
            typeq == [[true, false], [false, true]] ||
            typeq == [[false, false], [false, true]] {

            println!("=> H GATE on qubit {}", qubit);
            gate_seq.push(("h".to_string(), *qubit as u8, 0));
            append_h_no_phase(&mut decouple_mat, *qubit, num_qubits);
        }
        else if typeq == [[false, false], [true, true]] ||
            typeq == [[true, false], [true, true]] {
            println!("=> SH GATE on qubit {}", qubit);
            gate_seq.push(("s".to_string(), *qubit as u8, 0));
            append_s_no_phase(&mut decouple_mat, *qubit, num_qubits);
            gate_seq.push(("h".to_string(), *qubit as u8, 0));
            append_h_no_phase(&mut decouple_mat, *qubit, num_qubits);
        }
        else if typeq == [[true, true], [false, true]]  {
            println!("=> HS GATE on qubit {}", qubit);
            gate_seq.push(("h".to_string(), *qubit as u8, 0));
            append_h_no_phase(&mut decouple_mat, *qubit, num_qubits);
            gate_seq.push(("s".to_string(), *qubit as u8, 0));
            append_s_no_phase(&mut decouple_mat, *qubit, num_qubits);
        }
        else if typeq == [[false, true], [true, true]]  {
            println!("=> SHS GATE on qubit {}", qubit);
            gate_seq.push(("s".to_string(), *qubit as u8, 0));
            append_s_no_phase(&mut decouple_mat, *qubit, num_qubits);
            gate_seq.push(("h".to_string(), *qubit as u8, 0));
            append_h_no_phase(&mut decouple_mat, *qubit, num_qubits);
            gate_seq.push(("s".to_string(), *qubit as u8, 0));
            append_s_no_phase(&mut decouple_mat, *qubit, num_qubits);
        }
    }

    let mut a_qubits = Vec::<usize>::new();
    let mut b_qubits = Vec::<usize>::new();
    let mut c_qubits = Vec::<usize>::new();
    let mut d_qubits = Vec::<usize>::new();

    for qubit in qubit_list {
        let typeq = from_pair_paulis_to_type(pauli_x, pauli_z, *qubit);
        if A_CLASS.contains(&typeq) {
            a_qubits.push(*qubit);
        }
        else if B_CLASS.contains(&typeq) {
            b_qubits.push(*qubit);
        }
        else if C_CLASS.contains(&typeq) {
            c_qubits.push(*qubit);
        }
        else if D_CLASS.contains(&typeq) {
            d_qubits.push(*qubit);
        }
    }

    if a_qubits.len() % 2 != 1 {
        panic!("Symplectic elim fails");
    }

    if !a_qubits.contains(&min_qubit) {
        let qubit_a = a_qubits[0];
        println!("=> SWAP GATE on qubits {} and {}", min_qubit, qubit_a);
        gate_seq.push(("swap".to_string(), min_qubit as u8, qubit_a as u8));
        append_swap_no_phase(&mut decouple_mat, min_qubit, qubit_a, num_qubits);

        if b_qubits.contains(&min_qubit) {
            b_qubits.retain(|&x| x != min_qubit);
            b_qubits.push(qubit_a);
        }
        else if c_qubits.contains(&min_qubit) {
            c_qubits.retain(|&x| x != min_qubit);
            c_qubits.push(qubit_a);
        }
        else if d_qubits.contains(&min_qubit) {
            d_qubits.retain(|&x| x != min_qubit);
            d_qubits.push(qubit_a);
        }

        a_qubits.retain(|&x| x != qubit_a);
        a_qubits.push(min_qubit);
    }

    for qubit in c_qubits {
        println!("=> C_QUBITS: CX GATE on qubits {} and {}", min_qubit, qubit);
        gate_seq.push(("cx".to_string(), min_qubit as u8, qubit as u8));
        append_cx_no_phase(&mut decouple_mat, min_qubit, qubit, num_qubits);
    }

    for qubit in d_qubits {
        println!("=> D_QUBITS: CX GATE on qubits {} and {}", qubit, min_qubit);
        gate_seq.push(("cx".to_string(), qubit as u8, min_qubit as u8));
        append_cx_no_phase(&mut decouple_mat, qubit, min_qubit, num_qubits);
    }

    if b_qubits.len() > 1 {
        let qubit_b = b_qubits[0];
        for qubit in &b_qubits[1..] {
            println!("=> B_QUBITS: CX GATE on qubits {} and {}", qubit_b, qubit);
            gate_seq.push(("cx".to_string(), qubit_b as u8, *qubit as u8));
            append_cx_no_phase(&mut decouple_mat, *qubit, qubit_b, num_qubits);
        }
    }

    if b_qubits.len() > 0 {
        let qubit_b = b_qubits[0];
        println!("=> B_QUBITS: CX GATE on qubits {} and {}", min_qubit, qubit_b);
        gate_seq.push(("cx".to_string(), min_qubit as u8, qubit_b as u8));
        append_cx_no_phase(&mut decouple_mat, min_qubit, qubit_b, num_qubits);

        println!("=> B_QUBITS: H GATE on qubit {}", qubit_b);
        gate_seq.push(("h".to_string(), qubit_b as u8, 0));
        append_h_no_phase(&mut decouple_mat, qubit_b, num_qubits);

        println!("=> B_QUBITS: CX GATE on qubits {} and {}", qubit_b, min_qubit);
        gate_seq.push(("cx".to_string(), qubit_b as u8, min_qubit as u8));
        append_cx_no_phase(&mut decouple_mat, qubit_b, min_qubit, num_qubits);

    }

    let a_len: usize = (a_qubits.len() - 1) / 2;
    if a_len > 0 {
        a_qubits.retain(|&x| x != min_qubit);
    }

    for qubit in 0..a_len {
        println!("=> A_QUBITS: CX GATE on qubits {} and {}", a_qubits[2 * qubit + 1], a_qubits[2 * qubit]);
        gate_seq.push(("cx".to_string(), a_qubits[2 * qubit + 1] as u8, a_qubits[2 * qubit] as u8));
        append_cx_no_phase(&mut decouple_mat, a_qubits[2 * qubit + 1], a_qubits[2 * qubit], num_qubits);

        println!("=> A_QUBITS: CX GATE on qubits {} and {}", a_qubits[2 * qubit], min_qubit);
        gate_seq.push(("cx".to_string(), a_qubits[2 * qubit] as u8, min_qubit as u8));
        append_cx_no_phase(&mut decouple_mat, a_qubits[2 * qubit], min_qubit, num_qubits);

        println!("=> A_QUBITS: CX GATE on qubits {} and {}", min_qubit, a_qubits[2 * qubit + 1]);
        gate_seq.push(("cx".to_string(), min_qubit as u8, a_qubits[2 * qubit + 1] as u8));
        append_cx_no_phase(&mut decouple_mat, min_qubit, a_qubits[2 * qubit + 1], num_qubits);

    }
    println!("---------------------------------------------------------------------");

    decouple_mat
}



#[pyfunction]
#[pyo3(signature = (tableau))]
fn synth_clifford_greedy_new(tableau: PyReadonlyArray2<bool>) {
    let mut clifford_gates = CliffordSequenceVec::new();
    let clifford = tableau.as_array().to_owned();

    println!("I AM IN SYNTH_CLIFFORD_GREEDY_NEW!");

    synth_clifford_greedy_inner(&clifford);
}

#[pymodule]
pub fn clifford(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(synth_clifford_greedy_new, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::*;
    use ndarray::arr2;

    // for quicker development
    fn example_clifford() -> Array2<bool> {
        arr2(&[
            [
                false, false, true, true, false, true, false, false, false, true, true,
            ],
            [
                false, false, false, true, false, false, false, true, true, false, false,
            ],
            [
                false, true, true, true, false, false, false, false, false, false, true,
            ],
            [
                false, false, true, true, false, false, false, false, false, false, true,
            ],
            [
                false, false, true, false, false, true, false, true, true, false, true,
            ],
            [
                false, true, false, true, true, true, false, true, true, false, false,
            ],
            [
                true, false, false, false, true, true, false, true, true, false, false,
            ],
            [
                false, true, true, false, false, false, true, true, true, true, true,
            ],
            [
                true, false, true, true, false, true, true, true, false, false, false,
            ],
            [
                true, true, true, false, true, true, false, true, true, true, true,
            ],
        ])
    }

    #[test]
    fn test_example_clifford() {
        println!("===========================");
        println!("TEST!!!");
        let mut cliff = example_clifford();
        println!("{:?}", cliff);

        synth_clifford_greedy_inner(&cliff);
        println!("===========================");

    }

    #[test]
    fn test_compose() {
        let cliff1 = arr2 (
            &[
                [ true, false, false,  true,  true,  true, false],
                [false, false, false, false, false,  true,  true],
                [ true,  true, false, false, false,  true,  true],
                [ true, false, false, false, false, false, false],
                [ true, false,  true, false,  true,  true, false],
                [false, false, false, false,  true, false,  true],
            ]
        );
        let cliff2 = arr2 (
            &[[ true, false, false,  true, false, false,  true],
                 [ true,  true, false,  true,  true, false,  true],
                 [false, false,  true, false, false, false,  true],
                 [ true, false,  true, false,  true, false,  true],
                 [ true,  true, false,  true, false, false,  true],
                 [ true, false, false,  true, false,  true, false],
            ]
        );

        println!("{:?}", cliff1);
        println!("{:?}", cliff2);
        compose_ignore_phase(3,cliff1.view(), cliff2.view());
    }
}
