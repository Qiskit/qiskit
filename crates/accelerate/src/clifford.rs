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

use ndarray::{azip, s, ArrayView1, Array1};
use pyo3::prelude::*;
use std::collections::HashSet;

use numpy::prelude::*;

use numpy::{PyReadonlyArray2};

use crate::QiskitError;
use numpy::ndarray::{Array2};
use smallvec::{smallvec, SmallVec};

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{Param, StandardGate};
use qiskit_circuit::{circuit, Qubit};

use crate::linear_matrix::calc_inverse_matrix_inner;

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


fn compute_greedy_cost(symplectic_matrix: &SymplecticMatrix, qubit: usize, qubit_list: &Vec<usize>, num_qubits: usize) -> usize {
    let pauli_x = symplectic_matrix.smat.column(qubit + num_qubits);
    let pauli_z = symplectic_matrix.smat.column(qubit);

    let mut a_num = 0;
    let mut b_num = 0;
    let mut c_num = 0;
    let mut d_num = 0;

    let mut qubit_is_in_a = false;

    for q in qubit_list {
        let pair = from_pair_paulis_to_type(pauli_x, pauli_z, *q);

        if A_CLASS.contains(&pair) {
            a_num += 1;
            if *q == qubit {
                qubit_is_in_a = true;
            }
        } else if B_CLASS.contains(&pair) {
            b_num += 1;
        } else if C_CLASS.contains(&pair) {
            c_num += 1;
        } else if D_CLASS.contains(&pair) {
            d_num += 1;
        }
    }


    if a_num % 2 == 0 {
        panic!("Symplectic Gaussian elimination fails");
        // return Err(QiskitError::new_err(
        //     "Symplectic Gaussian elimination fails",
        // ));
    }

    let mut cnot_cost: usize =
        3 * (a_num - 1) / 2 + (b_num + 1) * ((b_num > 0) as usize) + c_num + d_num;

    if !qubit_is_in_a {
        cnot_cost += 3;
    }

    cnot_cost
}

type CliffordGatesVec = Vec<(StandardGate, SmallVec<[Qubit; 2]>)>;

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



// for now modify clifford in-place
fn synth_clifford_greedy_inner(clifford: &Array2<bool>) -> CliffordGatesVec {
    let mut clifford_gates = CliffordGatesVec::new();

    let num_qubits = clifford.shape()[0] / 2;

    let mut symplectic_matrix = SymplecticMatrix {
        num_qubits,
        smat: clifford.slice(s![.., 0..2 * num_qubits]).to_owned(),
    };

    // ToDo: this is a vector for now to be compatible with the python
    // implementation, but we should really turn it into a set
    let mut qubit_list: Vec<usize> = (0..num_qubits).collect();

    while qubit_list.len() > 0 {
        let mut list_greedy_cost = Vec::<(usize, usize)>::new();

        for qubit in &qubit_list {
            let cost = compute_greedy_cost(&symplectic_matrix, *qubit, &qubit_list, num_qubits);
            // println!("{}", cost);
            list_greedy_cost.push((cost, *qubit));
        }
        // println!("list_greedy_cost = {:?}", list_greedy_cost);
        let min_qubit = list_greedy_cost
            .iter()
            .min_by_key(|(cost, _qubit)| cost)
            .unwrap()
            .1;
        // println!("min_qubit = {:?}", min_qubit);
        // clifford[0 : self.num_qubits, :]

        let pauli_x = symplectic_matrix.smat.column(min_qubit + num_qubits);
        let pauli_z = symplectic_matrix.smat.column(min_qubit);

        let mut decouple_matrix = calc_decoupling(
            &mut clifford_gates,
            pauli_x,
            pauli_z,
            &qubit_list,
            min_qubit,
            num_qubits,
        );

        // println!("====================================================");
        // println!("DECOUPLER CLIFF:");
        // println!("{:?}", decouple_cliff);
        // println!("====================================================");

        decouple_matrix.adjoint();
        let composed_matrix = symplectic_matrix.compose(&decouple_matrix);
        symplectic_matrix = composed_matrix;



        // println!("====================================================");
        // println!("CLIFFORD CURRENT:");
        // println!("{:?}", symplectic_mat);
        // println!("====================================================");

        // qubit_list.remove(&min_qubit);
        qubit_list.retain(|&x| x != min_qubit);
    }

    // the symplectic matrix should be ok, but the phase is not
    // println!("====================================================");
    // println!("SMAT END:");
    // println!("{:?}", symplectic_mat);
    // println!("{:?}", clifford_gates);
    // println!("====================================================");

    fix_phase(&mut clifford_gates, clifford, num_qubits);

    clifford_gates
}

// Zip::from(clifford.column_mut(2 * num_qubits))
//     .and(&x)
//     .and(&z)
//     .for_each(|phase, &x, &z| *phase ^= x & z);
// Zip::from(clifford.column_mut(num_qubits + qubit))
//     .and(&x)
//     .for_each(|z, x| *z ^= x );

fn append_s(clifford: &mut Array2<bool>, qubit: usize, num_qubits: usize) {
    // println!("_append_s_clifford {}; num_qubits = {}", qubit, num_qubits);

    let (x, mut z, mut p) = clifford.multi_slice_mut((
        s![.., qubit],
        s![.., num_qubits + qubit],
        s![.., 2 * num_qubits],
    ));

    azip!((p in &mut p, &x in &x, &z in &z)  *p ^= x & z);
    azip!((z in &mut z, &x in &x) *z ^= x);
}

/*

    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    clifford.phase ^= x & ~z

 */
// fn append_sdg(mut clifford: &mut Array2<bool>, qubit: usize, num_qubits: usize) {
//     // println!("_append_sdg_clifford {}; num_qubits = {}", qubit, num_qubits);
//
//     let (x, mut z, mut p) = clifford.multi_slice_mut((
//         s![.., qubit],
//         s![.., num_qubits + qubit],
//         s![.., 2 * num_qubits],
//     ));
//
//     azip!((mut p in &mut p, &x in &x, &z in &z)  *p ^= x & !z);
//     azip!((mut z in &mut z, &x in &x) *z ^= x);
// }

fn append_h(clifford: &mut Array2<bool>, qubit: usize, num_qubits: usize) {
    // println!("_append_h_clifford {}; num_qubits = {}", qubit, num_qubits);

    let (mut x, mut z, mut p) = clifford.multi_slice_mut((
        s![.., qubit],
        s![.., num_qubits + qubit],
        s![.., 2 * num_qubits],
    ));

    azip!((p in &mut p, &x in &x, &z in &z)  *p ^= x & z);
    azip!((x in &mut x, z in &mut z)  (*x, *z) = (*z, *x));
}

fn append_swap(clifford: &mut Array2<bool>, qubit0: usize, qubit1: usize, num_qubits: usize) {
    let (mut x0, mut z0, mut x1, mut z1) = clifford.multi_slice_mut((
        s![.., qubit0],
        s![.., num_qubits + qubit0],
        s![.., qubit1],
        s![.., num_qubits + qubit1],
    ));
    azip!((x0 in &mut x0, x1 in &mut x1)  (*x0, *x1) = (*x1, *x0));
    azip!((z0 in &mut z0, z1 in &mut z1)  (*z0, *z1) = (*z1, *z0));
}

// x0, z0 - control, x1, z1 - target
fn append_cx(clifford: &mut Array2<bool>, qubit0: usize, qubit1: usize, num_qubits: usize) {
    // println!(
    //     "_append_cx_clifford {} {}; num_qubits = {}",
    //     qubit0, qubit1, num_qubits
    // );

    let (x0, mut z0, mut x1, z1, mut p) = clifford.multi_slice_mut((
        s![.., qubit0],
        s![.., num_qubits + qubit0],
        s![.., qubit1],
        s![.., num_qubits + qubit1],
        s![.., 2 * num_qubits],
    ));
    azip!((p in &mut p, &x0 in &x0, &z0 in &z0, &x1 in &x1, &z1 in &z1) *p ^= (x1 ^ z0 ^ true) & z1 & x0);
    azip!((x1 in &mut x1, &x0 in &x0) *x1 ^= x0);
    azip!((z0 in &mut z0, &z1 in &z1) *z0 ^= z1);
}

// CODE FOR SYMPLECTIC MATRICES!

/// Basic functionality for symplectic matrices
struct SymplecticMatrix {
    // number of qubits
    num_qubits: usize,
    // symplectic matrix with dimensions (2 * num_qubits) x (2 * num_qubits)
    smat: Array2<bool>,
}

impl SymplecticMatrix {

    // Modifies the matrix in-place by appending S-gate
    fn append_s(&mut self, qubit: usize) {
        let (x, mut z) = self.smat.multi_slice_mut((s![.., qubit], s![.., self.num_qubits + qubit]));
        azip!((z in &mut z, &x in &x) *z ^= x);
    }



    // Modifies the matrix in-place by appending H-gate
    fn append_h(&mut self, qubit: usize) {
        let (mut x, mut z) = self.smat.multi_slice_mut((s![.., qubit], s![.., self.num_qubits + qubit]));
        azip!((x in &mut x, z in &mut z)  (*x, *z) = (*z, *x));
    }

    // Modifies the matrix in-place by appending SWAP-gate
    fn append_swap(
        &mut self,
        qubit0: usize,
        qubit1: usize,
    ) {
        let (mut x0, mut z0, mut x1, mut z1) = self.smat.multi_slice_mut((
            s![.., qubit0],
            s![.., self.num_qubits + qubit0],
            s![.., qubit1],
            s![.., self.num_qubits + qubit1],
        ));
        azip!((x0 in &mut x0, x1 in &mut x1)  (*x0, *x1) = (*x1, *x0));
        azip!((z0 in &mut z0, z1 in &mut z1)  (*z0, *z1) = (*z1, *z0));
    }

    // Modifies the matrix in-place by appending SWAP-gate
    fn append_cx(
        &mut self,
        qubit0: usize,
        qubit1: usize,
    ) {
        let (x0, mut z0, mut x1, z1) = self.smat.multi_slice_mut((
            s![.., qubit0],
            s![.., self.num_qubits + qubit0],
            s![.., qubit1],
            s![.., self.num_qubits + qubit1],
        ));
        azip!((x1 in &mut x1, &x0 in &x0) *x1 ^= x0);
        azip!((z0 in &mut z0, &z1 in &z1) *z0 ^= z1);
    }

    // Multiplies two symplectic matrices and returns the result
    // other * self
    // ToDo: can we avoid all the copying?
    // ToDo: the order of operation is funny
    fn compose(
        &self, other: &SymplecticMatrix
    ) -> Self {
        // ToDo: get rid of slicing
        let num_qubits = self.num_qubits;
        let smat1 = self.smat.slice(s![.., 0..2 * num_qubits]).map(|v| *v as u8);
        let smat2 = other.smat.slice(s![.., 0..2 * num_qubits]).map(|v| *v as u8);
        let smat3 = smat2.dot(&smat1).map(|v| (*v % 2) == 1);
        SymplecticMatrix {
            num_qubits,
            smat: smat3,
        }
    }

    // Modifies the matrix in-place by computing its adjoint
    fn adjoint(&mut self) {
        // ToDo: get rid of this extra copying
        self.smat = self.smat.t().to_owned();
        // *(self.smat) = transposed;

        let (mut a, mut b, mut c, mut d) = self.smat.multi_slice_mut((
            s![0..self.num_qubits, 0..self.num_qubits],
            s![0..self.num_qubits, self.num_qubits..2 * self.num_qubits],
            s![self.num_qubits..2 * self.num_qubits, 0..self.num_qubits],
            s![self.num_qubits..2 * self.num_qubits, self.num_qubits..2 * self.num_qubits],
        ));

        azip!((a in &mut a, d in &mut d)  (*a, *d) = (*d, *a));
        azip!((b in &mut b, c in &mut c)  (*b, *c) = (*c, *b));
    }

}



/// Calculate a decoupling operator D:
/// D^{-1} * Ox * D = x1
/// D^{-1} * Oz * D = z1
/// and reduce the clifford such that it will act trivially on min_qubit
fn calc_decoupling(
    gate_seq: &mut CliffordGatesVec,
    pauli_x: ArrayView1<bool>,
    pauli_z: ArrayView1<bool>,
    qubit_list: &Vec<usize>,
    min_qubit: usize,
    num_qubits: usize,
) -> SymplecticMatrix {
    let mut decouple_matrix = SymplecticMatrix {
        num_qubits,
        smat: Array2::from_shape_fn((2 * num_qubits, 2 * num_qubits), |(i, j)| i == j),
    };


    for qubit in qubit_list {
        let typeq = from_pair_paulis_to_type(pauli_x, pauli_z, *qubit);

        if typeq == [[true, true], [false, false]]
            || typeq == [[true, true], [true, true]]
            || typeq == [[true, true], [true, false]]
        {
            gate_seq.push((StandardGate::SGate, smallvec![Qubit(*qubit as u32)]));
            decouple_matrix.append_s(*qubit);
        } else if typeq == [[true, false], [false, false]]
            || typeq == [[true, false], [true, false]]
            || typeq == [[true, false], [false, true]]
            || typeq == [[false, false], [false, true]]
        {
            gate_seq.push((StandardGate::HGate, smallvec![Qubit(*qubit as u32)]));
            decouple_matrix.append_h(*qubit);
        } else if typeq == [[false, false], [true, true]] || typeq == [[true, false], [true, true]]
        {
            gate_seq.push((StandardGate::SGate, smallvec![Qubit(*qubit as u32)]));
            decouple_matrix.append_s(*qubit);
            gate_seq.push((StandardGate::HGate, smallvec![Qubit(*qubit as u32)]));
            decouple_matrix.append_h(*qubit);
        } else if typeq == [[true, true], [false, true]] {
            gate_seq.push((StandardGate::HGate, smallvec![Qubit(*qubit as u32)]));
            decouple_matrix.append_h(*qubit);
            gate_seq.push((StandardGate::SGate, smallvec![Qubit(*qubit as u32)]));
            decouple_matrix.append_s(*qubit);
        } else if typeq == [[false, true], [true, true]] {
            gate_seq.push((StandardGate::SGate, smallvec![Qubit(*qubit as u32)]));
            decouple_matrix.append_s(*qubit);
            gate_seq.push((StandardGate::HGate, smallvec![Qubit(*qubit as u32)]));
            decouple_matrix.append_h(*qubit);
            gate_seq.push((StandardGate::SGate, smallvec![Qubit(*qubit as u32)]));
            decouple_matrix.append_s(*qubit);
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
        } else if B_CLASS.contains(&typeq) {
            b_qubits.push(*qubit);
        } else if C_CLASS.contains(&typeq) {
            c_qubits.push(*qubit);
        } else if D_CLASS.contains(&typeq) {
            d_qubits.push(*qubit);
        }
    }

    if a_qubits.len() % 2 != 1 {
        panic!("Symplectic elim fails");
    }

    if !a_qubits.contains(&min_qubit) {
        let qubit_a = a_qubits[0];
        gate_seq.push((
            StandardGate::SwapGate,
            smallvec![Qubit(min_qubit as u32), Qubit(qubit_a as u32)],
        ));
        decouple_matrix.append_swap(min_qubit, qubit_a);

        if b_qubits.contains(&min_qubit) {
            b_qubits.retain(|&x| x != min_qubit);
            b_qubits.push(qubit_a);
        } else if c_qubits.contains(&min_qubit) {
            c_qubits.retain(|&x| x != min_qubit);
            c_qubits.push(qubit_a);
        } else if d_qubits.contains(&min_qubit) {
            d_qubits.retain(|&x| x != min_qubit);
            d_qubits.push(qubit_a);
        }

        a_qubits.retain(|&x| x != qubit_a);
        a_qubits.push(min_qubit);
    }

    for qubit in c_qubits {
        gate_seq.push((
            StandardGate::CXGate,
            smallvec![Qubit(min_qubit as u32), Qubit(qubit as u32)],
        ));
        decouple_matrix.append_cx(min_qubit, qubit);
    }

    for qubit in d_qubits {
        gate_seq.push((
            StandardGate::CXGate,
            smallvec![Qubit(qubit as u32), Qubit(min_qubit as u32)],
        ));
        decouple_matrix.append_cx(qubit, min_qubit);
    }

    if b_qubits.len() > 1 {
        let qubit_b = b_qubits[0];
        for qubit in &b_qubits[1..] {
            gate_seq.push((
                StandardGate::CXGate,
                smallvec![Qubit(qubit_b as u32), Qubit(*qubit as u32)],
            ));
            decouple_matrix.append_cx(qubit_b, *qubit);
        }
    }

    if b_qubits.len() > 0 {
        let qubit_b = b_qubits[0];
        gate_seq.push((
            StandardGate::CXGate,
            smallvec![Qubit(min_qubit as u32), Qubit(qubit_b as u32)],
        ));
        decouple_matrix.append_cx(min_qubit, qubit_b);

        gate_seq.push((StandardGate::HGate, smallvec![Qubit(qubit_b as u32)]));
        decouple_matrix.append_h(qubit_b);

        gate_seq.push((
            StandardGate::CXGate,
            smallvec![Qubit(qubit_b as u32), Qubit(min_qubit as u32)],
        ));
        decouple_matrix.append_cx(qubit_b, min_qubit);
    }

    let a_len: usize = (a_qubits.len() - 1) / 2;
    if a_len > 0 {
        a_qubits.retain(|&x| x != min_qubit);
    }

    for qubit in 0..a_len {
        gate_seq.push((
            StandardGate::CXGate,
            smallvec![
                Qubit(a_qubits[2 * qubit + 1] as u32),
                Qubit(a_qubits[2 * qubit] as u32)
            ],
        ));
        decouple_matrix.append_cx(a_qubits[2 * qubit + 1], a_qubits[2 * qubit]);

        gate_seq.push((
            StandardGate::CXGate,
            smallvec![Qubit(a_qubits[2 * qubit] as u32), Qubit(min_qubit as u32)],
        ));
        decouple_matrix.append_cx(a_qubits[2 * qubit], min_qubit);

        gate_seq.push((
            StandardGate::CXGate,
            smallvec![
                Qubit(min_qubit as u32),
                Qubit(a_qubits[2 * qubit + 1] as u32)
            ],
        ));
        decouple_matrix.append_cx(min_qubit, a_qubits[2 * qubit + 1]);
    }

    decouple_matrix
}

fn clifford_sim(gate_seq: &CliffordGatesVec, num_qubits: usize) -> Array2<bool> {
    let mut current_clifford: Array2<bool> =
        Array2::from_shape_fn((2 * num_qubits, 2 * num_qubits + 1), |(i, j)| i == j);

    gate_seq.iter().for_each(|(gate, qubits)| match *gate {
        StandardGate::SGate => append_s(&mut current_clifford, qubits[0].0 as usize, num_qubits),
        StandardGate::HGate => append_h(&mut current_clifford, qubits[0].0 as usize, num_qubits),
        StandardGate::CXGate => append_cx(
            &mut current_clifford,
            qubits[0].0 as usize,
            qubits[1].0 as usize,
            num_qubits,
        ),
        StandardGate::SwapGate => append_swap(
            &mut current_clifford,
            qubits[0].0 as usize,
            qubits[1].0 as usize,
            num_qubits,
        ),
        _ => panic!("We should never get here!"),
    });
    current_clifford
}


/// Fixes the phase
fn fix_phase(gate_seq: &mut CliffordGatesVec, target_clifford: &Array2<bool>, num_qubits: usize) {
    // simulate the clifford circuit that we have constructed
    let simulated_clifford = clifford_sim(gate_seq, num_qubits);

    // compute phase difference
    let target_phase = target_clifford.column(2 * num_qubits);
    let sim_phase = simulated_clifford.column(2 * num_qubits);

    let delta_phase: Vec<bool> = target_phase
        .iter()
        .zip(sim_phase.iter())
        .map(|(&a, &b)| a ^ b)
        .collect();

    // compute inverse of the symplectic matrix
    let smat = target_clifford.slice(s![.., .. 2 * num_qubits]);
    let smat_inv = calc_inverse_matrix_inner(smat);

    // compute smat_inv * delta_phase
    let arr1 = smat_inv.map(|v| *v as usize);
    let vec2 : Vec<usize> = delta_phase.into_iter().map(|v| v as usize).collect();
    let arr2 = Array1::from(vec2);
    let delta_phase_pre = arr1.dot(&arr2).map(|v| v % 2 == 1);


    // println!("delta_phase_pre:");
    // println!("{:?}", delta_phase_pre);


    for qubit in 0..num_qubits {
        if delta_phase_pre[qubit] && delta_phase_pre[qubit + num_qubits] {
            // println!("=> Adding Y-gate on {}", qubit);
            gate_seq.push((StandardGate::YGate, smallvec![Qubit(qubit as u32)]));
        }
        else if delta_phase_pre[qubit] {
            // println!("=> Adding Z-gate on {}", qubit);
            gate_seq.push((StandardGate::ZGate, smallvec![Qubit(qubit as u32)]));
        }
        else if delta_phase_pre[qubit + num_qubits] {
            // println!("=> Adding X-gate on {}", qubit);
            gate_seq.push((StandardGate::XGate, smallvec![Qubit(qubit as u32)]));
        }
    }
}

#[pyfunction]
#[pyo3(signature = (tableau))]
fn synth_clifford_greedy_new(
    py: Python,
    tableau: PyReadonlyArray2<bool>,
) -> PyResult<Option<CircuitData>> {
    let clifford = tableau.as_array().to_owned();
    let num_qubits = clifford.shape()[0] / 2;
    let clifford_gates = synth_clifford_greedy_inner(&clifford);
    let circuit_data = CircuitData::from_standard_gates(
        py,
        num_qubits as u32,
        clifford_gates
            .into_iter()
            .map(|(gate, qubits)| (gate, smallvec![], qubits)),
        Param::Float(0.0),
    )
    .expect("Something went wrong on Qiskit's Python side, nothing to do here!");
    Ok(Some(circuit_data))
}

#[pymodule]
pub fn clifford(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(synth_clifford_greedy_new, m)?)?;
    Ok(())
}
