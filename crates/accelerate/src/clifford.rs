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

use ndarray::{azip, s, Array1, ArrayView1, ArrayView2};
use pyo3::prelude::*;
use std::collections::HashSet;
use hashbrown::HashMap;

use numpy::PyReadonlyArray2;

use crate::QiskitError;
use numpy::ndarray::Array2;
use smallvec::{smallvec, SmallVec};

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{Param, StandardGate};
use qiskit_circuit::Qubit;

use crate::linear_matrix::calc_inverse_matrix_inner;

/// Basic functionality for symplectic matrices
pub struct SymplecticMatrix {
    // number of qubits
    num_qubits: usize,
    // symplectic matrix with dimensions (2 * num_qubits) x (2 * num_qubits)
    smat: Array2<bool>,
}

impl SymplecticMatrix {
    // Modifies the matrix in-place by appending S-gate
    pub fn append_s(&mut self, qubit: usize) {
        let (x, mut z) = self
            .smat
            .multi_slice_mut((s![.., qubit], s![.., self.num_qubits + qubit]));
        azip!((z in &mut z, &x in &x) *z ^= x);
    }

    // Modifies the matrix in-place by prepending S-gate
    pub fn prepend_s(&mut self, qubit: usize) {
        let (x, mut z) = self
            .smat
            .multi_slice_mut((s![self.num_qubits + qubit, ..], s![qubit, ..]));
        azip!((z in &mut z, &x in &x) *z ^= x);
    }

    // Modifies the matrix in-place by appending H-gate
    pub fn append_h(&mut self, qubit: usize) {
        let (mut x, mut z) = self
            .smat
            .multi_slice_mut((s![.., qubit], s![.., self.num_qubits + qubit]));
        azip!((x in &mut x, z in &mut z)  (*x, *z) = (*z, *x));
    }

    // Modifies the matrix in-place by prepending H-gate
    pub fn prepend_h(&mut self, qubit: usize) {
        let (mut x, mut z) = self
            .smat
            .multi_slice_mut((s![qubit, ..], s![self.num_qubits + qubit, ..]));
        azip!((x in &mut x, z in &mut z)  (*x, *z) = (*z, *x));
    }

    // Modifies the matrix in-place by appending SWAP-gate
    pub fn append_swap(&mut self, qubit0: usize, qubit1: usize) {
        let (mut x0, mut z0, mut x1, mut z1) = self.smat.multi_slice_mut((
            s![.., qubit0],
            s![.., self.num_qubits + qubit0],
            s![.., qubit1],
            s![.., self.num_qubits + qubit1],
        ));
        azip!((x0 in &mut x0, x1 in &mut x1)  (*x0, *x1) = (*x1, *x0));
        azip!((z0 in &mut z0, z1 in &mut z1)  (*z0, *z1) = (*z1, *z0));
    }

    // Modifies the matrix in-place by prepending SWAP-gate
    pub fn prepend_swap(&mut self, qubit0: usize, qubit1: usize) {
        let (mut x0, mut z0, mut x1, mut z1) = self.smat.multi_slice_mut((
            s![qubit0, ..],
            s![self.num_qubits + qubit0, ..],
            s![qubit1, ..],
            s![self.num_qubits + qubit1, ..],
        ));
        azip!((x0 in &mut x0, x1 in &mut x1)  (*x0, *x1) = (*x1, *x0));
        azip!((z0 in &mut z0, z1 in &mut z1)  (*z0, *z1) = (*z1, *z0));
    }

    // Modifies the matrix in-place by appending CX-gate
    pub fn append_cx(&mut self, qubit0: usize, qubit1: usize) {
        let (x0, mut z0, mut x1, z1) = self.smat.multi_slice_mut((
            s![.., qubit0],
            s![.., self.num_qubits + qubit0],
            s![.., qubit1],
            s![.., self.num_qubits + qubit1],
        ));
        azip!((x1 in &mut x1, &x0 in &x0) *x1 ^= x0);
        azip!((z0 in &mut z0, &z1 in &z1) *z0 ^= z1);
    }

    // Modifies the matrix in-place by prepending CX-gate
    pub fn prepend_cx(&mut self, qubit0: usize, qubit1: usize) {
        let (x0, mut z0, mut x1, z1) = self.smat.multi_slice_mut((
            s![qubit1, ..],
            s![self.num_qubits + qubit1, ..],
            s![qubit0, ..],
            s![self.num_qubits + qubit0, ..],
        ));
        azip!((x1 in &mut x1, &x0 in &x0) *x1 ^= x0);
        azip!((z0 in &mut z0, &z1 in &z1) *z0 ^= z1);
    }
}

#[derive(Clone, Copy)]
enum PauliClass {
    ClassA,
    ClassB,
    ClassC,
    ClassD,
    ClassE,
}

struct GreedyCliffordSynthesis<'a> {
    // The Clifford to be synthesized.
    clifford: ArrayView2<'a, bool>,

    // The 16 pairs of Pauli operators are divided into 5 equivalence classes
    // under the action of single-qubit Cliffords.
    pauli_to_class: HashMap<[[bool; 2]; 2], PauliClass>,
}


// todo: next step is to change pauli_to_class map to be a map from 4-tuple of ints

impl GreedyCliffordSynthesis<'_> {

    fn new(clifford: ArrayView2<bool>) -> GreedyCliffordSynthesis<'_> {

        let pauli_to_class = HashMap::<[[bool; 2]; 2], PauliClass>::from(
            [
                ([[false, true], [true, true]], PauliClass::ClassA),    // 'XY'
                ([[false, true], [true, false]], PauliClass::ClassA),   // 'XZ'
                ([[true, true], [false, true]], PauliClass::ClassA),    // 'YX'
                ([[true, true], [true, false]], PauliClass::ClassA),    // 'YZ'
                ([[true, false], [false, true]], PauliClass::ClassA),   // 'ZX'
                ([[true, false], [true, true]], PauliClass::ClassA),    // 'ZY'
                ([[true, false], [true, false]], PauliClass::ClassB),   // 'ZZ'
                ([[false, true], [false, true]], PauliClass::ClassB),   // 'XX'
                ([[true, true], [true, true]], PauliClass::ClassB),     // 'YY'
                ([[true, false], [false, false]], PauliClass::ClassC),  // 'ZI'
                ([[false, true], [false, false]], PauliClass::ClassC),  // 'XI'
                ([[true, true], [false, false]], PauliClass::ClassC),   // 'YI'
                ([[false, false], [false, true]], PauliClass::ClassD),  // 'IX'
                ([[false, false], [true, false]], PauliClass::ClassD),  // 'IZ'
                ([[false, false], [true, true]], PauliClass::ClassD),   // 'IY'
                ([[false, false], [false, false]], PauliClass::ClassE), // 'II'
            ]
        );

        GreedyCliffordSynthesis {
            clifford,
            pauli_to_class,
        }
    }

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

    // todo: move more arguments to the main algo class
    fn compute_greedy_cost(
        &self,
        symplectic_matrix: &SymplecticMatrix,
        qubit: usize,
        qubit_list: &Vec<usize>,
        num_qubits: usize,
    ) -> usize {
        // todo: remove to_owned
        let pauli_x = symplectic_matrix.smat.column(qubit + num_qubits);
        let pauli_z = symplectic_matrix.smat.column(qubit);

        let mut a_num = 0;
        let mut b_num = 0;
        let mut c_num = 0;
        let mut d_num = 0;

        let mut qubit_is_in_a = false;

        for q in qubit_list {
            let pair = GreedyCliffordSynthesis::from_pair_paulis_to_type(pauli_x, pauli_z, *q);
            let pauli_class = self.pauli_to_class.get(&pair).unwrap();

            match pauli_class {
                PauliClass::ClassA => {
                    a_num += 1;
                    if *q == qubit {
                        qubit_is_in_a = true;
                    }
                },
                PauliClass::ClassB => {
                    b_num += 1;
                },
                PauliClass::ClassC => {
                    c_num += 1;
                },
                PauliClass::ClassD => {
                    d_num += 1;
                }
                PauliClass::ClassE => {
                }
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

    /// Calculate a decoupling operator D:
    /// D^{-1} * Ox * D = x1
    /// D^{-1} * Oz * D = z1
    /// and reduce the clifford such that it will act trivially on min_qubit
    fn calc_decoupling(
        &self,
        symplectic_matrix: &mut SymplecticMatrix,
        gate_seq: &mut CliffordGatesVec,
        qubit_list: &Vec<usize>,
        min_qubit: usize,
        num_qubits: usize,
    ) {
        let pauli_x = symplectic_matrix
            .smat
            .column(min_qubit + num_qubits)
            .to_owned();
        let pauli_z = symplectic_matrix.smat.column(min_qubit).to_owned();

        for qubit in qubit_list {

            let typeq = GreedyCliffordSynthesis::from_pair_paulis_to_type(pauli_x.view(), pauli_z.view(), *qubit);

            if typeq == [[true, true], [false, false]]
                || typeq == [[true, true], [true, true]]
                || typeq == [[true, true], [true, false]]
            {
                gate_seq.push((StandardGate::SGate, smallvec![Qubit(*qubit as u32)]));
                symplectic_matrix.prepend_s(*qubit);
            } else if typeq == [[true, false], [false, false]]
                || typeq == [[true, false], [true, false]]
                || typeq == [[true, false], [false, true]]
                || typeq == [[false, false], [false, true]]
            {
                gate_seq.push((StandardGate::HGate, smallvec![Qubit(*qubit as u32)]));
                symplectic_matrix.prepend_h(*qubit);
            } else if typeq == [[false, false], [true, true]] || typeq == [[true, false], [true, true]]
            {
                gate_seq.push((StandardGate::SGate, smallvec![Qubit(*qubit as u32)]));
                symplectic_matrix.prepend_s(*qubit);
                gate_seq.push((StandardGate::HGate, smallvec![Qubit(*qubit as u32)]));
                symplectic_matrix.prepend_h(*qubit);
            } else if typeq == [[true, true], [false, true]] {
                gate_seq.push((StandardGate::HGate, smallvec![Qubit(*qubit as u32)]));
                symplectic_matrix.prepend_h(*qubit);
                gate_seq.push((StandardGate::SGate, smallvec![Qubit(*qubit as u32)]));
                symplectic_matrix.prepend_s(*qubit);
            } else if typeq == [[false, true], [true, true]] {
                gate_seq.push((StandardGate::SGate, smallvec![Qubit(*qubit as u32)]));
                symplectic_matrix.prepend_s(*qubit);
                gate_seq.push((StandardGate::HGate, smallvec![Qubit(*qubit as u32)]));
                symplectic_matrix.prepend_h(*qubit);
                gate_seq.push((StandardGate::SGate, smallvec![Qubit(*qubit as u32)]));
                symplectic_matrix.prepend_s(*qubit);
            }
        }

        let mut a_qubits = Vec::<usize>::new();
        let mut b_qubits = Vec::<usize>::new();
        let mut c_qubits = Vec::<usize>::new();
        let mut d_qubits = Vec::<usize>::new();

        for qubit in qubit_list {
            let typeq = GreedyCliffordSynthesis::from_pair_paulis_to_type(pauli_x.view(), pauli_z.view(), *qubit);
            let pauli_class = self.pauli_to_class.get(&typeq).unwrap();
            match pauli_class {
                PauliClass::ClassA => a_qubits.push(*qubit),
                PauliClass::ClassB => b_qubits.push(*qubit),
                PauliClass::ClassC => c_qubits.push(*qubit),
                PauliClass::ClassD => d_qubits.push(*qubit),
                PauliClass::ClassE => { },

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
            symplectic_matrix.prepend_swap(min_qubit, qubit_a);

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
            symplectic_matrix.prepend_cx(min_qubit, qubit);
        }

        for qubit in d_qubits {
            gate_seq.push((
                StandardGate::CXGate,
                smallvec![Qubit(qubit as u32), Qubit(min_qubit as u32)],
            ));
            symplectic_matrix.prepend_cx(qubit, min_qubit);
        }

        if b_qubits.len() > 1 {
            let qubit_b = b_qubits[0];
            for qubit in &b_qubits[1..] {
                gate_seq.push((
                    StandardGate::CXGate,
                    smallvec![Qubit(qubit_b as u32), Qubit(*qubit as u32)],
                ));
                symplectic_matrix.prepend_cx(qubit_b, *qubit);
            }
        }

        if b_qubits.len() > 0 {
            let qubit_b = b_qubits[0];
            gate_seq.push((
                StandardGate::CXGate,
                smallvec![Qubit(min_qubit as u32), Qubit(qubit_b as u32)],
            ));
            symplectic_matrix.prepend_cx(min_qubit, qubit_b);

            gate_seq.push((StandardGate::HGate, smallvec![Qubit(qubit_b as u32)]));
            symplectic_matrix.prepend_h(qubit_b);

            gate_seq.push((
                StandardGate::CXGate,
                smallvec![Qubit(qubit_b as u32), Qubit(min_qubit as u32)],
            ));
            symplectic_matrix.prepend_cx(qubit_b, min_qubit);
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
            symplectic_matrix.prepend_cx(a_qubits[2 * qubit + 1], a_qubits[2 * qubit]);

            gate_seq.push((
                StandardGate::CXGate,
                smallvec![Qubit(a_qubits[2 * qubit] as u32), Qubit(min_qubit as u32)],
            ));
            symplectic_matrix.prepend_cx(a_qubits[2 * qubit], min_qubit);

            gate_seq.push((
                StandardGate::CXGate,
                smallvec![
                    Qubit(min_qubit as u32),
                    Qubit(a_qubits[2 * qubit + 1] as u32)
                ],
            ));
            symplectic_matrix.prepend_cx(min_qubit, a_qubits[2 * qubit + 1]);
        }
    }

    // The main synthesis function
    fn run(&self) -> CliffordGatesVec {
        let mut clifford_gates = CliffordGatesVec::new();

        let num_qubits = self.clifford.shape()[0] / 2;

        let mut symplectic_matrix = SymplecticMatrix {
            num_qubits,
            smat: self.clifford.slice(s![.., 0..2 * num_qubits]).to_owned(),
        };

        // ToDo: this is a vector for now to be compatible with the python
        // implementation, but we should really turn it into a set
        let mut qubit_list: Vec<usize> = (0..num_qubits).collect();

        while qubit_list.len() > 0 {
            let mut list_greedy_cost = Vec::<(usize, usize)>::new();

            for qubit in &qubit_list {
                let cost = self.compute_greedy_cost(&symplectic_matrix, *qubit, &qubit_list, num_qubits);
                // println!("{}", cost);
                list_greedy_cost.push((cost, *qubit));
            }
            let min_qubit = list_greedy_cost
                .iter()
                .min_by_key(|(cost, _qubit)| cost)
                .unwrap()
                .1;

            self.calc_decoupling(
                &mut symplectic_matrix,
                &mut clifford_gates,
                &qubit_list,
                min_qubit,
                num_qubits,
            );

            // qubit_list.remove(&min_qubit);
            qubit_list.retain(|&x| x != min_qubit);
        }

        fix_phase(&mut clifford_gates, self.clifford, num_qubits);

        clifford_gates
    }
}







type CliffordGatesVec = Vec<(StandardGate, SmallVec<[Qubit; 2]>)>;

fn synth_clifford_greedy_inner(clifford: &Array2<bool>) -> CliffordGatesVec {
    let greedy_synthesis = GreedyCliffordSynthesis::new(clifford.view());
    greedy_synthesis.run()
}




fn append_s(clifford: &mut Array2<bool>, qubit: usize, num_qubits: usize) {
    let (x, mut z, mut p) = clifford.multi_slice_mut((
        s![.., qubit],
        s![.., num_qubits + qubit],
        s![.., 2 * num_qubits],
    ));

    azip!((p in &mut p, &x in &x, &z in &z)  *p ^= x & z);
    azip!((z in &mut z, &x in &x) *z ^= x);
}

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

fn append_cx(clifford: &mut Array2<bool>, qubit0: usize, qubit1: usize, num_qubits: usize) {
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
fn fix_phase(gate_seq: &mut CliffordGatesVec, target_clifford: ArrayView2<bool>, num_qubits: usize) {
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
    let smat = target_clifford.slice(s![.., ..2 * num_qubits]);
    let smat_inv = calc_inverse_matrix_inner(smat);

    // compute smat_inv * delta_phase
    let arr1 = smat_inv.map(|v| *v as usize);
    let vec2: Vec<usize> = delta_phase.into_iter().map(|v| v as usize).collect();
    let arr2 = Array1::from(vec2);
    let delta_phase_pre = arr1.dot(&arr2).map(|v| v % 2 == 1);

    // println!("delta_phase_pre:");
    // println!("{:?}", delta_phase_pre);

    for qubit in 0..num_qubits {
        if delta_phase_pre[qubit] && delta_phase_pre[qubit + num_qubits] {
            // println!("=> Adding Y-gate on {}", qubit);
            gate_seq.push((StandardGate::YGate, smallvec![Qubit(qubit as u32)]));
        } else if delta_phase_pre[qubit] {
            // println!("=> Adding Z-gate on {}", qubit);
            gate_seq.push((StandardGate::ZGate, smallvec![Qubit(qubit as u32)]));
        } else if delta_phase_pre[qubit + num_qubits] {
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
