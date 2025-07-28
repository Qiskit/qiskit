// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::clifford::clifford_circuit::CliffordCircuit;
use crate::clifford::pauli_like::PauliLike;
use crate::clifford::pauli_set::PauliSet;

fn compute_phase_product_pauli(pset0: &PauliSet, vec: &[bool]) -> bool {
    let mut phase = false;
    for (j, item) in vec.iter().enumerate().take(2 * pset0.n) {
        phase ^= pset0.get_phase(j) & item;
    }
    let mut ifact: u8 = 0;
    for i in 0..pset0.n {
        if vec[i] & vec[i + pset0.n] {
            ifact += 1;
        }
    }
    ifact %= 4;
    for j in 0..pset0.n {
        let mut x: bool = false;
        let mut z: bool = false;
        for (i, item) in vec.iter().enumerate().take(2 * pset0.n) {
            if *item {
                let x1: bool = pset0.get_entry(j, i);
                let z1: bool = pset0.get_entry(j + pset0.n, i);
                let entry = (x1, z1, x, z);
                if LOOKUP_0.contains(&entry) {
                    ifact += 1;
                }
                if LOOKUP_1.contains(&entry) {
                    ifact += 3;
                }
                x ^= x1;
                z ^= z1;
                ifact %= 4;
            }
        }
    }
    (((ifact % 4) >> 1) != 0) ^ phase
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tableau {
    pub logicals: PauliSet,
}

impl Tableau {
    /// Allocates a new Tableau representing the identity operator over `n` qubits
    pub fn new(n: usize) -> Self {
        let mut logicals = PauliSet::new(n);
        for i in 0..2 * n {
            // fn insert_vec_bool(&mut self, axis: &Vec<bool>, phase: bool) -> usize {
            let mut vecbool = vec![false; 2 * n];
            vecbool[i] = true;
            logicals.insert_vec_bool(&vecbool, false);
        }
        Tableau { logicals }
    }
    /// Build the Tableau corresponding to a Clifford circuit
    pub fn from_circuit(circuit: &CliffordCircuit) -> Self {
        let mut tab = Self::new(circuit.nqbits);
        tab.conjugate_with_circuit(circuit);
        tab
    }
    /// Build a Tableau from a PauliSet
    pub fn from_operators(logicals: &Vec<(bool, String)>) -> Self {
        if logicals.is_empty() {
            return Self::new(0);
        }
        let nqbits = logicals[0].1.len();
        let mut pset = PauliSet::new(nqbits);
        for (phase, string) in logicals {
            pset.insert(string, *phase);
        }
        Self { logicals: pset }
    }
    /// Returns the inverse Tableau
    pub fn adjoint(&self) -> Self {
        let mut new_logicals = PauliSet::new(self.logicals.n);
        for i in 0..self.logicals.n {
            let (_, string) = self.logicals.get_inverse_x(i);
            new_logicals.insert(&string, self.logicals.get_phase(i));
        }
        for i in 0..self.logicals.n {
            let (_, string) = self.logicals.get_inverse_z(i);
            new_logicals.insert(&string, self.logicals.get_phase(i + self.logicals.n));
        }
        let prod = self.clone()
            * Tableau {
                logicals: new_logicals.clone(),
            };
        for i in 0..2 * self.logicals.n {
            new_logicals.set_phase(i, new_logicals.get_phase(i) ^ prod.logicals.get_phase(i));
        }
        Self {
            logicals: new_logicals,
        }
    }

    pub fn get_inverse_z(&self, qbit: usize) -> (bool, String) {
        let (_, string) = self.logicals.get_inverse_z(qbit);
        let mut as_vec_bool = vec![false; 2 * self.logicals.n];
        for qbit in 0..self.logicals.n {
            match string.chars().nth(qbit).unwrap() {
                'X' => {
                    as_vec_bool[qbit] = true;
                }
                'Y' => {
                    as_vec_bool[qbit] = true;
                    as_vec_bool[qbit + self.logicals.n] = true;
                }
                'Z' => {
                    as_vec_bool[qbit + self.logicals.n] = true;
                }
                _ => {}
            }
        }
        let phase = compute_phase_product_pauli(&self.logicals, &as_vec_bool);
        (phase, string)
    }
    pub fn get_inverse_x(&self, qbit: usize) -> (bool, String) {
        let (_, string) = self.logicals.get_inverse_x(qbit);
        let mut as_vec_bool = vec![false; 2 * self.logicals.n];
        for qbit in 0..self.logicals.n {
            match string.chars().nth(qbit).unwrap() {
                'X' => {
                    as_vec_bool[qbit] = true;
                }
                'Y' => {
                    as_vec_bool[qbit] = true;
                    as_vec_bool[qbit + self.logicals.n] = true;
                }
                'Z' => {
                    as_vec_bool[qbit + self.logicals.n] = true;
                }
                _ => {}
            }
        }
        let phase = compute_phase_product_pauli(&self.logicals, &as_vec_bool);
        (phase, string)
    }
}

impl PauliLike for Tableau {
    fn h(&mut self, i: usize) {
        self.logicals.h(i);
    }

    fn s(&mut self, i: usize) {
        self.logicals.s(i);
    }

    fn sd(&mut self, i: usize) {
        self.logicals.sd(i);
    }

    fn sqrt_x(&mut self, i: usize) {
        self.logicals.sqrt_x(i);
    }

    fn sqrt_xd(&mut self, i: usize) {
        self.logicals.sqrt_xd(i);
    }

    fn cnot(&mut self, i: usize, j: usize) {
        self.logicals.cnot(i, j);
    }
}

const LOOKUP_0: [(bool, bool, bool, bool); 3] = [
    (false, true, true, true),
    (true, false, false, true),
    (true, true, true, false),
];

const LOOKUP_1: [(bool, bool, bool, bool); 3] = [
    (false, true, true, false),
    (true, false, true, true),
    (true, true, false, true),
];

impl std::ops::Mul<Tableau> for Tableau {
    type Output = Tableau;
    fn mul(self, rhs: Tableau) -> Self::Output {
        assert_eq!(self.logicals.n, rhs.logicals.n);
        let mut new_tableau = Tableau::new(self.logicals.n);
        for i in 0..2 * self.logicals.n {
            let (mut phase, col) = rhs.logicals.get_as_vec_bool(i);
            for (j, item) in col.iter().enumerate().take(2 * self.logicals.n) {
                phase ^= self.logicals.get_phase(j) & item;
            }
            new_tableau.logicals.set_phase(i, phase);
        }
        let mut ifacts = rhs.logicals.get_i_factors();
        for (k, item) in ifacts.iter_mut().enumerate().take(2 * self.logicals.n) {
            for j in 0..self.logicals.n {
                let mut x: bool = false;
                let mut z: bool = false;
                for i in 0..2 * self.logicals.n {
                    if rhs.logicals.get_entry(i, k) {
                        let x1: bool = self.logicals.get_entry(j, i);
                        let z1: bool = self.logicals.get_entry(j + self.logicals.n, i);
                        let entry = (x1, z1, x, z);
                        if LOOKUP_0.contains(&entry) {
                            *item += 1;
                        }
                        if LOOKUP_1.contains(&entry) {
                            *item += 3;
                        }
                        x ^= x1;
                        z ^= z1;
                        *item %= 4;
                    }
                }
            }
            *item %= 4;
        }
        let p: Vec<bool> = ifacts.into_iter().map(|v| 0 != ((v % 4) >> 1)).collect();
        for (i, ph) in p.iter().enumerate() {
            new_tableau
                .logicals
                .set_phase(i, new_tableau.logicals.get_phase(i) ^ ph);
        }

        for i in 0..2 * self.logicals.n {
            for j in 0..2 * self.logicals.n {
                let (_, col) = rhs.logicals.get_as_vec_bool(j);
                new_tableau
                    .logicals
                    .set_raw_entry(i, j, self.logicals.and_row_acc(i, &col));
            }
        }
        new_tableau
    }
}
