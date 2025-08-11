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
use std::fmt;

use ndarray::{azip, s, Array2};

/// Symplectic matrix.
/// Currently this class is internal to the synthesis library.
pub struct SymplecticMatrix {
    /// Number of qubits.
    pub num_qubits: usize,
    /// Matrix with dimensions (2 * num_qubits) x (2 * num_qubits).
    pub smat: Array2<bool>,
}

/// Clifford.
/// Currently this class has a very different functionality from Qiskit's
/// python-based Clifford class.
#[derive(Clone)]
pub struct Clifford {
    /// Number of qubits.
    pub num_qubits: usize,
    /// Matrix with dimensions (2 * num_qubits) x (2 * num_qubits + 1).
    pub tableau: Array2<bool>,
}

impl Clifford {
    /// Creates the identity Clifford on num_qubits
    pub fn identity(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            tableau: Array2::from_shape_fn((2 * num_qubits, 2 * num_qubits + 1), |(i, j)| i == j),
        }
    }

    /// Modifies the tableau in-place by appending S-gate
    pub fn append_s(&mut self, qubit: usize) {
        let (x, mut z, mut p) = self.tableau.multi_slice_mut((
            s![.., qubit],
            s![.., self.num_qubits + qubit],
            s![.., 2 * self.num_qubits],
        ));

        azip!((p in &mut p, &x in &x, &z in &z)  *p ^= x & z);
        azip!((z in &mut z, &x in &x) *z ^= x);
    }

    /// Modifies the tableau in-place by appending Sdg-gate
    #[allow(dead_code)]
    pub fn append_sdg(&mut self, qubit: usize) {
        let (x, mut z, mut p) = self.tableau.multi_slice_mut((
            s![.., qubit],
            s![.., self.num_qubits + qubit],
            s![.., 2 * self.num_qubits],
        ));

        azip!((p in &mut p, &x in &x, &z in &z)  *p ^= x & !z);
        azip!((z in &mut z, &x in &x) *z ^= x);
    }

    /// Modifies the tableau in-place by appending SX-gate
    pub fn append_sx(&mut self, qubit: usize) {
        let (mut x, z, mut p) = self.tableau.multi_slice_mut((
            s![.., qubit],
            s![.., self.num_qubits + qubit],
            s![.., 2 * self.num_qubits],
        ));

        azip!((p in &mut p, &x in &x, &z in &z)  *p ^= !x & z);
        azip!((&z in &z, x in &mut x) *x ^= z);
    }

    /// Modifies the tableau in-place by appending SXDG-gate
    pub fn append_sxdg(&mut self, qubit: usize) {
        let (mut x, z, mut p) = self.tableau.multi_slice_mut((
            s![.., qubit],
            s![.., self.num_qubits + qubit],
            s![.., 2 * self.num_qubits],
        ));

        azip!((p in &mut p, &x in &x, &z in &z)  *p ^= x & z);
        azip!((&z in &z, x in &mut x) *x ^= z);
    }

    /// Modifies the tableau in-place by appending H-gate
    pub fn append_h(&mut self, qubit: usize) {
        let (mut x, mut z, mut p) = self.tableau.multi_slice_mut((
            s![.., qubit],
            s![.., self.num_qubits + qubit],
            s![.., 2 * self.num_qubits],
        ));

        azip!((p in &mut p, &x in &x, &z in &z)  *p ^= x & z);
        azip!((x in &mut x, z in &mut z)  (*x, *z) = (*z, *x));
    }

    /// Modifies the tableau in-place by appending SWAP-gate
    pub fn append_swap(&mut self, qubit0: usize, qubit1: usize) {
        let (mut x0, mut z0, mut x1, mut z1) = self.tableau.multi_slice_mut((
            s![.., qubit0],
            s![.., self.num_qubits + qubit0],
            s![.., qubit1],
            s![.., self.num_qubits + qubit1],
        ));
        azip!((x0 in &mut x0, x1 in &mut x1)  (*x0, *x1) = (*x1, *x0));
        azip!((z0 in &mut z0, z1 in &mut z1)  (*z0, *z1) = (*z1, *z0));
    }

    /// Modifies the tableau in-place by appending CX-gate
    pub fn append_cx(&mut self, qubit0: usize, qubit1: usize) {
        let (x0, mut z0, mut x1, z1, mut p) = self.tableau.multi_slice_mut((
            s![.., qubit0],
            s![.., self.num_qubits + qubit0],
            s![.., qubit1],
            s![.., self.num_qubits + qubit1],
            s![.., 2 * self.num_qubits],
        ));
        azip!((p in &mut p, &x0 in &x0, &z0 in &z0, &x1 in &x1, &z1 in &z1) *p ^= (x1 ^ z0 ^ true) & z1 & x0);
        azip!((x1 in &mut x1, &x0 in &x0) *x1 ^= x0);
        azip!((z0 in &mut z0, &z1 in &z1) *z0 ^= z1);
    }

    /// Modifies the tableau in-place by appending W-gate.
    /// This is equivalent to an Sdg gate followed by an H gate.
    pub fn append_v(&mut self, qubit: usize) {
        let (mut x, mut z) = self
            .tableau
            .multi_slice_mut((s![.., qubit], s![.., self.num_qubits + qubit]));

        azip!((x in &mut x, z in &mut z) (*x, *z) = (*x ^ *z, *x));
    }

    /// Modifies the tableau in-place by appending CZ-gate
    pub fn append_cz(&mut self, i: usize, j: usize) {
        self.append_h(j);
        self.append_cx(i, j);
        self.append_h(j);
    }

    /// Modifies the tableau in-place by appending V-gate.
    /// This is equivalent to two V gates.
    pub fn append_w(&mut self, qubit: usize) {
        let (mut x, mut z) = self
            .tableau
            .multi_slice_mut((s![.., qubit], s![.., self.num_qubits + qubit]));

        azip!((x in &mut x, z in &mut z)  (*x, *z) = (*z, *x ^ *z));
    }

    #[inline]
    pub fn get_entry(&self, row: usize, col: usize) -> bool {
        self.tableau[[col, row]]
    }

    #[inline]
    pub fn get_phase(&self, row: usize) -> bool {
        self.tableau[[row, 2 * self.num_qubits]]
    }

    /// ToDo: not sure if this is correct or whether row & column need to be switched somewhere
    /// Get the inverse Z output of the tableau
    /// Get the inverse Z output of the tableau
    pub fn get_inverse_z(&self, qbit: usize) -> (bool, String) {
        let mut string = String::new();
        let mut as_vec_bool = vec![false; 2 * self.num_qubits];

        for i in 0..self.num_qubits {
            let x_bit = self.get_entry(qbit, i + self.num_qubits);
            let z_bit = self.get_entry(qbit, i);
            match (x_bit, z_bit) {
                (false, false) => {
                    string.push('I');
                }
                (true, false) => {
                    string.push('X');
                    as_vec_bool[i] = true;
                }
                (false, true) => {
                    string.push('Z');
                    as_vec_bool[i + self.num_qubits] = true;
                }
                (true, true) => {
                    string.push('Y');
                    as_vec_bool[i] = true;
                    as_vec_bool[i + self.num_qubits] = true;
                }
            }
        }

        let phase = compute_phase_product_pauli(self, &as_vec_bool);
        (phase, string)
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
fn compute_phase_product_pauli(tableau: &Clifford, vec: &[bool]) -> bool {
    let phase = vec
        .iter()
        .enumerate()
        .fold(false, |acc, (j, &item)| acc ^ (tableau.get_phase(j) & item));

    let mut ifact: u8 = (0..tableau.num_qubits)
        .filter(|&i| vec[i] & vec[i + tableau.num_qubits])
        .count() as u8
        % 4;

    for j in 0..tableau.num_qubits {
        let mut x = false;
        let mut z = false;
        for (i, &item) in vec.iter().enumerate() {
            if item {
                let x1: bool = tableau.get_entry(j, i);
                let z1: bool = tableau.get_entry(j + tableau.num_qubits, i);
                let entry = (x1, z1, x, z);
                if LOOKUP_0.contains(&entry) {
                    ifact += 1;
                } else if LOOKUP_1.contains(&entry) {
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

impl fmt::Debug for Clifford {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f)?;
        writeln!(f, "Tableau:")?;
        for i in 0..2 * self.num_qubits {
            for j in 0..2 * self.num_qubits {
                write!(f, "{} ", self.get_entry(i, j) as u8)?;
            }
            writeln!(f)?;
        }
        writeln!(f, "Phases:")?;
        for i in 0..2 * self.num_qubits {
            write!(f, "{} ", self.get_phase(i) as u8)?;
        }
        writeln!(f)?;
        Ok(())
    }
}
