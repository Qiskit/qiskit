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

    /// Evolving the single-qubit Pauli-Z with Z on qubit qbit.
    /// Returns the evolved Pauli and the sign.
    pub fn get_inverse_z(&self, qbit: usize) -> (bool, String) {
        let mut string = String::new();
        let mut pauli = vec![false; 2 * self.num_qubits];

        for i in 0..self.num_qubits {
            let x_bit = self.tableau[[i + self.num_qubits, qbit]];
            let z_bit = self.tableau[[i, qbit]];
            match (x_bit, z_bit) {
                (false, false) => {
                    string.push('I');
                }
                (true, false) => {
                    string.push('X');
                    pauli[i] = true;
                }
                (false, true) => {
                    string.push('Z');
                    pauli[i + self.num_qubits] = true;
                }
                (true, true) => {
                    string.push('Y');
                    pauli[i] = true;
                    pauli[i + self.num_qubits] = true;
                }
            }
        }

        let phase = compute_phase_product_pauli(self, &pauli);
        (phase, string)
    }
}

/// Computes the sign (either +1 or -1) when conjugating a Pauli by a Clifford
fn compute_phase_product_pauli(clifford: &Clifford, pauli: &[bool]) -> bool {
    let phase = pauli.iter().enumerate().fold(false, |acc, (j, &item)| {
        acc ^ (clifford.tableau[[j, 2 * clifford.num_qubits]] & item)
    });

    let mut ifact: u8 = (0..clifford.num_qubits)
        .filter(|&i| pauli[i] & pauli[i + clifford.num_qubits])
        .count() as u8
        % 4;

    for j in 0..clifford.num_qubits {
        let mut x = false;
        let mut z = false;
        for (i, &item) in pauli.iter().enumerate() {
            if item {
                let x1: bool = clifford.tableau[[i, j]];
                let z1: bool = clifford.tableau[[i, j + clifford.num_qubits]];

                match (x1, z1, x, z) {
                    (false, true, true, true)
                    | (true, false, false, true)
                    | (true, true, true, false) => {
                        ifact += 1;
                    }
                    (false, true, true, false)
                    | (true, false, true, true)
                    | (true, true, false, true) => {
                        ifact += 3;
                    }
                    _ => {}
                };
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
            for j in 0..2 * self.num_qubits + 1 {
                write!(f, "{} ", self.tableau[[i, j]] as u8)?;
            }
            writeln!(f)?;
        }
        writeln!(f)?;
        Ok(())
    }
}
