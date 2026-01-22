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

use crate::sparse_observable::BitTerm;
use ndarray::{Array2, azip, s};
use qiskit_circuit::Qubit;

/// Symplectic matrix.
pub struct SymplecticMatrix {
    /// Number of qubits.
    pub num_qubits: usize,
    /// Matrix with dimensions (2 * num_qubits) x (2 * num_qubits).
    pub smat: Array2<bool>,
}

/// Clifford.
/// Currently this class offers a reduced functionality of Qiskit's
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

    /// Modifies the tableau in-place by appending CZ-gate
    pub fn append_cz(&mut self, qubit0: usize, qubit1: usize) {
        let (x0, mut z0, x1, mut z1, mut p) = self.tableau.multi_slice_mut((
            s![.., qubit0],
            s![.., self.num_qubits + qubit0],
            s![.., qubit1],
            s![.., self.num_qubits + qubit1],
            s![.., 2 * self.num_qubits],
        ));
        azip!((p in &mut p, &x0 in &x0, &z0 in &z0, &x1 in &x1, &z1 in &z1) *p ^= x0 & x1 & (z0 ^ z1));
        azip!((z1 in &mut z1, &x0 in &x0) *z1 ^= x0);
        azip!((z0 in &mut z0, &x1 in &x1) *z0 ^= x1);
    }

    /// Modifies the tableau in-place by appending CY-gate
    /// (todo: rewrite using native tableau manipulations)
    pub fn append_cy(&mut self, qubit0: usize, qubit1: usize) {
        self.append_sdg(qubit1);
        self.append_cx(qubit0, qubit1);
        self.append_s(qubit1);
    }

    /// Modifies the tableau in-place by appending X-gate
    pub fn append_x(&mut self, qubit: usize) {
        let (z, mut p) = self
            .tableau
            .multi_slice_mut((s![.., self.num_qubits + qubit], s![.., 2 * self.num_qubits]));

        azip!((p in &mut p, &z in &z)  *p ^= z);
    }

    /// Modifies the tableau in-place by appending Z-gate
    pub fn append_z(&mut self, qubit: usize) {
        let (x, mut p) = self
            .tableau
            .multi_slice_mut((s![.., qubit], s![.., 2 * self.num_qubits]));
        azip!((p in &mut p, &x in &x)  *p ^= x);
    }

    /// Modifies the tableau in-place by appending Y-gate
    pub fn append_y(&mut self, qubit: usize) {
        let (x, z, mut p) = self.tableau.multi_slice_mut((
            s![.., qubit],
            s![.., self.num_qubits + qubit],
            s![.., 2 * self.num_qubits],
        ));
        azip!((p in &mut p, &x in &x, &z in &z)  *p ^= x ^ z);
    }

    /// Modifies the tableau in-place by appending iSWAP-gate
    /// (todo: rewrite using native tableau manipulations)
    pub fn append_iswap(&mut self, qubit0: usize, qubit1: usize) {
        self.append_s(qubit0);
        self.append_s(qubit1);
        self.append_h(qubit0);
        self.append_cx(qubit0, qubit1);
        self.append_cx(qubit1, qubit0);
        self.append_h(qubit1);
    }

    /// Modifies the tableau in-place by appending ECR-gate
    /// (todo: rewrite using native tableau manipulations)
    pub fn append_ecr(&mut self, qubit0: usize, qubit1: usize) {
        self.append_s(qubit0);
        self.append_sx(qubit1);
        self.append_cx(qubit0, qubit1);
        self.append_x(qubit0);
    }

    /// Modifies the tableau in-place by appending DCX-gate
    /// (todo: rewrite using native tableau manipulations)
    pub fn append_dcx(&mut self, qubit0: usize, qubit1: usize) {
        self.append_cx(qubit0, qubit1);
        self.append_cx(qubit1, qubit0);
    }

    /// Modifies the tableau in-place by appending V-gate.
    /// This is equivalent to an Sdg gate followed by an H gate.
    pub fn append_v(&mut self, qubit: usize) {
        let (mut x, mut z) = self
            .tableau
            .multi_slice_mut((s![.., qubit], s![.., self.num_qubits + qubit]));

        azip!((x in &mut x, z in &mut z) (*x, *z) = (*x ^ *z, *x));
    }

    /// Modifies the tableau in-place by appending W-gate.
    /// This is equivalent to two V gates.
    pub fn append_w(&mut self, qubit: usize) {
        let (mut x, mut z) = self
            .tableau
            .multi_slice_mut((s![.., qubit], s![.., self.num_qubits + qubit]));

        azip!((x in &mut x, z in &mut z)  (*x, *z) = (*z, *x ^ *z));
    }

    /// Evolving the single-qubit Pauli-Z with Z on qubit qbit.
    /// Returns the evolved Pauli in the sparse format: (bool, bit_terms, indices)
    /// This is typically used for constructing a [`SparseObservable`] from the return.
    pub fn get_inverse_z(&self, qbit: usize) -> (bool, Vec<BitTerm>, Vec<Qubit>) {
        let mut bit_terms = Vec::with_capacity(self.num_qubits);
        let mut pauli_indices = Vec::<usize>::with_capacity(2 * self.num_qubits);
        // Compute the y-count to avoid recomputing it later
        let mut pauli_y_count: u32 = 0;

        let indices = (0..self.num_qubits)
            .filter_map(|i| {
                let z_bit = self.tableau[[i, qbit]];
                let x_bit = self.tableau[[i + self.num_qubits, qbit]];
                match [z_bit, x_bit] {
                    [true, true] => {
                        pauli_y_count += 1;
                        bit_terms.push(BitTerm::Y);
                        pauli_indices.push(i);
                        pauli_indices.push(i + self.num_qubits);
                        Some(Qubit::new(i))
                    }
                    [false, true] => {
                        bit_terms.push(BitTerm::X);
                        pauli_indices.push(i);
                        Some(Qubit::new(i))
                    }
                    [true, false] => {
                        bit_terms.push(BitTerm::Z);
                        pauli_indices.push(i + self.num_qubits);
                        Some(Qubit::new(i))
                    }
                    [false, false] => None,
                }
            })
            .collect();
        let phase = compute_phase_product_pauli(self, &pauli_indices, pauli_y_count);

        (phase, bit_terms, indices)
    }

    /// Evolving the single-qubit Pauli-Z with Z on qubit qbit.
    /// Returns the evolved Pauli in the sparse format: (bool, pauli_z, pauli_x, indices)
    /// This is typically used for constructing a [`PauliProductMeasurement`] from the return
    pub fn get_inverse_z_for_measurement(
        &self,
        qbit: usize,
    ) -> (bool, Vec<bool>, Vec<bool>, Vec<Qubit>) {
        let mut z = Vec::with_capacity(self.num_qubits);
        let mut x = Vec::with_capacity(self.num_qubits);
        let mut indices = Vec::with_capacity(self.num_qubits);
        let mut pauli_indices = Vec::<usize>::with_capacity(2 * self.num_qubits);
        // Compute the y-count to avoid recomputing it later
        let mut pauli_y_count: u32 = 0;
        for i in 0..self.num_qubits {
            let z_bit = self.tableau[[i, qbit]];
            let x_bit = self.tableau[[i + self.num_qubits, qbit]];
            if z_bit || x_bit {
                z.push(z_bit);
                x.push(x_bit);
                indices.push(Qubit::new(i));
                if x_bit {
                    pauli_indices.push(i);
                }
                if z_bit {
                    pauli_indices.push(i + self.num_qubits);
                }
                pauli_y_count += (x_bit && z_bit) as u32;
            }
        }
        let phase = compute_phase_product_pauli(self, &pauli_indices, pauli_y_count);

        (phase, z, x, indices)
    }
}

/// Computes the sign (either +1 or -1) when conjugating a Pauli by a Clifford
fn compute_phase_product_pauli(
    clifford: &Clifford,
    pauli_indices: &[usize],
    pauli_y_count: u32,
) -> bool {
    let phase = pauli_indices.iter().fold(false, |acc, &pauli_index| {
        acc ^ (clifford.tableau[[pauli_index, 2 * clifford.num_qubits]])
    });

    let mut ifact: u8 = pauli_y_count as u8 % 4;

    for j in 0..clifford.num_qubits {
        let mut x = false;
        let mut z = false;
        for &pauli_index in pauli_indices.iter() {
            let x1: bool = clifford.tableau[[pauli_index, j]];
            let z1: bool = clifford.tableau[[pauli_index, j + clifford.num_qubits]];

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
