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
use fixedbitset::FixedBitSet;
use ndarray::Array2;
use qiskit_circuit::Qubit;

/// Symplectic matrix.
pub struct SymplecticMatrix {
    /// Number of qubits.
    pub num_qubits: usize,
    /// Matrix with dimensions (2 * num_qubits) x (2 * num_qubits).
    pub smat: Array2<bool>,
}

/// SIMD accelerated Clifford.
///
/// Currently this class offers a reduced functionality of the python-based
/// Clifford class.
#[derive(Clone)]
pub struct Clifford {
    /// Number of qubits.
    pub num_qubits: usize,
    /// Matrix with dimensions (2 * num_qubits) x (2 * num_qubits + 1).
    pub tableau: Vec<FixedBitSet>,
}

impl Clifford {
    /// Creates the identity Clifford on num_qubits
    pub fn identity(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            tableau: (0..2 * num_qubits + 1)
                .map(|i| {
                    let mut row = FixedBitSet::with_capacity(2 * num_qubits);
                    // SAFETY: We know row is large enough since it's larger than the range
                    // i is from
                    unsafe {
                        row.insert_unchecked(i);
                    }
                    row
                })
                .collect(),
        }
    }

    fn get_phase_mut(&mut self) -> &mut FixedBitSet {
        self.tableau.get_mut(2 * self.num_qubits).unwrap()
    }

    pub fn get_phase(&self) -> &FixedBitSet {
        self.tableau.get(2 * self.num_qubits).unwrap()
    }

    pub fn get_z(&self, qubit: usize) -> &FixedBitSet {
        self.tableau.get(self.num_qubits + qubit).unwrap()
    }

    pub fn get_z_mut(&mut self, qubit: usize) -> &mut FixedBitSet {
        self.tableau.get_mut(self.num_qubits + qubit).unwrap()
    }

    /// Modifies the tableau in-place by appending S-gate
    pub fn append_s(&mut self, qubit: usize) {
        let x_and_z = &self.tableau[qubit] & self.get_z(qubit);
        *self.get_phase_mut() ^= x_and_z;
        let xor = self.get_z(qubit) ^ &self.tableau[qubit];
        *self.get_z_mut(qubit) = xor;
    }

    /// Modifies the tableau in-place by appending Sdg-gate
    pub fn append_sdg(&mut self, qubit: usize) {
        let x_and_not_z = if let Some(x) = self.tableau.get(qubit) {
            let mut not_z = self.get_z(qubit).clone();
            not_z.toggle_range(..);
            x & &not_z
        } else {
            unreachable!();
        };
        *self.get_phase_mut() ^= x_and_not_z;
        let xor = &self.tableau[qubit] ^ self.get_z(qubit);
        *self.get_z_mut(qubit) = xor;
    }

    /// Modifies the tableau in-place by appending SX-gate
    pub fn append_sx(&mut self, qubit: usize) {
        let not_x_and_z = if let Some(x) = self.tableau.get(qubit) {
            let z = self.get_z(qubit);
            let mut not_x = x.clone();
            not_x.toggle_range(..);
            &not_x & z
        } else {
            unreachable!();
        };
        *self.get_phase_mut() ^= not_x_and_z;
        let xor = &self.tableau[qubit] ^ self.get_z(qubit);
        self.tableau[qubit] = xor;
    }

    /// Modifies the tableau in-place by appending SXDG-gate
    pub fn append_sxdg(&mut self, qubit: usize) {
        let x_and_z = &self.tableau[qubit] & self.get_z(qubit);
        *self.get_phase_mut() ^= x_and_z;
        let xor = &self.tableau[qubit] ^ self.get_z(qubit);
        self.tableau[qubit] = xor;
    }

    /// Modifies the tableau in-place by appending H-gate
    pub fn append_h(&mut self, qubit: usize) {
        let x_and_z = if let Some(x) = self.tableau.get(qubit) {
            let z = self.get_z(qubit);
            x & z
        } else {
            unreachable!();
        };
        *self.get_phase_mut() ^= x_and_z;
        self.tableau.swap(qubit, self.num_qubits + qubit);
    }

    /// Modifies the tableau in-place by appending SWAP-gate
    pub fn append_swap(&mut self, qubit0: usize, qubit1: usize) {
        self.tableau.swap(qubit0, qubit1);
        self.tableau
            .swap(self.num_qubits + qubit0, self.num_qubits + qubit1);
    }

    /// Modifies the tableau in-place by appending CX-gate
    pub fn append_cx(&mut self, qubit0: usize, qubit1: usize) {
        let val = if let Some(x0) = self.tableau.get(qubit0) {
            let z0 = self.get_z(qubit0);
            let x1 = &self.tableau[qubit1];
            let z1 = self.get_z(qubit1);

            let mut x1_xor_z0 = x1 ^ z0;
            x1_xor_z0.toggle_range(..);
            let tmp = &x1_xor_z0 & z1;
            &tmp & x0
        } else {
            unreachable!();
        };
        *self.get_phase_mut() ^= val;
        let xor_x = &self.tableau[qubit1] ^ &self.tableau[qubit0];
        let xor_z = self.get_z(qubit0) ^ self.get_z(qubit1);
        self.tableau[qubit1] = xor_x;
        *self.get_z_mut(qubit0) = xor_z;
    }

    /// Modifies the tableau in-place by appending CZ-gate
    pub fn append_cz(&mut self, qubit0: usize, qubit1: usize) {
        let val = if let Some(x0) = self.tableau.get(qubit0) {
            let z0 = self.get_z(qubit0);
            let x1 = &self.tableau[qubit1];
            let z1 = self.get_z(qubit1);
            let z0_xor_z1 = z0 ^ z1;
            &(x0 & x1) & &z0_xor_z1
        } else {
            unreachable!();
        };
        *self.get_phase_mut() ^= val;
        let xor_z1_x0 = self.get_z(qubit1) ^ &self.tableau[qubit0];
        let xor_z0_x1 = self.get_z(qubit0) ^ &self.tableau[qubit1];
        *self.get_z_mut(qubit1) = xor_z1_x0;
        *self.get_z_mut(qubit0) = xor_z0_x1;
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
        let xor = self.get_phase() ^ self.get_z(qubit);
        *self.get_phase_mut() = xor;
    }

    /// Modifies the tableau in-place by appending Z-gate
    pub fn append_z(&mut self, qubit: usize) {
        let xor = self.get_phase() ^ &self.tableau[qubit];
        *self.get_phase_mut() = xor;
    }

    /// Modifies the tableau in-place by appending Y-gate
    pub fn append_y(&mut self, qubit: usize) {
        let xor = &self.tableau[qubit] ^ self.get_z(qubit);
        *self.get_phase_mut() ^= xor;
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
        let xor = &self.tableau[qubit] ^ self.get_z(qubit);
        self.tableau.swap(qubit, self.num_qubits + qubit);
        self.tableau[qubit] = xor;
    }

    /// Modifies the tableau in-place by appending W-gate.
    /// This is equivalent to two V gates.
    pub fn append_w(&mut self, qubit: usize) {
        let xor = &self.tableau[qubit] ^ self.get_z(qubit);
        self.tableau.swap(qubit, self.num_qubits + qubit);
        *self.get_z_mut(qubit) = xor;
    }

    /// Evolving the single-qubit Pauli-Z with Z on qubit qbit.
    /// Returns the evolved Pauli in the sparse format: (sign, paulis, indices).
    pub fn get_inverse_z(&self, qbit: usize) -> (bool, Vec<BitTerm>, Vec<Qubit>) {
        // Potentially overallocated, but this is temporary in the only use from litinski transform.
        let mut bit_terms = Vec::with_capacity(self.num_qubits);
        let mut pauli_indices = Vec::<usize>::with_capacity(2 * self.num_qubits);
        // Compute the y-count to avoid recomputing it later
        let mut pauli_y_count: u32 = 0;

        let indices = (0..self.num_qubits)
            .filter_map(|i| {
                let x_bit = self.tableau[qbit][i + self.num_qubits];
                let z_bit = self.tableau[qbit][i];
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
            let z_bit = self.tableau[qbit][i];
            let x_bit = self.tableau[qbit][i + self.num_qubits];
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
        acc ^ (clifford.tableau[2 * clifford.num_qubits][pauli_index])
    });

    let mut ifact: u8 = pauli_y_count as u8 % 4;

    for j in 0..clifford.num_qubits {
        let mut x = false;
        let mut z = false;
        for &pauli_index in pauli_indices.iter() {
            let x1: bool = clifford.tableau[j][pauli_index];
            let z1: bool = clifford.tableau[j + clifford.num_qubits][pauli_index];

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
                write!(f, "{} ", self.tableau[j][i] as u8)?;
            }
            writeln!(f)?;
        }
        writeln!(f)?;
        Ok(())
    }
}
