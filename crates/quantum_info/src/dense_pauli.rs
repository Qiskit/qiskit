// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::clifford::Clifford;
use fixedbitset::FixedBitSet;
use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64Mcg;
use thiserror::Error;

/// A dense Pauli operator class.
///
/// The `zx_phase` is a common bookkeeping convention that allows for faster Pauli
/// commutation and composition checks. This `zx_phase` and the group phase
/// (that is, the actual algebraic coefficient :math:`i^q` in front of the operator)
/// are related as `zx phase = (group phase + number of Y-terms) modulo 4`.
#[derive(Clone, Debug, PartialEq)]
pub struct DensePauli {
    /// z-component
    pub pauli_z: FixedBitSet,
    /// x-component
    pub pauli_x: FixedBitSet,
    /// zx-phase
    pub zx_phase: u8,
}

#[derive(Error, Debug)]
pub enum DensePauliError {
    #[error(
        "`DensePauli::from_sparse_bool` requires `z`, `x` and `indices` to have the same length"
    )]
    InvalidSparseInput,
    #[error("Invalid Pauli label: {0}")]
    InvalidLabel(String),
    #[error("Both Paulis are required to have the same length")]
    DifferentLengths,
    #[error(
        "Evolving `DensePauli` under a `Clifford` requires both to have the same number of qubits."
    )]
    DifferentLengthsEvolve,
}

impl DensePauli {
    /// Return the identity Pauli operator on ``num_qubits`` qubits.
    pub fn identity(num_qubits: usize) -> Self {
        let pauli_z = FixedBitSet::with_capacity(num_qubits);
        let pauli_x = FixedBitSet::with_capacity(num_qubits);
        let zx_phase = 0u8;
        DensePauli {
            pauli_z,
            pauli_x,
            zx_phase,
        }
    }

    /// Construct a dense Pauli operator from a sparse boolean representation.
    ///
    /// # Arguments
    ///
    /// * `num_qubits`: Number of qubits.
    /// * `x`: Boolean slice representing x-terms.
    /// * `z`: Boolean slice representing z-terms.
    /// * `indices`: Qubit indices corresponding to `x` and `z`.
    /// * `phase`: The phase of the Pauli operator, encoded modulo 4.
    /// * `is_group_phase`: If `true`, `phase` is interpreted as a group phase
    ///   and is converted to the internal ZX-phase representation.
    ///
    /// # Errors
    ///
    /// Returns [`DensePauliError::InvalidSparseInput`] if
    /// `z`, `x` or `indices` do not all have the same length.
    pub fn from_sparse_bool(
        num_qubits: usize,
        z: &[bool],
        x: &[bool],
        indices: &[u32],
        phase: u8,
        is_group_phase: bool,
    ) -> Result<Self, DensePauliError> {
        if (z.len() != indices.len()) || (x.len() != indices.len()) {
            return Err(DensePauliError::InvalidSparseInput);
        }

        let mut pauli_z = FixedBitSet::with_capacity(num_qubits);
        let mut pauli_x = FixedBitSet::with_capacity(num_qubits);
        let mut num_ys = 0;

        for (i, &q) in indices.iter().enumerate() {
            pauli_z.set(q as usize, z[i]);
            pauli_x.set(q as usize, x[i]);
            if x[i] & z[i] {
                num_ys += 1;
            }
        }

        let zx_phase = if is_group_phase {
            (phase + num_ys) & 3
        } else {
            phase
        };
        Ok(DensePauli {
            pauli_z,
            pauli_x,
            zx_phase,
        })
    }

    /// Construct a dense Pauli operator from a string label.
    ///
    /// # Arguments
    ///
    /// * `label`: A Pauli label string consisting of an optional minus sign, followed by
    ///   an optional `i` factor, followed by a sequence of `'I'`, `'X'`, `'Y'`, or `'Z'` characters.
    ///
    /// # Errors
    ///
    /// Returns [`DensePauliError::InvalidLabel`] if the label is invalid.
    ///
    /// .. note::
    ///
    ///     In Qiskit convention, the label is represented right-to-left,
    ///     for example "-iXIZY" is interpreted as `'X'` on qubit `3`, followed by `'I'` on qubit `2`,
    ///     followed by `'Z'` on qubit `1`, and finally by `'Y'` on qubit `0`.
    pub fn from_label(label: &str) -> Result<Self, DensePauliError> {
        let mut s = label;
        let mut zx_phase = 0u8;
        if let Some(r) = s.strip_prefix('-') {
            s = r;
            zx_phase = (zx_phase + 2) & 3;
        }
        if let Some(r) = s.strip_prefix('i') {
            s = r;
            zx_phase = (zx_phase + 1) & 3;
        }

        let num_qubits = s.len();
        let mut pauli_z = FixedBitSet::with_capacity(num_qubits);
        let mut pauli_x = FixedBitSet::with_capacity(num_qubits);

        for (i, c) in s.chars().rev().enumerate() {
            match c {
                'I' => {
                    pauli_z.set(i, false);
                    pauli_x.set(i, false);
                }
                'X' => {
                    pauli_z.set(i, false);
                    pauli_x.set(i, true);
                }
                'Z' => {
                    pauli_z.set(i, true);
                    pauli_x.set(i, false);
                }
                'Y' => {
                    pauli_z.set(i, true);
                    pauli_x.set(i, true);
                    zx_phase = (zx_phase + 1) & 3;
                }
                _ => {
                    return Err(DensePauliError::InvalidLabel(label.to_string()));
                }
            }
        }

        Ok(DensePauli {
            pauli_z,
            pauli_x,
            zx_phase,
        })
    }

    /// Construct a random dense Pauli operator on ``num_qubits`` qubits.
    ///     
    /// # Arguments
    ///
    /// * `num_qubits`: Number of qubits.
    /// * `seed`: Random seed used for generation.
    pub fn from_random(num_qubits: usize, seed: u64) -> Self {
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let mut pauli_z = FixedBitSet::with_capacity(num_qubits);
        let mut pauli_x = FixedBitSet::with_capacity(num_qubits);

        for i in 0..num_qubits {
            let val = rng.random_bool(0.5);
            pauli_z.set(i, val);
        }
        for i in 0..num_qubits {
            let val = rng.random_bool(0.5);
            pauli_x.set(i, val);
        }
        let zx_phase = rng.random::<u8>() & 3;

        DensePauli {
            pauli_z,
            pauli_x,
            zx_phase,
        }
    }

    /// Return the number of qubits in the Pauli.
    pub fn num_qubits(&self) -> usize {
        self.pauli_z.len()
    }

    /// Return the number of Y-terms in the Pauli.
    pub fn count_y(&self) -> u8 {
        let num_qubits = self.num_qubits();
        let mut cnt_y = 0;
        for i in 0..num_qubits {
            if self.pauli_x[i] & self.pauli_z[i] {
                cnt_y += 1;
            }
        }
        cnt_y
    }

    /// Construct a string representation for the Pauli operator.
    ///
    /// .. note::
    ///
    ///     In Qiskit convention, the label is represented right-to-left,
    ///     for example "-iXIZY" is interpreted as `'X'` on qubit `3`, followed by `'I'` on qubit `2`,
    ///     followed by `'Z'` on qubit `1`, and finally by `'Y'` on qubit `0`.
    pub fn to_label(&self) -> String {
        let mut s: String = Default::default();
        let n = self.num_qubits();

        let mut group_phase = self.zx_phase;
        for i in (0..n).rev() {
            match (self.pauli_z[i], self.pauli_x[i]) {
                (false, false) => {
                    s.push('I');
                }
                (true, false) => {
                    s.push('Z');
                }
                (false, true) => {
                    s.push('X');
                }
                (true, true) => {
                    s.push('Y');
                    group_phase = group_phase.wrapping_sub(1) & 3;
                }
            }
        }

        match group_phase {
            0 => {}
            1 => {
                s = String::from("i") + &s;
            }
            2 => {
                s = String::from("-") + &s;
            }
            3 => {
                s = String::from("-i") + &s;
            }
            _ => {
                unreachable!("The group phase is always kept reduced modulo 4.")
            }
        }

        s
    }

    /// Convert a dense Pauli operator to a sparse boolean representation,
    /// removing identity ('I') terms.
    ///
    /// # Arguments
    ///
    /// * `is_group_phase`: If `true`, the returned phase is a group phase. Otherwise,
    ///   it is the ZX-phase.
    ///
    /// # Returns
    ///
    /// A tuple containing x-terms, z-terms, qubits indices and phase. Qubit indices
    /// are sorted.
    pub fn to_sparse_bool(&self, is_group_phase: bool) -> (Vec<bool>, Vec<bool>, Vec<u32>, u8) {
        let num_qubits = self.num_qubits();
        let mut pauli_z_sparse: Vec<bool> = Vec::with_capacity(num_qubits);
        let mut pauli_x_sparse: Vec<bool> = Vec::with_capacity(num_qubits);
        let mut out_indices: Vec<u32> = Vec::with_capacity(num_qubits);

        for i in 0..num_qubits {
            if self.pauli_x[i] || self.pauli_z[i] {
                pauli_z_sparse.push(self.pauli_z[i]);
                pauli_x_sparse.push(self.pauli_x[i]);
                out_indices.push(i as u32);
            }
        }
        let phase = if is_group_phase {
            self.zx_phase.wrapping_sub(self.count_y()) & 3
        } else {
            self.zx_phase
        };

        (pauli_z_sparse, pauli_x_sparse, out_indices, phase)
    }

    /// Return `true` if `self` and `other` commute.
    ///
    /// This method **does not check** that the two Paulis act on the same
    /// number of qubits. Calling it with mismatched lengths results in an
    /// **undefined logical behavior**.
    ///
    /// See [`DensePauli::commutes`] for a safe version that validates input
    /// lengths.
    #[inline(always)]
    pub fn commutes_unchecked(&self, other: &DensePauli) -> bool {
        let num_qubits = self.num_qubits();
        let mut parity = false;
        for i in 0..num_qubits {
            parity ^= (self.pauli_z[i] & other.pauli_x[i]) ^ (self.pauli_x[i] & other.pauli_z[i]);
        }
        !parity
    }

    /// Return `true` if `self` and `other` commute.
    ///
    /// This method checks that both Paulis act on the same number of qubits.
    /// If they do not, an error is returned.
    ///
    /// See [`DensePauli::commutes_unchecked`] for a faster version without
    /// input validation.
    ///
    /// # Errors
    ///
    /// Returns [`DensePauliError::DifferentLengths`] if the two Paulis
    /// have different lengths.
    #[inline(always)]
    pub fn commutes(&self, other: &DensePauli) -> Result<bool, DensePauliError> {
        if self.num_qubits() != other.num_qubits() {
            return Err(DensePauliError::DifferentLengths);
        }
        Ok(self.commutes_unchecked(other))
    }

    /// Compose ``self`` and ``other``, modifying the current Pauli in-place.
    ///
    /// Computing ``self.compose_with_unchecked(other)`` is equivalent to
    /// computing ``other * self``.
    ///
    /// This method **does not check** that the two Paulis act on the same
    /// number of qubits. Calling it with mismatched lengths results in an
    /// **undefined logical behavior**.
    ///
    /// See [`DensePauli::compose_with`] for a safe version that validates input
    /// lengths.
    #[inline(always)]
    pub fn compose_with_unchecked(&mut self, other: &DensePauli) {
        let num_qubits = self.num_qubits();
        self.zx_phase = (self.zx_phase + other.zx_phase) & 3;
        for i in 0..num_qubits {
            if self.pauli_x[i] & other.pauli_z[i] {
                self.zx_phase = (self.zx_phase + 2) & 3;
            }
        }
        self.pauli_z ^= &other.pauli_z;
        self.pauli_x ^= &other.pauli_x;
    }

    /// Compose ``self`` and ``other``, modifying the current Pauli in-place.
    ///
    /// Computing ``self.compose_with(other)`` is equivalent to computing
    /// ``other * self``.
    ///
    /// This method checks that both Paulis act on the same number of qubits.
    /// If they do not, an error is returned.
    ///
    /// # Errors
    ///
    /// Returns [`DensePauliError::DifferentLengths`] if the two Paulis
    /// have different lengths.
    #[inline(always)]
    pub fn compose_with(&mut self, other: &DensePauli) -> Result<(), DensePauliError> {
        if self.num_qubits() != other.num_qubits() {
            return Err(DensePauliError::DifferentLengths);
        }
        self.compose_with_unchecked(other);
        Ok(())
    }

    /// Compose ``self`` and ``other``, returning the result in a new Pauli.
    ///
    /// Computing ``self.compose_unchecked(other)`` is equivalent to computing
    /// ``other * self``.
    ///
    /// This method **does not check** that the two Paulis act on the same
    /// number of qubits. Calling it with mismatched lengths results in an
    /// **undefined logical behavior**.
    ///
    /// See [`DensePauli::compose`] for a safe version that validates input
    /// lengths.
    #[inline(always)]
    pub fn compose_unchecked(&self, other: &DensePauli) -> DensePauli {
        let mut zx_phase = (self.zx_phase + other.zx_phase) & 3;
        let num_qubits = self.num_qubits();
        for i in 0..num_qubits {
            if self.pauli_x[i] && other.pauli_z[i] {
                zx_phase = (zx_phase + 2) & 3;
            }
        }

        let pauli_z = &self.pauli_z ^ &other.pauli_z;
        let pauli_x = &self.pauli_x ^ &other.pauli_x;
        DensePauli {
            pauli_z,
            pauli_x,
            zx_phase,
        }
    }

    /// Compose ``self`` and ``other``, returning the result in a new Pauli.
    ///
    /// Computing ``self.compose(other)`` is equivalent to computing ``other * self``.
    ///
    /// This method checks that both Paulis act on the same number of qubits.
    /// If they do not, an error is returned.
    ///
    /// # Errors
    ///
    /// Returns [`DensePauliError::DifferentLengths`] if the two Paulis
    /// have different lengths.
    #[inline(always)]
    pub fn compose(&self, other: &DensePauli) -> Result<DensePauli, DensePauliError> {
        if self.num_qubits() != other.num_qubits() {
            return Err(DensePauliError::DifferentLengths);
        }
        Ok(self.compose_unchecked(other))
    }
}

/// Evolve a Pauli :math:`P` by a Clifford :math:`C`, using the Heisenberg picture,
/// namely compute :math:`C^\dagger P C`.
///
/// # Arguments
///
/// * `pauli`: The pauli to evolve.
/// * `cliff`: The Clifford to evolve by.
///
/// # Errors
///
/// Returns [`DensePauliError::DifferentLengthsEvolve`] if `pauli` and `clifford`
/// are not defined for the same number of qubits.
pub fn evolve_pauli_by_clifford(
    pauli: &DensePauli,
    cliff: &Clifford,
) -> Result<DensePauli, DensePauliError> {
    if pauli.num_qubits() != cliff.num_qubits {
        return Err(DensePauliError::DifferentLengthsEvolve);
    }
    let num_qubits = cliff.num_qubits;
    let mut out_pauli = DensePauli::identity(num_qubits);

    // Decompose pauli as a tensor product of single qubit paulis on each of the qubits.
    for qbit in 0..num_qubits {
        // evolve the singe qubit pauli by cliff
        let pz = pauli.pauli_z[qbit];
        let px = pauli.pauli_x[qbit];

        if (pz, px) != (false, false) {
            // single qubit pauli is not I (only X, Y, Z)
            let evolved_pauli = cliff.evolve_single_qubit_pauli_dense(pz, px, qbit);

            // compose the ouput evolved dense paulies
            out_pauli.compose_with_unchecked(&evolved_pauli);
        }
    }

    out_pauli.zx_phase = (out_pauli.zx_phase + pauli.zx_phase) & 3;
    Ok(out_pauli)
}

#[cfg(test)]
mod tests {
    use fixedbitset::FixedBitSet;

    use crate::clifford::Clifford;
    use crate::dense_pauli::{DensePauli, DensePauliError, evolve_pauli_by_clifford};

    fn pauli_from_label(label: &str) -> DensePauli {
        DensePauli::from_label(label).unwrap()
    }

    #[test]
    fn test_identity() {
        let num_qubits = 3;
        let pauli = DensePauli::identity(num_qubits);
        assert_eq!(pauli.num_qubits(), 3);
        assert_eq!(pauli.count_y(), 0);
        assert_eq!(pauli.pauli_x, FixedBitSet::with_capacity(num_qubits));
        assert_eq!(pauli.pauli_z, FixedBitSet::with_capacity(num_qubits));
        assert_eq!(pauli.zx_phase, 0);
    }

    #[test]
    fn test_from_sparse_bool_and_back() {
        let pauli =
            DensePauli::from_sparse_bool(3, &[true, true], &[true, false], &[1, 2], 2, true)
                .unwrap();
        let mut expected_x = FixedBitSet::with_capacity(3);
        expected_x.set(1, true);
        let mut expected_z = FixedBitSet::with_capacity(3);
        expected_z.set(1, true);
        expected_z.set(2, true);
        let expected_zx_phase = 3;
        assert_eq!(pauli.num_qubits(), 3);
        assert_eq!(pauli.count_y(), 1);
        assert_eq!(pauli.pauli_x, expected_x);
        assert_eq!(pauli.pauli_z, expected_z);
        assert_eq!(pauli.zx_phase, expected_zx_phase);

        let (z, x, indices, phase) = pauli.to_sparse_bool(true);
        assert_eq!(x, vec![true, false]);
        assert_eq!(z, vec![true, true]);
        assert_eq!(indices, vec![1, 2]);
        assert_eq!(phase, 2);
    }

    #[test]
    fn test_from_label_and_back() {
        let pauli = pauli_from_label("-iXZ");
        let mut expected_x = FixedBitSet::with_capacity(2);
        expected_x.set(1, true);
        let mut expected_z = FixedBitSet::with_capacity(2);
        expected_z.set(0, true);
        let expected_zx_phase = 3;
        assert_eq!(pauli.num_qubits(), 2);
        assert_eq!(pauli.count_y(), 0);
        assert_eq!(pauli.pauli_x, expected_x);
        assert_eq!(pauli.pauli_z, expected_z);
        assert_eq!(pauli.zx_phase, expected_zx_phase);

        let pauli_label = pauli.to_label();
        assert_eq!(pauli_label, String::from("-iXZ"));
    }

    #[test]
    fn test_invalid_label() {
        let result = DensePauli::from_label("XI-Z");
        assert!(matches!(result, Err(DensePauliError::InvalidLabel(_))));
    }

    /// Assert that commuting two Paulis P and Q gives the expected result.
    fn assert_commute(p_label: &str, q_label: &str, expected: bool) {
        let p = pauli_from_label(p_label);
        let q = pauli_from_label(q_label);
        let computed_pq = p.commutes_unchecked(&q);
        let computed_qp = p.commutes_unchecked(&q);
        assert_eq!(computed_pq, expected);
        assert_eq!(computed_qp, expected);
    }

    #[test]
    fn test_pauli_commute() {
        assert_commute("XX", "YY", true);
        assert_commute("XXX", "YYY", false);
        assert_commute("XZ", "iZY", true);
        assert_commute("III", "-iXYZ", true);
        assert_commute("-XIXI", "iIZZI", false);
        assert_commute("IXYZ", "IYZX", false);
    }

    /// Assert that multiplying Paulis P and Q gives the expected result.
    fn assert_multiply(p_label: &str, q_label: &str, expected_label: &str) {
        let p = pauli_from_label(p_label);
        let q = pauli_from_label(q_label);
        let expected = pauli_from_label(expected_label);
        let computed = q.compose_unchecked(&p);
        assert_eq!(computed, expected);
    }

    #[test]
    fn test_pauli_multiply() {
        assert_multiply("X", "Y", "iZ");
        assert_multiply("Y", "X", "-iZ");
        assert_multiply("X", "Z", "-iY");
        assert_multiply("Z", "X", "iY");
        assert_multiply("Y", "Z", "iX");
        assert_multiply("Z", "Y", "-iX");
        assert_multiply("I", "X", "X");
        assert_multiply("-iX", "iI", "X");
        assert_multiply("iZ", "iZ", "-I");
        assert_multiply("XX", "XY", "iIZ");
        assert_multiply("XY", "XX", "-iIZ");
    }

    #[test]
    fn test_compose_with() {
        let mut p = DensePauli::identity(3);
        p.compose_with_unchecked(&pauli_from_label("iXYZ"));
        assert_eq!(p, pauli_from_label("iXYZ"));
        p.compose_with_unchecked(&pauli_from_label("ZII"));
        assert_eq!(p, pauli_from_label("-YYZ"));
        p.compose_with_unchecked(&pauli_from_label("-YYZ"));
        assert_eq!(p, pauli_from_label("III"));
    }

    /// Assert that evolving P under Cliff gives the expected result.
    fn assert_evolve(p_label: &str, cliff: &Clifford, expected_label: &str) {
        let p = pauli_from_label(p_label);
        let computed = evolve_pauli_by_clifford(&p, cliff).unwrap();
        let expected = pauli_from_label(expected_label);
        assert_eq!(computed, expected);
    }

    #[test]
    fn test_evolve_1_qubit() {
        use ndarray::Array2;

        let cliff_s = Clifford::from_array(
            Array2::from(vec![[true, true, false], [false, true, false]]).view(),
        );
        let cliff_h = Clifford::from_array(
            Array2::from(vec![[false, true, false], [true, false, false]]).view(),
        );
        let cliff_sdg = Clifford::from_array(
            Array2::from(vec![[true, true, true], [false, true, false]]).view(),
        );
        let cliff_sx = Clifford::from_array(
            Array2::from(vec![[true, false, false], [true, true, true]]).view(),
        );
        let cliff_sxdg = Clifford::from_array(
            Array2::from(vec![[true, false, false], [true, true, false]]).view(),
        );

        assert_evolve("I", &cliff_s, "I");
        assert_evolve("X", &cliff_s, "-Y");
        assert_evolve("Z", &cliff_s, "Z");
        assert_evolve("Y", &cliff_s, "X");
        assert_evolve("-I", &cliff_s, "-I");
        assert_evolve("-X", &cliff_s, "Y");
        assert_evolve("-Z", &cliff_s, "-Z");
        assert_evolve("-Y", &cliff_s, "-X");

        assert_evolve("I", &cliff_sdg, "I");
        assert_evolve("X", &cliff_sdg, "Y");
        assert_evolve("Z", &cliff_sdg, "Z");
        assert_evolve("Y", &cliff_sdg, "-X");
        assert_evolve("-I", &cliff_sdg, "-I");
        assert_evolve("-X", &cliff_sdg, "-Y");
        assert_evolve("-Z", &cliff_sdg, "-Z");
        assert_evolve("-Y", &cliff_sdg, "X");

        assert_evolve("I", &cliff_h, "I");
        assert_evolve("X", &cliff_h, "Z");
        assert_evolve("Z", &cliff_h, "X");
        assert_evolve("Y", &cliff_h, "-Y");
        assert_evolve("-I", &cliff_h, "-I");
        assert_evolve("-X", &cliff_h, "-Z");
        assert_evolve("-Z", &cliff_h, "-X");
        assert_evolve("-Y", &cliff_h, "Y");

        assert_evolve("I", &cliff_sx, "I");
        assert_evolve("X", &cliff_sx, "X");
        assert_evolve("Z", &cliff_sx, "Y");
        assert_evolve("Y", &cliff_sx, "-Z");
        assert_evolve("-I", &cliff_sx, "-I");
        assert_evolve("-X", &cliff_sx, "-X");
        assert_evolve("-Z", &cliff_sx, "-Y");
        assert_evolve("-Y", &cliff_sx, "Z");

        assert_evolve("I", &cliff_sxdg, "I");
        assert_evolve("X", &cliff_sxdg, "X");
        assert_evolve("Z", &cliff_sxdg, "-Y");
        assert_evolve("Y", &cliff_sxdg, "Z");
        assert_evolve("-I", &cliff_sxdg, "-I");
        assert_evolve("-X", &cliff_sxdg, "-X");
        assert_evolve("-Z", &cliff_sxdg, "Y");
        assert_evolve("-Y", &cliff_sxdg, "-Z");
    }

    #[test]
    fn test_evolve_2_qubits() {
        use ndarray::Array2;

        // The assertions below were generated using the Clifford class in python, where the
        // Clifford below was generated via ``random_clifford(2, seed=1234)``.
        let cliff = Clifford::from_array(
            Array2::from(vec![
                [false, true, false, true, false],
                [false, true, true, true, true],
                [true, false, true, true, false],
                [true, false, true, false, true],
            ])
            .view(),
        );

        assert_evolve("XX", &cliff, "-XZ");
        assert_evolve("YY", &cliff, "-ZX");
        assert_evolve("XY", &cliff, "-IY");
        assert_evolve("-IY", &cliff, "ZI");
        assert_evolve("ZI", &cliff, "-ZZ");
        assert_evolve("-YZ", &cliff, "XI");
        assert_evolve("YX", &cliff, "YI");
        assert_evolve("iYX", &cliff, "iYI");
        assert_evolve("-YX", &cliff, "-YI");
        assert_evolve("-iYX", &cliff, "-iYI");
        assert_evolve("II", &cliff, "II");
    }
}
