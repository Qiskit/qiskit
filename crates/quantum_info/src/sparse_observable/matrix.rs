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

use ndarray::{Array2, ArrayViewMut1};
use num_complex::Complex64;
use thiserror::Error;

use crate::sparse_observable::SparseTermView;

use super::{BitTerm, SparseObservable};

/// The error returned for matrix operations.
#[derive(Debug, Error)]
#[error("{0} qubit matrix not supported on this system")]
pub struct MatrixError(u32);

impl SparseObservable {
    /// Expand the observable into its matrix form.
    ///
    /// If the observable contains projectors to the eigenstate of X, Y, or Z,
    /// the observable is expanded into its Pauli-only form. See
    /// [`SparseObservable::as_paulis`].
    ///
    /// # Warning
    ///
    /// The total length of the matrix scales exponentially with the number of
    /// qubits. For example, an 8 qubit matrix requires 4 KB of memory, whereas
    /// 16 qubits requires 69 GB!
    ///
    /// # Errors
    ///
    /// Returns an error if the total number of matrix elements would exceed
    /// [`usize::MAX`].
    pub fn to_matrix(&self) -> Result<Array2<Complex64>, MatrixError> {
        let mut matrix = create_empty_matrix(self.num_qubits)?;
        let terms = compress_terms(self);

        for (i, mut row) in matrix.rows_mut().into_iter().enumerate() {
            fill_matrix_row(&mut row, i, &terms);
        }

        Ok(matrix)
    }
}

fn create_empty_matrix(num_qubits: u32) -> Result<Array2<Complex64>, MatrixError> {
    let dim = 1usize
        .checked_shl(num_qubits)
        .ok_or(MatrixError(num_qubits))?;

    let len = dim.checked_mul(dim).ok_or(MatrixError(num_qubits))?;
    let data = vec![Complex64::ZERO; len];

    let matrix = Array2::from_shape_vec((dim, dim), data).expect("shape fits len");
    Ok(matrix)
}

struct ZXTerm {
    coeff: Complex64,
    x: u32,
    z: u32,
}

fn compress_terms(observable: &SparseObservable) -> Vec<ZXTerm> {
    if has_projectors(observable) {
        observable.iter().map(|term| compress_term(&term)).collect()
    } else {
        let observable = observable.as_paulis();
        observable.iter().map(|term| compress_term(&term)).collect()
    }
}

fn has_projectors(observable: &SparseObservable) -> bool {
    observable
        .bit_terms()
        .iter()
        .any(|bit_term| !matches!(bit_term, BitTerm::X | BitTerm::Y | BitTerm::Z))
}

fn compress_term(term: &SparseTermView<'_>) -> ZXTerm {
    let mut pauli = ZXTerm {
        coeff: term.coeff,
        x: 0,
        z: 0,
    };

    for (bit_term, qubit) in term.bit_terms.iter().zip(term.indices) {
        let enable_qubit = |qubits: &mut u32| *qubits |= 1 << qubit;

        match bit_term {
            BitTerm::X => {
                enable_qubit(&mut pauli.x);
            }
            BitTerm::Y => {
                enable_qubit(&mut pauli.x);
                enable_qubit(&mut pauli.z);
                pauli.coeff *= -Complex64::i();
            }
            BitTerm::Z => {
                enable_qubit(&mut pauli.z);
            }
            _ => (),
        }
    }

    pauli
}

fn fill_matrix_row(row: &mut ArrayViewMut1<Complex64>, i: usize, terms: &[ZXTerm]) {
    for term in terms {
        let qubit_col = i ^ term.x as usize;

        if (i as u32 & term.z).count_ones().is_multiple_of(2) {
            row[qubit_col] += term.coeff;
        } else {
            row[qubit_col] -= term.coeff;
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;
    use num_complex::c64;

    use super::*;

    #[test]
    fn test_2xi_xy_3iz() {
        let terms = &[(2.0.into(), "XI"), (1.0.into(), "XY"), (3.0.into(), "IZ")];
        let observable = parse_observable(terms);
        let expect = arr2(&[
            [c64(3.0, 0.0), c64(0.0, 0.0), c64(2.0, 0.0), c64(0.0, -1.0)],
            [c64(0.0, 0.0), c64(-3.0, 0.0), c64(0.0, 1.0), c64(2.0, 0.0)],
            [c64(2.0, 0.0), c64(0.0, -1.0), c64(3.0, 0.0), c64(0.0, 0.0)],
            [c64(0.0, 1.0), c64(2.0, 0.0), c64(0.0, 0.0), c64(-3.0, 0.0)],
        ]);

        let result = observable.to_matrix().expect("is supported");
        assert_eq!(result, expect);
    }

    #[test]
    fn test_5yz_2xx_3iy() {
        let terms = &[(5.0.into(), "YZ"), (2.0.into(), "XX"), (3.0.into(), "IY")];
        let observable = parse_observable(terms);
        let expect = arr2(&[
            [c64(0.0, 0.0), c64(0.0, -3.0), c64(0.0, -5.0), c64(2.0, 0.0)],
            [c64(0.0, 3.0), c64(0.0, 0.0), c64(2.0, 0.0), c64(0.0, 5.0)],
            [c64(0.0, 5.0), c64(2.0, 0.0), c64(0.0, 0.0), c64(0.0, -3.0)],
            [c64(2.0, 0.0), c64(0.0, -5.0), c64(0.0, 3.0), c64(0.0, 0.0)],
        ]);

        let result = observable.to_matrix().expect("is supported");
        assert_eq!(result, expect);
    }

    #[test]
    fn test_3yy() {
        let observable = parse_observable(&[(3.0.into(), "YY")]);
        let expect = arr2(&[
            [c64(0.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(-3.0, 0.0)],
            [c64(0.0, 0.0), c64(0.0, 0.0), c64(3.0, 0.0), c64(0.0, 0.0)],
            [c64(0.0, 0.0), c64(3.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0)],
            [c64(-3.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0)],
        ]);

        let result = observable.to_matrix().expect("is supported");
        assert_eq!(result, expect);
    }

    fn parse_observable<'a>(
        terms: impl IntoIterator<Item = &'a (Complex64, &'a str)>,
    ) -> SparseObservable {
        let mut num_qubits = 0;
        let mut coeffs = Vec::new();
        let mut bit_terms = Vec::new();
        let mut indicies = Vec::new();
        let mut boundaries = vec![0];

        for (coeff, term) in terms {
            num_qubits = term.len() as u32;
            coeffs.push(*coeff);

            let mut num_bit_terms = 0;
            for (i, bit_term) in term.as_bytes().iter().rev().enumerate() {
                let bit_term = BitTerm::try_from_u8(*bit_term).expect("is bit term char");

                if let Some(non_identity) = bit_term {
                    bit_terms.push(non_identity);
                    num_bit_terms += 1;
                    indicies.push(i as u32);
                }
            }

            let next_boundary = boundaries.last().expect("non empty") + num_bit_terms;
            boundaries.push(next_boundary);
        }

        SparseObservable::new(num_qubits, coeffs, bit_terms, indicies, boundaries)
            .expect("is coherent")
    }
}
