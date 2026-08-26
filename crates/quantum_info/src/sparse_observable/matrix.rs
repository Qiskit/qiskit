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

use std::{cmp::Reverse, result};

use bytemuck::zeroed_vec;
use ndarray::{Array2, ArrayView2, linalg::kron};
use num_complex::Complex64;
use thiserror::Error;

use super::{BitTerm, SparseObservable, SparseTermView};

pub type Result<T> = result::Result<T, MatrixError>;

#[derive(Debug, Error)]
pub enum MatrixError {
    #[error("number of qubits is 0")]
    ZeroQubits,
    #[error("{0} qubit matrix too large for this system")]
    TooManyQubits(u32),
}

impl SparseObservable {
    pub fn to_matrix(&self) -> Result<Array2<Complex64>> {
        let mut matrix = create_matrix_with_zeros(self.num_qubits)?;

        for term in self.iter() {
            add_term(&mut matrix, &term)
        }

        Ok(matrix)
    }
}

fn create_matrix_with_zeros(num_qubits: u32) -> Result<Array2<Complex64>> {
    if num_qubits == 0 {
        return Err(MatrixError::ZeroQubits);
    }

    let dim = 1usize
        .checked_shl(num_qubits)
        .ok_or(MatrixError::TooManyQubits(num_qubits))?;

    let len = dim
        .checked_mul(dim)
        .ok_or(MatrixError::TooManyQubits(num_qubits))?;

    let data = zeroed_vec(len);
    let matrix = Array2::from_shape_vec((dim, dim), data).expect("shape fits data");
    Ok(matrix)
}

struct PauliTerm {
    coeff: Complex64,
    x: u32,
    z: u32,
}

fn add_term(matrix: &mut Array2<Complex64>, term: &SparseTermView) {
    if let Some(pauli) = maybe_compress_pauli(term) {
        add_term_pauli(matrix, &pauli);
    } else {
        add_term_kron(matrix, term);
    }
}

fn maybe_compress_pauli(term: &SparseTermView) -> Option<PauliTerm> {
    let mut pauli = PauliTerm {
        coeff: term.coeff,
        x: 0,
        z: 0,
    };

    for (bit_term, qubit_idx) in term.bit_terms.iter().zip(term.indices) {
        let set_qubit_op = |qubit_ops: &mut u32| *qubit_ops |= 1 << qubit_idx;

        match bit_term {
            BitTerm::X => {
                set_qubit_op(&mut pauli.x);
            }
            BitTerm::Y => {
                set_qubit_op(&mut pauli.x);
                set_qubit_op(&mut pauli.z);
                pauli.coeff *= -Complex64::i();
            }
            BitTerm::Z => {
                set_qubit_op(&mut pauli.z);
            }
            _ => return None,
        }
    }

    Some(pauli)
}

fn add_term_pauli(matrix: &mut Array2<Complex64>, term: &PauliTerm) {
    for (i, mut row) in matrix.rows_mut().into_iter().enumerate() {
        let qubit_col = i ^ term.x as usize;

        if (i as u32 & term.z).count_ones().is_multiple_of(2) {
            row[qubit_col] += term.coeff;
        } else {
            row[qubit_col] -= term.coeff;
        }
    }
}

fn add_term_kron(matrix: &mut Array2<Complex64>, term: &SparseTermView) {
    let n = matrix.nrows().trailing_zeros();
    let m = term.indices.len();

    let mut order: Vec<usize> = (0..m).collect();
    order.sort_unstable_by_key(|&idx| Reverse(term.indices[idx]));

    let mut local = Array2::from_elem((1, 1), term.coeff);
    for &idx in &order {
        local = kron(&local, &get_bit_term_matrix(term.bit_terms[idx]));
    }

    let local_bit_to_qubit: Vec<u32> = order.iter().rev().map(|&idx| term.indices[idx]).collect();

    let support: u32 = term.indices.iter().fold(0u32, |acc, &q| acc | (1 << q));
    let identity_qubits: Vec<u32> = (0..n).filter(|q| support & (1 << q) == 0).collect();

    let n_free = 1usize << identity_qubits.len();
    let mut free_patterns = vec![0usize; n_free];
    for (bit_pos, &q) in identity_qubits.iter().enumerate() {
        for (pattern_idx, pattern) in free_patterns.iter_mut().enumerate() {
            if (pattern_idx >> bit_pos) & 1 == 1 {
                *pattern |= 1usize << q;
            }
        }
    }

    let scatter = |local_idx: usize| -> usize {
        local_bit_to_qubit
            .iter()
            .enumerate()
            .fold(0usize, |acc, (bit_pos, &q)| {
                acc | (((local_idx >> bit_pos) & 1) << q)
            })
    };

    let local_dim = local.nrows();
    for local_row in 0..local_dim {
        let base_row = scatter(local_row);
        for local_col in 0..local_dim {
            let val = local[(local_row, local_col)];
            if val == Complex64::ZERO {
                continue;
            }
            let base_col = scatter(local_col);
            for &pattern in &free_patterns {
                matrix[(base_row | pattern, base_col | pattern)] += val;
            }
        }
    }
}

fn get_bit_term_matrix(bit_term: BitTerm) -> ArrayView2<'static, Complex64> {
    let data = match bit_term {
        BitTerm::X => const { &[re(0.0), re(1.0), re(1.0), re(0.0)] },
        BitTerm::Y => const { &[re(0.0), im(-1.0), im(1.0), re(0.0)] },
        BitTerm::Z => const { &[re(1.0), re(0.0), re(0.0), re(-1.0)] },
        BitTerm::Plus => const { &[re(0.5), re(0.5), re(0.5), re(0.5)] },
        BitTerm::Minus => const { &[re(0.5), re(-0.5), re(-0.5), re(0.5)] },
        BitTerm::Right => const { &[re(0.5), im(-0.5), im(0.5), re(0.5)] },
        BitTerm::Left => const { &[re(0.5), im(0.5), im(-0.5), re(0.5)] },
        BitTerm::Zero => const { &[re(1.0), re(0.0), re(0.0), re(0.0)] },
        BitTerm::One => const { &[re(0.0), re(0.0), re(0.0), re(1.0)] },
    };

    ArrayView2::from_shape((2, 2), data).expect("shape fits data")
}

const fn re(n: f64) -> Complex64 {
    Complex64::new(n, 0.0)
}

const fn im(n: f64) -> Complex64 {
    Complex64::new(0.0, n)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use num_complex::c64;

    use super::*;

    #[test]
    fn test_zero_qubits() {
        let num_qubits = 0;

        let res = SparseObservable::new(num_qubits, vec![], vec![], vec![], vec![0])
            .expect("is coherent")
            .to_matrix();

        assert!(matches!(res, Err(MatrixError::ZeroQubits)));
    }

    #[test]
    fn test_too_many_qubits() {
        let num_qubits = 0usize.count_zeros();

        let res = SparseObservable::new(num_qubits, vec![], vec![], vec![], vec![0])
            .expect("is coherent")
            .to_matrix();

        assert!(matches!(res, Err(MatrixError::TooManyQubits(_))));
    }

    #[test]
    fn test_dimension_scaling() {
        const DIMS: &[usize] = &[2, 4, 8, 16, 32, 64, 128, 256];

        for (i, dim) in DIMS.iter().copied().enumerate() {
            let num_qubits = (i + 1) as u32;
            let obs = SparseObservable::new(num_qubits, vec![], vec![], vec![], vec![0])
                .expect("is coherent");

            let matrix = obs.to_matrix().expect("no errors");
            assert_eq!(matrix.dim(), (dim, dim));
        }
    }

    #[test]
    fn test_paulis() {
        let terms = &[
            (c64(-3.0, 0.0), "XI"),
            (c64(0.0, 4.4), "YZ"),
            (c64(0.2, -0.1), "YY"),
            (c64(66.12, 0.0), "ZZ"),
        ];
        let obs = create_obs(terms);
        let res = obs.to_matrix().expect("no errors");

        let data = &[
            // Row 1
            c64(66.12, 0.0),
            c64(0.0, 0.0),
            c64(1.4, 0.0),
            c64(-0.2, 0.1),
            // Row 2
            c64(0.0, 0.0),
            c64(-66.12, 0.0),
            c64(0.2, -0.1),
            c64(-7.4, 0.0),
            // Row 3
            c64(-7.4, 0.0),
            c64(0.2, -0.1),
            c64(-66.12, 0.0),
            c64(0.0, 0.0),
            // Row 4
            c64(-0.2, 0.1),
            c64(1.4, 0.0),
            c64(0.0, 0.0),
            c64(66.12, 0.0),
        ];

        let exp = ArrayView2::from_shape((4, 4), data).expect("shape fits data");
        assert_abs_diff_eq!(res, exp, epsilon = 0.0001);
    }

    #[test]
    fn test_paulis_and_projectors() {
        let terms = &[
            (c64(0.5, -1.0), "X+"),
            (c64(8.1, 0.0), "Y-"),
            (c64(0.7, -0.1), "Zr"),
            (c64(9.1, 0.0), "Il"),
            (c64(2.0, 0.0), "I0"),
            (c64(0.5, 0.0), "I1"),
        ];
        let obs = create_obs(terms);
        let res = obs.to_matrix().expect("no errors");

        let exp = obs.as_paulis().to_matrix().expect("no errors");
        assert_abs_diff_eq!(res, exp, epsilon = 0.0001);
    }

    #[test]
    fn test_zero_coeff() {
        let terms = &[(c64(0.0, 0.0), "II")];
        let obs = create_obs(terms);
        let res = obs.to_matrix().expect("no errors");

        let data = &[Complex64::ZERO; 16];
        let exp = ArrayView2::from_shape((4, 4), data).expect("shape fits data");
        assert_eq!(res, exp);
    }

    fn create_obs(terms: &[(Complex64, &str)]) -> SparseObservable {
        let mut num_qubits = 0;
        let mut coeffs = vec![];
        let mut bit_terms = vec![];
        let mut indices = vec![];
        let mut boundaries = vec![0];

        for (coeff, term) in terms {
            num_qubits = term.len();
            coeffs.push(*coeff);

            let mut num_bit_terms = 0;
            for (i, ch) in term.as_bytes().iter().rev().enumerate() {
                let bit_term = BitTerm::try_from_u8(*ch).expect("ch is bit term");

                if let Some(non_identity) = bit_term {
                    bit_terms.push(non_identity);
                    indices.push(i as u32);
                    num_bit_terms += 1;
                }
            }

            let next_boundary = boundaries.last().expect("non empty") + num_bit_terms;
            boundaries.push(next_boundary);
        }

        SparseObservable::new(num_qubits as u32, coeffs, bit_terms, indices, boundaries)
            .expect("is coherent")
    }
}
