use std::result;

use ndarray::Array2;
use num_complex::Complex64;
use thiserror::Error;

use crate::sparse_observable::SparseTermView;

use super::{BitTerm, SparseObservable};

pub type Result<T> = result::Result<T, MatrixError>;

#[derive(Debug, Error)]
#[error("{0} qubit matrix too large for this system")]
pub struct MatrixError(u32);

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
    let dim = 1usize
        .checked_shl(num_qubits)
        .ok_or(MatrixError(num_qubits))?;

    let len = dim.checked_mul(dim).ok_or(MatrixError(num_qubits))?;
    let data = vec![Complex64::ZERO; len];

    let matrix = Array2::from_shape_vec((dim, dim), data).expect("shape fits len");
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
    todo!()
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

        let result = observable.to_matrix_old().expect("is supported");
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

        let result = observable.to_matrix_old().expect("is supported");
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

        let result = observable.to_matrix_old().expect("is supported");
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
