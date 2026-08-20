use ndarray::{Array2, ArrayViewMut1};
use num_complex::{Complex64, c64};
use thiserror::Error;

use crate::sparse_observable::SparseTermView;

use super::{BitTerm, SparseObservable};

#[derive(Debug, Error)]
#[error("{0} qubit matrix not supported on this system")]
pub struct MatrixError(u32);

impl SparseObservable {
    pub fn to_matrix(&self) -> Result<Array2<Complex64>, MatrixError> {
        let mut matrix = create_empty_matrix(self.num_qubits)?;
        let terms = compress_pauli_terms(self);

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

fn fill_matrix_row(row: &mut ArrayViewMut1<Complex64>, i: usize, terms: &[Term]) {
    for term in terms {
        match &term.kind {
            TermKind::Pauli { x, z } => {
                accumulate_pauli(row, i, term.coeff, *x, *z);
            }
            TermKind::Projector(bit_terms) => {
                accumulate_projector(row, i, term.coeff, bit_terms);
            }
        }
    }
}

fn accumulate_pauli(
    row: &mut ArrayViewMut1<Complex64>,
    i: usize,
    coeff: Complex64,
    x: u32,
    z: u32,
) {
    let qubit_col = i ^ x as usize;
    let coeff = count_phase(coeff, i, x, z);
    row[qubit_col] += coeff;
}

fn count_phase(coeff: Complex64, row: usize, x: u32, z: u32) -> Complex64 {
    let mask = row as u32;
    let y = x & z;

    if y != 0 && (mask & y).count_ones() % 2 != 0 {
        coeff * Complex64::I
    } else if y != 0 {
        coeff * -Complex64::I
    } else if (mask & z).count_ones() % 2 != 0 {
        -coeff
    } else {
        coeff
    }
}

fn accumulate_projector(
    row: &mut ArrayViewMut1<Complex64>,
    i: usize,
    mut coeff: Complex64,
    qubit_terms: &[QubitTerm],
) {
    let mut qubit_col = i;

    for qubit_term in qubit_terms.iter() {
        let is_qubit_one = || qubit_col & (1usize << qubit_term.qubit_idx) != 0;
        let move_qubit_col = || qubit_col ^ (1usize << qubit_term.qubit_idx);

        match qubit_term.bit_term {
            BitTerm::X => {
                qubit_col = move_qubit_col();
            }
            BitTerm::Y if is_qubit_one() => {
                coeff *= Complex64::I;
                qubit_col = move_qubit_col();
            }
            BitTerm::Y => {
                coeff *= -Complex64::I;
                qubit_col = move_qubit_col();
            }
            BitTerm::Z if is_qubit_one() => {
                coeff = -coeff;
            }
            BitTerm::Plus | BitTerm::Minus => {
                coeff /= c64(2_f64.sqrt(), 0.0);
            }
            BitTerm::Right if is_qubit_one() => {
                coeff *= Complex64::I;
            }
            BitTerm::Right => {
                coeff *= -Complex64::I;
            }
            BitTerm::Left if is_qubit_one() => {
                coeff *= -Complex64::I;
            }
            BitTerm::Left => {
                coeff *= Complex64::I;
            }
            BitTerm::Zero if is_qubit_one() => {
                return;
            }
            BitTerm::One if !is_qubit_one() => {
                return;
            }
            _ => (),
        }
    }

    row[qubit_col] += coeff;
}

#[derive(Debug, Clone)]
struct Term {
    coeff: Complex64,
    kind: TermKind,
}

#[derive(Debug, Clone)]
enum TermKind {
    Pauli { x: u32, z: u32 },
    Projector(Vec<QubitTerm>),
}

#[derive(Debug, Clone)]
struct QubitTerm {
    bit_term: BitTerm,
    qubit_idx: u32,
}

fn compress_pauli_terms(operator: &SparseObservable) -> Vec<Term> {
    operator
        .iter()
        .map(|term| maybe_compress_term(&term))
        .collect()
}

fn maybe_compress_term(term: &SparseTermView<'_>) -> Term {
    let mut x = 0;
    let mut z = 0;

    for (bit_term, qubit_idx) in term.bit_terms.iter().zip(term.indices) {
        let enable = |qubits: &mut u32| *qubits |= 1 << qubit_idx;

        match bit_term {
            BitTerm::X => {
                enable(&mut x);
            }
            BitTerm::Y => {
                enable(&mut x);
                enable(&mut z);
            }
            BitTerm::Z => {
                enable(&mut z);
            }
            _ => return map_projector(term),
        }
    }

    Term {
        coeff: term.coeff,
        kind: TermKind::Pauli { x, z },
    }
}

fn map_projector(term: &SparseTermView<'_>) -> Term {
    let qubit_terms = term
        .bit_terms
        .iter()
        .copied()
        .zip(term.indices.iter().copied())
        .map(|(bit_term, qubit_idx)| QubitTerm {
            qubit_idx,
            bit_term,
        })
        .collect();

    Term {
        coeff: term.coeff,
        kind: TermKind::Projector(qubit_terms),
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;

    use super::*;

    #[test]
    fn test_pauli_only() {
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
    fn test_pauli_with_projectors() {
        let terms = &[
            (2.0.into(), "X+"),
            (1.0.into(), "Y-"),
            (3.5.into(), "Zr"),
            (2.0.into(), "Il"),
            (0.5.into(), "X0"),
            (1.0.into(), "Y1"),
        ];

        let observable = parse_observable(terms);
        let expect: Array2<Complex64> = todo!();

        let result = observable.to_matrix().expect("is_supported");
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
