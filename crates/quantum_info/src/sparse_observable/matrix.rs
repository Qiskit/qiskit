use ndarray::{Array2, ArrayViewMut1};
use num_complex::Complex64;
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
    mut coeff: Complex64,
    x: u32,
    z: u32,
) {
    let target_col = i ^ x as usize;
    let is_negative = !(i & z as usize).count_ones().is_multiple_of(2);

    if is_negative {
        coeff *= -1.0;
    }

    row[target_col] += coeff;
}

fn accumulate_projector(
    row: &mut ArrayViewMut1<Complex64>,
    i: usize,
    mut coeff: Complex64,
    qubit_terms: &[QubitTerm],
) {
    let mut target_col = i;

    for qubit_term in qubit_terms.iter() {
        let toggle = || target_col ^ 1usize << qubit_term.idx;
        let is_enabled = || target_col & (1usize << qubit_term.idx) != 0;

        match qubit_term.bit {
            BitTerm::X => {
                target_col = toggle();
            }
            BitTerm::Y if is_enabled() => {
                coeff *= Complex64::i();
                target_col = toggle();
            }
            BitTerm::Y => {
                coeff *= -Complex64::i();
                target_col = toggle();
            }
            BitTerm::Z if is_enabled() => {
                coeff = -coeff;
            }
            BitTerm::Plus | BitTerm::Minus => {
                coeff /= Complex64::from(2.0_f64.sqrt());
            }
            BitTerm::Right if is_enabled() => {
                coeff *= Complex64::i();
            }
            BitTerm::Right => {
                coeff *= -Complex64::i();
            }
            BitTerm::Left if is_enabled() => {
                coeff *= -Complex64::i();
            }
            BitTerm::Left => {
                coeff *= Complex64::i();
            }
            BitTerm::Zero if is_enabled() => {
                return;
            }
            BitTerm::One if !is_enabled() => {
                return;
            }
            _ => (),
        }
    }

    row[target_col] += coeff;
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
    bit: BitTerm,
    idx: u32,
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
    let bit_terms = term
        .bit_terms
        .iter()
        .copied()
        .zip(term.indices.iter().copied())
        .map(|(bit, idx)| QubitTerm { idx, bit })
        .collect();

    Term {
        coeff: term.coeff,
        kind: TermKind::Projector(bit_terms),
    }
}
