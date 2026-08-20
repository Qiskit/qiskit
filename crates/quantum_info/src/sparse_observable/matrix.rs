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
        let observable = self.as_paulis();

        let mut matrix = create_empty_matrix(observable.num_qubits)?;
        let terms: Vec<_> = observable.iter().map(|term| compress_term(&term)).collect();

        for (i, mut row) in matrix.rows_mut().into_iter().enumerate() {
            fill_matrix_row(&mut row, i, &terms);
        }

        Ok(matrix)
    }
}

struct PauliTerm {
    coeff: Complex64,
    x: u32,
    z: u32,
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

fn fill_matrix_row(row: &mut ArrayViewMut1<Complex64>, i: usize, terms: &[PauliTerm]) {
    for term in terms {
        let qubit_col = i ^ term.x as usize;
        let coeff = count_phase(term.coeff, i, term.x, term.z);
        row[qubit_col] += coeff;
    }
}

fn count_phase(coeff: Complex64, row: usize, x: u32, z: u32) -> Complex64 {
    let mask = row as u32;
    let y = x & z;

    if y != 0 {
        if (mask & y).count_ones() % 2 != 0 {
            coeff * Complex64::I
        } else {
            coeff * -Complex64::I
        }
    } else if (mask & z).count_ones() % 2 != 0 {
        -coeff
    } else {
        coeff
    }
}

fn compress_term(term: &SparseTermView<'_>) -> PauliTerm {
    let mut x = 0;
    let mut z = 0;

    for (bit_term, qubit) in term.bit_terms.iter().zip(term.indices) {
        let enable_qubit = |qubits: &mut u32| *qubits |= 1 << qubit;

        match bit_term {
            BitTerm::X => {
                enable_qubit(&mut x);
            }
            BitTerm::Y => {
                enable_qubit(&mut x);
                enable_qubit(&mut z);
            }
            BitTerm::Z => {
                enable_qubit(&mut z);
            }
            _ => (),
        }
    }

    PauliTerm {
        coeff: term.coeff,
        x,
        z,
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;

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
