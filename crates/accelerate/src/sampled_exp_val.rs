// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use num_complex::Complex64;

use hashbrown::HashMap;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::pauli_exp_val::fast_sum;
use qiskit_circuit::util::c64;
use qiskit_quantum_info::sparse_observable::BitTerm;
use qiskit_quantum_info::sparse_observable::PySparseObservable;
use qiskit_quantum_info::sparse_observable::SparseObservable;

const OPER_TABLE_SIZE: usize = (b'Z' as usize) + 1;
const fn generate_oper_table() -> [[f64; 2]; OPER_TABLE_SIZE] {
    let mut table = [[0.; 2]; OPER_TABLE_SIZE];
    table[b'Z' as usize] = [1., -1.];
    table[b'0' as usize] = [1., 0.];
    table[b'1' as usize] = [0., 1.];
    table
}

static OPERS: [[f64; 2]; OPER_TABLE_SIZE] = generate_oper_table();

fn bitstring_expval(dist: &HashMap<String, f64>, mut oper_str: String) -> f64 {
    let inds: Vec<usize> = oper_str
        .char_indices()
        .filter_map(|(index, oper)| if oper != 'I' { Some(index) } else { None })
        .collect();
    oper_str.retain(|c| !r#"I"#.contains(c));
    let denom: f64 = fast_sum(&dist.values().copied().collect::<Vec<f64>>());
    let exp_val: f64 = dist
        .iter()
        .map(|(bits, val)| {
            let temp_product: f64 = oper_str.bytes().enumerate().fold(1.0, |acc, (idx, oper)| {
                let diagonal: [f64; 2] = OPERS[oper as usize];
                let index_char: char = bits.as_bytes()[inds[idx]] as char;
                let index: usize = index_char.to_digit(10).unwrap() as usize;
                acc * diagonal[index]
            });
            val * temp_product
        })
        .sum();
    exp_val / denom
}

/// Compute the expectation value from a sampled distribution
#[pyfunction]
#[pyo3(text_signature = "(oper_strs, coeff, dist, /)")]
pub fn sampled_expval_float(
    oper_strs: Vec<String>,
    coeff: PyReadonlyArray1<f64>,
    dist: HashMap<String, f64>,
) -> PyResult<f64> {
    let coeff_arr = coeff.as_slice()?;
    let out = oper_strs
        .into_iter()
        .enumerate()
        .map(|(idx, string)| coeff_arr[idx] * bitstring_expval(&dist, string))
        .sum();
    Ok(out)
}

/// Compute the expectation value from a sampled distribution
#[pyfunction]
#[pyo3(text_signature = "(oper_strs, coeff, dist, /)")]
pub fn sampled_expval_complex(
    oper_strs: Vec<String>,
    coeff: PyReadonlyArray1<Complex64>,
    dist: HashMap<String, f64>,
) -> PyResult<f64> {
    let coeff_arr = coeff.as_slice()?;
    let out: Complex64 = oper_strs
        .into_iter()
        .enumerate()
        .map(|(idx, string)| coeff_arr[idx] * c64(bitstring_expval(&dist, string), 0.))
        .sum::<Complex64>();
    Ok(out.re)
}

// Compute the expectation value from a sampled distribution for SparseObservable objects
#[pyfunction]
#[pyo3(text_signature = "(sparse_obs, dist, /)")]
pub fn sampled_expval_sparse_observable(
    sparse_obs: PyRef<PySparseObservable>,
    dist: HashMap<String, f64>,
) -> PyResult<f64> {
    // Access the inner SparseObservable through the RwLock
    let sparse_obs_guard = sparse_obs.inner.read().unwrap();
    let sparse_obs: &SparseObservable = &sparse_obs_guard;
    let n = sparse_obs.num_qubits();

    // Convert SparseObservable to operator strings and coefficients
    let result: Result<Complex64, PyErr> =
        sparse_obs
            .iter()
            .enumerate()
            .try_fold(Complex64::new(0.0, 0.0), |acc, (_idx, term)| {
                let mut full_op = vec!["I"; n as usize];
                for (bit_term, &index) in term.bit_terms.iter().zip(term.indices.iter()) {
                    let char = match bit_term {
                        BitTerm::X => "X",
                        BitTerm::Y => "Y",
                        BitTerm::Z => "Z",
                        BitTerm::Plus => "+",
                        BitTerm::Minus => "-",
                        BitTerm::Right => "r",
                        BitTerm::Left => "l",
                        BitTerm::Zero => "0",
                        BitTerm::One => "1",
                    };
                    full_op[(n - 1 - index) as usize] = char;
                }
                let oper_str = full_op.join("");

                // Validating that all operators are diagonal
                if !oper_str.chars().all(|c| ['I', 'Z', '0', '1'].contains(&c)) {
                    return Err(PyValueError::new_err(format!(
                        "Operator string '{}' contains non-diagonal terms",
                        oper_str
                    )));
                }
                Ok(acc + term.coeff * Complex64::new(bitstring_expval(&dist, oper_str), 0.0))
            });
    Ok(result?.re)
}

pub fn sampled_exp_val(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(sampled_expval_float))?;
    m.add_wrapped(wrap_pyfunction!(sampled_expval_complex))?;
    m.add_wrapped(wrap_pyfunction!(sampled_expval_sparse_observable))?;
    Ok(())
}
