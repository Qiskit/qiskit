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

use std::convert::TryInto;

use num_complex::Complex64;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyOverflowError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;

use crate::getenv_use_multiple_threads;

const LANES: usize = 8;
const PARALLEL_THRESHOLD: usize = 19;

// Based on the sum implementation in:
// https://stackoverflow.com/a/67191480/14033130
// and adjust for f64 usage
#[inline]
pub fn fast_sum(values: &[f64]) -> f64 {
    let chunks = values.chunks_exact(LANES);
    let remainder = chunks.remainder();

    let sum = chunks.fold([0.; LANES], |mut acc, chunk| {
        let chunk: [f64; LANES] = chunk.try_into().unwrap();
        for i in 0..LANES {
            acc[i] += chunk[i];
        }
        acc
    });
    let remainder: f64 = remainder.iter().copied().sum();

    let mut reduced = 0.;
    for val in sum {
        reduced += val;
    }
    reduced + remainder
}

/// Compute the pauli expectatation value of a statevector without x
#[pyfunction]
#[pyo3(text_signature = "(data, num_qubits, z_mask, /)")]
pub fn expval_pauli_no_x(
    data: PyReadonlyArray1<Complex64>,
    num_qubits: usize,
    z_mask: usize,
) -> PyResult<f64> {
    if num_qubits >= usize::BITS as usize {
        return Err(PyOverflowError::new_err(format!(
            "The value for num_qubits, {num_qubits}, is too large and would overflow"
        )));
    }
    let data_arr = data.as_slice()?;
    let size = 1_usize << num_qubits;
    let run_in_parallel = getenv_use_multiple_threads();
    let map_fn = |i: usize| -> f64 {
        let mut val: f64 = data_arr[i].re * data_arr[i].re + data_arr[i].im * data_arr[i].im;
        if (i & z_mask).count_ones() & 1 != 0 {
            val *= -1.;
        }
        val
    };

    if num_qubits < PARALLEL_THRESHOLD || !run_in_parallel {
        Ok(fast_sum(&(0..size).map(map_fn).collect::<Vec<f64>>()))
    } else {
        Ok((0..size).into_par_iter().map(map_fn).sum())
    }
}

/// Compute the pauli expectatation value of a statevector with x
#[pyfunction]
#[pyo3(text_signature = "(data, num_qubits, z_mask, x_mask, phase, x_max, /)")]
pub fn expval_pauli_with_x(
    data: PyReadonlyArray1<Complex64>,
    num_qubits: usize,
    z_mask: usize,
    x_mask: usize,
    phase: Complex64,
    x_max: u32,
) -> PyResult<f64> {
    if num_qubits > usize::BITS as usize {
        return Err(PyOverflowError::new_err(format!(
            "The value for num_qubits, {num_qubits}, is too large and would overflow",
        )));
    }
    let data_arr = data.as_slice()?;
    let mask_u = !(2_usize.pow(x_max + 1) - 1);
    let mask_l = 2_usize.pow(x_max) - 1;
    let size = 1_usize << (num_qubits - 1);
    let run_in_parallel = getenv_use_multiple_threads();
    let map_fn = |i: usize| -> f64 {
        let index_0 = ((i << 1) & mask_u) | (i & mask_l);
        let index_1 = index_0 ^ x_mask;
        let val_0 = (phase
            * Complex64::new(
                data_arr[index_1].re * data_arr[index_0].re
                    + data_arr[index_1].im * data_arr[index_0].im,
                data_arr[index_1].im * data_arr[index_0].re
                    - data_arr[index_1].re * data_arr[index_0].im,
            ))
        .re;
        let val_1 = (phase
            * Complex64::new(
                data_arr[index_0].re * data_arr[index_1].re
                    + data_arr[index_0].im * data_arr[index_1].im,
                data_arr[index_0].im * data_arr[index_1].re
                    - data_arr[index_0].re * data_arr[index_1].im,
            ))
        .re;
        let mut val = val_0;
        if (index_0 & z_mask).count_ones() & 1 != 0 {
            val *= -1.
        }
        if (index_1 & z_mask).count_ones() & 1 != 0 {
            val -= val_1;
        } else {
            val += val_1;
        }
        val
    };
    if num_qubits < PARALLEL_THRESHOLD || !run_in_parallel {
        Ok(fast_sum(&(0..size).map(map_fn).collect::<Vec<f64>>()))
    } else {
        Ok((0..size).into_par_iter().map(map_fn).sum())
    }
}

/// Compute the pauli expectatation value of a density matrix without x
#[pyfunction]
#[pyo3(text_signature = "(data, num_qubits, z_mask, /)")]
pub fn density_expval_pauli_no_x(
    data: PyReadonlyArray1<Complex64>,
    num_qubits: usize,
    z_mask: usize,
) -> PyResult<f64> {
    if num_qubits >= usize::BITS as usize {
        return Err(PyOverflowError::new_err(format!(
            "The value for num_qubits, {num_qubits}, is too large and would overflow",
        )));
    }
    let data_arr = data.as_slice()?;
    let num_rows = 1_usize << num_qubits;
    let stride = 1 + num_rows;
    let run_in_parallel = getenv_use_multiple_threads();
    let map_fn = |i: usize| -> f64 {
        let index = i * stride;
        let mut val = data_arr[index].re;
        if (i & z_mask).count_ones() & 1 != 0 {
            val *= -1.;
        }
        val
    };
    if num_qubits < PARALLEL_THRESHOLD || !run_in_parallel {
        Ok(fast_sum(&(0..num_rows).map(map_fn).collect::<Vec<f64>>()))
    } else {
        Ok((0..num_rows).into_par_iter().map(map_fn).sum())
    }
}

/// Compute the pauli expectatation value of a density matrix with x
#[pyfunction]
#[pyo3(text_signature = "(data, num_qubits, z_mask, x_mask, phase, x_max, /)")]
pub fn density_expval_pauli_with_x(
    data: PyReadonlyArray1<Complex64>,
    num_qubits: usize,
    z_mask: usize,
    x_mask: usize,
    phase: Complex64,
    x_max: u32,
) -> PyResult<f64> {
    if num_qubits >= usize::BITS as usize {
        return Err(PyOverflowError::new_err(format!(
            "The value for num_qubits, {num_qubits}, is too large and would overflow",
        )));
    }
    let data_arr = data.as_slice()?;
    let mask_u = !(2_usize.pow(x_max + 1) - 1);
    let mask_l = 2_usize.pow(x_max) - 1;
    let num_rows = 1_usize << num_qubits;
    let run_in_parallel = getenv_use_multiple_threads();
    let map_fn = |i: usize| -> f64 {
        let index_vec = ((i << 1) & mask_u) | (i & mask_l);
        let index_mat = (index_vec ^ x_mask) + num_rows * index_vec;
        let mut val = 2. * (phase * data_arr[index_mat]).re;
        if (index_vec & z_mask).count_ones() & 1 != 0 {
            val *= -1.
        }
        val
    };
    if num_qubits < PARALLEL_THRESHOLD || !run_in_parallel {
        Ok(fast_sum(
            &(0..num_rows >> 1).map(map_fn).collect::<Vec<f64>>(),
        ))
    } else {
        Ok((0..num_rows >> 1).into_par_iter().map(map_fn).sum())
    }
}

#[pymodule]
pub fn pauli_expval(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(expval_pauli_no_x))?;
    m.add_wrapped(wrap_pyfunction!(expval_pauli_with_x))?;
    m.add_wrapped(wrap_pyfunction!(density_expval_pauli_with_x))?;
    m.add_wrapped(wrap_pyfunction!(density_expval_pauli_no_x))?;
    Ok(())
}
