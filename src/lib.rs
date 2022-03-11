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
use std::env;

use pyo3::prelude::*;
use pyo3::wrap_pymodule;
use pyo3::Python;

mod dense_layout;
mod edge_collections;
mod nlayout;
mod pauli_exp_val;
mod stochastic_swap;

#[inline]
pub fn getenv_use_multiple_threads() -> bool {
    let parallel_context = env::var("QISKIT_IN_PARALLEL")
        .unwrap_or_else(|_| "FALSE".to_string())
        .to_uppercase()
        == "TRUE";
    let force_threads = env::var("QISKIT_FORCE_THREADS")
        .unwrap_or_else(|_| "FALSE".to_string())
        .to_uppercase()
        == "TRUE";
    !parallel_context || force_threads
}

const LANES: usize = 8;

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

#[pymodule]
fn _accelerate(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(stochastic_swap::stochastic_swap))?;
    m.add_wrapped(wrap_pymodule!(pauli_exp_val::pauli_expval))?;
    m.add_wrapped(wrap_pymodule!(dense_layout::dense_layout))?;
    Ok(())
}
