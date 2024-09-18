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

use std::env;

use pyo3::prelude::*;
use pyo3::Python;

mod convert_2q_block_matrix;
mod dense_layout;
mod edge_collections;
mod error_map;
mod euler_one_qubit_decomposer;
mod nlayout;
mod optimize_1q_gates;
mod pauli_exp_val;
mod results;
mod sabre_layout;
mod sabre_swap;
mod sampled_exp_val;
mod sparse_pauli_op;
mod stochastic_swap;
mod vf2_layout;

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

#[inline(always)]
#[doc(hidden)]
fn add_submodule<F>(py: Python, m: &PyModule, constructor: F, name: &str) -> PyResult<()>
where
    F: FnOnce(Python, &PyModule) -> PyResult<()>,
{
    let new_mod = PyModule::new(py, name)?;
    constructor(py, new_mod)?;
    m.add_submodule(new_mod)
}

#[pymodule]
fn _accelerate(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    add_submodule(py, m, nlayout::nlayout, "nlayout")?;
    add_submodule(py, m, stochastic_swap::stochastic_swap, "stochastic_swap")?;
    add_submodule(py, m, sabre_swap::sabre_swap, "sabre_swap")?;
    add_submodule(py, m, pauli_exp_val::pauli_expval, "pauli_expval")?;
    add_submodule(py, m, dense_layout::dense_layout, "dense_layout")?;
    add_submodule(py, m, error_map::error_map, "error_map")?;
    add_submodule(py, m, sparse_pauli_op::sparse_pauli_op, "sparse_pauli_op")?;
    add_submodule(py, m, results::results, "results")?;
    add_submodule(
        py,
        m,
        optimize_1q_gates::optimize_1q_gates,
        "optimize_1q_gates",
    )?;
    add_submodule(py, m, sampled_exp_val::sampled_exp_val, "sampled_exp_val")?;
    add_submodule(py, m, sabre_layout::sabre_layout, "sabre_layout")?;
    add_submodule(py, m, vf2_layout::vf2_layout, "vf2_layout")?;
    add_submodule(
        py,
        m,
        euler_one_qubit_decomposer::euler_one_qubit_decomposer,
        "euler_one_qubit_decomposer",
    )?;
    add_submodule(
        py,
        m,
        convert_2q_block_matrix::convert_2q_block_matrix,
        "convert_2q_block_matrix",
    )?;
    Ok(())
}
