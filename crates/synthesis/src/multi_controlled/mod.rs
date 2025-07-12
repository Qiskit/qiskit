// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use mcx::{c3x, c4x, synth_mcx_n_dirty_i15, synth_mcx_noaux_hp24, synth_mcx_noaux_v24};
use pyo3::prelude::*;
use qiskit_circuit::circuit_data::CircuitData;

mod mcmt;
mod mcx;

#[pyfunction]
#[pyo3(name="synth_mcx_n_dirty_i15", signature = (num_controls, relative_phase=false, action_only=false))]
fn py_synth_mcx_n_dirty_i15(
    num_controls: usize,
    relative_phase: bool,
    action_only: bool,
) -> PyResult<CircuitData> {
    synth_mcx_n_dirty_i15(num_controls, relative_phase, action_only)
}

#[pyfunction]
#[pyo3(name="synth_mcx_noaux_v24", signature = (num_controls))]
fn py_synth_mcx_noaux_v24(py: Python, num_controls: usize) -> PyResult<CircuitData> {
    synth_mcx_noaux_v24(py, num_controls)
}

#[pyfunction]
#[pyo3(name="synth_mcx_noaux_hp24", signature = (num_controls))]
fn py_synth_mcx_noaux_hp24(num_controls: usize) -> PyResult<CircuitData> {
    synth_mcx_noaux_hp24(num_controls)
}

pub fn multi_controlled(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(c3x, m)?)?;
    m.add_function(wrap_pyfunction!(c4x, m)?)?;
    m.add_function(wrap_pyfunction!(py_synth_mcx_n_dirty_i15, m)?)?;
    m.add_function(wrap_pyfunction!(py_synth_mcx_noaux_v24, m)?)?;
    m.add_function(wrap_pyfunction!(py_synth_mcx_noaux_hp24, m)?)?;
    m.add_function(wrap_pyfunction!(mcmt::mcmt_v_chain, m)?)?;
    Ok(())
}
