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

use mcx::{c3x, c4x, synth_mcx_n_dirty_i15, synth_mcx_noaux_v24};
use pyo3::prelude::*;
use qiskit_circuit::circuit_data::CircuitData;

mod mcmt;
mod mcx;

/// Synthesize a multi-controlled X gate with :math:`k` controls based on the paper
/// by Iten et al. [1].
///
/// For :math:`k\ge 4` the method uses :math:`k - 2` dirty ancillary qubits, producing a circuit
/// with :math:`2 * k - 1` qubits and at most :math:`8 * k - 6` CX gates. For :math:`k\le 3`
/// explicit efficient circuits are used instead.
///
/// # Arguments
/// - num_controls: the number of control qubits.
/// - relative_phase: when set to `true`, the method applies the optimized multi-controlled
///   X gate up to a relative phase, in a way that the relative phases of the `action part`
///   cancel out with the relative phases of the `reset part`.
/// - action_only: when set to `true`, the methods applies only the `action part`.
///
/// # References
///
/// 1. Iten et al., *Quantum Circuits for Isometries*, Phys. Rev. A 93, 032318 (2016),
/// [arXiv:1501.06911] (http://arxiv.org/abs/1501.06911).
#[pyfunction]
#[pyo3(name="synth_mcx_n_dirty_i15", signature = (num_controls, relative_phase=false, action_only=false))]
fn py_synth_mcx_n_dirty_i15(
    num_controls: usize,
    relative_phase: bool,
    action_only: bool,
) -> PyResult<CircuitData> {
    synth_mcx_n_dirty_i15(num_controls, relative_phase, action_only)
}

/// Synthesize a multi-controlled X gate with :math:`k` controls based on
/// the implementation for `MCPhaseGate`.
///
/// In turn, the MCPhase gate uses the decomposition for multi-controlled
/// special unitaries described in [1].
///
/// # Arguments
/// - num_controls: the number of control qubits.
///
/// # Returns
///
/// A quantum circuit with :math:`k + 1` qubits. The number of CX-gates is
/// quadratic in :math:`k`.
///
/// # References
///
/// 1. Vale et. al., *Circuit Decomposition of Multicontrolled Special Unitary
/// Single-Qubit Gates*, IEEE TCAD 43(3) (2024),
/// [arXiv:2302.06377] (https://arxiv.org/abs/2302.06377).
#[pyfunction]
#[pyo3(name="synth_mcx_noaux_v24", signature = (num_controls))]
fn py_synth_mcx_noaux_v24(py: Python, num_controls: usize) -> PyResult<CircuitData> {
    synth_mcx_noaux_v24(py, num_controls)
}

pub fn multi_controlled(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(c3x, m)?)?;
    m.add_function(wrap_pyfunction!(c4x, m)?)?;
    m.add_function(wrap_pyfunction!(py_synth_mcx_n_dirty_i15, m)?)?;
    m.add_function(wrap_pyfunction!(py_synth_mcx_noaux_v24, m)?)?;
    m.add_function(wrap_pyfunction!(mcmt::mcmt_v_chain, m)?)?;
    Ok(())
}
