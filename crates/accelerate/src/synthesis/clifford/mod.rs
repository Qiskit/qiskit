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

mod bm_synthesis;
mod greedy_synthesis;
mod utils;

use crate::synthesis::clifford::bm_synthesis::synth_clifford_bm_inner;
use crate::synthesis::clifford::greedy_synthesis::GreedyCliffordSynthesis;
use crate::QiskitError;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::Param;

/// Create a circuit that synthesizes a given Clifford operator represented as a tableau.
///
/// This is an implementation of the "greedy Clifford compiler" presented in
/// Appendix A of the paper "Clifford Circuit Optimization with Templates and Symbolic
/// Pauli Gates" by Bravyi, Shaydulin, Hu, and Maslov (2021), `<https://arxiv.org/abs/2105.02291>`__.
///
/// This method typically yields better CX cost compared to the Aaronson-Gottesman method.
///
/// Note that this function only implements the greedy Clifford compiler and not the
/// templates and symbolic Pauli gates optimizations that are also described in the paper.
#[pyfunction]
#[pyo3(signature = (clifford))]
fn synth_clifford_greedy(py: Python, clifford: PyReadonlyArray2<bool>) -> PyResult<CircuitData> {
    let tableau = clifford.as_array();
    let mut greedy_synthesis =
        GreedyCliffordSynthesis::new(tableau.view()).map_err(QiskitError::new_err)?;
    let (num_qubits, clifford_gates) = greedy_synthesis.run().map_err(QiskitError::new_err)?;

    CircuitData::from_standard_gates(py, num_qubits as u32, clifford_gates, Param::Float(0.0))
}

/// Create a circuit that optimally synthesizes a given Clifford operator represented as
/// a tableau for Cliffords up to 3 qubits.
///
/// This implementation follows the paper "Hadamard-free circuits expose the structure
/// of the Clifford group" by S. Bravyi, D. Maslov (2020), `<https://arxiv.org/abs/2003.09412>`__.
#[pyfunction]
#[pyo3(signature = (clifford))]
fn synth_clifford_bm(py: Python, clifford: PyReadonlyArray2<bool>) -> PyResult<CircuitData> {
    let tableau = clifford.as_array();
    let (num_qubits, clifford_gates) =
        synth_clifford_bm_inner(tableau).map_err(QiskitError::new_err)?;
    CircuitData::from_standard_gates(py, num_qubits as u32, clifford_gates, Param::Float(0.0))
}

#[pymodule]
pub fn clifford(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(synth_clifford_greedy, m)?)?;
    m.add_function(wrap_pyfunction!(synth_clifford_bm, m)?)?;
    Ok(())
}
