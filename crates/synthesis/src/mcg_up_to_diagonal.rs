// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use core::num;

use numpy::PyReadonlyArray2;
use pyo3::Python;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use qiskit_circuit::operations::OperationRef::StandardGate;

use crate::ucrz::get_ucrz;
use crate::uc_gate::dec_ucg_inner;

use nalgebra::{Matrix2, MatrixView2, Vector2};
use num_complex::Complex64;
use qiskit_circuit::Qubit;
use qiskit_circuit::bit::ShareableQubit;
use qiskit_circuit::circuit_data::{CircuitData, CircuitDataError};
use qiskit_circuit::operations::Param;

pub(crate) fn mcg_up_to_diagonal_inner(
    gate: Matrix2<Complex64>,
    num_ctrls: u32,
) -> Result<(CircuitData, Vec<Complex64>), CircuitDataError> {
    let num_qubits = num_ctrls + 1;

    let mut gates = vec![Matrix2::identity(); 2usize.pow(num_ctrls)];
    let last = gates.len() - 1;
    gates[last] = gate;
    dec_ucg_inner(gates, num_qubits, true, true)
    //let mut circuit = CircuitData::with_capacity(num_qubits, 0, 0, Param::Float(0.0))?;
    //let diagonal:Vec<Complex64> = Vec::with_capacity(num_qubits as usize);
    
}

#[pyfunction]
pub fn mcg_up_to_diagonal_synth(
    py: Python,
    gate: PyReadonlyArray2<Complex64>,
    num_ctrls: u32,
) -> PyResult<(Py<PyAny>,Vec<Complex64>)> {
    let gate_rs: Matrix2<Complex64> = gate
        .try_as_matrix()
        .map(|m: MatrixView2<Complex64>| m.into_owned())
        .ok_or_else(|| {
            PyValueError::new_err("expected a 2x2 unitary matrix for each single-qubit gate")
        })?;

    let (circuit, diag) = mcg_up_to_diagonal_inner(gate_rs, num_ctrls).map_err(PyErr::from)?;
    let qc = circuit.into_py_quantum_circuit(py)?;
    qc.setattr("name", "mcg")?;
    Ok((qc.unbind(), diag))
}

pub fn mcg_up_to_diagonal(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mcg_up_to_diagonal_synth, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {}
