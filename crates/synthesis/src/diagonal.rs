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

use pyo3::Python;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::qsd::append;
use crate::ucrz::get_ucrz;
use qiskit_circuit::Qubit;
use qiskit_circuit::bit::ShareableQubit;
use qiskit_circuit::circuit_data::{CircuitData, CircuitDataError};
use qiskit_circuit::operations::Param;

pub fn diagonal_gate_circuit(
    diag_phases: &mut [f64],
    num_qubits: usize,
) -> Result<CircuitData, CircuitDataError> {
    let out_qubits = (0..num_qubits)
        .map(|_| ShareableQubit::new_anonymous())
        .collect::<Vec<_>>();
    let mut circuit = CircuitData::new(Some(out_qubits), None, Param::Float(0.))?;

    let mut n = diag_phases.len();

    while n >= 2 {
        let mut angles_rz = Vec::<f64>::new();
        for i in (0..n).step_by(2) {
            let phi1 = diag_phases[i];
            let phi2 = diag_phases[i + 1];
            diag_phases[i / 2] = (phi1 + phi2) / 2.0;
            angles_rz.push(phi2 - phi1);
        }
        let num_act_qubits = n.trailing_zeros() as usize;
        let target_qubit = num_qubits - num_act_qubits;
        let ucrz = get_ucrz(num_act_qubits, &mut angles_rz, true)?;

        let quibit_map: Vec<Qubit> = (0..num_act_qubits)
            .map(|q| Qubit((q + target_qubit) as u32))
            .collect();
        append(&mut circuit, ucrz, &quibit_map)?;
        n /= 2;
    }
    circuit.add_global_phase(&Param::Float(diag_phases[0]))?;
    Ok(circuit)
}

#[pyfunction]
pub fn py_synth_diagonal(
    py: Python,
    diag_pahses: Vec<f64>,
    num_qubits: u32,
) -> PyResult<Py<PyAny>> {
    let mut phases = diag_pahses;
    let circuit = diagonal_gate_circuit(&mut phases, num_qubits as usize).map_err(PyErr::from)?;
    let qc = circuit.into_py_quantum_circuit(py)?;
    qc.setattr("name", "diagonal")?;
    Ok(qc.unbind())
}

pub fn diagonal(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_synth_diagonal, m)?)?;
    Ok(())
}
