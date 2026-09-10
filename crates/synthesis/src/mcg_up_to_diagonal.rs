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
use pyo3::exceptions::PyValueError;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::ucrz::get_ucrz;
use qiskit_circuit::Qubit;
use qiskit_circuit::bit::ShareableQubit;
use qiskit_circuit::circuit_data::{CircuitData, CircuitDataError};
use qiskit_circuit::operations::Param;
use nalgebra::{Matrix2, MatrixView2, Vector2};
use num_complex::Complex64;

// pub(crate) fn diagonal_gate_circuit(
//     diag_phases: &mut [f64],
//     num_qubits: usize,
// ) -> Result<CircuitData, CircuitDataError> {
//     let out_qubits = (0..num_qubits)
//         .map(|_| ShareableQubit::new_anonymous())
//         .collect::<Vec<_>>();
//     let mut circuit = CircuitData::new(Some(out_qubits), None, Param::Float(0.))?;

//     let mut n = diag_phases.len();

//     while n >= 2 {
//         let mut angles_rz = Vec::<f64>::new();
//         for i in (0..n).step_by(2) {
//             let phi1 = diag_phases[i];
//             let phi2 = diag_phases[i + 1];
//             diag_phases[i / 2] = (phi1 + phi2) / 2.0;
//             angles_rz.push(phi2 - phi1);
//         }
//         let num_act_qubits = n.trailing_zeros() as usize;
//         let target_qubit = num_qubits - num_act_qubits;
//         let ucrz = get_ucrz(num_act_qubits, &mut angles_rz, true)?;

//         let qubit_map: Vec<Qubit> = (0..num_act_qubits)
//             .map(|q| Qubit((q + target_qubit) as u32))
//             .collect();
//         append(&mut circuit, ucrz, &qubit_map)?;
//         n /= 2;
//     }
//     circuit.add_global_phase(&Param::Float(diag_phases[0]))?;
//     Ok(circuit)
// }

 pub(crate) fn mcg_up_to_diagonal_inner(gate: Matrix2<Complex64>, num_ctrls: u32)
 ->Result<(CircuitData, Vec<Complex64>), CircuitDataError> 
 {
    let num_qubits = num_ctrls + 1;
    let mut circuit = CircuitData::with_capacity(num_qubits, 0, 0, Param::Float(0.0))?;    
    let diagonal:Vec<Complex64> = Vec::with_capacity(num_qubits as usize);
    Ok((circuit, diagonal))

 }

#[pyfunction]
pub fn mcg_up_to_diagonal_synth(py: Python,  gate: PyReadonlyArray2<Complex64>, num_ctrls: u32) -> PyResult<Py<PyAny>> {
    // let expected = 1u64 << num_qubits;
    // let got = diag_phases.len();
    // if got as u64 != expected {
    //     return Err(PyValueError::new_err(format!(
    //         "expected {expected} diagonal phases for {num_qubits} qubits, got {got}"
    //     )));
    // }
    //let mut phases = diag_phases;
    let gate_rs: Matrix2<Complex64> = gate.try_as_matrix().map(|m: MatrixView2<Complex64>| m.into_owned()).ok_or_else(|| {
                PyValueError::new_err("expected a 2x2 unitary matrix for each single-qubit gate")
            })?;

    let (circuit,_) = mcg_up_to_diagonal_inner(gate_rs, num_ctrls).map_err(PyErr::from)?;
    let qc = circuit.into_py_quantum_circuit(py)?;
    qc.setattr("name", "mcg")?;
    Ok(qc.unbind())
}

pub fn mcg_up_to_diagonal(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mcg_up_to_diagonal_synth, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
   
}
