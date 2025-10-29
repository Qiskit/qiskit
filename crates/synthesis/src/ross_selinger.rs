// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use ndarray::ArrayView2;
use num_complex::Complex64;

use pyo3::prelude::*;

use numpy::PyReadonlyArray2;

use rsgridsynth::config::config_from_theta_epsilon;
use rsgridsynth::gridsynth::gridsynth_gates;

use crate::QiskitError;
use crate::euler_one_qubit_decomposer::params_zxz_inner;
use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{Param, StandardGate};

use std::f64::consts::PI;
const PI4: f64 = PI / 4.0;

/// Creates a circuit from the (iterator over) gate names that are returned by rsgridsynth.
fn circuit_from_string<S>(s: S, phase: f64, instruction_capacity: usize) -> PyResult<CircuitData>
where
    S: IntoIterator<Item = char>,
{
    let qubit = [Qubit(0)];
    let mut circuit =
        CircuitData::with_capacity(1u32, 0, instruction_capacity, Param::Float(phase))?;
    for c in s {
        match c {
            'I' => {}
            'H' => {
                circuit.push_standard_gate(StandardGate::H, &[], &qubit)?;
            }
            'S' => {
                circuit.push_standard_gate(StandardGate::S, &[], &qubit)?;
            }
            'T' => {
                circuit.push_standard_gate(StandardGate::T, &[], &qubit)?;
            }
            'X' => {
                circuit.push_standard_gate(StandardGate::X, &[], &qubit)?;
            }
            'W' => {
                // Note that W in gridsynth represents the global phase update by pi/4,
                // and not the Clifford W-gate used in Qiskit.
                circuit.add_global_phase(&Param::Float(PI4))?;
            }
            _ => {
                return Err(QiskitError::new_err(format!(
                    "Unknown character in gridsynth output: {c}"
                )));
            }
        };
    }
    Ok(circuit)
}

#[pyfunction]
pub fn approximate_rz_rotation(theta: f64, epsilon: f64) -> PyResult<CircuitData> {
    let gates = gridsynth_gates(&mut config_from_theta_epsilon(theta, epsilon, 0u64));
    let gates_iter = gates.chars();
    let instruction_capacity = gates.len();
    circuit_from_string(gates_iter, 0., instruction_capacity)
}

/// Approximates 1q unitary matrix using Ross-Selinger algorithm
/// as implemented in https://github.com/qiskit-community/rsgridsynth.
pub fn approximate_1q_unitary_inner(
    mat: ArrayView2<Complex64>,
    epsilon: f64,
) -> PyResult<CircuitData> {
    // Run ZXZ decomposiition
    let [theta, phi, lambda, phase] = params_zxz_inner(mat);

    // Approximate each of the RZ, RX, RZ rotations using rsgridsynth and join the results.
    let gates_theta = gridsynth_gates(&mut config_from_theta_epsilon(theta, epsilon, 0u64));
    let gates_phi = gridsynth_gates(&mut config_from_theta_epsilon(phi, epsilon, 0u64));
    let gates_lambda = gridsynth_gates(&mut config_from_theta_epsilon(lambda, epsilon, 0u64));
    let instruction_capacity = gates_theta.len() + gates_phi.len() + gates_lambda.len() + 2;
    let gates_iter = gates_lambda
        .chars()
        .chain(std::iter::once('H'))
        .chain(gates_theta.chars())
        .chain(std::iter::once('H'))
        .chain(gates_phi.chars());
    circuit_from_string(gates_iter, phase, instruction_capacity)
}

/// Approximates 1q unitary matrix using Ross-Selinger algorithm
/// as implemented in https://github.com/qiskit-community/rsgridsynth.
#[pyfunction]
pub fn approximate_1q_unitary(
    unitary: PyReadonlyArray2<Complex64>,
    epsilon: f64,
) -> PyResult<CircuitData> {
    approximate_1q_unitary_inner(unitary.as_array(), epsilon)
}

pub fn ross_selinger_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(approximate_rz_rotation, m)?)?;
    m.add_function(wrap_pyfunction!(approximate_1q_unitary, m)?)?;
    Ok(())
}
