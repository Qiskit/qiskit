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

use pyo3::prelude::*;

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::circuit_data::CircuitError;
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::operations::Operation;
use qiskit_circuit::operations::OperationRef;
use qiskit_circuit::operations::Param;
use qiskit_circuit::operations::PauliProductMeasurement;
use qiskit_circuit::operations::StandardGate;
use qiskit_circuit::operations::StandardInstruction;
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{Clbit, Qubit};

/// Synthesizes a PauliProductMeasurement instruction.
/// This function is used in HighLevelSynthesis and is exposed to Python's class definition method.
pub fn synthesize_ppm(ppm: &PauliProductMeasurement) -> PyResult<CircuitData> {
    // The code is somewhat similar to pauli_evolution synthesis, however combining the two branches
    // is tricky.
    let num_qubits = ppm.num_qubits();

    let mut circuit = CircuitData::with_capacity(num_qubits, 1, 0, Param::Float(0.0))?;

    // Filtering 'I's
    let pauli_qubits = ppm
        .z
        .iter()
        .zip(ppm.x.iter())
        .enumerate()
        .filter(|(_i, (z, x))| **z || **x)
        .map(|(i, (z, x))| (i, *z, *x))
        .collect::<Vec<(usize, bool, bool)>>();

    let first_qubit = if !pauli_qubits.is_empty() {
        Qubit(pauli_qubits[0].0 as u32)
    } else {
        return Err(CircuitError::new_err(
            "A pauli measurement gate must have at least one non-'I' bit.",
        ));
    };

    // Basis change layer
    for (i, z, x) in pauli_qubits.iter() {
        match (z, x) {
            (_, false) => {}
            (false, true) => {
                circuit.push_standard_gate(StandardGate::H, &[], &[Qubit(*i as u32)])?;
            }
            (true, true) => {
                circuit.push_standard_gate(StandardGate::SX, &[], &[Qubit(*i as u32)])?;
            }
        }
    }

    if ppm.phase == 2 {
        circuit.push_standard_gate(StandardGate::X, &[], &[first_qubit])?;
    }

    // CX-layer
    for w in pauli_qubits.windows(2).rev() {
        circuit.push_standard_gate(
            StandardGate::CX,
            &[],
            &[Qubit(w[1].0 as u32), Qubit(w[0].0 as u32)],
        )?;
    }

    // Z-measurement on first qubit
    circuit.push_packed_operation(
        PackedOperation::from_standard_instruction(StandardInstruction::Measure),
        &[],
        &[first_qubit],
        &[Clbit(0)],
    )?;

    // CX-layer
    for w in pauli_qubits.windows(2) {
        circuit.push_standard_gate(
            StandardGate::CX,
            &[],
            &[Qubit(w[1].0 as u32), Qubit(w[0].0 as u32)],
        )?;
    }

    // Basis change layer
    if ppm.phase == 2 {
        circuit.push_standard_gate(StandardGate::X, &[], &[first_qubit])?;
    }

    // Basis change layer
    for (i, z, x) in pauli_qubits.iter() {
        match (z, x) {
            (_, false) => {}
            (false, true) => {
                circuit.push_standard_gate(StandardGate::H, &[], &[Qubit(*i as u32)])?;
            }
            (true, true) => {
                circuit.push_standard_gate(StandardGate::SXdg, &[], &[Qubit(*i as u32)])?;
            }
        }
    }

    Ok(circuit)
}

#[pyfunction]
fn synth_pauli_product_measurement(operation: &Bound<PyAny>) -> PyResult<CircuitData> {
    let op_from_python = operation.extract::<OperationFromPython>()?;

    if let OperationRef::PauliProductMeasurement(ppm) = op_from_python.operation.view() {
        synthesize_ppm(ppm)
    } else {
        Err(CircuitError::new_err(
            "Calling pauli product measurement synthesis on a non-pauli-product-measurement.",
        ))
    }
}

pub fn pauli_product_measurement_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(synth_pauli_product_measurement))?;
    Ok(())
}
