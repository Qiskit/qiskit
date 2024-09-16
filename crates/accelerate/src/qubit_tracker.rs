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

use ahash::HashSet;
use pyo3::prelude::*;
use qiskit_circuit::Qubit;

/// Track qubits by their state.
#[pyclass]
pub struct QubitTracker {
    qubits: Vec<Qubit>,
    clean: HashSet<Qubit>,
    dirty: HashSet<Qubit>,
}

#[pymethods]
impl QubitTracker {
    #[new]
    pub fn new(qubits: Vec<Qubit>, clean: HashSet<Qubit>, dirty: HashSet<Qubit>) -> Self {
        QubitTracker {
            qubits,
            clean,
            dirty,
        }
    }

    pub fn print(&self) {
        println!("Qubits: {:?}", self.qubits);
        println!("Clean: {:?}", self.clean);
        println!("Clean: {:?}", self.dirty);
    }
}

pub fn qubit_tracker_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<QubitTracker>()?;
    Ok(())
}
