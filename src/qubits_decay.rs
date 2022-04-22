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

use numpy::IntoPyArray;
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::Python;

/// An unsigned integer Vector based layout class
///
/// This class tracks the layout (or mapping between virtual qubits in the the
/// circuit and physical qubits on the physical device) efficiently
///
/// Args:
///     qubit_indices (dict): A dictionary mapping the virtual qubit index in the circuit to the
///         physical qubit index on the coupling graph.
///     logical_qubits (int): The number of logical qubits in the layout
///     physical_qubits (int): The number of physical qubits in the layout
#[pyclass(module = "qiskit._accelerate.sabre_swap")]
#[pyo3(text_signature = "(qubit_indices, logical_qubits, physical_qubits, /)")]
#[derive(Clone, Debug)]
pub struct QubitsDecay {
    pub decay: Vec<f64>,
}

#[pymethods]
impl QubitsDecay {
    #[new]
    pub fn new(qubit_count: usize) -> Self {
        QubitsDecay {
            decay: vec![1.; qubit_count],
        }
    }

    // Mapping Protocol
    pub fn __len__(&self) -> usize {
        self.decay.len()
    }

    pub fn __contains__(&self, object: f64) -> bool {
        self.decay.contains(&object)
    }

    pub fn __getitem__(&self, object: usize) -> PyResult<f64> {
        match self.decay.get(object) {
            Some(val) => Ok(*val),
            None => Err(PyIndexError::new_err(format!(
                "Index {} out of range for this EdgeList",
                object
            ))),
        }
    }

    pub fn __setitem__(mut slf: PyRefMut<Self>, object: usize, value: f64) -> PyResult<()> {
        if object > slf.decay.len() {
            return Err(PyIndexError::new_err(format!(
                "Index {} out of range for this EdgeList",
                object
            )));
        }
        slf.decay[object] = value;
        Ok(())
    }

    pub fn __array__(&self, py: Python) -> PyObject {
        self.decay.clone().into_pyarray(py).into()
    }

    pub fn reset(mut slf: PyRefMut<Self>) {
        for v in &mut slf.decay {
            *v = 1.;
        }
    }
}
