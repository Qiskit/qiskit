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

/// A container for qubit decay values for each qubit
///
/// This class tracks the qubit decay for the sabre heuristic. When initialized
/// all qubits are set to a value of ``1.``. This class implements the sequence
/// protocol and can be modified in place like any python sequence.
///
/// Args:
///     qubit_count (int): The number of qubits
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
        if object >= slf.decay.len() {
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

    pub fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.decay))
    }

    /// Reset decay for all qubits back to default ``1.``
    #[pyo3(text_signature = "(self, /)")]
    pub fn reset(mut slf: PyRefMut<Self>) {
        slf.decay.fill_with(|| 1.);
    }
}
