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

use super::{Qargs, Target};
use pyo3::{prelude::*, pyclass};

/**
 A representation of an ``InstructionProperties`` object.
*/
#[pyclass(
    subclass,
    name = "BaseInstructionProperties",
    module = "qiskit._accelerate.target"
)]
#[derive(Clone, Debug)]
pub struct InstructionProperties {
    #[pyo3(get, set, name = "_duration")]
    pub duration: Option<f64>,
    #[pyo3(get, set, name = "_error")]
    pub error: Option<f64>,
}

#[pymethods]
impl InstructionProperties {
    /// Create a new ``BaseInstructionProperties`` object
    ///
    /// Args:
    ///     duration (Option<f64>): The duration, in seconds, of the instruction on the
    ///         specified set of qubits
    ///     error (Option<f64>): The average error rate for the instruction on the specified
    ///         set of qubits.
    ///     calibration (Option<PyObject>): The pulse representation of the instruction.
    #[new]
    #[pyo3(signature = (duration=None, error=None))]
    pub fn new(duration: Option<f64>, error: Option<f64>) -> Self {
        Self { error, duration }
    }

    fn __getstate__(&self) -> (Option<f64>, Option<f64>) {
        (self.duration, self.error)
    }

    fn __setstate__(&mut self, state: (Option<f64>, Option<f64>)) {
        self.duration = state.0;
        self.error = state.1;
    }

    fn __repr__(&self, _py: Python<'_>) -> String {
        format!(
            "InstructionProperties(duration={}, error={})",
            if let Some(duration) = self.duration {
                duration.to_string()
            } else {
                "None".to_string()
            },
            if let Some(error) = self.error {
                error.to_string()
            } else {
                "None".to_string()
            }
        )
    }
}

/// A mutable view of an `InstructionProperties` object that's stored solely in Rust.
/// Only exposed to python when accessing the gate map in the `Target`.
#[pyclass(
    weakref,
    mapping,
    name = "InstructionPropertiesViewMut",
    module = "qiskit._accelerate.target"
)]
pub(super) struct InstructionPropertiesViewMut {
    pub(super) source: Py<Target>,
    pub(super) key: String,
    pub(super) sub_key: Box<Option<Qargs>>,
}

#[pymethods]
impl InstructionPropertiesViewMut {
    #[getter]
    fn get_duration(&self, py: Python) -> Option<f64> {
        let borrowed_source = self.source.borrow(py);
        if let Some(inst_prop) = borrowed_source[&self.key][self.sub_key.as_ref().as_ref()].as_ref()
        {
            inst_prop.duration.as_ref().copied()
        } else {
            None
        }
    }

    #[getter]
    fn get_error(&self, py: Python) -> Option<f64> {
        let borrowed_source = self.source.borrow(py);
        if let Some(inst_prop) = borrowed_source[&self.key][self.sub_key.as_ref().as_ref()].as_ref()
        {
            inst_prop.error.as_ref().copied()
        } else {
            None
        }
    }

    #[setter]
    fn set_duration(&self, py: Python, new_val: Option<f64>) {
        let mut borrowed_source = self.source.borrow_mut(py);
        if let Some(Some(inst_prop)) =
            borrowed_source.gate_map[&self.key].get_mut(self.sub_key.as_ref().as_ref())
        {
            inst_prop.duration = new_val;
        } else {
            unreachable!("This entry should exist in the target");
        }
    }

    #[setter]
    fn set_error(&self, py: Python, new_val: Option<f64>) {
        let mut borrowed_source = self.source.borrow_mut(py);
        if let Some(Some(inst_prop)) =
            borrowed_source.gate_map[&self.key].get_mut(self.sub_key.as_ref().as_ref())
        {
            inst_prop.error = new_val;
        } else {
            unreachable!("This entry should exist in the target");
        }
    }

    fn __repr__(&self, py: Python) -> String {
        format!(
            "InstructionPropertiesViewMut(duration={}, error={})",
            if let Some(duration) = self.get_duration(py) {
                duration.to_string()
            } else {
                "None".to_string()
            },
            if let Some(error) = self.get_error(py) {
                error.to_string()
            } else {
                "None".to_string()
            }
        )
    }
}
