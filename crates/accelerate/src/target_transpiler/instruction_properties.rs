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

use pyo3::{prelude::*, pyclass};

/**
 A representation of the properties of a gate implementation.

This class provides the optional properties that a backend can provide
about an instruction. These represent the set that the transpiler can
currently work with if present. However, if your backend provides additional
properties for instructions you should subclass this to add additional
custom attributes for those custom/additional properties by the backend.
*/
#[pyclass(subclass, module = "qiskit._accelerate.target")]
#[derive(Clone, Debug)]
pub struct BaseInstructionProperties {
    #[pyo3(get, set)]
    pub duration: Option<f64>,
    #[pyo3(get, set)]
    pub error: Option<f64>,
}

#[pymethods]
impl BaseInstructionProperties {
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
    pub fn new(_py: Python<'_>, duration: Option<f64>, error: Option<f64>) -> Self {
        Self { error, duration }
    }

    fn __getstate__(&self) -> PyResult<(Option<f64>, Option<f64>)> {
        Ok((self.duration, self.error))
    }

    fn __setstate__(
        &mut self,
        _py: Python<'_>,
        state: (Option<f64>, Option<f64>, Bound<PyAny>),
    ) -> PyResult<()> {
        self.duration = state.0;
        self.error = state.1;
        Ok(())
    }

    fn __repr__(&self, _py: Python<'_>) -> PyResult<String> {
        let mut output = "InstructionProperties(".to_owned();
        if let Some(duration) = self.duration {
            output.push_str("duration=");
            output.push_str(duration.to_string().as_str());
            output.push_str(", ");
        } else {
            output.push_str("duration=None, ");
        }

        if let Some(error) = self.error {
            output.push_str("error=");
            output.push_str(error.to_string().as_str());
            output.push_str(", ");
        } else {
            output.push_str("error=None, ");
        }
        Ok(output)
    }
}