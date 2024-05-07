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

use pyo3::{
    prelude::*,
    pyclass,
    types::{IntoPyDict, PyType},
};

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
pub struct InstructionProperties {
    #[pyo3(get)]
    pub duration: Option<f64>,
    #[pyo3(get, set)]
    pub error: Option<f64>,
    #[pyo3(get)]
    _calibration: PyObject,
}

#[pymethods]
impl InstructionProperties {
    /**
    Create a new ``InstructionProperties`` object

    Args:
        duration (Option<f64>): The duration, in seconds, of the instruction on the
            specified set of qubits
        error (Option<f64>): The average error rate for the instruction on the specified
            set of qubits.
        calibration (Option<PyObject>): The pulse representation of the instruction.
    */
    #[new]
    #[pyo3(text_signature = "(/, duration: float | None = None,
        error: float | None = None,
        calibration: Schedule | ScheduleBlock | CalibrationEntry | None = None,)")]
    pub fn new(
        py: Python<'_>,
        duration: Option<f64>,
        error: Option<f64>,
        calibration: Option<Bound<PyAny>>,
    ) -> Self {
        let mut instruction_prop = InstructionProperties {
            error,
            duration,
            _calibration: py.None(),
        };
        if let Some(calibration) = calibration {
            let _ = instruction_prop.set_calibration(py, calibration);
        }
        instruction_prop
    }

    /**
    The pulse representation of the instruction.

    .. note::

        This attribute always returns a Qiskit pulse program, but it is internally
        wrapped by the :class:`.CalibrationEntry` to manage unbound parameters
        and to uniformly handle different data representation,
        for example, un-parsed Pulse Qobj JSON that a backend provider may provide.

        This value can be overridden through the property setter in following manner.
        When you set either :class:`.Schedule` or :class:`.ScheduleBlock` this is
        always treated as a user-defined (custom) calibration and
        the transpiler may automatically attach the calibration data to the output circuit.
        This calibration data may appear in the wire format as an inline calibration,
        which may further update the backend standard instruction set architecture.

        If you are a backend provider who provides a default calibration data
        that is not needed to be attached to the transpiled quantum circuit,
        you can directly set :class:`.CalibrationEntry` instance to this attribute,
        in which you should set :code:`user_provided=False` when you define
        calibration data for the entry. End users can still intentionally utilize
        the calibration data, for example, to run pulse-level simulation of the circuit.
        However, such entry doesn't appear in the wire format, and backend must
        use own definition to compile the circuit down to the execution format.
    */
    #[getter]
    pub fn get_calibration(&self, py: Python<'_>) -> PyResult<PyObject> {
        if !&self._calibration.is_none(py) {
            return self._calibration.call_method0(py, "get_schedule");
        }
        Ok(py.None())
    }

    #[setter]
    pub fn set_calibration(&mut self, py: Python<'_>, calibration: Bound<PyAny>) -> PyResult<()> {
        let module = py.import_bound("qiskit.pulse.schedule")?;
        // Import Schedule and ScheduleBlock types.
        let schedule_type = module.getattr("Schedule")?;
        let schedule_type = schedule_type.downcast::<PyType>()?;
        let schedule_block_type = module.getattr("ScheduleBlock")?;
        let schedule_block_type = schedule_block_type.downcast::<PyType>()?;
        if calibration.is_instance(schedule_block_type)?
            || calibration.is_instance(schedule_type)?
        {
            // Import the calibration_entries module
            let calibration_entries = py.import_bound("qiskit.pulse.calibration_entries")?;
            // Import the schedule def class.
            let schedule_def = calibration_entries.getattr("ScheduleDef")?;
            // Create a ScheduleDef instance.
            let new_entry: Bound<PyAny> = schedule_def.call0()?;
            // Definethe schedule, make sure it is user provided.
            let args = (calibration,);
            let kwargs = [("user_provided", true)].into_py_dict_bound(py);
            new_entry.call_method("define", args, Some(&kwargs))?;
            self._calibration = new_entry.unbind();
        } else {
            self._calibration = calibration.unbind();
        }
        Ok(())
    }

    fn __getstate__(&self) -> PyResult<(Option<f64>, Option<f64>, Option<&PyObject>)> {
        Ok((self.duration, self.error, Some(&self._calibration)))
    }

    fn __setstate__(
        &mut self,
        py: Python<'_>,
        state: (Option<f64>, Option<f64>, Bound<PyAny>),
    ) -> PyResult<()> {
        self.duration = state.0;
        self.error = state.1;
        self.set_calibration(py, state.2)?;
        Ok(())
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
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

        if !self.get_calibration(py)?.is_none(py) {
            output.push_str(
                format!(
                    "calibration={:?})",
                    self.get_calibration(py)?
                        .call_method0(py, "__str__")?
                        .extract::<String>(py)?
                )
                .as_str(),
            );
        } else {
            output.push_str("calibration=None)");
        }
        Ok(output)
    }
}
