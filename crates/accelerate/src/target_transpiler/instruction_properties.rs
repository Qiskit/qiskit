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

use pyo3::{prelude::*, pyclass, types::IntoPyDict};
use qiskit_circuit::imports::ImportOnceCell;

static SCHEDULE: ImportOnceCell = ImportOnceCell::new("qiskit.pulse.schedule", "Schedule");
static SCHEDULE_BLOCK: ImportOnceCell =
    ImportOnceCell::new("qiskit.pulse.schedule", "ScheduleBlock");
static SCHEDULE_DEF: ImportOnceCell =
    ImportOnceCell::new("qiskit.pulse.calibration_entries", "ScheduleDef");

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
    #[pyo3(get, set)]
    pub duration: Option<f64>,
    #[pyo3(get, set)]
    pub error: Option<f64>,
    #[pyo3(get)]
    _calibration: Option<PyObject>,
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
    #[pyo3(signature = (duration=None, error=None, calibration=None))]
    pub fn new(
        py: Python,
        duration: Option<f64>,
        error: Option<f64>,
        calibration: Option<PyObject>,
    ) -> Self {
        let mut instance = Self {
            error,
            duration,
            _calibration: None,
        };
        if let Some(calibration) = calibration {
            let _ = instance.set_calibration(calibration.into_bound(py));
        }
        instance
    }

    /// The pulse representation of the instruction.
    ///
    /// .. note::
    ///
    ///     This attribute always returns a Qiskit pulse program, but it is internally
    ///     wrapped by the :class:`.CalibrationEntry` to manage unbound parameters
    ///     and to uniformly handle different data representation,
    ///     for example, un-parsed Pulse Qobj JSON that a backend provider may provide.
    ///
    ///     This value can be overridden through the property setter in following manner.
    ///     When you set either :class:`.Schedule` or :class:`.ScheduleBlock` this is
    ///     always treated as a user-defined (custom) calibration and
    ///     the transpiler may automatically attach the calibration data to the output circuit.
    ///     This calibration data may appear in the wire format as an inline calibration,
    ///     which may further update the backend standard instruction set architecture.
    ///
    ///     If you are a backend provider who provides a default calibration data
    ///     that is not needed to be attached to the transpiled quantum circuit,
    ///     you can directly set :class:`.CalibrationEntry` instance to this attribute,
    ///     in which you should set :code:`user_provided=False` when you define
    ///     calibration data for the entry. End users can still intentionally utilize
    ///     the calibration data, for example, to run pulse-level simulation of the circuit.
    ///     However, such entry doesn't appear in the wire format, and backend must
    ///     use own definition to compile the circuit down to the execution format.
    #[getter]
    fn get_calibration(&self, py: Python) -> PyResult<PyObject> {
        if let Some(calib) = &self._calibration {
            Ok(calib.call_method0(py, "get_schedule")?)
        } else {
            Ok(py.None())
        }
    }

    #[setter]
    fn set_calibration(&mut self, calibration: Bound<PyAny>) -> PyResult<()> {
        let py = calibration.py();
        if calibration.is_instance(SCHEDULE.get_bound(py))?
            || calibration.is_instance(SCHEDULE_BLOCK.get_bound(py))?
        {
            let new_entry = SCHEDULE_DEF.get_bound(py).call0()?;
            new_entry.call_method(
                "define",
                (calibration,),
                Some(&[("user_provided", true)].into_py_dict_bound(py)),
            )?;
            self._calibration = Some(new_entry.into());
        } else {
            self._calibration = Some(calibration.into());
        }
        Ok(())
    }

    fn __getnewargs__(&self, py: Python) -> PyResult<(Option<f64>, Option<f64>, Option<PyObject>)> {
        Ok((
            self.duration,
            self.error,
            self._calibration
                .as_ref()
                .map(|calibration| calibration.clone_ref(py)),
        ))
    }

    fn __repr__(slf: Bound<Self>) -> PyResult<String> {
        let duration = slf.getattr("duration")?.str()?.to_string();
        let error = slf.getattr("error")?.str()?.to_string();
        let calibration = slf.getattr("_calibration")?.str()?.to_string();
        Ok(format!(
            "InstructionProperties(duration={}, error={}, calibration={})",
            duration, error, calibration,
        ))
    }
}
