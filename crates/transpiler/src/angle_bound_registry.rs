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

use hashbrown::HashMap;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict};
use pyo3::Python;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::PhysicalQubit;

#[derive(Clone)]
pub(crate) enum CallbackType {
    Python(PyObject),
    Native(fn(&[f64], &[PhysicalQubit]) -> DAGCircuit),
}

impl CallbackType {
    fn call(&self, angles: &[f64], qubits: &[PhysicalQubit]) -> PyResult<DAGCircuit> {
        match self {
            Self::Python(inner) => {
                let qubits: Vec<usize> = qubits.iter().map(|x| x.index()).collect();
                Python::with_gil(|py| inner.bind(py).call1((angles, qubits))?.extract())
            }
            Self::Native(inner) => Ok(inner(angles, qubits)),
        }
    }
}

/// Registry of Angle Wrapping function
///
///
#[pyclass(module = "qiskit._accelerate.angle_bound_registry")]
#[pyo3(name = "WrapAngleRegistry")]
pub struct PyWrapAngleRegistry(WrapAngleRegistry);

#[pymethods]
impl PyWrapAngleRegistry {
    #[new]
    pub fn new() -> Self {
        PyWrapAngleRegistry(WrapAngleRegistry::new())
    }

    /// Get a replacement circuit for
    pub fn substitute_angle_bounds(
        &self,
        name: &str,
        angles: Vec<f64>,
        qubits: Vec<u32>,
    ) -> PyResult<Option<DAGCircuit>> {
        let qubits: Vec<PhysicalQubit> = qubits.into_iter().map(PhysicalQubit::new).collect();
        self.0.substitute_angle_bounds(name, &angles, &qubits)
    }

    pub fn add_wrapper(&mut self, name: String, callback: PyObject) {
        self.0.registry.insert(name, CallbackType::Python(callback));
    }

    fn __getstate__(&self, py: Python) -> PyResult<Py<PyDict>> {
        let bounds_dict = PyDict::new(py);
        for (name, bound) in self.get_inner().registry.iter() {
            match bound {
                CallbackType::Python(obj) => {
                    bounds_dict.set_item(name, obj.clone_ref(py))?;
                }
                CallbackType::Native(func) => {
                    bounds_dict.set_item(name, PyCapsule::new(py, *func, None)?)?;
                }
            }
        }
        Ok(bounds_dict.unbind())
    }

    fn __setstate__(&mut self, data: Bound<PyDict>) -> PyResult<()> {
        for (key, val) in data.iter() {
            let name: String = key.extract()?;
            if let Ok(capsule) = val.downcast::<PyCapsule>() {
                // SAFETY: It isn't
                unsafe {
                    self.0.add_native(
                        name,
                        *capsule.reference::<fn(&[f64], &[PhysicalQubit]) -> DAGCircuit>(),
                    );
                }
            } else {
                self.add_wrapper(name, val.unbind());
            }
        }
        Ok(())
    }
}

impl PyWrapAngleRegistry {
    pub fn get_inner(&self) -> &WrapAngleRegistry {
        &self.0
    }
}

pub struct WrapAngleRegistry {
    registry: HashMap<String, CallbackType>,
}

impl WrapAngleRegistry {
    pub fn new() -> Self {
        WrapAngleRegistry {
            registry: HashMap::new(),
        }
    }

    pub fn add_native(
        &mut self,
        name: String,
        callback: fn(&[f64], &[PhysicalQubit]) -> DAGCircuit,
    ) {
        self.registry.insert(name, CallbackType::Native(callback));
    }

    /// Get a replacement circuit for
    pub fn substitute_angle_bounds(
        &self,
        name: &str,
        angles: &[f64],
        qubits: &[PhysicalQubit],
    ) -> PyResult<Option<DAGCircuit>> {
        if let Some(callback) = self.registry.get(name) {
            Some(callback.call(angles, qubits)).transpose()
        } else {
            Err(PyKeyError::new_err("Name: {} not in registry"))
        }
    }
}

impl Default for WrapAngleRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PyWrapAngleRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub fn angle_bound_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyWrapAngleRegistry>()?;
    Ok(())
}
