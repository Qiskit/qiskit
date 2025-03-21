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
use pyo3::sync::GILOnceCell;
use pyo3::PyTypeInfo;

static BOOL_TYPE: GILOnceCell<Py<PyBool>> = GILOnceCell::new();
static DURATION_TYPE: GILOnceCell<Py<PyDuration>> = GILOnceCell::new();
static FLOAT_TYPE: GILOnceCell<Py<PyFloat>> = GILOnceCell::new();

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Type {
    Bool,
    Duration,
    Float,
    Uint(u16),
}

impl<'py> IntoPyObject<'py> for Type {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Type::Bool => Ok(PyBool::new(py).into_bound(py).into_any()),
            Type::Duration => Ok(PyDuration::new(py).into_bound(py).into_any()),
            Type::Float => Ok(PyFloat::new(py).into_bound(py).into_any()),
            Type::Uint(n) => Ok(PyUint::new(py, n).into_bound(py).into_any()),
        }
    }
}

impl<'py> FromPyObject<'py> for Type {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let PyType(kind) = ob.extract()?;
        Ok(match kind {
            TypeKind::Bool => Type::Bool,
            TypeKind::Duration => Type::Duration,
            TypeKind::Float => Type::Float,
            TypeKind::Uint => {
                let PyUint(n) = ob.extract()?;
                Type::Uint(n)
            }
        })
    }
}

#[pyclass(
    eq,
    hash,
    subclass,
    frozen,
    name = "Type",
    module = "qiskit._accelerate.circuit"
)]
#[derive(PartialEq, Clone, Copy, Debug, Hash)]
struct PyType(TypeKind);

#[pymethods]
impl PyType {
    #[getter]
    fn get_kind(&self, py: Python) -> Py<PyAny> {
        match self.0 {
            TypeKind::Bool => PyBool::type_object(py).into_any().unbind(),
            TypeKind::Duration => PyDuration::type_object(py).into_any().unbind(),
            TypeKind::Float => PyFloat::type_object(py).into_any().unbind(),
            TypeKind::Uint => PyUint::type_object(py).into_any().unbind(),
        }
    }
}

#[repr(u8)]
#[derive(PartialEq, Clone, Copy, Debug, Hash)]
enum TypeKind {
    Bool,
    Duration,
    Float,
    Uint,
}

#[pyclass(eq, hash, extends = PyType, frozen, name = "Bool", module = "qiskit._accelerate.circuit")]
#[derive(PartialEq, Clone, Copy, Debug, Hash)]
struct PyBool;

#[pymethods]
impl PyBool {
    #[new]
    fn new(py: Python) -> Py<Self> {
        BOOL_TYPE
            .get_or_init(py, || {
                Py::new(py, (PyBool, PyType(TypeKind::Bool))).unwrap()
            })
            .clone_ref(py)
    }
}

#[pyclass(eq, hash, extends = PyType, frozen, name = "Duration", module = "qiskit._accelerate.circuit")]
#[derive(PartialEq, Clone, Copy, Debug, Hash)]
struct PyDuration;

#[pymethods]
impl PyDuration {
    #[new]
    fn new(py: Python) -> Py<Self> {
        DURATION_TYPE
            .get_or_init(py, || {
                Py::new(py, (PyDuration, PyType(TypeKind::Duration))).unwrap()
            })
            .clone_ref(py)
    }
}

#[pyclass(eq, hash, extends = PyType, frozen, name = "Float", module = "qiskit._accelerate.circuit")]
#[derive(PartialEq, Clone, Copy, Debug, Hash)]
struct PyFloat;

#[pymethods]
impl PyFloat {
    #[new]
    fn new(py: Python) -> Py<Self> {
        FLOAT_TYPE
            .get_or_init(py, || {
                Py::new(py, (PyFloat, PyType(TypeKind::Float))).unwrap()
            })
            .clone_ref(py)
    }
}

#[pyclass(eq, hash, extends = PyType, frozen, name = "Uint", module = "qiskit._accelerate.circuit")]
#[derive(PartialEq, Clone, Copy, Debug, Hash)]
struct PyUint(u16);

#[pymethods]
impl PyUint {
    #[new]
    fn new(py: Python, width: u16) -> Py<Self> {
        Py::new(py, (PyUint(width), PyType(TypeKind::Uint))).unwrap()
    }
}

pub(crate) fn register_python(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyType>()?;
    m.add_class::<PyBool>()?;
    m.add_class::<PyDuration>()?;
    m.add_class::<PyFloat>()?;
    m.add_class::<PyUint>()?;
    Ok(())
}
