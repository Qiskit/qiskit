// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::classical::expr::{ExprKind, PyExpr};
use crate::classical::types::Type;
use crate::duration::Duration;
use num_bigint::BigUint;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use pyo3::{IntoPyObjectExt, intern};

/// A single scalar or 1-D array value expression.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Duration(Duration),
    Float { raw: f64, ty: Type },
    Uint { raw: BigUint, ty: Type },
    Array { elems: Vec<Value>, ty: Type },
}

impl Value {
    /// Construct a [`Value`] from a Python object and a resolved type.
    fn from_py(value: &Bound<PyAny>, ty: Type) -> PyResult<Self> {
        match ty {
            Type::Array { size, .. } => {
                let seq = value.extract::<Vec<Bound<PyAny>>>().map_err(|_| {
                    PyTypeError::new_err("array Value requires a sequence of scalar elements")
                })?;
                if seq.len() != size as usize {
                    return Err(PyValueError::new_err(format!(
                        "array Value has {} elements but type is '{}'",
                        seq.len(),
                        format_type(ty)
                    )));
                }
                let Some(elem_ty) = ty.array_element() else {
                    unreachable!("Type::Array always has an element type")
                };
                let mut elems = Vec::with_capacity(seq.len());
                for item in seq {
                    let inner = Self::from_py(&item, elem_ty)?;
                    if matches!(inner, Value::Array { .. }) {
                        return Err(PyTypeError::new_err(
                            "nested arrays are not supported as Value elements",
                        ));
                    }
                    elems.push(inner);
                }
                Ok(Value::Array { elems, ty })
            }
            _ => {
                if let Ok(raw) = value.extract::<BigUint>() {
                    Ok(Value::Uint { raw, ty })
                } else if let Ok(raw) = value.extract::<f64>() {
                    Ok(Value::Float { raw, ty })
                } else {
                    Ok(Value::Duration(value.extract()?))
                }
            }
        }
    }

    fn payload_into_py(&self, py: Python) -> PyResult<Py<PyAny>> {
        match self {
            Value::Duration(d) => d.into_py_any(py),
            Value::Float { raw, .. } => raw.into_py_any(py),
            Value::Uint { raw, .. } => raw.into_py_any(py),
            Value::Array { elems, .. } => {
                let list = PyList::empty(py);
                for elem in elems {
                    list.append(elem.payload_into_py(py)?)?;
                }
                list.into_py_any(py)
            }
        }
    }

    pub fn ty(&self) -> Type {
        match self {
            Value::Duration(_) => Type::Duration,
            Value::Float { ty, .. } | Value::Uint { ty, .. } | Value::Array { ty, .. } => *ty,
        }
    }
}

fn format_type(ty: Type) -> String {
    match ty {
        Type::Bool => "Bool()".to_string(),
        Type::Duration => "Duration()".to_string(),
        Type::Float => "Float()".to_string(),
        Type::Uint(w) => format!("Uint({w})"),
        Type::Array {
            elem,
            elem_width,
            size,
        } => format!(
            "Array({}, {size})",
            format_type(Type::from_scalar(elem, elem_width))
        ),
    }
}

impl<'py> IntoPyObject<'py> for Value {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Bound::new(py, (PyValue(self), PyExpr(ExprKind::Value)))?.into_any())
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Value {
    type Error = <PyValue as FromPyObject<'a, 'py>>::Error;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let PyValue(v) = ob.extract()?;
        Ok(v)
    }
}

/// A single scalar or 1-D array value.
#[pyclass(
    eq,
    extends = PyExpr,
    name = "Value",
    module = "qiskit._accelerate.circuit.classical.expr",
    from_py_object
)]
#[derive(PartialEq, Clone, Debug)]
pub struct PyValue(Value);

#[pymethods]
impl PyValue {
    #[new]
    #[pyo3(text_signature = "(value, type)")]
    fn new(py: Python, value: Bound<PyAny>, ty: Type) -> PyResult<Py<Self>> {
        Py::new(
            py,
            (
                PyValue(Value::from_py(&value, ty)?),
                PyExpr(ExprKind::Value),
            ),
        )
    }

    #[getter]
    fn get_value(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.payload_into_py(py)
    }

    #[getter]
    fn get_const(&self) -> bool {
        true
    }

    #[getter]
    fn get_type(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.ty().into_py_any(py)
    }

    fn accept<'py>(
        slf: PyRef<'py, Self>,
        visitor: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        visitor.call_method1(intern!(visitor.py(), "visit_value"), (slf,))
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        (
            py.get_type::<Self>(),
            (self.get_value(py)?, self.get_type(py)?),
        )
            .into_pyobject(py)
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        Ok(format!(
            "Value({}, {})",
            self.get_value(py)?.bind(py).repr()?,
            self.get_type(py)?.bind(py).repr()?,
        ))
    }
}
