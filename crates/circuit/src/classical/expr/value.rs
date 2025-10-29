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

use crate::classical::expr::{ExprKind, PyExpr};
use crate::classical::types::Type;
use crate::duration::Duration;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use pyo3::{IntoPyObjectExt, intern};

/// A single scalar value expression.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Duration(Duration),
    Float(f64),
    Bool(bool),
    Uint(Vec<u8>, u16), // Vec<u8> encodes the unbounded integer value, u16 encoded the width of the value
}

impl Value {
    pub fn new_small_int(value: u8, width: u16) -> Value {
        Value::Uint(vec![value], width)
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

/// A single scalar value.
#[pyclass(eq, extends = PyExpr, name = "Value", module = "qiskit._accelerate.circuit.classical.expr")]
#[derive(PartialEq, Clone, Debug)]
pub struct PyValue(Value);

#[pymethods]
impl PyValue {
    #[new]
    #[pyo3(text_signature = "(value, type)")]
    fn new(py: Python, py_value: Bound<PyAny>, ty: Type) -> PyResult<Py<Self>> {
        let value = match ty {
            Type::Bool => Value::Bool(py_value.extract()?),
            Type::Float => Value::Float(py_value.extract()?),
            Type::Duration => Value::Duration(py_value.extract()?),
            Type::Uint(width) => {
                let bit_length: usize = py_value.call_method0("bit_length")?.extract()?;
                let byte_len = ((bit_length + 7) / 8).max(1);
                let py_bytes: Bound<PyBytes> = py_value
                    .call_method1("to_bytes", (byte_len, "big"))?
                    .extract()?;
                Value::Uint(py_bytes.as_bytes().to_vec(), width)
            }
        };
        Py::new(py, (PyValue(value), PyExpr(ExprKind::Value)))
    }
    // fn new(py: Python, value: Bound<PyAny>, ty: Type) -> PyResult<Py<Self>> {
    //     let value = if let Ok(raw) = value.extract::<u64>() {
    //         println!("Creating Value::Uint with raw {:?} and ty {:?}", raw, ty);
    //         Value::Uint { raw, ty }
    //     } else if let Ok(raw) = value.extract::<f64>() {
    //         println!("Creating Value::Float with raw {:?} and ty {:?}", raw, ty);
    //         Value::Float { raw, ty }
    //     } else {
    //         Value::Duration(value.extract()?)
    //     };
    //     Py::new(py, (PyValue(value), PyExpr(ExprKind::Value)))
    // }

    #[getter]
    fn get_value(&self, py: Python) -> PyResult<Py<PyAny>> {
        match &self.0 {
            Value::Duration(d) => d.into_py_any(py),
            Value::Float(f) => f.into_py_any(py),
            Value::Bool(b) => b.into_py_any(py),
            Value::Uint(raw, _width) => py
                .import("builtins")?
                .getattr("int")?
                .call_method1("from_bytes", (&raw, "big"))?
                .into_py_any(py),
        }
    }

    #[getter]
    fn get_const(&self) -> bool {
        true
    }

    #[getter]
    fn get_type(&self, py: Python) -> PyResult<Py<PyAny>> {
        match &self.0 {
            Value::Duration(_) => Type::Duration.into_py_any(py),
            Value::Float(_) => Type::Float.into_py_any(py),
            Value::Bool(_) => Type::Bool.into_py_any(py),
            Value::Uint(_raw, width) => Type::Uint(*width).into_py_any(py),
        }
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
