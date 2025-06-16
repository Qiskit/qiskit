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

use crate::classical::expr::{Expr, ExprKind, PyExpr, Value};
use crate::classical::types::Type;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::{intern, IntoPyObjectExt};

/// A range expression that represents a sequence of values.
#[derive(Clone, Debug, PartialEq)]
pub struct Range {
    pub start: Expr,
    pub stop: Expr,
    pub step: Option<Expr>,
    pub ty: Type,
    pub constant: bool,
}

impl<'py> IntoPyObject<'py> for Range {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Bound::new(py, (PyRange(self), PyExpr(ExprKind::Range)))?.into_any())
    }
}

impl<'py> FromPyObject<'py> for Range {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let PyRange(r) = ob.extract()?;
        Ok(r)
    }
}

/// Helper function to convert Python values to Expr::Value
fn py_value_to_expr(_py: Python, value: &Bound<PyAny>) -> PyResult<Expr> {
    if let Ok(raw) = value.extract::<i64>() {
        Ok(Value::Uint {
            raw: raw as u64,
            ty: Type::Uint(64),
        }
    .into())
    } else if let Ok(raw) = value.extract::<f64>() {
        Ok(Value::Float {
            raw,
            ty: Type::Float,
        }.into())
    } else if let Ok(expr) = value.extract::<Expr>() {
        Ok(expr)
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Expected integer, float, or Expr, got {}",
             value.get_type().name()?
             )))
    }
}

/// A range expression.
///
/// Args:
///     start: The start value of the range.
///     stop: The stop value of the range.
///     step: Optional step value for the range. Defaults to 1.
///     type: The resolved type of the result.
#[pyclass(
    eq,
    subclass,
    frozen,
    extends = PyExpr,
    name = "Range",
    module = "qiskit._accelerate.circuit.classical.expr"
)]
#[derive(PartialEq, Clone, Debug)]
pub struct PyRange(pub Range);

#[pymethods]
impl PyRange {
    #[new]
    #[pyo3(signature=(start, stop, step=None, ty=None), text_signature="(start, stop, step=None, type=None)")]
    fn new(py: Python,
        start: &Bound<PyAny>,
        stop: &Bound<PyAny>,
        step: Option<&Bound<PyAny>>,
        ty: Option<Type>
    ) -> PyResult<(Self, PyExpr)> {
        let start_expr = py_value_to_expr(py, start)?;
        let stop_expr = py_value_to_expr(py, stop)?;
        let step_expr = if let Some(step) = step {
            Some(py_value_to_expr(py, step)?)
        } else {
            None
        };
        let constant = start_expr.is_const()
            && stop_expr.is_const()
            && step_expr.as_ref().map_or(true, |s| s.is_const());
        let ty = ty.unwrap_or_else(|| start_expr.ty());
        Ok((
            PyRange(Range {
                start: start_expr,
                stop: stop_expr,
                step: step_expr,
                ty,
                constant,
            }),
            PyExpr(ExprKind::Range),
        ))
    }

    #[getter]
    fn get_start(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.start.clone().into_py_any(py)
    }

    #[getter]
    fn get_stop(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.stop.clone().into_py_any(py)
    }

    #[getter]
    fn get_step(&self, py: Python) -> PyResult<Py<PyAny>> {
        match &self.0.step {
            Some(step) => step.clone().into_py_any(py),
            None => Ok(py.None()),
        }
    }

    #[getter]
    fn get_const(&self) -> bool {
        self.0.constant
    }

    #[getter]
    fn get_type(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.ty.into_py_any(py)
    }

    fn accept<'py>(
        slf: PyRef<'py, Self>,
        visitor: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        visitor.call_method1(intern!(visitor.py(), "visit_range"), (slf,))
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        let start = self.0.start.clone().into_py_any(py)?.bind(py).repr()?;
        let stop = self.0.stop.clone().into_py_any(py)?.bind(py).repr()?;
        let step = if let Some(step) = &self.0.step {
            format!(", step={}", step.clone().into_py_any(py)?.bind(py).repr()?)
        } else {
            String::new()
        };
        let ty = self.0.ty.into_py_any(py)?.bind(py).repr()?;
        Ok(format!("Range(start={}, stop={}{}, ty={})",
         start, stop, step, ty
         ))
    }

    fn __str__(&self, py: Python) -> PyResult<String> {
        // Create longer-lived bindings for start
        let start_py = self.0.start.clone().into_py_any(py)?;
        let start = start_py.bind(py);

        let stop_py = self.0.stop.clone().into_py_any(py)?;
        let stop = stop_py.bind(py);
        let start_str = match start.getattr("name") {
            Ok(name) => name.extract::<String>()?,
            Err(_) => start.str()?.to_string(),
        };
        let stop_str = match stop.getattr("name") {
            Ok(name) => name.extract::<String>()?,
            Err(_) => stop.str()?.to_string(),
        };
        let step_str = if let Some(step) = &self.0.step {
            let step_py = step.clone().into_py_any(py)?;
            let step = step_py.bind(py);
            match step.getattr("name") {
                Ok(name) => format!(", step={}", name.extract::<String>()?),
                Err(_) => format!(", step={}", step.str()?.to_string()),
            }
        } else {
            String::new()
        };

        Ok(format!("Range({}, {}{})", start_str, stop_str, step_str))
    }
}
