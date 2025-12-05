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

use crate::classical::expr::cast::Cast;
use crate::classical::expr::{Expr, ExprKind, PyExpr, Value};
use crate::classical::types::Type;
use num_bigint::BigUint;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};
use pyo3::{IntoPyObjectExt, intern};
use std::boxed::Box;

/// A range expression that represents a sequence of values.
///
/// The step is always Some(Expr) as it defaults to 1 when not explicitly provided.
/// This avoids the need for unwrap() calls throughout the codebase.
#[derive(Clone, Debug, PartialEq)]
pub struct Range {
    pub start: Expr,
    pub stop: Expr,
    pub step: Expr, // Always Some(Expr) - defaults to 1 when not provided
    pub ty: Type,
    pub constant: bool,
}

// Manual Eq implementation since Expr only implements PartialEq, not Eq.
// This is safe because Eq is just a marker trait indicating that PartialEq
// is an equivalence relation (reflexive, symmetric, transitive).
impl Eq for Range {}

impl<'py> IntoPyObject<'py> for Range {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Bound::new(py, (PyRangeExpr(self), PyExpr(ExprKind::Range)))?.into_any())
    }
}

impl<'py> IntoPyObject<'py> for &'_ Range {
    type Target = <Range as IntoPyObject<'py>>::Target;
    type Output = <Range as IntoPyObject<'py>>::Output;
    type Error = <Range as IntoPyObject<'py>>::Error;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        self.clone().into_pyobject(py)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Range {
    type Error = <PyRangeExpr as FromPyObject<'a, 'py>>::Error;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let PyRangeExpr(r) = ob.extract()?;
        Ok(r)
    }
}

impl Range {
    /// Returns the length of the range.
    ///
    /// Note: Returns 1 as the length is generally undefined for dynamic ranges.
    pub fn len(&self) -> usize {
        1
    }

    /// Returns whether the range is empty.
    ///
    /// Note: Returns false if the object exists, as the emptiness cannot be
    /// determined statically for dynamic ranges.
    pub fn is_empty(&self) -> bool {
        false
    }
}

/// Helper function to convert Python values to Expr::Value
fn py_value_to_expr(_py: Python, value: &Bound<PyAny>) -> PyResult<Expr> {
    if let Ok(raw) = value.extract::<i64>() {
        if raw < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Range values must be non-negative integers",
            ));
        }
        Ok(Value::Uint {
            raw: BigUint::from(raw as u64),
            ty: Type::Uint(64),
        }
        .into())
    } else if let Ok(expr) = value.extract::<Expr>() {
        // Ensure the expression is of unsigned integer type
        if !matches!(expr.ty(), Type::Uint(_)) {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Range values must be of unsigned integer type",
            ));
        }
        Ok(expr)
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Expected non-negative integer or Expr of unsigned integer type, got {}",
            value.get_type()
        )))
    }
}

/// Helper function to determine the common type with maximum bit width
fn determine_common_max_type(types: &[Type]) -> Type {
    let mut max_type = types[0];

    for &ty in types.iter().skip(1) {
        if let (Type::Uint(max_width), Type::Uint(width)) = (&max_type, &ty) {
            // If both types are Uint, take the one with the largest bit width
            if width > max_width {
                max_type = ty;
            }
        }
        // For any other type combinations, keep the first type
    }

    max_type
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
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct PyRangeExpr(pub Range);

#[pymethods]
impl PyRangeExpr {
    #[new]
    #[pyo3(signature=(start, stop, step=None, ty=None), text_signature="(start, stop, step=None, type=None)")]
    fn new(
        py: Python,
        start: &Bound<PyAny>,
        stop: &Bound<PyAny>,
        step: Option<&Bound<PyAny>>,
        ty: Option<Type>,
    ) -> PyResult<(Self, PyExpr)> {
        let start_expr = py_value_to_expr(py, start)?;
        let stop_expr = py_value_to_expr(py, stop)?;
        // Store whether the type was explicitly specified or implicitly determined
        let is_implicit_promotion = ty.is_none();

        // Determine the target type for the Range and its expressions
        let target_ty = if let Some(explicit_ty) = ty {
            // Verify that the explicit type is indeed a Uint
            if !matches!(explicit_ty, Type::Uint(_)) {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Range type must be an unsigned integer type",
                ));
            }
            explicit_ty
        } else {
            // When no explicit type is provided, determine the common type from start and stop
            // If step is provided, include it in the type determination
            let mut types = vec![start_expr.ty(), stop_expr.ty()];
            if let Some(step) = step {
                // If step is provided, we need to convert it first to get its type
                let step_expr_temp = py_value_to_expr(py, step)?;
                types.push(step_expr_temp.ty());
            }
            determine_common_max_type(&types)
        };

        // Create step expression - either from provided value or default to 1 in the target type
        let step_expr = if let Some(step) = step {
            py_value_to_expr(py, step)?
        } else {
            // Create a default step value of 1 in the target type
            match target_ty {
                Type::Uint(_) => Value::Uint {
                    raw: BigUint::from(1u64),
                    ty: target_ty,
                }
                .into(),
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "Range type must be an unsigned integer type",
                    ));
                }
            }
        };

        let constant = start_expr.is_const() && stop_expr.is_const() && step_expr.is_const();

        // Apply casts to any expressions with types different from the target type
        // For Range expressions, we only handle Uint types
        if !matches!(target_ty, Type::Uint(_)) {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Range type must be an unsigned integer type",
            ));
        }

        let (start_expr, stop_expr, step_expr) = {
            // Create necessary Cast expressions for start if needed
            let start_ty = start_expr.ty();
            let start_const = start_expr.is_const();
            let start_expr = if target_ty != start_ty {
                Expr::Cast(Box::new(Cast {
                    operand: start_expr,
                    ty: target_ty,
                    constant: start_const,
                    // Mark as implicit if the type was determined via promotion
                    implicit: is_implicit_promotion,
                }))
            } else {
                start_expr
            };

            // Create necessary Cast expressions for stop if needed
            let stop_ty = stop_expr.ty();
            let stop_const = stop_expr.is_const();
            let stop_expr = if target_ty != stop_ty {
                Expr::Cast(Box::new(Cast {
                    operand: stop_expr,
                    ty: target_ty,
                    constant: stop_const,
                    // Mark as implicit if the type was determined via promotion
                    implicit: is_implicit_promotion,
                }))
            } else {
                stop_expr
            };

            // Create necessary Cast expressions for step if needed
            let step_ty = step_expr.ty();
            let step_const = step_expr.is_const();
            let step_expr = if target_ty != step_ty {
                Expr::Cast(Box::new(Cast {
                    operand: step_expr,
                    ty: target_ty,
                    constant: step_const,
                    // Mark as implicit if the type was determined via promotion
                    implicit: is_implicit_promotion,
                }))
            } else {
                step_expr
            };

            (start_expr, stop_expr, step_expr)
        };

        Ok((
            PyRangeExpr(Range {
                start: start_expr,
                stop: stop_expr,
                step: step_expr,
                ty: target_ty,
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
        self.0.step.clone().into_py_any(py)
    }

    #[getter]
    fn get_const(&self) -> bool {
        self.0.constant
    }

    #[getter]
    fn get_type(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.0.ty.into_py_any(py)
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn accept<'py>(
        slf: PyRef<'py, Self>,
        visitor: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        visitor.call_method1(intern!(visitor.py(), "visit_range"), (slf,))
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        let start = self.get_start(py)?.bind(py).repr()?;
        let stop = self.get_stop(py)?.bind(py).repr()?;
        let step = self.get_step(py)?.bind(py).repr()?;
        let ty = self.get_type(py)?.bind(py).repr()?;
        Ok(format!("Range({}, {}, {}, {})", start, stop, step, ty))
    }

    fn __str__(&self, py: Python) -> PyResult<String> {
        // Use get_start and get_stop methods for consistency
        let start_py = self.get_start(py)?;
        let start = start_py.bind(py);

        let stop_py = self.get_stop(py)?;
        let stop = stop_py.bind(py);
        let start_str = match start.getattr("name") {
            Ok(name) => name.extract::<String>()?,
            Err(_) => start.str()?.to_string(),
        };
        let stop_str = match stop.getattr("name") {
            Ok(name) => name.extract::<String>()?,
            Err(_) => stop.str()?.to_string(),
        };
        let step_py = self.get_step(py)?;
        let step = step_py.bind(py);
        let step_str = match step.getattr("name") {
            Ok(name) => format!(", step={}", name.extract::<String>()?),
            Err(_) => format!(", step={}", step.str()?),
        };

        Ok(format!("Range({}, {}{})", start_str, stop_str, step_str))
    }

    fn __copy__(slf: PyRef<Self>) -> PyRef<Self> {
        // I am immutable...
        slf
    }

    fn __deepcopy__<'py>(slf: PyRef<'py, Self>, _memo: Bound<'py, PyAny>) -> PyRef<'py, Self> {
        // ... as are all my constituent parts.
        slf
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        (
            py.get_type::<Self>(),
            (
                self.get_start(py)?,
                self.get_stop(py)?,
                self.get_step(py)?,
                self.get_type(py)?,
            ),
        )
            .into_pyobject(py)
    }
}
