// This code is part of Qiskit.
//
// (C) Copyright IBM 2023, 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// ParameterExpression class for symbolic equation on Rust / interface to Python

use hashbrown::HashMap;
use num_complex::Complex64;
use pyo3::exceptions::{PyRuntimeError, PyZeroDivisionError};
use pyo3::types::{PyDict, PySet, PyString};
use thiserror::Error;

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;

use crate::parameter::symbol_expr;
use crate::parameter::symbol_expr::SymbolExpr;
use crate::parameter::symbol_parser::parse_expression;

use super::symbol_expr::Parameter;

#[derive(Error, Debug)]
pub enum ParameterError {
    #[error("Encountered unbound parameter.")]
    UnboundParameter,
    #[error("Division by zero.")]
    ZeroDivisionError,
    #[error("Binding to infinite value.")]
    BindingInf,
    #[error("Binding to NaN.")]
    BindingNaN,
}

impl From<ParameterError> for PyErr {
    fn from(value: ParameterError) -> Self {
        match value {
            ParameterError::ZeroDivisionError => {
                PyZeroDivisionError::new_err("zero division occurs while binding parameter")
            }
            ParameterError::BindingInf => {
                PyZeroDivisionError::new_err("attempted to bind infinite value to parameter")
            }
            _ => PyRuntimeError::new_err(value.to_string()),
        }
    }
}

// Python interface to SymbolExpr
#[pyclass(sequence, module = "qiskit._accelerate.circuit")]
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterExpression {
    expr: SymbolExpr,
}

#[inline]
fn _extract_value(value: &Bound<PyAny>) -> Option<ParameterExpression> {
    if let Ok(i) = value.extract::<i64>() {
        Some(ParameterExpression {
            expr: SymbolExpr::Value(symbol_expr::Value::from(i)),
        })
    } else if let Ok(c) = value.extract::<Complex64>() {
        Some(ParameterExpression {
            expr: SymbolExpr::Value(symbol_expr::Value::from(c)),
        })
    } else if let Ok(r) = value.extract::<f64>() {
        Some(ParameterExpression {
            expr: SymbolExpr::Value(symbol_expr::Value::from(r)),
        })
    } else if let Ok(s) = value.extract::<String>() {
        if let Ok(expr) = parse_expression(&s) {
            Some(ParameterExpression { expr })
        } else {
            None
        }
    } else {
        value.extract::<ParameterExpression>().ok()
    }
}

impl ParameterExpression {
    pub fn sin(&self) -> Self {
        Self {
            expr: self.expr.sin(),
        }
    }
    pub fn cos(&self) -> Self {
        Self {
            expr: self.expr.cos(),
        }
    }
    pub fn tan(&self) -> Self {
        Self {
            expr: self.expr.tan(),
        }
    }
    pub fn asin(&self) -> Self {
        Self {
            expr: self.expr.asin(),
        }
    }
    pub fn acos(&self) -> Self {
        Self {
            expr: self.expr.acos(),
        }
    }
    pub fn atan(&self) -> Self {
        Self {
            expr: self.expr.atan(),
        }
    }
    pub fn exp(&self) -> Self {
        Self {
            expr: self.expr.exp(),
        }
    }
    pub fn log(&self) -> Self {
        Self {
            expr: self.expr.log(),
        }
    }
    pub fn abs(&self) -> Self {
        Self {
            expr: self.expr.abs(),
        }
    }
    pub fn sign(&self) -> Self {
        Self {
            expr: self.expr.sign(),
        }
    }

    /// clone expression
    pub fn copy(&self) -> Self {
        Self {
            expr: self.expr.clone(),
        }
    }
    /// return conjugate of expression
    pub fn conjugate(&self) -> Self {
        Self {
            expr: self.expr.conjugate(),
        }
    }
    /// return derivative of this expression for param
    pub fn derivative(&self, param: &Self) -> Result<Self, String> {
        self.expr.derivative(&param.expr).map(|expr| Self { expr })
    }

    /// expand expression
    pub fn expand(&self) -> Self {
        Self {
            expr: self.expr.expand(),
        }
    }

    /// substitute symbols to expressions (or values) given by hash map
    pub fn subs(&self, in_maps: &HashMap<String, Self>) -> Self {
        let maps: HashMap<String, SymbolExpr> = in_maps
            .into_iter()
            .map(|(key, val)| (key.clone(), val.expr.clone()))
            .collect();
        Self {
            expr: self.expr.subs(&maps),
        }
    }

    pub fn name(&self) -> String {
        if let SymbolExpr::Symbol(s) = &self.expr {
            return s.name();
        }
        match self.expr.eval(true) {
            Some(e) => e.to_string(),
            None => self.expr.optimize().to_string(),
        }
    }

    pub fn bind(&self, map: &HashMap<String, symbol_expr::Value>) -> Result<Self, ParameterError> {
        let bound = self.expr.bind(map);
        match bound.eval(true) {
            Some(v) => match &v {
                symbol_expr::Value::Real(r) => {
                    if r.is_infinite() {
                        Err(ParameterError::BindingInf)
                    } else if r.is_nan() {
                        Err(ParameterError::BindingNaN)
                    } else {
                        Ok(Self {
                            expr: SymbolExpr::Value(v),
                        })
                    }
                }
                symbol_expr::Value::Int(_) => Ok(Self {
                    expr: SymbolExpr::Value(v),
                }),
                symbol_expr::Value::Complex(c) => {
                    if c.re.is_infinite() || c.im.is_infinite() {
                        Err(ParameterError::ZeroDivisionError) // TODO this should probs be BindingInf
                    } else if c.re.is_nan() || c.im.is_nan() {
                        Err(ParameterError::BindingNaN)
                    } else if (-symbol_expr::SYMEXPR_EPSILON..symbol_expr::SYMEXPR_EPSILON)
                        .contains(&c.im)
                    {
                        Ok(Self {
                            expr: SymbolExpr::Value(symbol_expr::Value::Real(c.re)),
                        })
                    } else {
                        Ok(Self {
                            expr: SymbolExpr::Value(v),
                        })
                    }
                }
            },
            None => Ok(Self { expr: bound }),
        }
    }
}

#[pymethods]
impl ParameterExpression {
    /// parse expression from string
    #[new]
    #[pyo3(signature = (in_expr=None))]
    pub fn new(in_expr: Option<String>) -> PyResult<Self> {
        match in_expr {
            Some(e) => match parse_expression(&e) {
                Ok(expr) => Ok(ParameterExpression { expr }),
                Err(s) => Err(pyo3::exceptions::PyRuntimeError::new_err(s)),
            },
            None => Ok(ParameterExpression {
                expr: SymbolExpr::Value(symbol_expr::Value::Int(0)),
            }),
        }
    }

    /// create new expression as a symbol
    #[allow(non_snake_case)]
    #[staticmethod]
    pub fn Symbol(name: String) -> Self {
        // check if expr contains replacements for sympy
        let name = name
            .replace("__begin_sympy_replace__", "$\\")
            .replace("__end_sympy_replace__", "$");

        ParameterExpression {
            expr: SymbolExpr::Symbol(Parameter::new(&name)),
        }
    }
    /// create new expression as a value
    #[allow(non_snake_case)]
    #[staticmethod]
    pub fn Value(value: &Bound<PyAny>) -> PyResult<Self> {
        match _extract_value(value) {
            Some(v) => Ok(v),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Unsupported data type to initialize SymbolExpr as a value",
            )),
        }
    }

    /// create new expression from string
    #[allow(non_snake_case)]
    #[staticmethod]
    pub fn Expression(expr: String) -> PyResult<Self> {
        // check if expr contains replacements for sympy
        let expr = expr
            .replace("__begin_sympy_replace__", "$\\")
            .replace("__end_sympy_replace__", "$");
        match parse_expression(&expr) {
            // substitute 'I' to imaginary number i before returning expression
            Ok(expr) => Ok(ParameterExpression {
                expr: expr.bind(&HashMap::from([(
                    "I".to_string(),
                    symbol_expr::Value::from(Complex64::i()),
                )])),
            }),
            Err(s) => Err(pyo3::exceptions::PyRuntimeError::new_err(s)),
        }
    }

    /// return value if expression does not contain any symbols
    pub fn value(&self, py: Python) -> PyResult<PyObject> {
        match self.expr.eval(true) {
            Some(v) => match v {
                symbol_expr::Value::Real(r) => r.into_py_any(py),
                symbol_expr::Value::Int(i) => i.into_py_any(py),
                symbol_expr::Value::Complex(c) => {
                    if (-symbol_expr::SYMEXPR_EPSILON..symbol_expr::SYMEXPR_EPSILON).contains(&c.im)
                    {
                        c.re.into_py_any(py)
                    } else {
                        c.into_py_any(py)
                    }
                }
            },
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Expression has some undefined symbols.",
            )),
        }
    }

    pub fn parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        PySet::new(py, self.expr.parameters())
    }

    pub fn name_map<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (name, param) in self.expr.name_map().iter() {
            let py_name = PyString::new(py, name.as_str());
            dict.set_item(py_name, param.clone())?;
        }
        Ok(dict)
    }

    #[pyo3(name = "sin")]
    pub fn py_sin(&self) -> Self {
        self.sin()
    }

    #[pyo3(name = "cos")]
    pub fn py_cos(&self) -> Self {
        self.cos()
    }

    #[pyo3(name = "tan")]
    pub fn py_tan(&self) -> Self {
        self.tan()
    }

    #[pyo3(name = "asin")]
    pub fn py_asin(&self) -> Self {
        self.asin()
    }

    #[pyo3(name = "acos")]
    pub fn py_acos(&self) -> Self {
        self.acos()
    }

    #[pyo3(name = "atan")]
    pub fn py_atan(&self) -> Self {
        self.atan()
    }

    #[pyo3(name = "exp")]
    pub fn py_exp(&self) -> Self {
        self.exp()
    }

    #[pyo3(name = "log")]
    pub fn py_log(&self) -> Self {
        self.log()
    }

    #[pyo3(name = "abs")]
    pub fn py_abs(&self) -> Self {
        self.abs()
    }

    #[pyo3(name = "sign")]
    pub fn py_sign(&self) -> Self {
        self.sign()
    }

    #[pyo3(name = "copy")]
    pub fn py_copy(&self) -> Self {
        self.copy()
    }

    #[pyo3(name = "conjugate")]
    pub fn py_conjugate(&self) -> Self {
        self.conjugate()
    }

    /// Return derivative of this expression for param
    pub fn gradient(&self, param: &Self) -> PyResult<Self> {
        self.derivative(param).map_err(PyRuntimeError::new_err)
    }

    /// return all values in this equation
    pub fn values(&self, py: Python) -> PyResult<Vec<PyObject>> {
        self.expr
            .values()
            .iter()
            .map(|val| match val {
                symbol_expr::Value::Real(r) => r.into_py_any(py),
                symbol_expr::Value::Int(i) => i.into_py_any(py),
                symbol_expr::Value::Complex(c) => c.into_py_any(py),
            })
            .collect()
    }

    /// return expression as a string
    #[getter]
    #[pyo3(name = "name")]
    pub fn py_name(&self) -> String {
        self.name()
    }

    /// substitute symbols to expressions (or values) given by hash map
    #[pyo3(name = "subs")]
    pub fn py_subs(&self, in_maps: HashMap<String, Self>) -> Self {
        self.subs(&in_maps)
    }

    // bind values to symbols given by input hashmap
    #[pyo3(name = "bind")]
    pub fn py_bind(&self, map: HashMap<String, Bound<PyAny>>) -> PyResult<Self> {
        let map: HashMap<String, symbol_expr::Value> = map
            .into_iter()
            .filter_map(|(key, val)| {
                if let Ok(i) = val.extract::<i64>() {
                    Some((key, symbol_expr::Value::from(i)))
                } else if let Ok(r) = val.extract::<f64>() {
                    Some((key, symbol_expr::Value::from(r)))
                } else if let Ok(c) = val.extract::<Complex64>() {
                    Some((key, symbol_expr::Value::from(c)))
                } else {
                    // if unsupported data type, insert nothing
                    None
                }
            })
            .collect();

        self.bind(&map).map_err(|e| e.into())
    }

    // ====================================
    // operator overrides
    // ====================================
    pub fn __eq__(&self, rhs: &Bound<PyAny>) -> bool {
        match _extract_value(rhs) {
            Some(rhs) => match rhs.expr {
                SymbolExpr::Value(v) => match self.expr.eval(true) {
                    Some(e) => e == v,
                    None => false,
                },
                _ => self.expr == rhs.expr,
            },
            None => false,
        }
    }
    pub fn __ne__(&self, rhs: &Bound<PyAny>) -> bool {
        match _extract_value(rhs) {
            Some(rhs) => match rhs.expr {
                SymbolExpr::Value(v) => match self.expr.eval(true) {
                    Some(e) => e != v,
                    None => true,
                },
                _ => self.expr != rhs.expr,
            },
            None => true,
        }
    }

    pub fn __neg__(&self) -> Self {
        Self { expr: -&self.expr }
    }

    pub fn __add__(&self, rhs: &Bound<PyAny>) -> PyResult<Self> {
        match _extract_value(rhs) {
            Some(rhs) => Ok(Self {
                expr: &self.expr + &rhs.expr,
            }),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __add__",
            )),
        }
    }
    pub fn __radd__(&self, lhs: &Bound<PyAny>) -> PyResult<Self> {
        match _extract_value(lhs) {
            Some(lhs) => Ok(Self {
                expr: &lhs.expr + &self.expr,
            }),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __radd__",
            )),
        }
    }
    pub fn __sub__(&self, rhs: &Bound<PyAny>) -> PyResult<Self> {
        match _extract_value(rhs) {
            Some(rhs) => Ok(Self {
                expr: &self.expr - &rhs.expr,
            }),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __sub__",
            )),
        }
    }
    pub fn __rsub__(&self, lhs: &Bound<PyAny>) -> PyResult<Self> {
        match _extract_value(lhs) {
            Some(lhs) => Ok(Self {
                expr: &lhs.expr - &self.expr,
            }),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __rsub__",
            )),
        }
    }
    pub fn __mul__(&self, rhs: &Bound<PyAny>) -> PyResult<Self> {
        match _extract_value(rhs) {
            Some(rhs) => Ok(Self {
                expr: &self.expr * &rhs.expr,
            }),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __mul__",
            )),
        }
    }
    pub fn __rmul__(&self, lhs: &Bound<PyAny>) -> PyResult<Self> {
        match _extract_value(lhs) {
            Some(lhs) => Ok(Self {
                expr: &lhs.expr * &self.expr,
            }),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __rmul__",
            )),
        }
    }

    pub fn __truediv__(&self, rhs: &Bound<PyAny>) -> PyResult<Self> {
        match _extract_value(rhs) {
            Some(rhs) => Ok(Self {
                expr: &self.expr / &rhs.expr,
            }),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __truediv__",
            )),
        }
    }
    pub fn __rtruediv__(&self, lhs: &Bound<PyAny>) -> PyResult<Self> {
        match _extract_value(lhs) {
            Some(lhs) => Ok(Self {
                expr: &lhs.expr / &self.expr,
            }),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __rtruediv__",
            )),
        }
    }
    pub fn __pow__(&self, rhs: &Bound<PyAny>, _modulo: Option<i32>) -> PyResult<Self> {
        match _extract_value(rhs) {
            Some(rhs) => Ok(Self {
                expr: self.expr.pow(&rhs.expr),
            }),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __pow__",
            )),
        }
    }
    pub fn __rpow__(&self, lhs: &Bound<PyAny>, _modulo: Option<i32>) -> PyResult<Self> {
        match _extract_value(lhs) {
            Some(lhs) => Ok(Self {
                expr: lhs.expr.pow(&self.expr),
            }),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __rpow__",
            )),
        }
    }

    pub fn __str__(&self) -> String {
        self.name()
    }

    pub fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.expr.to_string().hash(&mut hasher);
        hasher.finish()
    }

    // for pickle, we can reproduce equation from expression string
    fn __getstate__(&self) -> String {
        self.__str__()
    }
    fn __setstate__(&mut self, state: String) {
        if let Ok(expr) = parse_expression(&state) {
            self.expr = expr;
        }
    }
}

impl Default for ParameterExpression {
    // default constructor returns zero
    fn default() -> Self {
        Self {
            expr: SymbolExpr::Value(symbol_expr::Value::Int(0)),
        }
    }
}

impl fmt::Display for ParameterExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expr)
    }
}

// rust native implementation will be added in PR #14207
