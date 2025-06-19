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

use hashbrown::{HashMap, HashSet};
use num_complex::Complex64;
use pyo3::exceptions::{
    PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError, PyZeroDivisionError,
};
use pyo3::types::{IntoPyDict, PyDict, PySet, PyString};
use thiserror::Error;
use uuid::Uuid;

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;

use crate::circuit_data::CircuitError;
use crate::imports::UUID;
use crate::parameter::symbol_expr;
use crate::parameter::symbol_expr::SymbolExpr;
use crate::parameter::symbol_parser::parse_expression;

use super::symbol_expr::{Symbol, Value, SYMEXPR_EPSILON};

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
    #[error("Cannot bind Parameter {0:?} not present in expression")]
    UnknownParameter(Symbol),
    #[error("Name conflict adding parameters.")]
    NameConflict,
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
            ParameterError::UnknownParameter(_) | ParameterError::NameConflict => {
                CircuitError::new_err(value.to_string())
            }
            _ => PyRuntimeError::new_err(value.to_string()),
        }
    }
}

// Python interface to SymbolExpr
#[pyclass(
    subclass,
    sequence,
    module = "qiskit._accelerate.circuit",
    name = "ParameterExpression"
)]
#[derive(Clone, Debug, PartialEq)]
pub struct PyParameterExpression {
    expr: SymbolExpr,
    // Cached information about the expression.
    name_map: Option<HashMap<String, Symbol>>,
}

impl Hash for PyParameterExpression {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.expr.to_string().hash(state);
    }
}

#[inline]
fn _extract_value(value: &Bound<PyAny>) -> Option<PyParameterExpression> {
    if let Ok(i) = value.extract::<i64>() {
        Some(PyParameterExpression::new(SymbolExpr::Value(
            symbol_expr::Value::from(i),
        )))
    } else if let Ok(c) = value.extract::<Complex64>() {
        Some(PyParameterExpression::new(SymbolExpr::Value(
            symbol_expr::Value::from(c),
        )))
    } else if let Ok(r) = value.extract::<f64>() {
        Some(PyParameterExpression::new(SymbolExpr::Value(
            symbol_expr::Value::from(r),
        )))
    } else if let Ok(s) = value.extract::<String>() {
        if let Ok(expr) = parse_expression(&s) {
            Some(PyParameterExpression::new(expr))
        } else {
            None
        }
    } else {
        value.extract::<PyParameterExpression>().ok()
    }
}

impl PyParameterExpression {
    pub fn new(expr: SymbolExpr) -> Self {
        Self {
            expr: expr.clone(),
            name_map: None,
        }
    }

    pub fn sin(&self) -> Self {
        Self {
            expr: self.expr.sin(),
            name_map: self.name_map.clone(), // the name map remains the same
        }
    }

    pub fn cos(&self) -> Self {
        Self {
            expr: self.expr.cos(),
            name_map: self.name_map.clone(),
        }
    }
    pub fn tan(&self) -> Self {
        Self {
            expr: self.expr.tan(),
            name_map: self.name_map.clone(),
        }
    }
    pub fn asin(&self) -> Self {
        Self {
            expr: self.expr.asin(),
            name_map: self.name_map.clone(),
        }
    }
    pub fn acos(&self) -> Self {
        Self {
            expr: self.expr.acos(),
            name_map: self.name_map.clone(),
        }
    }
    pub fn atan(&self) -> Self {
        Self {
            expr: self.expr.atan(),
            name_map: self.name_map.clone(),
        }
    }
    pub fn exp(&self) -> Self {
        Self {
            expr: self.expr.exp(),
            name_map: self.name_map.clone(),
        }
    }
    pub fn log(&self) -> Self {
        Self {
            expr: self.expr.log(),
            name_map: self.name_map.clone(),
        }
    }
    pub fn abs(&self) -> Self {
        Self {
            expr: self.expr.abs(),
            name_map: self.name_map.clone(),
        }
    }
    pub fn sign(&self) -> Self {
        Self {
            expr: self.expr.sign(),
            name_map: self.name_map.clone(),
        }
    }

    /// clone expression
    pub fn copy(&self) -> Self {
        Self {
            expr: self.expr.clone(),
            name_map: self.name_map.clone(),
        }
    }
    /// return conjugate of expression
    pub fn conjugate(&self) -> Self {
        Self {
            expr: self.expr.conjugate(),
            name_map: self.name_map.clone(),
        }
    }
    /// return derivative of this expression for param
    pub fn derivative(&self, param: &Self) -> Result<Self, String> {
        self.expr
            .derivative(&param.expr)
            .map(|expr| Self::new(expr))
    }

    /// expand expression
    pub fn expand(&self) -> Self {
        Self {
            expr: self.expr.expand(),
            name_map: self.name_map.clone(),
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

    pub fn get_name_map(&self) -> HashMap<String, Symbol> {
        if let Some(map) = &self.name_map {
            map.clone()
        } else {
            let map = self.expr.name_map();
            // self.name_map = Some(map.clone());
            map
        }
    }

    /// substitute symbols to expressions (or values) given by hash map
    pub fn subs(
        &self,
        map: &HashMap<Symbol, Self>,
        allow_unknown_parameters: bool,
    ) -> Result<Self, ParameterError> {
        // we do not allow substituting if the resulting expression has duplicate names,
        // so we create a hashmap containing all symbols in the replacement expressions and
        // check that there are no name conflicts
        let replacements = map
            .values()
            .into_iter()
            .flat_map(|expr| {
                expr.expr
                    .parameters()
                    .into_iter()
                    .map(|symbol| (symbol.name(), symbol.clone()))
            })
            .collect();
        let replacing = map
            .keys()
            .map(|symbol| symbol.name())
            .collect::<HashSet<String>>();

        if self.has_name_conflicts(&replacements, Some(&replacing)) {
            return Err(ParameterError::NameConflict);
        }

        let maps: HashMap<Symbol, SymbolExpr> = map
            .into_iter()
            .map(|(key, val)| {
                // if we only allow known parameters, check that the symbol exists
                if allow_unknown_parameters || self.expr.has_symbol(&key.name()) {
                    Ok((key.clone(), val.expr.clone()))
                } else {
                    Err(ParameterError::UnknownParameter(key.clone()))
                }
            })
            .collect::<Result<_, ParameterError>>()?;
        Ok(Self::new(self.expr.subs(&maps)))
    }

    pub fn bind(
        &self,
        map: &HashMap<Symbol, symbol_expr::Value>,
        allow_unknown_parameters: bool,
    ) -> Result<Self, ParameterError> {
        if !allow_unknown_parameters {
            map.iter()
                .map(|(symbol, _)| {
                    if !self.expr.has_symbol(&symbol.name()) {
                        Err(ParameterError::UnknownParameter(symbol.clone()))
                    } else {
                        Ok(())
                    }
                })
                .collect::<Result<(), ParameterError>>()?;
        }

        let bound = self.expr.bind(map);
        match bound.eval(true) {
            Some(v) => match &v {
                symbol_expr::Value::Real(r) => {
                    if r.is_infinite() {
                        Err(ParameterError::BindingInf)
                    } else if r.is_nan() {
                        Err(ParameterError::BindingNaN)
                    } else {
                        Ok(Self::new(SymbolExpr::Value(v)))
                    }
                }
                symbol_expr::Value::Int(_) => Ok(Self::new(SymbolExpr::Value(v))),
                symbol_expr::Value::Complex(c) => {
                    if c.re.is_infinite() || c.im.is_infinite() {
                        Err(ParameterError::ZeroDivisionError) // TODO this should probs be BindingInf
                    } else if c.re.is_nan() || c.im.is_nan() {
                        Err(ParameterError::BindingNaN)
                    } else if (-symbol_expr::SYMEXPR_EPSILON..symbol_expr::SYMEXPR_EPSILON)
                        .contains(&c.im)
                    {
                        Ok(Self::new(SymbolExpr::Value(symbol_expr::Value::Real(c.re))))
                    } else {
                        Ok(Self::new(SymbolExpr::Value(v)))
                    }
                }
            },
            None => Ok(Self::new(bound)),
        }
    }

    /// Check whether a hashmap of incoming parameters have a name conflict with the expression.
    ///
    /// Args:
    ///     - inbound_parameters: The hashmap of incoming parameters. Can e.g. be a map of parameters
    ///         for subtitution, or the parameters of another expression that we merge with.
    ///     - replacement: Set to ``true`` for substitutions, ``false`` for merge.
    fn has_name_conflicts(
        &self,
        inbound_parameters: &HashMap<String, Symbol>,
        outbound: Option<&HashSet<String>>,
    ) -> bool {
        let name_map = self.get_name_map();
        for (name, symbol) in inbound_parameters.iter() {
            if let Some(existing_symbol) = name_map.get(name) {
                if let Some(outbound) = outbound {
                    if outbound.contains(name) {
                        continue;
                    }
                }
                if symbol != existing_symbol {
                    return true;
                }
            }
        }
        false
    }
}

#[pymethods]
impl PyParameterExpression {
    /// parse expression from string
    #[new]
    #[pyo3(signature = (in_expr=None))]
    pub fn py_new(in_expr: Option<String>) -> PyResult<Self> {
        match in_expr {
            Some(e) => match parse_expression(&e) {
                Ok(expr) => Ok(PyParameterExpression::new(expr)),
                Err(s) => Err(pyo3::exceptions::PyRuntimeError::new_err(s)),
            },
            None => Ok(PyParameterExpression::new(SymbolExpr::Value(
                symbol_expr::Value::Int(0),
            ))),
        }
    }

    #[staticmethod]
    pub fn from_parameter(param: &Bound<'_, Symbol>) -> Self {
        PyParameterExpression::new(SymbolExpr::Symbol(param.borrow().clone()))
    }

    /// TODO: remove -- this is to be done via PyParameter
    #[allow(non_snake_case)]
    #[staticmethod]
    pub fn Symbol(name: String) -> Self {
        // check if expr contains replacements for sympy
        let name = name
            .replace("__begin_sympy_replace__", "$\\")
            .replace("__end_sympy_replace__", "$");

        PyParameterExpression::new(SymbolExpr::Symbol(Symbol::new(&name, None)))
    }

    /// TODO remove
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

    /// TODO remove
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
            Ok(expr) => Ok(PyParameterExpression::new(expr.bind_by_name(
                &HashMap::from([("I".to_string(), symbol_expr::Value::from(Complex64::i()))]),
            ))),
            Err(s) => Err(pyo3::exceptions::PyRuntimeError::new_err(s)),
        }
    }

    /// return value if expression does not contain any symbols
    pub fn numeric(&self, py: Python) -> PyResult<PyObject> {
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

    pub fn sympify(&self) -> PyResult<()> {
        Err(PyNotImplementedError::new_err("sympify is todo!"))
    }

    #[getter]
    pub fn parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        let py_parameters: Vec<Py<PyParameter>> = self
            .expr
            .parameters()
            .iter()
            .map(|s| Py::new(py, PyParameter::from_symbol(s)))
            .collect::<PyResult<_>>()?;
        PySet::new(py, py_parameters)
    }

    pub fn name_map<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (name, param) in self.get_name_map().iter() {
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

    pub fn arcsin(&self) -> Self {
        self.asin()
    }

    pub fn arccos(&self) -> Self {
        self.acos()
    }

    pub fn arctan(&self) -> Self {
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

    #[pyo3(name = "is_real")]
    pub fn py_is_real(&self) -> Option<bool> {
        self.expr.is_real()
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
    #[pyo3(signature = (parameter_map, allow_unknown_parameters=false))]
    pub fn py_subs(
        &self,
        parameter_map: HashMap<PyParameter, Self>,
        allow_unknown_parameters: bool,
    ) -> PyResult<Self> {
        let map = parameter_map
            .iter()
            .map(|(param, expr)| (param.symbol.clone(), expr.clone()))
            .collect();
        self.subs(&map, allow_unknown_parameters)
            .map_err(|e| e.into())
    }

    // bind values to symbols given by input hashmap
    #[pyo3(name = "bind")]
    #[pyo3(signature = (parameter_values, allow_unknown_parameters=false))]
    pub fn py_bind(
        &self,
        parameter_values: HashMap<PyParameter, Bound<PyAny>>,
        allow_unknown_parameters: bool,
    ) -> PyResult<Self> {
        let map = parameter_values
            .iter()
            .map(|(param, value)| {
                let value = value.extract()?;
                Ok((param.symbol.clone(), value))
            })
            .collect::<PyResult<_>>()?;

        self.bind(&map, allow_unknown_parameters)
            .map_err(|e| e.into())
    }

    #[pyo3(name = "assign")]
    pub fn py_assign(&self, parameter: PyParameter, value: &Bound<PyAny>) -> PyResult<Self> {
        let symbol = parameter.symbol.clone();

        if let Ok(expr) = value.downcast::<Self>() {
            let map = [(symbol, expr.borrow().clone())].into_iter().collect();
            self.subs(&map, false).map_err(|e| e.into())
        } else if let Ok(value) = value.extract::<Value>() {
            let map = [(symbol, value)].into_iter().collect();
            self.bind(&map, false).map_err(|e| e.into())
        } else {
            Err(PyValueError::new_err(
                "Unexpected value in assign: {replacement:?}",
            ))
        }
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

    pub fn __abs__(&self) -> Self {
        self.abs()
    }

    pub fn __pos__(&self) -> Self {
        self.copy()
    }

    pub fn __neg__(&self) -> Self {
        Self {
            expr: -&self.expr,
            name_map: self.name_map.clone(),
        }
    }

    pub fn __add__(&self, rhs: &Bound<PyAny>) -> PyResult<Self> {
        match _extract_value(rhs) {
            Some(rhs) => Ok(Self::new(&self.expr + &rhs.expr)),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __add__",
            )),
        }
    }
    pub fn __radd__(&self, lhs: &Bound<PyAny>) -> PyResult<Self> {
        match _extract_value(lhs) {
            Some(lhs) => Ok(Self::new(&lhs.expr + &self.expr)),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __radd__",
            )),
        }
    }
    pub fn __sub__(&self, rhs: &Bound<PyAny>) -> PyResult<Self> {
        match _extract_value(rhs) {
            Some(rhs) => Ok(Self::new(&self.expr - &rhs.expr)),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __sub__",
            )),
        }
    }
    pub fn __rsub__(&self, lhs: &Bound<PyAny>) -> PyResult<Self> {
        match _extract_value(lhs) {
            Some(lhs) => Ok(Self::new(&lhs.expr - &self.expr)),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __rsub__",
            )),
        }
    }
    pub fn __mul__(&self, rhs: &Bound<PyAny>) -> PyResult<Self> {
        match _extract_value(rhs) {
            Some(rhs) => Ok(Self::new(&self.expr * &rhs.expr)),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __mul__",
            )),
        }
    }
    pub fn __rmul__(&self, lhs: &Bound<PyAny>) -> PyResult<Self> {
        match _extract_value(lhs) {
            Some(lhs) => Ok(Self::new(&lhs.expr * &self.expr)),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __rmul__",
            )),
        }
    }

    pub fn __truediv__(&self, rhs: &Bound<PyAny>) -> PyResult<Self> {
        match _extract_value(rhs) {
            Some(rhs) => {
                if rhs.expr.is_zero() {
                    Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                        "Division by 0.",
                    ))
                } else {
                    Ok(Self::new(&self.expr / &rhs.expr))
                }
            }
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __truediv__",
            )),
        }
    }
    pub fn __rtruediv__(&self, lhs: &Bound<PyAny>) -> PyResult<Self> {
        match _extract_value(lhs) {
            Some(lhs) => Ok(Self::new(&lhs.expr / &self.expr)),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __rtruediv__",
            )),
        }
    }
    pub fn __pow__(&self, rhs: &Bound<PyAny>, _modulo: Option<i32>) -> PyResult<Self> {
        match _extract_value(rhs) {
            Some(rhs) => Ok(Self::new(self.expr.pow(&rhs.expr))),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __pow__",
            )),
        }
    }
    pub fn __rpow__(&self, lhs: &Bound<PyAny>, _modulo: Option<i32>) -> PyResult<Self> {
        match _extract_value(lhs) {
            Some(lhs) => Ok(Self::new(lhs.expr.pow(&self.expr))),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __rpow__",
            )),
        }
    }

    pub fn __int__(&self) -> PyResult<i64> {
        match self.expr.eval(true) {
            Some(value) => match value {
                symbol_expr::Value::Complex(_) => Err(PyTypeError::new_err(
                    "Cannot cast complex parameter to float.",
                )),
                symbol_expr::Value::Real(r) => {
                    let rounded = r.round();
                    if (r - rounded).abs() > SYMEXPR_EPSILON {
                        Err(PyTypeError::new_err("Cannot cast float to int."))
                    } else {
                        Ok(rounded as i64)
                    }
                }
                symbol_expr::Value::Int(i) => Ok(i),
            },
            None => {
                let free_symbols = self.expr.parameters();
                Err(PyTypeError::new_err(format!(
                "Parameter expression with unbound parameters {free_symbols:?} cannot be cast to int.")
            ))
            }
        }
    }

    pub fn __float__(&self) -> PyResult<f64> {
        match self.expr.eval(true) {
            Some(value) => match value {
                symbol_expr::Value::Complex(c) => {
                    if c.im.abs() > SYMEXPR_EPSILON {
                        Err(PyTypeError::new_err(
                            "Cannot cast complex parameter to float.",
                        ))
                    } else {
                        Ok(c.re)
                    }
                }
                symbol_expr::Value::Real(r) => Ok(r),
                symbol_expr::Value::Int(i) => Ok(i as f64),
            },
            None => {
                let free_symbols = self.expr.parameters();
                Err(PyTypeError::new_err(format!(
                "Parameter expression with unbound parameters {free_symbols:?} cannot be cast to float.")
            ))
            }
        }
    }

    pub fn __complex__(&self) -> PyResult<Complex64> {
        match self.expr.eval(true) {
            Some(value) => match value {
                symbol_expr::Value::Complex(c) => Ok(c),
                symbol_expr::Value::Real(r) => Ok(Complex64::new(r, 0.)),
                symbol_expr::Value::Int(i) => Ok(Complex64::new(i as f64, 0.)),
            },
            None => {
                let free_symbols = self.expr.parameters();
                Err(PyTypeError::new_err(format!(
                "Parameter expression with unbound parameters {free_symbols:?} cannot be cast to complex.")
            ))
            }
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

impl Default for PyParameterExpression {
    // default constructor returns zero
    fn default() -> Self {
        Self {
            expr: SymbolExpr::Value(symbol_expr::Value::Int(0)),
            name_map: Some(HashMap::new()), // no parameters, hence empty name map
        }
    }
}

impl fmt::Display for PyParameterExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expr)
    }
}

// rust native implementation will be added in PR #14207
#[pyclass(sequence, subclass, module="qiskit._accelerate.circuit", extends=PyParameterExpression, name="Parameter")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyParameter {
    pub symbol: Symbol,
}

impl Hash for PyParameter {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.symbol.hash(state);
    }
}

impl PyParameter {
    fn from_symbol(symbol: &Symbol) -> PyClassInitializer<Self> {
        let expr = SymbolExpr::Symbol(symbol.clone());

        let py_parameter = Self {
            symbol: symbol.clone(),
        };
        let py_expr = PyParameterExpression::new(expr);

        PyClassInitializer::from(py_expr).add_subclass(py_parameter)
    }
}

#[pymethods]
impl PyParameter {
    #[new]
    #[pyo3(signature = (name, uuid=None))]
    fn py_new(
        py: Python<'_>,
        name: String,
        uuid: Option<PyObject>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let uuid = if let Some(val) = uuid {
            // construct from u128
            let as_u128 = if let Ok(as_u128) = val.extract::<u128>(py) {
                as_u128
            // construct from Python UUID type
            } else if val.bind(py).is_exact_instance(UUID.get_bound(py)) {
                val.getattr(py, "int")?.extract::<u128>(py)?
            // invalid format
            } else {
                return Err(PyTypeError::new_err("not a UUID!"));
            };
            Some(Uuid::from_u128(as_u128))
        } else {
            None
        };

        let symbol = Symbol::new(name.as_str(), uuid);
        let expr = SymbolExpr::Symbol(symbol.clone());

        let py_parameter = Self { symbol };
        let py_expr = PyParameterExpression::new(expr);

        Ok(PyClassInitializer::from(py_expr).add_subclass(py_parameter))
    }

    /// The UUID of the parameter.
    #[getter]
    fn uuid(&self, py: Python<'_>) -> PyResult<PyObject> {
        let uuid = self.symbol.py_uuid();
        let kwargs = [("int", uuid)].into_py_dict(py)?;
        Ok(UUID.get_bound(py).call((), Some(&kwargs))?.unbind())
    }

    pub fn __repr__<'py>(&self, py: Python<'py>) -> Bound<'py, PyString> {
        let str = format!(
            "Parameter(name={}, uuid={})",
            self.symbol.name(),
            self.symbol.py_uuid()
        );
        PyString::new(py, str.as_str())
    }
}

#[pyclass(sequence, subclass, module="qiskit._accelerate.circuit", extends=PyParameter, name="ParameterVectorElement")]
#[derive(Clone, Debug)]
pub struct PyParameterVectorElement {
    index: u64,
    vector: PyObject,
}

#[pymethods]
impl PyParameterVectorElement {
    #[new]
    #[pyo3(signature = (vector, index, uuid=None))]
    pub fn py_new(
        py: Python<'_>,
        vector: PyObject,
        index: u64,
        uuid: Option<PyObject>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let vector_name = vector.getattr(py, "name")?.extract::<String>(py)?;

        let name = format!("{}[{}]", vector_name, index);

        let py_parameter = PyParameter::py_new(py, name, uuid)?;
        let py_element = Self {
            index,
            vector: vector.clone_ref(py),
        };

        Ok(py_parameter.add_subclass(py_element))
    }

    #[getter]
    pub fn index(&self) -> u64 {
        self.index
    }

    #[getter]
    pub fn vector(&self, py: Python<'_>) -> PyObject {
        self.vector.clone_ref(py)
    }
}
