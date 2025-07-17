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

use hashbrown::hash_map::Entry;
use hashbrown::{HashMap, HashSet};
use num_complex::Complex64;
use pyo3::conversion::FromPyObjectBound;
use pyo3::exceptions::{
    PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError, PyZeroDivisionError,
};
use pyo3::types::{IntoPyDict, PyNotImplemented, PySet, PyString};
use thiserror::Error;
use uuid::Uuid;

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;

use crate::circuit_data::CircuitError;
use crate::imports::{BUILTIN_HASH, SYMPIFY_PARAMETER_EXPRESSION, UUID};
use crate::parameter::symbol_expr::SymbolExpr;
use crate::parameter::symbol_parser::parse_expression;
use crate::parameter::{self, symbol_expr};
use crate::util::c64;

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
    #[error("Cannot bind following parameters not present in expression: {0:?}")]
    UnknownParameters(HashSet<Symbol>),
    #[error("Name conflict adding parameters.")]
    NameConflict,
    #[error("Invalid cast to OpCode: {0}")]
    InvalidU8ToOpCode(u8),
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
            ParameterError::UnknownParameter(_)
            | ParameterError::UnknownParameters(_)
            | ParameterError::NameConflict => CircuitError::new_err(value.to_string()),
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
#[derive(Clone, Debug)]
pub struct PyParameterExpression {
    // The symbolic expression.
    pub expr: SymbolExpr,
    // A map keeping track of all symbols, with their name. This map *must* be kept
    // up to date upon any operation performed on the expression.
    pub name_map: HashMap<String, PyParameter>,
}

impl Hash for PyParameterExpression {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.expr.to_string().hash(state);
    }
}

impl PartialEq for PyParameterExpression {
    fn eq(&self, other: &Self) -> bool {
        self.expr.eq(&other.expr)
    }
}

impl Eq for PyParameterExpression {}

#[inline]
fn _extract_value(value: &Bound<PyAny>) -> Option<PyParameterExpression> {
    if let Ok(i) = value.extract::<i64>() {
        Some(PyParameterExpression::new(SymbolExpr::Value(
            symbol_expr::Value::from(i),
        )))
    } else if let Ok(c) = value.extract::<Complex64>() {
        if c.is_infinite() || c.is_nan() {
            return None;
        }
        Some(PyParameterExpression::new(SymbolExpr::Value(
            symbol_expr::Value::from(c),
        )))
    } else if let Ok(r) = value.extract::<f64>() {
        if r.is_infinite() || r.is_nan() {
            return None;
        }
        Some(PyParameterExpression::new(SymbolExpr::Value(
            symbol_expr::Value::from(r),
        )))
    } else if let Ok(parameter) = value.extract::<PyParameter>() {
        Some(parameter.symbol.as_expr())
    } else if let Ok(element) = value.extract::<PyParameterVectorElement>() {
        Some(element.symbol.as_expr())
    } else {
        value.extract::<PyParameterExpression>().ok()
    }
}

impl PyParameterExpression {
    pub fn new(expr: SymbolExpr) -> Self {
        Self {
            expr: expr.clone(),
            name_map: expr
                .name_map()
                .into_iter()
                .map(|(name, symbol)| (name, PyParameter { symbol }))
                .collect(),
        }
    }

    pub fn from_qpy(replay: &[OPReplay]) -> Result<Self, ParameterError> {
        let binary_ops: HashSet<OpCode> = [
            OpCode::ADD,
            OpCode::SUB,
            OpCode::MUL,
            OpCode::DIV,
            OpCode::POW,
            OpCode::RSUB,
            OpCode::RDIV,
            OpCode::RPOW,
        ]
        .into();
        let mut stack: Vec<PyParameterExpression> = Vec::new();
        for inst in replay.iter() {
            let OPReplay { op, lhs, rhs } = inst;
            if let Some(value) = lhs {
                stack.push(value.clone().into());
            }
            if let Some(value) = rhs {
                stack.push(value.clone().into());
            }

            let rhs = if binary_ops.contains(op) {
                Some(stack.pop().expect("Pop from empty stack"))
            } else {
                None
            };

            let lhs = stack.pop().expect("Pop from empty stack");

            let result: PyParameterExpression = match op {
                OpCode::ADD => lhs.add(&rhs.unwrap())?,
                OpCode::MUL => lhs.mul(&rhs.unwrap())?,
                OpCode::SUB => lhs.sub(&rhs.unwrap())?,
                OpCode::RSUB => lhs.rsub(&rhs.unwrap())?,
                OpCode::POW => lhs.pow(&rhs.unwrap())?,
                OpCode::RPOW => lhs.rpow(&rhs.unwrap())?,
                OpCode::DIV => lhs.div(&rhs.unwrap())?,
                OpCode::RDIV => lhs.rdiv(&rhs.unwrap())?,
                OpCode::ABS => lhs.py_abs(),
                OpCode::SIN => lhs.sin(),
                OpCode::ASIN => lhs.asin(),
                OpCode::COS => lhs.cos(),
                OpCode::ACOS => lhs.acos(),
                OpCode::TAN => lhs.tan(),
                OpCode::ATAN => lhs.atan(),
                OpCode::CONJ => lhs.conjugate(),
                OpCode::LOG => lhs.log(),
                OpCode::EXP => lhs.exp(),
                OpCode::SIGN => lhs.sign(),
                OpCode::GRAD | OpCode::SUBSTITUTE => {
                    unreachable!("GRAD and SUBSTITUTE are not supported.")
                }
            };

            stack.push(result);
        }

        Ok(stack.pop().expect("Final pop is empty"))
    }

    pub fn add(&self, rhs: &PyParameterExpression) -> Result<Self, ParameterError> {
        let name_map = self.update_name_map(&rhs)?;
        Ok(Self {
            expr: &self.expr + &rhs.expr,
            name_map,
        })
    }

    pub fn mul(&self, rhs: &PyParameterExpression) -> Result<Self, ParameterError> {
        let name_map = self.update_name_map(&rhs)?;
        Ok(Self {
            expr: &self.expr * &rhs.expr,
            name_map,
        })
    }

    pub fn sub(&self, rhs: &PyParameterExpression) -> Result<Self, ParameterError> {
        let name_map = self.update_name_map(&rhs)?;
        Ok(Self {
            expr: &self.expr - &rhs.expr,
            name_map,
        })
    }

    pub fn rsub(&self, lhs: &PyParameterExpression) -> Result<Self, ParameterError> {
        let name_map = self.update_name_map(&lhs)?;
        Ok(Self {
            expr: &lhs.expr - &self.expr,
            name_map,
        })
    }

    pub fn div(&self, rhs: &PyParameterExpression) -> Result<Self, ParameterError> {
        if rhs.expr.is_zero() {
            return Err(ParameterError::ZeroDivisionError);
        }

        let name_map = self.update_name_map(&rhs)?;
        Ok(Self {
            expr: &self.expr / &rhs.expr,
            name_map,
        })
    }

    pub fn rdiv(&self, lhs: &PyParameterExpression) -> Result<Self, ParameterError> {
        if self.expr.is_zero() {
            return Err(ParameterError::ZeroDivisionError);
        }

        let name_map = self.update_name_map(&lhs)?;
        Ok(Self {
            expr: &lhs.expr / &self.expr,
            name_map,
        })
    }

    pub fn pow(&self, rhs: &PyParameterExpression) -> Result<Self, ParameterError> {
        let name_map = self.update_name_map(&rhs)?;
        Ok(Self {
            expr: self.expr.pow(&rhs.expr),
            name_map,
        })
    }

    pub fn rpow(&self, lhs: &PyParameterExpression) -> Result<Self, ParameterError> {
        let name_map = self.update_name_map(&lhs)?;
        Ok(Self {
            expr: lhs.expr.pow(&self.expr),
            name_map,
        })
    }

    pub fn sin(&self) -> Self {
        Self {
            expr: self.expr.sin(),
            name_map: self.name_map.clone(),
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

    /// Clone the expression.
    pub fn copy(&self) -> Self {
        Self {
            expr: self.expr.clone(),
            name_map: self.name_map.clone(),
        }
    }

    /// Complex conjugate the expression.
    pub fn conjugate(&self) -> Self {
        Self {
            expr: self.expr.conjugate(),
            name_map: self.name_map.clone(),
        }
    }

    /// Compute the derivative of the expression with respect to the provided symbol.
    pub fn derivative(&self, param: &Symbol) -> Result<Self, String> {
        self.expr.derivative(param).map(Self::new)
    }

    /// Expand the expression.
    pub fn expand(&self) -> Self {
        Self {
            expr: self.expr.expand(),
            name_map: self.name_map.clone(),
        }
    }

    /// Substitute symbols with [PyParameterExpression]s.
    pub fn subs(
        &self,
        map: &HashMap<Symbol, Self>,
        allow_unknown_parameters: bool,
    ) -> Result<Self, ParameterError> {
        // Build the outgoing name map. In the process we check for any duplicates.
        let mut name_map: HashMap<String, PyParameter> = HashMap::new();
        let mut symbol_map: HashMap<Symbol, SymbolExpr> = HashMap::new();

        if !allow_unknown_parameters {
            let existing: HashSet<&Symbol> = self.name_map.values().map(|p| &p.symbol).collect();
            let to_replace: HashSet<&Symbol> = map.keys().collect();
            let difference: HashSet<Symbol> = to_replace
                .difference(&existing)
                .map(|&symbol| symbol.clone())
                .collect();
            if !difference.is_empty() {
                return Err(ParameterError::UnknownParameters(difference));
            }
        }

        for (name, py_param) in self.name_map.iter() {
            let symbol = &py_param.symbol;

            // check if the symbol will get replaced
            if let Some(replacement) = map.get(symbol) {
                // If yes, update the name_map. This also checks for duplicates.
                for (replacement_name, replacement_symbol) in replacement.name_map.iter() {
                    if let Some(duplicate) = name_map.get(replacement_name) {
                        if duplicate != replacement_symbol {
                            return Err(ParameterError::NameConflict);
                        } else {
                            // symbol already exists, nothing to do
                        }
                    } else {
                        // SAFETY: We know the key does not exist yet.
                        unsafe {
                            name_map.insert_unique_unchecked(
                                replacement_name.clone(),
                                replacement_symbol.clone(),
                            )
                        };
                    }
                }

                // If we got until here, there were no duplicates, so we are safe to
                // add this symbol to the internal replacement map.
                symbol_map.insert(symbol.clone(), replacement.expr.clone());
            } else {
                // no replacement for this symbol, carry on
                // TODO maybe this needs to do clone_ref(py) ?
                match name_map.entry(name.clone()) {
                    Entry::Occupied(duplicate) => {
                        if duplicate.get() != py_param {
                            return Err(ParameterError::NameConflict);
                        }
                    }
                    Entry::Vacant(e) => {
                        e.insert(py_param.clone());
                    }
                }
            }
        }

        let res = self.expr.subs(&symbol_map);
        // println!("Called subs -- output is {res:?} and name map {name_map:?}");
        Ok(Self {
            expr: res,
            name_map,
        })
    }

    pub fn bind(
        &self,
        map: &HashMap<Symbol, symbol_expr::Value>,
        allow_unknown_parameters: bool,
    ) -> Result<Self, ParameterError> {
        // The set of symbols we will bind. Used twice, hence pre-computed here.
        let bind_symbols: HashSet<Symbol> = map.keys().cloned().collect();

        if !allow_unknown_parameters {
            let existing_symbols: HashSet<Symbol> = self
                .name_map
                .values()
                .map(|py_parameter| py_parameter.symbol.clone())
                .collect();
            let difference: HashSet<Symbol> = bind_symbols
                .difference(&existing_symbols)
                .cloned()
                .collect();
            if !difference.is_empty() {
                return Err(ParameterError::UnknownParameters(difference));
            }
        }

        // bind the symbol expression and then check the outcome for inf/nan, or numeric values
        let bound_expr = self.expr.bind(map);
        let bound = match bound_expr.eval(true) {
            Some(v) => match &v {
                symbol_expr::Value::Real(r) => {
                    if r.is_infinite() {
                        Err(ParameterError::BindingInf)
                    } else if r.is_nan() {
                        Err(ParameterError::BindingNaN)
                    } else {
                        Ok(SymbolExpr::Value(v))
                    }
                }
                symbol_expr::Value::Int(_) => Ok(SymbolExpr::Value(v)),
                symbol_expr::Value::Complex(c) => {
                    if c.re.is_infinite() || c.im.is_infinite() {
                        Err(ParameterError::ZeroDivisionError) // TODO this should probs be BindingInf
                    } else if c.re.is_nan() || c.im.is_nan() {
                        Err(ParameterError::BindingNaN)
                    } else if (-symbol_expr::SYMEXPR_EPSILON..symbol_expr::SYMEXPR_EPSILON)
                        .contains(&c.im)
                    {
                        Ok(SymbolExpr::Value(symbol_expr::Value::Real(c.re)))
                    } else {
                        Ok(SymbolExpr::Value(v))
                    }
                }
            },
            None => Ok(bound_expr),
        }?;

        // update the name map by removing the bound parameters
        let bound_name_map: HashMap<String, PyParameter> = self
            .name_map
            .iter()
            .filter(|(_, py_param)| !bind_symbols.contains(&py_param.symbol))
            .map(|(name, symbol)| (name.clone(), symbol.clone()))
            .collect();

        // println!("- Called bind");
        // println!(
        //     "-- self is {:?} and name map {:?}",
        //     self.expr, self.name_map
        // );
        // println!("-- bind instr is {:?}", map);
        // println!("-- output is {bound:?} and name map {bound_name_map:?}");
        Ok(Self {
            expr: bound,
            name_map: bound_name_map,
        })
    }

    /// Check whether a hashmap of incoming parameters have a name conflict with the expression.
    ///
    /// Args:
    ///     - inbound_parameters: The hashmap of incoming parameters. Can e.g. be a map of parameters
    ///         for subtitution, or the parameters of another expression that we merge with.
    ///     - replacement: Set to ``true`` for substitutions, ``false`` for merge.
    fn has_name_conflicts(
        &self,
        inbound_parameters: &HashMap<String, PyParameter>,
        outbound: Option<&HashSet<String>>,
    ) -> bool {
        for (name, param) in inbound_parameters.iter() {
            if let Some(existing_param) = self.name_map.get(name) {
                if let Some(outbound) = outbound {
                    if outbound.contains(name) {
                        continue;
                    }
                }
                if param.symbol != existing_param.symbol {
                    return true;
                }
            }
        }
        false
    }

    /// Merge name maps. Returns an error if there is a name conflict.
    ///
    /// Args:
    ///     - other: The other parameter expression whose symbols we add to self.
    fn update_name_map(
        &self,
        other: &Self,
    ) -> Result<HashMap<String, PyParameter>, ParameterError> {
        let mut merged = self.name_map.clone();
        for (name, param) in other.name_map.iter() {
            match merged.get(name) {
                Some(existing_param) => {
                    if param != existing_param {
                        return Err(ParameterError::NameConflict);
                    }
                }
                None => {
                    // SAFETY: We ensured the key is unique
                    let _ = unsafe { merged.insert_unique_unchecked(name.clone(), param.clone()) };
                }
            }
        }
        Ok(merged)
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

        PyParameterExpression::new(SymbolExpr::Symbol(Symbol::new(&name, None, None)))
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
            Ok(expr) => {
                let expr = PyParameterExpression::new(expr);
                if let Some(imag_symbol) = expr.name_map.get("I") {
                    let bind_map: HashMap<Symbol, Value> = HashMap::from([(
                        imag_symbol.symbol.clone(),
                        symbol_expr::Value::Complex(c64(0., 1.)),
                    )]);
                    Ok(expr.bind(&bind_map, false)?)
                } else {
                    Ok(expr)
                }
            }
            Err(s) => Err(pyo3::exceptions::PyRuntimeError::new_err(s)),
        }
    }

    /// Return an error if names in the other expression collide with existing names.
    fn raise_if_name_conflict(&self, other: &Self) -> PyResult<()> {
        if self.has_name_conflicts(&other.name_map, None) {
            Err(CircuitError::new_err(
                "Name conflict applying operation __add__",
            ))
        } else {
            Ok(())
        }
    }

    /// Check if the expression corresponds to a plain symbol.
    pub fn is_symbol(&self) -> bool {
        match &self.expr {
            SymbolExpr::Symbol(_) => true,
            _ => false,
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
            None => {
                let free_symbols = self.expr.parameters();
                Err(PyTypeError::new_err(format!(
                    "Parameter expression with unbound parameters {free_symbols:?} is not numeric."
                )))
            }
        }
    }

    pub fn sympify(&self, py: Python) -> PyResult<PyObject> {
        let py_sympify = SYMPIFY_PARAMETER_EXPRESSION.get(py);
        py_sympify.call1(py, (self.clone(),))
    }

    #[getter]
    pub fn parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        let py_parameters: Vec<PyObject> = self
            .name_map
            .values()
            .map(|param| {
                let s = &param.symbol;
                match (s.index, &s.vector) {
                    // if index and vector is set, it is an element
                    (Some(_index), Some(_vector)) => {
                        Ok(Py::new(py, PyParameterVectorElement::from_symbol(s))?.into_any())
                    }
                    // else, a normal parameter
                    _ => Ok(Py::new(py, PyParameter::from_symbol(s))?.into_any()),
                }
            })
            .collect::<PyResult<_>>()?;
        PySet::new(py, py_parameters)
    }

    /// Sine of the expression.
    #[pyo3(name = "sin")]
    pub fn py_sin(&self) -> Self {
        self.sin()
    }

    /// Cosine of the expression.
    #[pyo3(name = "cos")]
    pub fn py_cos(&self) -> Self {
        self.cos()
    }

    /// Tangent of the expression.
    #[pyo3(name = "tan")]
    pub fn py_tan(&self) -> Self {
        self.tan()
    }

    /// Arcsine of the expression.
    pub fn arcsin(&self) -> Self {
        self.asin()
    }

    /// Arccosine of the expression.
    pub fn arccos(&self) -> Self {
        self.acos()
    }

    /// Arctangent of the expression.
    pub fn arctan(&self) -> Self {
        self.atan()
    }

    /// Exponentiate the expression
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

    fn __copy__(slf: PyRef<Self>) -> PyRef<Self> {
        // ParameterExpression is immutable.
        slf
    }

    fn __deepcopy__<'py>(slf: PyRef<'py, Self>, _memo: Bound<'py, PyAny>) -> PyRef<'py, Self> {
        // Everything a ParameterExpression contains is immutable.
        slf
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
    pub fn gradient(&self, param: &Bound<'_, PyAny>) -> PyResult<Self> {
        let symbol = symbol_from_py_parameter(param)?;
        self.derivative(&symbol).map_err(PyRuntimeError::new_err)
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
        self.to_string()
    }

    /// substitute symbols to expressions (or values) given by hash map
    #[pyo3(name = "subs")]
    #[pyo3(signature = (parameter_map, allow_unknown_parameters=false))]
    pub fn py_subs(
        &self,
        parameter_map: HashMap<PyParameter, Self>,
        allow_unknown_parameters: bool,
    ) -> PyResult<Self> {
        println!("called expr subs");
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

        println!("called expr assign");
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
            Some(rhs) => self.add(&rhs).map_err(|e| e.into()),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __add__",
            )),
        }
    }

    pub fn __radd__(&self, lhs: &Bound<PyAny>) -> PyResult<Self> {
        match _extract_value(lhs) {
            Some(lhs) => {
                let name_map = self.update_name_map(&lhs)?;
                Ok(Self {
                    expr: &lhs.expr + &self.expr,
                    name_map,
                })
            }
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __radd__",
            )),
        }
    }

    pub fn __sub__(&self, rhs: &Bound<PyAny>) -> PyResult<Self> {
        match _extract_value(rhs) {
            Some(rhs) => self.sub(&rhs).map_err(|e| e.into()),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __sub__",
            )),
        }
    }

    pub fn __rsub__(&self, lhs: &Bound<PyAny>) -> PyResult<Self> {
        match _extract_value(lhs) {
            // TODO do we need .rsub or can I just  do lhs.sub(&self) ???
            Some(lhs) => self.rsub(&lhs).map_err(|e| e.into()),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __rsub__",
            )),
        }
    }

    pub fn __mul__<'py>(&self, rhs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = rhs.py();
        match _extract_value(rhs) {
            Some(rhs) => match self.mul(&rhs) {
                Ok(result) => result.into_bound_py_any(py),
                Err(e) => Err(PyErr::from(e)),
            },
            None => PyNotImplemented::get(py).into_bound_py_any(py),
        }
    }

    pub fn __rmul__(&self, lhs: &Bound<PyAny>) -> PyResult<Self> {
        match _extract_value(lhs) {
            Some(lhs) => {
                let name_map = self.update_name_map(&lhs)?;
                Ok(Self {
                    expr: &lhs.expr * &self.expr,
                    name_map,
                })
            }
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __rmul__",
            )),
        }
    }

    pub fn __truediv__(&self, rhs: &Bound<PyAny>) -> PyResult<Self> {
        match _extract_value(rhs) {
            Some(rhs) => self.div(&rhs).map_err(|e| e.into()),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __truediv__",
            )),
        }
    }
    pub fn __rtruediv__(&self, lhs: &Bound<PyAny>) -> PyResult<Self> {
        match _extract_value(lhs) {
            Some(lhs) => self.rdiv(&lhs).map_err(|e| e.into()),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __rtruediv__",
            )),
        }
    }
    pub fn __pow__(&self, rhs: &Bound<PyAny>, _modulo: Option<i32>) -> PyResult<Self> {
        match _extract_value(rhs) {
            Some(rhs) => self.pow(&rhs).map_err(|e| e.into()),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __pow__",
            )),
        }
    }
    pub fn __rpow__(&self, lhs: &Bound<PyAny>, _modulo: Option<i32>) -> PyResult<Self> {
        match _extract_value(lhs) {
            Some(lhs) => self.rpow(&lhs).map_err(|e| e.into()),
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
                    let rounded = r.floor();
                    Ok(rounded as i64)
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
                            "Could not cast complex parameter expression to float.",
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
        self.to_string()
    }

    pub fn __hash__(&self, py: Python) -> PyResult<u64> {
        match self.expr.eval(true) {
            // if a value, we promise to match the hash of the raw value!
            Some(value) => {
                let py_hash = BUILTIN_HASH.get_bound(py);
                match value {
                    symbol_expr::Value::Complex(c) => py_hash.call1((c,))?.extract::<u64>(),
                    symbol_expr::Value::Real(r) => py_hash.call1((r,))?.extract::<u64>(),
                    symbol_expr::Value::Int(i) => py_hash.call1((i,))?.extract::<u64>(),
                }
            }
            None => {
                let mut hasher = DefaultHasher::new();
                self.expr.string_id().hash(&mut hasher);
                Ok(hasher.finish())
            }
        }
    }

    // for pickle, we can reproduce equation from expression string
    fn __getstate__(&self) -> (Vec<OPReplay>, HashMap<String, PyParameter>) {
        // (self.expr.to_string(), self.name_map.clone())
        (self._qpy_replay(), self.name_map.clone())
    }

    fn __setstate__(
        &mut self,
        state: (Vec<OPReplay>, HashMap<String, PyParameter>),
    ) -> PyResult<()> {
        self.name_map = state.1;
        let from_qpy = Self::from_qpy(&state.0)?;
        // if let Ok(expr) = parse_expression(&state.0) {
        //     self.expr = expr;
        // }
        self.expr = from_qpy.expr;
        // println!("-- expr post pickle: {:?} {:?}", self.expr, self.name_map);
        Ok(())
    }

    #[getter]
    fn _qpy_replay(&self) -> Vec<OPReplay> {
        // let input = ParameterValueType::Expression(self.clone());
        let mut replay = Vec::new();
        qpy_replay(&self, &self.name_map, &mut replay);
        replay
    }
}

impl Default for PyParameterExpression {
    // default constructor returns zero
    fn default() -> Self {
        Self {
            expr: SymbolExpr::Value(symbol_expr::Value::Int(0)),
            name_map: HashMap::new(), // no parameters, hence empty name map
        }
    }
}

impl fmt::Display for PyParameterExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", {
            if let SymbolExpr::Symbol(s) = &self.expr {
                s.name()
            } else {
                match self.expr.eval(true) {
                    Some(e) => e.to_string(),
                    None => self.expr.optimize().to_string(),
                }
            }
        })
    }
}

// rust native implementation will be added in PR #14207
#[pyclass(sequence, subclass, module="qiskit._accelerate.circuit", extends=PyParameterExpression, name="Parameter")]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd)]
pub struct PyParameter {
    pub symbol: Symbol,
}

impl Hash for PyParameter {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.symbol.hash(state);
    }
}

impl<'py> IntoPyObject<'py> for PyParameter {
    type Target = PyParameter;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let symbol = &self.symbol;
        let expr = SymbolExpr::Symbol(symbol.clone());
        let py_expr = PyParameterExpression::new(expr);

        Ok(Py::new(py, (self, py_expr))?.into_bound(py))
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
        let uuid = uuid_from_py(py, uuid)?;
        let symbol = Symbol::new(name.as_str(), uuid, None);
        let expr = SymbolExpr::Symbol(symbol.clone());

        let py_parameter = Self { symbol };
        let py_expr = PyParameterExpression::new(expr);

        Ok(PyClassInitializer::from(py_expr).add_subclass(py_parameter))
    }

    /// The UUID of the parameter.
    #[getter]
    fn uuid(&self, py: Python<'_>) -> PyResult<PyObject> {
        // let uuid = self.symbol.py_uuid();
        // let kwargs = [("int", uuid)].into_py_dict(py)?;
        // Ok(UUID.get_bound(py).call((), Some(&kwargs))?.unbind())
        uuid_to_py(py, self.symbol.uuid)
    }

    pub fn __repr__<'py>(&self, py: Python<'py>) -> Bound<'py, PyString> {
        let str = format!(
            "Parameter(name={}, uuid={})",
            self.symbol.name(),
            self.symbol.py_uuid()
        );
        PyString::new(py, str.as_str())
    }

    pub fn __getnewargs__(&self) -> (String, u128) {
        (self.symbol.name(), self.symbol.py_uuid())
    }

    pub fn __getstate__(&self) -> (String, u128) {
        (self.symbol.name(), self.symbol.py_uuid())
    }

    pub fn __setstate__(&mut self, state: (String, u128)) {
        let name = state.0.as_str();
        let uuid = Uuid::from_u128(state.1);
        let symbol = Symbol::new(name, Some(uuid), None);
        self.symbol = symbol;
        // println!("--- param post pickle: {:?}", self.symbol);
    }

    /// This function is to get the types right on Python side
    pub fn sympify(&self, py: Python) -> PyResult<PyObject> {
        let py_sympify = SYMPIFY_PARAMETER_EXPRESSION.get(py);
        py_sympify.call1(py, (self.clone(),))
    }

    fn __copy__(slf: PyRef<Self>) -> PyRef<Self> {
        // ParameterExpression is immutable.
        slf
    }

    fn __deepcopy__<'py>(slf: PyRef<'py, Self>, _memo: Bound<'py, PyAny>) -> PyRef<'py, Self> {
        // Everything a ParameterExpression contains is immutable.
        slf
    }

    #[pyo3(name = "subs")]
    #[pyo3(signature = (parameter_map, allow_unknown_parameters=false))]
    pub fn py_subs<'py>(
        &self,
        py: Python<'py>,
        parameter_map: HashMap<PyParameter, Bound<'py, PyAny>>,
        allow_unknown_parameters: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        // This is such that x.subs({x: y}) remains a Parameter, and is not upgraded to an expression
        println!("called param subs");
        match parameter_map.get(self) {
            None => {
                if allow_unknown_parameters {
                    self.clone().into_bound_py_any(py)
                } else {
                    Err(CircuitError::new_err(
                        "Cannot bind parameters not present in parameter.",
                    ))
                }
            }
            Some(replacement) => {
                if allow_unknown_parameters || parameter_map.len() == 1 {
                    Ok(replacement.clone())
                } else {
                    Err(CircuitError::new_err(
                        "Cannot bind parameters not present in parameter.",
                    ))
                }
            }
        }
    }

    #[pyo3(name = "bind")]
    #[pyo3(signature = (parameter_values, allow_unknown_parameters=false))]
    pub fn py_bind<'py>(
        &self,
        py: Python<'py>,
        parameter_values: HashMap<PyParameter, Bound<'py, PyAny>>,
        allow_unknown_parameters: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        // return PyObject to cover Parameter and ParameterExpression(value)
        match parameter_values.get(self) {
            None => {
                if allow_unknown_parameters {
                    self.clone().into_bound_py_any(py)
                } else {
                    Err(CircuitError::new_err(
                        "Cannot bind parameters not present in parameter.",
                    ))
                }
            }
            Some(replacement) => {
                if allow_unknown_parameters || parameter_values.len() == 1 {
                    if let Some(expr) = _extract_value(replacement) {
                        if let SymbolExpr::Value(_) = expr.expr {
                            return expr.into_bound_py_any(py);
                        }
                    }
                    Err(PyValueError::new_err("Invalid binding value."))
                } else {
                    Err(CircuitError::new_err(
                        "Cannot bind parameters not present in parameter.",
                    ))
                }
            }
        }
    }

    #[pyo3(name = "assign")]
    pub fn py_assign<'py>(
        &self,
        py: Python<'py>,
        parameter: PyParameter,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Ok(_) = value.extract::<Value>() {
            let map = [(parameter, value.clone())].into_iter().collect();
            self.py_bind(py, map, false).map_err(|e| e.into())
        } else if let Ok(_) = value.downcast::<PyParameterExpression>() {
            let map = [(parameter, value.clone())].into_iter().collect();
            self.py_subs(py, map, false).map_err(|e| e.into())
        } else {
            Err(PyValueError::new_err(
                "Unexpected value in assign: {replacement:?}",
            ))
        }
    }
}

#[pyclass(sequence, subclass, module="qiskit._accelerate.circuit", extends=PyParameter, name="ParameterVectorElement")]
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd)]
pub struct PyParameterVectorElement {
    symbol: Symbol,
}

impl<'py> IntoPyObject<'py> for PyParameterVectorElement {
    type Target = PyParameterVectorElement;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let symbol = &self.symbol;
        let py_param = PyParameter::from_symbol(symbol);
        let py_element = py_param.add_subclass(self);

        Ok(Py::new(py, py_element)?.into_bound(py))
    }
}

impl PyParameterVectorElement {
    fn from_symbol(symbol: &Symbol) -> PyClassInitializer<Self> {
        let py_element = Self {
            symbol: symbol.clone(),
        };
        let py_parameter = PyParameter::from_symbol(symbol);

        py_parameter.add_subclass(py_element)
    }
}

#[pymethods]
impl PyParameterVectorElement {
    #[new]
    #[pyo3(signature = (vector, index, uuid=None))]
    pub fn py_new(
        py: Python<'_>,
        vector: PyObject,
        index: u32,
        uuid: Option<PyObject>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let vector_name = vector.getattr(py, "name")?.extract::<String>(py)?;
        let uuid = uuid_from_py(py, uuid)?.unwrap_or(Uuid::new_v4());

        let symbol = Symbol::py_new(
            &vector_name,
            Some(uuid.as_u128()),
            Some(index),
            Some(vector.clone_ref(py)),
        )?;

        let py_parameter = PyParameter::from_symbol(&symbol);
        let py_element = Self { symbol };

        Ok(py_parameter.add_subclass(py_element))
    }

    pub fn __getnewargs__(&self, py: Python) -> PyResult<(PyObject, u32, Option<PyObject>)> {
        let vector = self
            .symbol
            .vector
            .clone()
            .expect("vector element should have a vector");
        let index = self
            .symbol
            .index
            .expect("vector element should have an index");
        let uuid = uuid_to_py(py, self.symbol.uuid)?;
        Ok((vector, index, Some(uuid)))
    }

    pub fn __getstate__(&self, py: Python) -> PyResult<(PyObject, u32, Option<PyObject>)> {
        self.__getnewargs__(py)
    }

    pub fn __setstate__(
        &mut self,
        py: Python,
        state: (PyObject, u32, Option<PyObject>),
    ) -> PyResult<()> {
        let vector = state.0;
        let index = state.1;
        let vector_name = vector.getattr(py, "name")?.extract::<String>(py)?;
        let uuid = uuid_from_py(py, state.2)?.map(|id| id.as_u128());
        self.symbol = Symbol::py_new(&vector_name, uuid, Some(index), Some(vector))?;
        Ok(())
    }

    #[getter]
    pub fn index(&self) -> u32 {
        self.symbol
            .index
            .expect("A vector element should have an index")
    }

    #[getter]
    pub fn vector(&self) -> PyObject {
        self.symbol
            .clone()
            .vector
            .expect("A vector element should have a vector")
    }

    /// wrong backward compatibility -- this should not be used, but some methods do use it
    #[getter]
    pub fn _vector(&self) -> PyObject {
        self.vector()
    }

    fn __copy__(slf: PyRef<Self>) -> PyRef<Self> {
        // ParameterExpression is immutable.
        slf
    }

    fn __deepcopy__<'py>(slf: PyRef<'py, Self>, _memo: Bound<'py, PyAny>) -> PyRef<'py, Self> {
        // Everything a ParameterExpression contains is immutable.
        slf
    }
}

fn uuid_from_py(py: Python<'_>, uuid: Option<PyObject>) -> PyResult<Option<Uuid>> {
    if let Some(val) = uuid {
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
        Ok(Some(Uuid::from_u128(as_u128)))
    } else {
        Ok(None)
    }
}

fn uuid_to_py(py: Python<'_>, uuid: Uuid) -> PyResult<PyObject> {
    let uuid = uuid.as_u128();
    let kwargs = [("int", uuid)].into_py_dict(py)?;
    Ok(UUID.get_bound(py).call((), Some(&kwargs))?.unbind())
}

fn symbol_from_py_parameter(param: &Bound<'_, PyAny>) -> PyResult<Symbol> {
    if let Ok(element) = param.extract::<PyParameterVectorElement>() {
        Ok(element.symbol.clone())
    } else if let Ok(parameter) = param.extract::<PyParameter>() {
        Ok(parameter.symbol.clone())
    } else {
        Err(PyValueError::new_err("Could not extract parameter"))
    }
}

#[derive(IntoPyObject, FromPyObject, Clone, Debug)]
pub enum ParameterValueType {
    Int(i64),
    Float(f64),
    Complex(Complex64),
    Parameter(PyParameter),
    VectorElement(PyParameterVectorElement),
}

impl ParameterValueType {
    fn extract_from_expr(expr: &Box<SymbolExpr>) -> Option<ParameterValueType> {
        if let Some(value) = expr.eval(true) {
            match value {
                Value::Int(i) => Some(ParameterValueType::Int(i)),
                Value::Real(r) => Some(ParameterValueType::Float(r)),
                Value::Complex(c) => Some(ParameterValueType::Complex(c)),
            }
        } else {
            if let SymbolExpr::Symbol(symbol) = expr.as_ref() {
                match symbol.index {
                    None => {
                        let param = PyParameter {
                            symbol: symbol.clone(),
                        };
                        Some(ParameterValueType::Parameter(param))
                    }
                    Some(_) => {
                        let param = PyParameterVectorElement {
                            symbol: symbol.clone(),
                        };
                        Some(ParameterValueType::VectorElement(param))
                    }
                }
            } else {
                // ParameterExpressions have the value None, as they must be constructed
                None
            }
        }
    }
}

impl From<ParameterValueType> for PyParameterExpression {
    fn from(value: ParameterValueType) -> Self {
        match value {
            ParameterValueType::Parameter(param) => {
                let expr = SymbolExpr::Symbol(param.symbol);
                Self::new(expr)
            }
            ParameterValueType::VectorElement(param) => {
                let expr = SymbolExpr::Symbol(param.symbol);
                Self::new(expr)
            }
            ParameterValueType::Int(i) => {
                let expr = SymbolExpr::Value(Value::Int(i));
                Self::new(expr)
            }
            ParameterValueType::Float(f) => {
                let expr = SymbolExpr::Value(Value::Real(f));
                Self::new(expr)
            }
            ParameterValueType::Complex(c) => {
                let expr = SymbolExpr::Value(Value::Complex(c));
                Self::new(expr)
            }
        }
    }
}

#[pyclass(module = "qiskit._accelerate.circuit")]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum OpCode {
    ADD = 0,
    SUB = 1,
    MUL = 2,
    DIV = 3,
    POW = 4,
    SIN = 5,
    COS = 6,
    TAN = 7,
    ASIN = 8,
    ACOS = 9,
    EXP = 10,
    LOG = 11,
    SIGN = 12,
    GRAD = 13, // for backward compatibility, unused in Rust's ParameterExpression
    CONJ = 14,
    SUBSTITUTE = 15, // for backward compatibility, unused in Rust's ParameterExpression
    ABS = 16,
    ATAN = 17,
    RSUB = 18,
    RDIV = 19,
    RPOW = 20,
}

impl From<OpCode> for u8 {
    fn from(value: OpCode) -> Self {
        value as u8
    }
}

unsafe impl ::bytemuck::CheckedBitPattern for OpCode {
    type Bits = u8;

    #[inline(always)]
    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits <= 20
    }
}

unsafe impl ::bytemuck::NoUninit for OpCode {}

#[pymethods]
impl OpCode {
    #[new]
    fn py_new(value: u8) -> PyResult<Self> {
        let code: OpCode = ::bytemuck::checked::try_cast(value)
            .map_err(|_| ParameterError::InvalidU8ToOpCode(value))?;
        Ok(code)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        if let Ok(code) = other.downcast::<OpCode>() {
            *code.borrow() == *self
        } else {
            false
        }
    }

    fn __hash__(&self) -> u8 {
        *self as u8
    }

    fn __getnewargs__(&self) -> (u8,) {
        (*self as u8,)
    }
}

// enum for QPY replay
#[pyclass(sequence, module = "qiskit._accelerate.circuit")]
#[derive(Clone, Debug)]
pub struct OPReplay {
    pub op: OpCode,
    pub lhs: Option<ParameterValueType>,
    pub rhs: Option<ParameterValueType>,
}

#[pymethods]
impl OPReplay {
    #[new]
    pub fn py_new(
        op: OpCode,
        lhs: Option<ParameterValueType>,
        rhs: Option<ParameterValueType>,
    ) -> OPReplay {
        OPReplay { op, lhs, rhs }
    }

    #[getter]
    fn op(&self) -> OpCode {
        self.op
    }

    #[getter]
    fn lhs(&self) -> Option<ParameterValueType> {
        self.lhs.clone()
    }

    #[getter]
    fn rhs(&self) -> Option<ParameterValueType> {
        self.rhs.clone()
    }

    fn __getnewargs__(
        &self,
    ) -> (
        OpCode,
        Option<ParameterValueType>,
        Option<ParameterValueType>,
    ) {
        (self.op, self.lhs.clone(), self.rhs.clone())
    }
}

/// Internal helper. Extract one part of the expression tree, keeping the name map up to date.
///
/// Example: Given expr1 + expr2, each being [PyParameterExpression], we need the ability to
/// extract one of the expressions with the proper name map.
///
/// Args:
///     - joint_parameter_expr: The full expression, e.g. expr1 + expr2.
///     - sub_expr: The sub expression, on whose symbols we restrict the name map.
fn filter_name_map(
    sub_expr: &Box<SymbolExpr>,
    name_map: &HashMap<String, PyParameter>,
) -> PyParameterExpression {
    let sub_symbols = sub_expr.parameters();
    let restricted_name_map: HashMap<String, PyParameter> = name_map
        .iter()
        .filter(|(_, param)| sub_symbols.contains(&param.symbol))
        .map(|(name, param)| (name.clone(), param.clone()))
        .collect();

    PyParameterExpression {
        expr: *sub_expr.clone(),
        name_map: restricted_name_map,
    }
}

pub fn qpy_replay(
    expr: &PyParameterExpression,
    name_map: &HashMap<String, PyParameter>,
    replay: &mut Vec<OPReplay>,
) {
    match &expr.expr {
        SymbolExpr::Value(_) | SymbolExpr::Symbol(_) => {
            // nothing to do here, we only need to traverse instructions
        }
        SymbolExpr::Unary { op, expr } => {
            let op = match op {
                symbol_expr::UnaryOp::Abs => OpCode::ABS,
                symbol_expr::UnaryOp::Acos => OpCode::ACOS,
                symbol_expr::UnaryOp::Asin => OpCode::ASIN,
                symbol_expr::UnaryOp::Atan => OpCode::ATAN,
                symbol_expr::UnaryOp::Conj => OpCode::CONJ,
                symbol_expr::UnaryOp::Cos => OpCode::COS,
                symbol_expr::UnaryOp::Exp => OpCode::EXP,
                symbol_expr::UnaryOp::Log => OpCode::LOG,
                symbol_expr::UnaryOp::Neg => OpCode::MUL,
                symbol_expr::UnaryOp::Sign => OpCode::SIGN,
                symbol_expr::UnaryOp::Sin => OpCode::SIN,
                symbol_expr::UnaryOp::Tan => OpCode::TAN,
            };
            // TODO filter shouldn't be necessary for unary ops
            // let lhs = ParameterValueType::extract_from_expr(expr, name_map);
            // recurse on the instruction
            let lhs = filter_name_map(expr, name_map);
            qpy_replay(&lhs, name_map, replay);

            let lhs_value = ParameterValueType::extract_from_expr(expr);

            // MUL is special: we implement ``neg`` as multiplication by -1
            if let OpCode::MUL = &op {
                replay.push(OPReplay {
                    op,
                    lhs: lhs_value,
                    rhs: Some(ParameterValueType::Int(-1)),
                });
            } else {
                replay.push(OPReplay {
                    op,
                    lhs: lhs_value,
                    rhs: None,
                });
            }
        }
        SymbolExpr::Binary { op, lhs, rhs } => {
            let lhs_value = ParameterValueType::extract_from_expr(lhs);
            let rhs_value = ParameterValueType::extract_from_expr(rhs);

            // recurse on the parameter expressions
            let lhs = filter_name_map(lhs, name_map);
            let rhs = filter_name_map(rhs, name_map);
            qpy_replay(&lhs, name_map, replay);
            qpy_replay(&rhs, name_map, replay);

            // add the expression to the replay
            match lhs_value {
                None | Some(ParameterValueType::Parameter(_)) => {
                    let op = match op {
                        symbol_expr::BinaryOp::Add => OpCode::ADD,
                        symbol_expr::BinaryOp::Sub => OpCode::SUB,
                        symbol_expr::BinaryOp::Mul => OpCode::MUL,
                        symbol_expr::BinaryOp::Div => OpCode::DIV,
                        symbol_expr::BinaryOp::Pow => OpCode::POW,
                    };
                    replay.push(OPReplay {
                        op,
                        lhs: lhs_value,
                        rhs: rhs_value,
                    });
                }
                _ => {
                    let op = match op {
                        symbol_expr::BinaryOp::Add => OpCode::ADD,
                        symbol_expr::BinaryOp::Sub => OpCode::RSUB,
                        symbol_expr::BinaryOp::Mul => OpCode::MUL,
                        symbol_expr::BinaryOp::Div => OpCode::RDIV,
                        symbol_expr::BinaryOp::Pow => OpCode::RPOW,
                    };
                    if let OpCode::ADD | OpCode::MUL = op {
                        replay.push(OPReplay {
                            op,
                            lhs: lhs_value,
                            rhs: rhs_value,
                        });
                    } else {
                        // this covers RSUB, RDIV, RPOW, hence we swap lhs and rhs
                        replay.push(OPReplay {
                            op,
                            rhs: rhs_value,
                            lhs: lhs_value,
                        });
                    }
                }
            }
        }
    }
}
