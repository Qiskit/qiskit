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
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError, PyZeroDivisionError};
use pyo3::types::{IntoPyDict, PyNotImplemented, PySet, PyString};
use thiserror::Error;
use uuid::Uuid;

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;

use crate::circuit_data::CircuitError;
use crate::imports::{BUILTIN_HASH, SYMPIFY_PARAMETER_EXPRESSION, UUID};
use crate::parameter::symbol_expr;
use crate::parameter::symbol_expr::SymbolExpr;
use crate::parameter::symbol_parser::parse_expression;

use super::symbol_expr::{Symbol, Value, SYMEXPR_EPSILON};

/// Errors for dealing with parameters and parameter expressions.
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

/// A parameter expression.
///
/// This is backed by Qiskit's symbolic expression engine and a cache
/// for the parameters inside the expression.
#[pyclass(subclass, sequence, module = "qiskit._accelerate.circuit")]
#[derive(Clone, Debug)]
pub struct ParameterExpression {
    // The symbolic expression.
    pub expr: SymbolExpr,
    // A map keeping track of all symbols, with their name. This map *must* be kept
    // up to date upon any operation performed on the expression.
    // TODO it would be nicer to just store [Symbol]s as values, which can maybe be done
    // with a secondary PyParameterExpression storing the [PyParameter] in a map.
    pub name_map: HashMap<String, PyParameter>,
}

impl Hash for ParameterExpression {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.expr.to_string().hash(state);
    }
}

impl PartialEq for ParameterExpression {
    fn eq(&self, other: &Self) -> bool {
        self.expr.eq(&other.expr)
    }
}

impl Eq for ParameterExpression {}

impl Default for ParameterExpression {
    /// The default constructor returns zero.
    fn default() -> Self {
        Self {
            expr: SymbolExpr::Value(Value::Int(0)),
            name_map: HashMap::new(), // no parameters, hence empty name map
        }
    }
}

impl fmt::Display for ParameterExpression {
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

/// Attempt to extract a [PyParameterExpression] from a bound [PyAny].
///
/// If no valid value can be extracted, none is returned (e.g. for inf or nan values).
#[inline]
fn _extract_value(value: &Bound<PyAny>) -> Option<ParameterExpression> {
    if let Ok(i) = value.extract::<i64>() {
        Some(ParameterExpression::new(SymbolExpr::Value(Value::from(i))))
    } else if let Ok(c) = value.extract::<Complex64>() {
        if c.is_infinite() || c.is_nan() {
            return None;
        }
        Some(ParameterExpression::new(SymbolExpr::Value(Value::from(c))))
    } else if let Ok(r) = value.extract::<f64>() {
        if r.is_infinite() || r.is_nan() {
            return None;
        }
        Some(ParameterExpression::new(SymbolExpr::Value(Value::from(r))))
    } else if let Ok(parameter) = value.extract::<PyParameter>() {
        Some(parameter.symbol.as_expr())
    } else if let Ok(element) = value.extract::<PyParameterVectorElement>() {
        Some(element.symbol.as_expr())
    } else {
        value.extract::<ParameterExpression>().ok()
    }
}

/// Lookup for which operations are binary (i.e. require two operands).
static BINARY_OPS: [OpCode; 8] = [
    // a HashSet would be better but requires unstable features
    OpCode::ADD,
    OpCode::SUB,
    OpCode::MUL,
    OpCode::DIV,
    OpCode::POW,
    OpCode::RSUB,
    OpCode::RDIV,
    OpCode::RPOW,
];

impl ParameterExpression {
    /// Construct from a [SymbolExpr].
    ///
    /// This populates the name map with the symbols in the expression.
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

    /// Load from a sequence of [OPReplay]s. Used in serialization.
    pub fn from_qpy(replay: &[OPReplay]) -> Result<Self, ParameterError> {
        // the stack contains the latest lhs and rhs values
        let mut stack: Vec<ParameterExpression> = Vec::new();

        for inst in replay.iter() {
            let OPReplay { op, lhs, rhs } = inst;

            // put the values on the stack, if they exist
            if let Some(value) = lhs {
                stack.push(value.clone().into());
            }
            if let Some(value) = rhs {
                stack.push(value.clone().into());
            }

            // if we need two operands, pop rhs from the stack
            let rhs = if BINARY_OPS.contains(op) {
                Some(stack.pop().expect("Pop from empty stack"))
            } else {
                None
            };

            // pop lhs from the stack, this we always need
            let lhs = stack.pop().expect("Pop from empty stack");

            // apply the operation and put the result onto the stack for the next replay
            let result: ParameterExpression = match op {
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

        // once we're done, just return the last element in the stack
        Ok(stack
            .pop()
            .expect("OPReplay was empty, no expression could be built"))
    }

    /// Add an expression; ``self + rhs``.
    pub fn add(&self, rhs: &ParameterExpression) -> Result<Self, ParameterError> {
        let name_map = self.update_name_map(rhs)?;
        Ok(Self {
            expr: &self.expr + &rhs.expr,
            name_map,
        })
    }

    /// Multiply with an expression; ``self * rhs``.
    pub fn mul(&self, rhs: &ParameterExpression) -> Result<Self, ParameterError> {
        let name_map = self.update_name_map(rhs)?;
        Ok(Self {
            expr: &self.expr * &rhs.expr,
            name_map,
        })
    }

    /// Subtract another expression; ``self - rhs``.
    pub fn sub(&self, rhs: &ParameterExpression) -> Result<Self, ParameterError> {
        let name_map = self.update_name_map(rhs)?;
        Ok(Self {
            expr: &self.expr - &rhs.expr,
            name_map,
        })
    }

    /// Subtract this expression from another one; ``lhs - self``.
    pub fn rsub(&self, lhs: &ParameterExpression) -> Result<Self, ParameterError> {
        let name_map = self.update_name_map(lhs)?;
        Ok(Self {
            expr: &lhs.expr - &self.expr,
            name_map,
        })
    }

    /// Divide by another expression; ``self / rhs``.
    pub fn div(&self, rhs: &ParameterExpression) -> Result<Self, ParameterError> {
        if rhs.expr.is_zero() {
            return Err(ParameterError::ZeroDivisionError);
        }

        let name_map = self.update_name_map(rhs)?;
        Ok(Self {
            expr: &self.expr / &rhs.expr,
            name_map,
        })
    }

    /// Divide another expression by this one; ``lhs / self``.
    pub fn rdiv(&self, lhs: &ParameterExpression) -> Result<Self, ParameterError> {
        if self.expr.is_zero() {
            return Err(ParameterError::ZeroDivisionError);
        }

        let name_map = self.update_name_map(lhs)?;
        Ok(Self {
            expr: &lhs.expr / &self.expr,
            name_map,
        })
    }

    /// Raise this expression to a power; ``self ^ rhs``.
    pub fn pow(&self, rhs: &ParameterExpression) -> Result<Self, ParameterError> {
        let name_map = self.update_name_map(rhs)?;
        Ok(Self {
            expr: self.expr.pow(&rhs.expr),
            name_map,
        })
    }

    /// Raise an expression to the power of this expression; ``lhs ^ self``.
    pub fn rpow(&self, lhs: &ParameterExpression) -> Result<Self, ParameterError> {
        let name_map = self.update_name_map(lhs)?;
        Ok(Self {
            expr: lhs.expr.pow(&self.expr),
            name_map,
        })
    }

    /// Apply the sine to this expression; ``sin(self)``.
    pub fn sin(&self) -> Self {
        Self {
            expr: self.expr.sin(),
            name_map: self.name_map.clone(),
        }
    }

    /// Apply the cosine to this expression; ``cos(self)``.
    pub fn cos(&self) -> Self {
        Self {
            expr: self.expr.cos(),
            name_map: self.name_map.clone(),
        }
    }

    /// Apply the tangent to this expression; ``tan(self)``.
    pub fn tan(&self) -> Self {
        Self {
            expr: self.expr.tan(),
            name_map: self.name_map.clone(),
        }
    }

    /// Apply the arcsine to this expression; ``asin(self)``.
    pub fn asin(&self) -> Self {
        Self {
            expr: self.expr.asin(),
            name_map: self.name_map.clone(),
        }
    }

    /// Apply the arccosine to this expression; ``acos(self)``.
    pub fn acos(&self) -> Self {
        Self {
            expr: self.expr.acos(),
            name_map: self.name_map.clone(),
        }
    }

    /// Apply the arctangent to this expression; ``atan(self)``.
    pub fn atan(&self) -> Self {
        Self {
            expr: self.expr.atan(),
            name_map: self.name_map.clone(),
        }
    }

    /// Exponentiate this expression; ``exp(self)``.
    pub fn exp(&self) -> Self {
        Self {
            expr: self.expr.exp(),
            name_map: self.name_map.clone(),
        }
    }

    /// Take the (natural) logarithm of this expression; ``log(self)``.
    pub fn log(&self) -> Self {
        Self {
            expr: self.expr.log(),
            name_map: self.name_map.clone(),
        }
    }

    /// Take the absolute value of this expression; ``|self|``.
    pub fn abs(&self) -> Self {
        Self {
            expr: self.expr.abs(),
            name_map: self.name_map.clone(),
        }
    }

    /// Return the sign of this expression; ``sign(self)``.
    pub fn sign(&self) -> Self {
        Self {
            expr: self.expr.sign(),
            name_map: self.name_map.clone(),
        }
    }

    /// Clone the expression.
    pub fn copy(&self) -> Self {
        // TODO: Is this necessary? Doesn't ParameterExpression.clone() implement the same thing?
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
    ///
    /// Note that this keeps the name map unchanged. Meaning that computing the derivative
    /// of ``x`` will yield ``1`` but the expression still owns the symbol ``x``. This is
    /// done such that we can still bind the value ``x`` in an automated process.
    pub fn derivative(&self, param: &Symbol) -> Result<Self, String> {
        Ok(Self {
            expr: self.expr.derivative(param)?,
            name_map: self.name_map.clone(),
        })
    }

    /// Expand the expression.
    pub fn expand(&self) -> Self {
        Self {
            expr: self.expr.expand(),
            name_map: self.name_map.clone(),
        }
    }

    /// Substitute symbols with [PyParameterExpression]s.
    ///
    /// Args:
    ///     - map: A hashmap with [Symbol] keys and [PyParameterExpression]s to replace these
    ///         symbols with.
    ///     - allow_unknown_parameters: If ``false``, returns an error if any symbol in the
    ///         hashmap is not present in the expression. If ``true``, unknown symbols are ignored.
    ///         Setting to ``true`` is slightly faster as it does not involve additional checks.
    ///
    /// Returns:
    ///     A parameter expression with the substituted symbols.
    pub fn subs(
        &self,
        map: &HashMap<Symbol, Self>,
        allow_unknown_parameters: bool,
    ) -> Result<Self, ParameterError> {
        // Build the outgoing name map. In the process we check for any duplicates.
        let mut name_map: HashMap<String, PyParameter> = HashMap::new();
        let mut symbol_map: HashMap<Symbol, SymbolExpr> = HashMap::new();

        // If we don't allow for unknown parameters, check if there are any.
        if !allow_unknown_parameters {
            let existing: HashSet<&Symbol> = self.name_map.values().map(|p| &p.symbol).collect();
            let to_replace: HashSet<&Symbol> = map.keys().collect();

            // This could be done a little bit more efficiently, but we want to
            // show the user all symbols that are not present.
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
                        // If a symbol with the same name already exists, check whether it is
                        // the same symbol (fine) or a different symbol with the same name (conflict)!
                        if duplicate != replacement_symbol {
                            return Err(ParameterError::NameConflict);
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
        Ok(Self {
            expr: res,
            name_map,
        })
    }

    /// Bind symbols to values.
    ///
    /// Args:
    ///     - map: A hashmap with [Symbol] keys and [Value]s to replace these
    ///         symbols with.
    ///     - allow_unknown_parameters: If ``false``, returns an error if any symbol in the
    ///         hashmap is not present in the expression. If ``true``, unknown symbols are ignored.
    ///         Setting to ``true`` is slightly faster as it does not involve additional checks.
    ///
    /// Returns:
    ///     A parameter expression with the bound symbols.
    pub fn bind(
        &self,
        map: &HashMap<Symbol, Value>,
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
                Value::Real(r) => {
                    if r.is_infinite() {
                        Err(ParameterError::BindingInf)
                    } else if r.is_nan() {
                        Err(ParameterError::BindingNaN)
                    } else {
                        Ok(SymbolExpr::Value(v))
                    }
                }
                Value::Int(_) => Ok(SymbolExpr::Value(v)),
                Value::Complex(c) => {
                    if c.re.is_infinite() || c.im.is_infinite() {
                        Err(ParameterError::ZeroDivisionError) // TODO this should probs be BindingInf
                    } else if c.re.is_nan() || c.im.is_nan() {
                        Err(ParameterError::BindingNaN)
                    } else if (-symbol_expr::SYMEXPR_EPSILON..symbol_expr::SYMEXPR_EPSILON)
                        .contains(&c.im)
                    {
                        Ok(SymbolExpr::Value(Value::Real(c.re)))
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
impl ParameterExpression {
    /// This is a **strictly internal** constructor and **should not be used**.
    /// It is subject to arbitrary change in between Qiskit versions and cannot be relied on.
    /// Parameter expressions should always be constructed from applying operations on
    /// parameters, or by loading via QPY.
    ///
    /// The input values are allowed to be None for pickling purposes.
    #[new]
    #[pyo3(signature = (name_map=None, expr=None))]
    pub fn py_new(
        name_map: Option<HashMap<String, PyParameter>>,
        expr: Option<String>,
    ) -> PyResult<Self> {
        match (name_map, expr) {
            (None, None) => Ok(Self::default()),
            (Some(name_map), Some(expr)) => {
                // We first parse the expression and then update the symbols with the ones
                // the user provided. The replacement relies on the names to match.
                // This is hacky and we likely want a more reliably conversion from a SymPy object,
                // if we decide we want to continue supporting this.
                let expr = parse_expression(&expr)
                    .map_err(|_| PyRuntimeError::new_err("Failed parsing input expression"))?;
                let symbol_map: HashMap<String, Symbol> = name_map
                    .iter()
                    .map(|(string, param)| (string.clone(), param.symbol.clone()))
                    .collect();

                let replaced_expr = replace_symbol(&expr, &symbol_map);
                Ok(ParameterExpression {
                    expr: replaced_expr,
                    name_map,
                })
            }
            _ => Err(PyValueError::new_err(
                "Pass either both a name_map and expr, or neither",
            )),
        }
    }

    /// TODO Check whether this is required, this was not part of the parameters API before.
    /// Marked as private for now.
    #[allow(non_snake_case)]
    #[staticmethod]
    pub fn _Value(value: &Bound<PyAny>) -> PyResult<Self> {
        match _extract_value(value) {
            Some(v) => Ok(v),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Unsupported data type to initialize SymbolExpr as a value",
            )),
        }
    }

    /// Raise an error if names in the other expression collide with existing names.
    ///
    /// Args:
    ///     other: The other :class:`.ParameterExpression` to check name conflicts with.
    ///
    /// Raises:
    ///     CircuitError: If there is a name conflict.
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
    ///
    /// Returns:
    ///     ``True`` is this expression corresponds to a symbol, ``False`` otherwise.
    pub fn is_symbol(&self) -> bool {
        matches!(&self.expr, SymbolExpr::Symbol(_))
    }

    /// Cast this expression to a numeric value.
    ///
    /// Args:
    ///     strict: If ``True`` (default) this function raises an error if there are any
    ///         unbound symbols in the expression. If ``False``, this allows casting
    ///         if the expression represents a numeric value, regardless of unbound symbols.
    ///         For example ``(0 * Parameter("x"))`` is 0 but has the symbol ``x`` present.
    #[pyo3(signature = (strict=true))]
    pub fn numeric(&self, py: Python, strict: bool) -> PyResult<PyObject> {
        // Check if we have unbound symbols. Then we'll always say we are non-numeric,
        // even if the expression is 0. (Example: (0 * x).numeric() fails.)
        if strict && !self.name_map.is_empty() {
            let free_symbols = self.name_map.values();
            return Err(PyTypeError::new_err(format!(
                "Parameter expression with unbound parameters {free_symbols:?} is not numeric."
            )));
        }

        match self.expr.eval(true) {
            Some(v) => match v {
                Value::Real(r) => r.into_py_any(py),
                Value::Int(i) => i.into_py_any(py),
                Value::Complex(c) => {
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

    /// Return a SymPy equivalent of this expression.
    ///
    /// Returns:
    ///     A SymPy equivalent of this expression.
    pub fn sympify(&self, py: Python) -> PyResult<PyObject> {
        let py_sympify = SYMPIFY_PARAMETER_EXPRESSION.get(py);
        py_sympify.call1(py, (self.clone(),))
    }

    /// Get the parameters present in the expression.
    ///
    /// .. note::
    ///     
    ///     Qiskit guarantees equality (via ``==``) of parameters retrieved from an expression
    ///     with the original :class:`.Parameter` objects used to create this expression,
    ///     but does _not guarantee_ ``is`` comparisons to succeed.
    ///
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

    /// Exponentiate the expression.
    #[pyo3(name = "exp")]
    pub fn py_exp(&self) -> Self {
        self.exp()
    }

    /// Take the natural logarithm of the expression.
    #[pyo3(name = "log")]
    pub fn py_log(&self) -> Self {
        self.log()
    }

    /// Take the absolute value of the expression.
    #[pyo3(name = "abs")]
    pub fn py_abs(&self) -> Self {
        self.abs()
    }

    /// Return the sign of the expression.
    #[pyo3(name = "sign")]
    pub fn py_sign(&self) -> Self {
        self.sign()
    }

    /// Return the complex conjugate of the expression.
    #[pyo3(name = "conjugate")]
    pub fn py_conjugate(&self) -> Self {
        self.conjugate()
    }

    /// Check whether the expression represents a real number.
    ///
    /// Note that this will return ``None`` if there are unbound parameters, in which case
    /// it cannot be determined whether the expression is real.
    #[pyo3(name = "is_real")]
    pub fn py_is_real(&self) -> Option<bool> {
        self.expr.is_real()
    }

    /// Return derivative of this expression with respect to the input parameter.
    ///
    /// Args:
    ///     param: The parameter with respect to which the derivative is calculated.
    ///
    /// Returns:
    ///     The derivative.
    pub fn gradient(&self, param: &Bound<'_, PyAny>) -> PyResult<Self> {
        let symbol = symbol_from_py_parameter(param)?;
        self.derivative(&symbol).map_err(PyRuntimeError::new_err)
    }

    /// Return all values in this equation.
    pub fn _values(&self, py: Python) -> PyResult<Vec<PyObject>> {
        self.expr
            .values()
            .iter()
            .map(|val| match val {
                Value::Real(r) => r.into_py_any(py),
                Value::Int(i) => i.into_py_any(py),
                Value::Complex(c) => c.into_py_any(py),
            })
            .collect()
    }

    /// Returns a new expression with replacement parameters.
    ///
    /// Args:
    ///     parameter_map: Mapping from :class:`.Parameter`\ s in ``self`` to the
    ///         :class:`.ParameterExpression` instances with which they should be replaced.
    ///     allow_unknown_parameters: If ``False``, raises an error if ``parameter_map``
    ///         contains :class:`.Parameter`\ s in the keys outside those present in the expression.
    ///         If ``True``, any such parameters are simply ignored.
    ///
    /// Raises:
    ///     CircuitError:
    ///         - If parameter_map contains parameters outside those in self.
    ///         - If the replacement parameters in ``parameter_map`` would result in
    ///           a name conflict in the generated expression.
    ///
    /// Returns:
    ///     A new expression with the specified parameters replaced.
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

    /// Binds the provided set of parameters to their corresponding values.
    ///
    /// Args:
    ///     parameter_values: Mapping of :class:`.Parameter` instances to the numeric value to which
    ///         they will be bound.
    ///     allow_unknown_parameters: If ``False``, raises an error if ``parameter_values``
    ///         contains :class:`.Parameter`\ s in the keys outside those present in the expression.
    ///         If ``True``, any such parameters are simply ignored.
    ///
    /// Raises:
    ///     CircuitError:
    ///         - If parameter_values contains parameters outside those in self.
    ///         - If a non-numeric value is passed in ``parameter_values``.
    ///     ZeroDivisionError:
    ///         - If binding the provided values requires division by zero.
    ///
    /// Returns:
    ///     A new expression parameterized by any parameters which were not bound by
    ///     ``parameter_values``.
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

    /// Assign one parameter to a value, which can either be numeric or another parameter
    /// expression.
    ///
    /// Args:
    ///     parameter: A parameter in this expression whose value will be updated.
    ///     value: The new value to bind to.
    ///
    /// Returns:
    ///     A new expression parameterized by any parameters which were not bound by assignment.
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

    fn __copy__(slf: PyRef<Self>) -> PyRef<Self> {
        // ParameterExpression is immutable.
        slf
    }

    fn __deepcopy__<'py>(slf: PyRef<'py, Self>, _memo: Bound<'py, PyAny>) -> PyRef<'py, Self> {
        // Everything a ParameterExpression contains is immutable.
        slf
    }

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
                Value::Complex(_) => Err(PyTypeError::new_err(
                    "Cannot cast complex parameter to float.",
                )),
                Value::Real(r) => {
                    let rounded = r.floor();
                    Ok(rounded as i64)
                }
                Value::Int(i) => Ok(i),
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
                Value::Complex(c) => {
                    if c.im.abs() > SYMEXPR_EPSILON {
                        Err(PyTypeError::new_err(
                            "Could not cast complex parameter expression to float.",
                        ))
                    } else {
                        Ok(c.re)
                    }
                }
                Value::Real(r) => Ok(r),
                Value::Int(i) => Ok(i as f64),
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
                Value::Complex(c) => Ok(c),
                Value::Real(r) => Ok(Complex64::new(r, 0.)),
                Value::Int(i) => Ok(Complex64::new(i as f64, 0.)),
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
                    Value::Complex(c) => py_hash.call1((c,))?.extract::<u64>(),
                    Value::Real(r) => py_hash.call1((r,))?.extract::<u64>(),
                    Value::Int(i) => py_hash.call1((i,))?.extract::<u64>(),
                }
            }
            None => {
                let mut hasher = DefaultHasher::new();
                self.expr.string_id().hash(&mut hasher);
                Ok(hasher.finish())
            }
        }
    }

    fn __getstate__(&self) -> (Vec<OPReplay>, HashMap<String, PyParameter>) {
        // To pickle the object we use the QPY replay and rebuild from that.
        (self._qpy_replay(), self.name_map.clone())
    }

    fn __setstate__(
        &mut self,
        state: (Vec<OPReplay>, HashMap<String, PyParameter>),
    ) -> PyResult<()> {
        self.name_map = state.1;
        let from_qpy = Self::from_qpy(&state.0)?;
        self.expr = from_qpy.expr;
        Ok(())
    }

    #[getter]
    fn _qpy_replay(&self) -> Vec<OPReplay> {
        let mut replay = Vec::new();
        qpy_replay(self, &self.name_map, &mut replay);
        replay
    }
}

/// A compile-time symbolic parameter.
///
/// The value of a :class:`.Parameter` must be entirely determined before a circuit begins execution.
/// Typically this will mean that you should supply values for all :class:`.Parameter`\ s in a
/// circuit using :meth:`.QuantumCircuit.assign_parameters`, though certain hardware vendors may
/// allow you to give them a circuit in terms of these parameters, provided you also pass the values
/// separately.
///
/// This is the atom of :class:`.ParameterExpression`, and is itself an expression.  The numeric
/// value of a parameter need not be fixed while the circuit is being defined.
///
/// Examples:
///
///     Construct a variable-rotation X gate using circuit parameters.
///
///     .. plot::
///         :alt: Circuit diagram output by the previous code.
///         :include-source:
///
///         from qiskit.circuit import QuantumCircuit, Parameter
///
///         # create the parameter
///         phi = Parameter("phi")
///         qc = QuantumCircuit(1)
///
///         # parameterize the rotation
///         qc.rx(phi, 0)
///         qc.draw("mpl")
///
///         # bind the parameters after circuit to create a bound circuit
///         bc = qc.assign_parameters({phi: 3.14})
///         bc.measure_all()
///         bc.draw("mpl")
#[pyclass(sequence, subclass, module="qiskit._accelerate.circuit", extends=ParameterExpression, name="Parameter")]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd)]
pub struct PyParameter {
    pub symbol: Symbol,
}

impl Hash for PyParameter {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.symbol.hash(state);
    }
}

// This needs to be implemented manually, since PyO3 does not provide this conversion
// for subclasses.
impl<'py> IntoPyObject<'py> for PyParameter {
    type Target = PyParameter;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let symbol = &self.symbol;
        let expr = SymbolExpr::Symbol(symbol.clone());
        let py_expr = ParameterExpression::new(expr);

        Ok(Py::new(py, (self, py_expr))?.into_bound(py))
    }
}

impl PyParameter {
    /// Get a Python class initialization from a symbol.
    fn from_symbol(symbol: &Symbol) -> PyClassInitializer<Self> {
        let expr = SymbolExpr::Symbol(symbol.clone());

        let py_parameter = Self {
            symbol: symbol.clone(),
        };
        let py_expr = ParameterExpression::new(expr);

        PyClassInitializer::from(py_expr).add_subclass(py_parameter)
    }
}

#[pymethods]
impl PyParameter {
    /// Args:
    ///     name: name of the parameter, used for visual representation. This can
    ///         be any Unicode string, e.g. ``"ϕ"``.
    ///     uuid: For advanced usage only.  Override the UUID of this parameter, in order to make it
    ///         compare equal to some other parameter object.  By default, two parameters with the
    ///         same name do not compare equal to help catch shadowing bugs when two circuits
    ///         containing the same named parameters are spurious combined.  Setting the ``uuid``
    ///         field when creating two parameters to the same thing (along with the same name)
    ///         allows them to be equal.  This is useful during serialization and deserialization.
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
        let py_expr = ParameterExpression::new(expr);

        Ok(PyClassInitializer::from(py_expr).add_subclass(py_parameter))
    }

    /// Returns the name of the :class:`.Parameter`.
    #[getter]
    fn name<'py>(&self, py: Python<'py>) -> Bound<'py, PyString> {
        self.symbol.py_name(py)
    }

    /// Returns the :class:`~uuid.UUID` of the :class:`Parameter`.
    ///
    /// In advanced use cases, this property can be passed to the
    /// :class:`.Parameter` constructor to produce an instance that compares
    /// equal to another instance.
    #[getter]
    fn uuid(&self, py: Python<'_>) -> PyResult<PyObject> {
        uuid_to_py(py, self.symbol.uuid)
    }

    pub fn __repr__<'py>(&self, py: Python<'py>) -> Bound<'py, PyString> {
        let str = format!("Parameter({})", self.symbol.name(),);
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
    }

    pub fn sympify(&self, py: Python) -> PyResult<PyObject> {
        // TODO can this be removed as it's the same as the parent class one?
        let py_sympify = SYMPIFY_PARAMETER_EXPRESSION.get(py);
        py_sympify.call1(py, (self.clone(),))
    }

    fn __copy__(slf: PyRef<Self>) -> PyRef<Self> {
        // Parameter is immutable. Note that this **cannot** be deferred to the parent class
        // since PyO3 would then always return the parent type.
        slf
    }

    fn __deepcopy__<'py>(slf: PyRef<'py, Self>, _memo: Bound<'py, PyAny>) -> PyRef<'py, Self> {
        // Everything inside a Parameter is immutable. Note that this **cannot** be deferred to the
        // parent class since PyO3 would then always return the parent type.
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
        // We implement this method on this class, and do not defer to the parent, such
        // that x.subs({x: y}) remains a Parameter, and is not upgraded to an expression.
        // Also this should be faster than going via ParameterExpression, which constructs
        // intermediary HashMaps we don't need here.
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
        // Returns PyAny to cover Parameter and ParameterExpression(value).
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
        if value.downcast::<ParameterExpression>().is_ok() {
            let map = [(parameter, value.clone())].into_iter().collect();
            self.py_subs(py, map, false)
        } else if value.extract::<Value>().is_ok() {
            let map = [(parameter, value.clone())].into_iter().collect();
            self.py_bind(py, map, false)
        } else {
            Err(PyValueError::new_err(
                "Unexpected value in assign: {replacement:?}",
            ))
        }
    }
}

/// An element of a :class:`.ParameterVector`.
///
/// .. note::
///     There is very little reason to ever construct this class directly.  Objects of this type are
///     automatically constructed efficiently as part of creating a :class:`.ParameterVector`.
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

    pub fn __repr__<'py>(&self, py: Python<'py>) -> Bound<'py, PyString> {
        let str = format!("ParameterVectorElement({})", self.symbol.name(),);
        PyString::new(py, str.as_str())
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

    /// Get the index of this element in the parent vector.
    #[getter]
    pub fn index(&self) -> u32 {
        self.symbol
            .index
            .expect("A vector element should have an index")
    }

    /// Get the parent vector instance.
    #[getter]
    pub fn vector(&self) -> PyObject {
        self.symbol
            .clone()
            .vector
            .expect("A vector element should have a vector")
    }

    /// For backward compatibility only. This should not be used and we ought to update those
    /// usages!
    #[getter]
    pub fn _vector(&self) -> PyObject {
        self.vector()
    }

    fn __copy__(slf: PyRef<Self>) -> PyRef<Self> {
        // ParameterVectorElement is immutable.
        slf
    }

    fn __deepcopy__<'py>(slf: PyRef<'py, Self>, _memo: Bound<'py, PyAny>) -> PyRef<'py, Self> {
        // Everything a ParameterVectorElement contains is immutable.
        slf
    }
}

/// Try to extract a Uuid from a Python object, which could be a Python UUID or int.
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

/// Convert a Rust Uuid object to a Python UUID object.
fn uuid_to_py(py: Python<'_>, uuid: Uuid) -> PyResult<PyObject> {
    let uuid = uuid.as_u128();
    let kwargs = [("int", uuid)].into_py_dict(py)?;
    Ok(UUID.get_bound(py).call((), Some(&kwargs))?.unbind())
}

/// Extract a [Symbol] for a Python object, which could either be a Parameter or a
/// ParameterVectorElement.
fn symbol_from_py_parameter(param: &Bound<'_, PyAny>) -> PyResult<Symbol> {
    if let Ok(element) = param.extract::<PyParameterVectorElement>() {
        Ok(element.symbol.clone())
    } else if let Ok(parameter) = param.extract::<PyParameter>() {
        Ok(parameter.symbol.clone())
    } else {
        Err(PyValueError::new_err("Could not extract parameter"))
    }
}

/// A singular parameter value used for QPY serialization. This covers anything
/// but a [PyParameterExpression], which is represented by [None] in the serialization.
#[derive(IntoPyObject, FromPyObject, Clone, Debug)]
pub enum ParameterValueType {
    Int(i64),
    Float(f64),
    Complex(Complex64),
    Parameter(PyParameter),
    VectorElement(PyParameterVectorElement),
}

impl ParameterValueType {
    fn extract_from_expr(expr: &SymbolExpr) -> Option<ParameterValueType> {
        if let Some(value) = expr.eval(true) {
            match value {
                Value::Int(i) => Some(ParameterValueType::Int(i)),
                Value::Real(r) => Some(ParameterValueType::Float(r)),
                Value::Complex(c) => Some(ParameterValueType::Complex(c)),
            }
        } else if let SymbolExpr::Symbol(symbol) = expr {
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

impl From<ParameterValueType> for ParameterExpression {
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
    sub_expr: &SymbolExpr,
    name_map: &HashMap<String, PyParameter>,
) -> ParameterExpression {
    let sub_symbols = sub_expr.parameters();
    let restricted_name_map: HashMap<String, PyParameter> = name_map
        .iter()
        .filter(|(_, param)| sub_symbols.contains(&param.symbol))
        .map(|(name, param)| (name.clone(), param.clone()))
        .collect();

    ParameterExpression {
        expr: sub_expr.clone(),
        name_map: restricted_name_map,
    }
}

pub fn qpy_replay(
    expr: &ParameterExpression,
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
            let lhs = filter_name_map(expr, name_map);

            // recurse on the instruction
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
                None
                | Some(ParameterValueType::Parameter(_))
                | Some(ParameterValueType::VectorElement(_)) => {
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
                            lhs: rhs_value,
                            rhs: lhs_value,
                        });
                    }
                }
            }
        }
    }
}

/// Replace [Symbol]s in a [SymbolExpr] according to the name map. This
/// is used to reconstruct a parameter expression from a string.
fn replace_symbol(symbol_expr: &SymbolExpr, name_map: &HashMap<String, Symbol>) -> SymbolExpr {
    match symbol_expr {
        SymbolExpr::Symbol(existing_symbol) => {
            let name = existing_symbol.name();
            if let Some(new_symbol) = name_map.get(&name) {
                SymbolExpr::Symbol(new_symbol.clone())
            } else {
                symbol_expr.clone()
            }
        }
        SymbolExpr::Value(_) => symbol_expr.clone(), // nothing to do
        SymbolExpr::Binary { op, lhs, rhs } => SymbolExpr::Binary {
            op: op.clone(),
            lhs: Arc::new(replace_symbol(lhs, name_map)),
            rhs: Arc::new(replace_symbol(rhs, name_map)),
        },
        SymbolExpr::Unary { op, expr } => SymbolExpr::Unary {
            op: op.clone(),
            expr: Arc::new(replace_symbol(expr, name_map)),
        },
    }
}
