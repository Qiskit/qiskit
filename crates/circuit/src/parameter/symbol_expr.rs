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

use hashbrown::HashMap;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyTypeError;
use pyo3::exceptions::PyValueError;
use std::cmp::Ordering;
use std::cmp::PartialOrd;
use std::convert::From;
use std::fmt;
use std::hash::Hash;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;
use uuid::Uuid;

use num_complex::Complex64;
use pyo3::prelude::*;

use crate::parameter::parameter_expression::PyParameter;
use crate::parameter::parameter_expression::PyParameterVectorElement;

// epsilon for SymbolExpr is heuristically defined
pub const SYMEXPR_EPSILON: f64 = f64::EPSILON * 8.0;

#[derive(Debug, Clone)]
pub struct Symbol {
    name: String,                  // the name of the symbol
    pub uuid: Uuid,                // the unique identifier
    pub index: Option<u32>,        // an optional index, if part of a vector
    pub vector: Option<Py<PyAny>>, // Python only: a reference to the vector, if it is an element
}

/// Custom implementations of Eq, PartialEq, PartialOrd and Hash to ignore the ``vector`` field
impl PartialEq for Symbol {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.uuid == other.uuid && self.index == other.index
    }
}

impl Eq for Symbol {}

impl PartialOrd for Symbol {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (&self.name, self.uuid, self.index).partial_cmp(&(&other.name, other.uuid, other.index))
    }
}

impl Hash for Symbol {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (&self.name, self.uuid, self.index).hash(state);
    }
}

impl<'py> FromPyObject<'py> for Symbol {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(py_vector_element) = ob.extract::<PyParameterVectorElement>() {
            Ok(py_vector_element.symbol().clone())
        } else if let Ok(py_param) = ob.extract::<PyParameter>() {
            Ok(py_param.symbol().clone())
        } else {
            Err(PyTypeError::new_err("Cannot extract Symbol from {ob:?}"))
        }
    }
}

impl<'py> IntoPyObject<'py> for Symbol {
    type Target = PyAny; // to cover PyParameter and PyParameterVectorElement
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match (&self.index, &self.vector) {
            (Some(_), Some(_)) => Py::new(py, PyParameterVectorElement::from_symbol(self.clone()))?
                .into_bound_py_any(py),
            _ => Py::new(py, PyParameter::from_symbol(self.clone()))?.into_bound_py_any(py),
        }
    }
}

impl Symbol {
    pub fn new(name: &str, uuid: Option<Uuid>, index: Option<u32>) -> Self {
        Self {
            name: name.to_string(),
            uuid: uuid.unwrap_or(Uuid::new_v4()),
            index,
            vector: None, // Python only
        }
    }

    /// In addition to ``new``, this also takes a vector.
    pub fn py_new(
        name: &str,
        uuid: Option<u128>,
        index: Option<u32>,
        vector: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        if index.is_some() != vector.is_some() {
            return Err(PyValueError::new_err(
                "Either both of vector and index must be provided, or neither of them",
            ));
        }

        Ok(Self {
            name: name.to_string(),
            uuid: uuid.map_or_else(Uuid::new_v4, Uuid::from_u128),
            index,
            vector,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn repr(&self, with_uuid: bool) -> String {
        match (self.index, with_uuid) {
            (Some(i), true) => format!("{}[{}]_{}", self.name, i, self.uuid.as_u128()),
            (Some(i), false) => format!("{}[{}]", self.name, i),
            (None, true) => format!("{}_{}", self.name, self.uuid.as_u128()),
            (None, false) => self.name.clone(),
        }
    }
}

/// node types of expression tree
#[derive(Debug, Clone)]
pub enum SymbolExpr {
    Symbol(Arc<Symbol>),
    Value(Value),
    Unary {
        op: UnaryOp,
        expr: Arc<SymbolExpr>,
    },
    Binary {
        op: BinaryOp,
        lhs: Arc<SymbolExpr>,
        rhs: Arc<SymbolExpr>,
    },
}

/// Value type, can be integer, real or complex number
#[derive(Debug, Clone, Copy, IntoPyObject, IntoPyObjectRef)]
pub enum Value {
    Real(f64),
    Int(i64),
    Complex(Complex64),
    Rational { numerator: i64, denominator: i64 },
}

impl<'py> FromPyObject<'py> for Value {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(i) = ob.extract::<i64>() {
            Ok(Value::Int(i))
        } else if let Ok(r) = ob.extract::<f64>() {
            Ok(Value::Real(r))
        } else if let Ok(c) = ob.extract::<Complex64>() {
            Ok(Value::Complex(c))
        } else {
            Err(PyValueError::new_err(
                "Could not cast Bound<PyAny> to Value.",
            ))
        }
    }
}

/// definition of unary operations
#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Abs,
    Neg,
    Sin,
    Asin,
    Cos,
    Acos,
    Tan,
    Atan,
    Exp,
    Log,
    Sign,
    Conj,
}

/// definition of binary operations
#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

// functions to make new expr for add
#[inline(always)]
fn _add(lhs: SymbolExpr, rhs: SymbolExpr) -> SymbolExpr {
    if let SymbolExpr::Unary {
        op: UnaryOp::Neg, ..
    } = &rhs
    {
        return match rhs.neg_opt() {
            Some(e) => SymbolExpr::Binary {
                op: BinaryOp::Sub,
                lhs: Arc::new(lhs),
                rhs: Arc::new(e),
            },
            None => SymbolExpr::Binary {
                op: BinaryOp::Sub,
                lhs: Arc::new(lhs),
                rhs: Arc::new(_neg(rhs)),
            },
        };
    }
    SymbolExpr::Binary {
        op: BinaryOp::Add,
        lhs: Arc::new(lhs),
        rhs: Arc::new(rhs),
    }
}

// functions to make new expr for sub
#[inline(always)]
fn _sub(lhs: SymbolExpr, rhs: SymbolExpr) -> SymbolExpr {
    if let SymbolExpr::Unary {
        op: UnaryOp::Neg, ..
    } = &rhs
    {
        return match rhs.neg_opt() {
            Some(e) => SymbolExpr::Binary {
                op: BinaryOp::Add,
                lhs: Arc::new(lhs),
                rhs: Arc::new(e),
            },
            None => SymbolExpr::Binary {
                op: BinaryOp::Sub,
                lhs: Arc::new(lhs),
                rhs: Arc::new(rhs),
            },
        };
    }
    SymbolExpr::Binary {
        op: BinaryOp::Sub,
        lhs: Arc::new(lhs),
        rhs: Arc::new(rhs),
    }
}

// functions to make new expr for mul
#[inline(always)]
fn _mul(lhs: SymbolExpr, rhs: SymbolExpr) -> SymbolExpr {
    SymbolExpr::Binary {
        op: BinaryOp::Mul,
        lhs: Arc::new(lhs),
        rhs: Arc::new(rhs),
    }
}

// functions to make new expr for div
#[inline(always)]
fn _div(lhs: SymbolExpr, rhs: SymbolExpr) -> SymbolExpr {
    SymbolExpr::Binary {
        op: BinaryOp::Div,
        lhs: Arc::new(lhs),
        rhs: Arc::new(rhs),
    }
}

// functions to make new expr for pow
#[inline(always)]
fn _pow(lhs: SymbolExpr, rhs: SymbolExpr) -> SymbolExpr {
    SymbolExpr::Binary {
        op: BinaryOp::Pow,
        lhs: Arc::new(lhs),
        rhs: Arc::new(rhs),
    }
}

// functions to make new expr for neg
#[inline(always)]
fn _neg(expr: SymbolExpr) -> SymbolExpr {
    match expr.neg_opt() {
        Some(e) => e,
        None => SymbolExpr::Unary {
            op: UnaryOp::Neg,
            expr: Arc::new(expr),
        },
    }
}

impl fmt::Display for SymbolExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.repr(false))
    }
}

impl SymbolExpr {
    /// bind value to symbol node
    pub fn bind(&self, maps: &HashMap<&Symbol, Value>) -> SymbolExpr {
        match self {
            SymbolExpr::Symbol(e) => match maps.get(e.as_ref()) {
                Some(v) => SymbolExpr::Value(*v),
                None => self.clone(),
            },
            SymbolExpr::Value(e) => SymbolExpr::Value(*e),
            SymbolExpr::Unary { op, expr } => SymbolExpr::Unary {
                op: op.clone(),
                expr: Arc::new(expr.bind(maps)),
            },
            SymbolExpr::Binary { op, lhs, rhs } => {
                let new_lhs = lhs.bind(maps);
                let new_rhs = rhs.bind(maps);
                match op {
                    BinaryOp::Add => new_lhs + new_rhs,
                    BinaryOp::Sub => new_lhs - new_rhs,
                    BinaryOp::Mul => new_lhs * new_rhs,
                    BinaryOp::Div => new_lhs / new_rhs,
                    BinaryOp::Pow => _pow(new_lhs, new_rhs),
                }
            }
        }
    }

    /// substitute symbol node to other expression
    /// allows unknown expressions
    /// does not allow duplicate names with different UUID
    pub fn subs(&self, maps: &HashMap<Symbol, SymbolExpr>) -> SymbolExpr {
        match self {
            SymbolExpr::Symbol(e) => match maps.get(e.as_ref()) {
                Some(v) => v.clone(),
                None => self.clone(),
            },
            SymbolExpr::Value(e) => SymbolExpr::Value(*e),
            SymbolExpr::Unary { op, expr } => SymbolExpr::Unary {
                op: op.clone(),
                expr: Arc::new(expr.subs(maps)),
            },
            SymbolExpr::Binary { op, lhs, rhs } => {
                let new_lhs = lhs.subs(maps);
                let new_rhs = rhs.subs(maps);
                match op {
                    BinaryOp::Add => new_lhs + new_rhs,
                    BinaryOp::Sub => new_lhs - new_rhs,
                    BinaryOp::Mul => new_lhs * new_rhs,
                    BinaryOp::Div => new_lhs / new_rhs,
                    BinaryOp::Pow => _pow(new_lhs, new_rhs),
                }
            }
        }
    }

    /// evaluate the equation
    /// if recursive is false, only this node will be evaluated
    pub fn eval(&self, recurse: bool) -> Option<Value> {
        match self {
            SymbolExpr::Symbol(_) => None,
            SymbolExpr::Value(e) => Some(*e),
            SymbolExpr::Unary { op, expr } => {
                let val: Value;
                if recurse {
                    match expr.eval(recurse) {
                        Some(v) => val = v,
                        None => return None,
                    }
                } else {
                    match expr.as_ref() {
                        SymbolExpr::Value(e) => val = *e,
                        _ => return None,
                    }
                }
                let ret = match op {
                    UnaryOp::Abs => val.abs(),
                    UnaryOp::Neg => -val,
                    UnaryOp::Sin => val.sin(),
                    UnaryOp::Asin => val.asin(),
                    UnaryOp::Cos => val.cos(),
                    UnaryOp::Acos => val.acos(),
                    UnaryOp::Tan => val.tan(),
                    UnaryOp::Atan => val.atan(),
                    UnaryOp::Exp => val.exp(),
                    UnaryOp::Log => val.log(),
                    UnaryOp::Sign => val.sign(),
                    UnaryOp::Conj => match val {
                        Value::Complex(v) => Value::Complex(v.conj()),
                        _ => val,
                    },
                };
                match ret {
                    Value::Real(_) => Some(ret),
                    Value::Int(_) => Some(ret),
                    Value::Complex(c) => {
                        if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.im) {
                            Some(Value::Real(c.re))
                        } else {
                            Some(ret)
                        }
                    }
                    Value::Rational { .. } => Some(ret),
                }
            }
            SymbolExpr::Binary { op, lhs, rhs } => {
                let lval: Value;
                let rval: Value;
                if recurse {
                    match (lhs.eval(true), rhs.eval(true)) {
                        (Some(left), Some(right)) => {
                            lval = left;
                            rval = right;
                        }
                        (Some(left), None) => {
                            if left.is_zero() {
                                return Some(left);
                            } else {
                                return None;
                            }
                        }
                        (None, Some(right)) => {
                            if right.is_zero() {
                                return Some(right);
                            } else {
                                return None;
                            }
                        }
                        (_, _) => return None,
                    }
                } else {
                    match (lhs.as_ref(), rhs.as_ref()) {
                        (SymbolExpr::Value(l), SymbolExpr::Value(r)) => {
                            lval = *l;
                            rval = *r;
                        }
                        _ => return None,
                    }
                }
                let ret = match op {
                    BinaryOp::Add => lval + rval,
                    BinaryOp::Sub => lval - rval,
                    BinaryOp::Mul => lval * rval,
                    BinaryOp::Div => lval / rval,
                    BinaryOp::Pow => lval.pow(&rval),
                };
                match ret {
                    Value::Real(_) => Some(ret),
                    Value::Int(_) => Some(ret),
                    Value::Complex(c) => {
                        if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.im) {
                            Some(Value::Real(c.re))
                        } else {
                            Some(ret)
                        }
                    }
                    Value::Rational { .. } => Some(ret),
                }
            }
        }
    }

    /// calculate derivative of the equantion for a symbol passed by param
    pub fn derivative(&self, param: &Symbol) -> Result<SymbolExpr, String> {
        if let SymbolExpr::Symbol(s) = self {
            if s.as_ref() == param {
                return Ok(SymbolExpr::Value(Value::Real(1.0)));
            }
        }
        if self.is_rational() {
            return Ok(SymbolExpr::Value(Value::Real(0.0)));
        }

        match self {
            SymbolExpr::Value(_) | SymbolExpr::Symbol(_) => Ok(SymbolExpr::Value(Value::Real(0.0))),
            SymbolExpr::Unary { op, expr } => {
                let expr_d = expr.derivative(param)?;
                match op {
                    UnaryOp::Abs => Ok(&(expr.as_ref() * &expr_d)
                        / &SymbolExpr::Unary {
                            op: op.clone(),
                            expr: Arc::new(expr.as_ref().clone()),
                        }),
                    UnaryOp::Neg => Ok(SymbolExpr::Unary {
                        op: UnaryOp::Neg,
                        expr: Arc::new(expr_d),
                    }),
                    UnaryOp::Sin => {
                        let lhs = SymbolExpr::Unary {
                            op: UnaryOp::Cos,
                            expr: Arc::new(expr.as_ref().clone()),
                        };
                        Ok(lhs * expr_d)
                    }
                    UnaryOp::Asin => {
                        let d =
                            &SymbolExpr::Value(Value::Real(1.0)) - &(expr.as_ref() * expr.as_ref());
                        let rhs = match d {
                            SymbolExpr::Value(v) => SymbolExpr::Value(v.sqrt()),
                            _ => _pow(d, SymbolExpr::Value(Value::Real(0.5))),
                        };
                        Ok(&expr_d / &rhs)
                    }
                    UnaryOp::Cos => {
                        let lhs = SymbolExpr::Unary {
                            op: UnaryOp::Sin,
                            expr: Arc::new(expr.as_ref().clone()),
                        };
                        Ok(&-&lhs * &expr_d)
                    }
                    UnaryOp::Acos => {
                        let d =
                            &SymbolExpr::Value(Value::Real(1.0)) - &(expr.as_ref() * expr.as_ref());
                        let rhs = match d {
                            SymbolExpr::Value(v) => SymbolExpr::Value(v.sqrt()),
                            _ => _pow(d, SymbolExpr::Value(Value::Real(0.5))),
                        };
                        Ok(&-&expr_d / &rhs)
                    }
                    UnaryOp::Tan => {
                        let d = SymbolExpr::Unary {
                            op: UnaryOp::Cos,
                            expr: Arc::new(expr.as_ref().clone()),
                        };
                        Ok(&(&expr_d / &d) / &d)
                    }
                    UnaryOp::Atan => {
                        let d =
                            &SymbolExpr::Value(Value::Real(1.0)) + &(expr.as_ref() * expr.as_ref());
                        Ok(&expr_d / &d)
                    }
                    UnaryOp::Exp => Ok(&SymbolExpr::Unary {
                        op: UnaryOp::Exp,
                        expr: Arc::new(expr.as_ref().clone()),
                    } * &expr_d),
                    UnaryOp::Log => Ok(&expr_d / expr.as_ref()),
                    UnaryOp::Sign => {
                        Err("SymbolExpr::derivative does not support sign function.".to_string())
                    }
                    UnaryOp::Conj => {
                        // we assume real parameters, hence Conj acts as identity
                        Ok(SymbolExpr::Value(Value::Real(1.0)))
                    }
                }
            }
            SymbolExpr::Binary { op, lhs, rhs } => match op {
                BinaryOp::Add => Ok(lhs.derivative(param)? + rhs.derivative(param)?),
                BinaryOp::Sub => Ok(lhs.derivative(param)? - rhs.derivative(param)?),
                BinaryOp::Mul => Ok(&(&lhs.derivative(param)? * rhs.as_ref())
                    + &(lhs.as_ref() * &rhs.derivative(param)?)),
                BinaryOp::Div => Ok(&(&(&(&lhs.derivative(param)? * rhs.as_ref())
                    - &(lhs.as_ref() * &rhs.derivative(param)?))
                    / rhs.as_ref())
                    / rhs.as_ref()),
                BinaryOp::Pow => {
                    if !lhs.has_symbol(param) {
                        if !rhs.has_symbol(param) {
                            Ok(SymbolExpr::Value(Value::Real(0.0)))
                        } else {
                            Ok(_mul(
                                SymbolExpr::Binary {
                                    op: BinaryOp::Pow,
                                    lhs: Arc::new(lhs.as_ref().clone()),
                                    rhs: Arc::new(rhs.as_ref().clone()),
                                },
                                SymbolExpr::Unary {
                                    op: UnaryOp::Log,
                                    expr: Arc::new(lhs.as_ref().clone()),
                                },
                            ))
                        }
                    } else if !rhs.has_symbol(param) {
                        Ok(rhs.as_ref()
                            * &SymbolExpr::Binary {
                                op: BinaryOp::Pow,
                                lhs: Arc::new(lhs.as_ref().clone()),
                                rhs: Arc::new(rhs.as_ref() - &SymbolExpr::Value(Value::Real(1.0))),
                            })
                    } else {
                        let new_expr = SymbolExpr::Unary {
                            op: UnaryOp::Exp,
                            expr: Arc::new(_mul(
                                SymbolExpr::Unary {
                                    op: UnaryOp::Log,
                                    expr: Arc::new(lhs.as_ref().clone()),
                                },
                                rhs.as_ref().clone(),
                            )),
                        };
                        new_expr.derivative(param)
                    }
                }
            },
        }
    }

    /// expand the equation
    pub fn expand(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Symbol(_) => self.clone(),
            SymbolExpr::Value(_) => self.clone(),
            SymbolExpr::Unary { op, expr } => {
                let ex = expr.expand();
                match op {
                    UnaryOp::Neg => match ex.neg_opt() {
                        Some(ne) => ne,
                        None => _neg(ex),
                    },
                    _ => SymbolExpr::Unary {
                        op: op.clone(),
                        expr: Arc::new(ex),
                    },
                }
            }
            SymbolExpr::Binary { op, lhs, rhs } => {
                if self.is_rational() {
                    return self.clone();
                }

                match op {
                    BinaryOp::Mul => match lhs.mul_expand(rhs) {
                        Some(e) => e,
                        None => _mul(lhs.as_ref().clone(), rhs.as_ref().clone()),
                    },
                    BinaryOp::Div => match lhs.div_expand(rhs) {
                        Some(e) => e,
                        None => _div(lhs.as_ref().clone(), rhs.as_ref().clone()),
                    },
                    BinaryOp::Add => {
                        let ex_lhs = lhs.expand();
                        let ex_rhs = rhs.expand();
                        match ex_lhs.expand().add_opt(&ex_rhs, true) {
                            Some(e) => e,
                            None => _add(ex_lhs, ex_rhs),
                        }
                    }
                    BinaryOp::Sub => {
                        let ex_lhs = lhs.expand();
                        let ex_rhs = rhs.expand();
                        match ex_lhs.expand().sub_opt(&ex_rhs, true) {
                            Some(e) => e,
                            None => _sub(ex_lhs, ex_rhs),
                        }
                    }
                    _ => _pow(lhs.expand(), rhs.expand()), // TO DO : add expand for pow
                }
            }
        }
    }

    /// sign operator
    pub fn sign(&self) -> SymbolExpr {
        SymbolExpr::Unary {
            op: UnaryOp::Sign,
            expr: Arc::new(self.clone()),
        }
    }

    /// return real number if equation can be evaluated
    pub fn real(&self) -> Option<f64> {
        match self.eval(true) {
            Some(v) => match v {
                Value::Real(r) => Some(r),
                Value::Int(r) => Some(r as f64),
                Value::Complex(c) => Some(c.re),
                Value::Rational {
                    numerator,
                    denominator,
                } => Some(numerator as f64 / denominator as f64),
            },
            None => None,
        }
    }
    /// return imaginary part of the value if equation can be evaluated as complex number
    pub fn imag(&self) -> Option<f64> {
        match self.eval(true) {
            Some(v) => match v {
                Value::Real(_) => Some(0.0),
                Value::Int(_) => Some(0.0),
                Value::Complex(c) => Some(c.im),
                Value::Rational { .. } => Some(0.0),
            },
            None => None,
        }
    }
    /// return complex number if equation can be evaluated as complex
    pub fn complex(&self) -> Option<Complex64> {
        match self.eval(true) {
            Some(v) => match v {
                Value::Real(r) => Some(r.into()),
                Value::Int(i) => Some((i as f64).into()),
                Value::Complex(c) => Some(c),
                Value::Rational {
                    numerator,
                    denominator,
                } => Some((numerator as f64 / denominator as f64).into()),
            },
            None => None,
        }
    }

    /// return integer rational number as tuple
    #[inline(always)]
    pub fn rational(&self) -> Option<(i64, i64)> {
        match self {
            SymbolExpr::Value(Value::Rational {
                numerator,
                denominator,
            }) => Some((*numerator, *denominator)),
            SymbolExpr::Binary { op, lhs, rhs } => match (op, lhs.as_ref(), rhs.as_ref()) {
                (
                    BinaryOp::Div,
                    SymbolExpr::Value(Value::Int(numerator)),
                    SymbolExpr::Value(Value::Int(denominator)),
                ) => Some((*numerator, *denominator)),
                (_, _, _) => None,
            },
            _ => None,
        }
    }

    /// Iterate over all symbols this equation contains.
    pub fn iter_symbols(&self) -> Box<dyn Iterator<Item = &Symbol> + '_> {
        // This could maybe be more elegantly resolved with a SymbolIter type>
        match self {
            SymbolExpr::Symbol(e) => Box::new(::std::iter::once(e.as_ref())),
            SymbolExpr::Value(_) => Box::new(::std::iter::empty()),
            SymbolExpr::Unary { op: _, expr } => expr.iter_symbols(),
            SymbolExpr::Binary { op: _, lhs, rhs } => {
                Box::new(lhs.iter_symbols().chain(rhs.iter_symbols()))
            }
        }
    }

    /// Map of parameter name to the parameter.
    pub fn name_map(&self) -> HashMap<String, Symbol> {
        self.iter_symbols()
            .map(|param| (param.repr(false), param.clone()))
            .collect()
    }

    /// return all numbers in the equation
    pub fn values(&self) -> Vec<Value> {
        match self {
            SymbolExpr::Symbol(_) => Vec::<Value>::new(),
            SymbolExpr::Value(v) => Vec::<Value>::from([*v]),
            SymbolExpr::Unary { op: _, expr } => expr.values(),
            SymbolExpr::Binary { op: _, lhs, rhs } => {
                if let Some((numerator, denominator)) = self.rational() {
                    return Vec::from([Value::Real(numerator as f64 / denominator as f64)]);
                }
                let mut l = lhs.values();
                let r = rhs.values();
                l.extend(r);
                l
            }
        }
    }

    /// check if a symbol is in this equation
    pub fn has_symbol(&self, param: &Symbol) -> bool {
        match self {
            SymbolExpr::Symbol(e) => e.as_ref().eq(param),
            SymbolExpr::Value(_) => false,
            SymbolExpr::Unary { op: _, expr } => expr.has_symbol(param),
            SymbolExpr::Binary { op: _, lhs, rhs } => lhs.has_symbol(param) | rhs.has_symbol(param),
        }
    }

    /// return reciprocal of the equation
    pub fn rcp(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Symbol(e) => _div(
                SymbolExpr::Value(Value::Int(1)),
                SymbolExpr::Symbol(e.clone()),
            ),
            SymbolExpr::Value(e) => match e {
                Value::Int(i) => SymbolExpr::Value(_rational(1, *i)),
                _ => SymbolExpr::Value(e.rcp()),
            },
            SymbolExpr::Unary { .. } => _div(SymbolExpr::Value(Value::Int(1)), self.clone()),
            SymbolExpr::Binary { op, lhs, rhs } => match op {
                BinaryOp::Div => SymbolExpr::Binary {
                    op: op.clone(),
                    lhs: rhs.clone(),
                    rhs: lhs.clone(),
                },
                _ => _div(SymbolExpr::Value(Value::Int(1)), self.clone()),
            },
        }
    }
    /// return square root of the equation
    pub fn sqrt(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(v) => SymbolExpr::Value(v.sqrt()),
            _ => self.pow(&SymbolExpr::Value(Value::Real(0.5))),
        }
    }

    /// return conjugate of the equation
    pub fn conjugate(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Symbol(_) => SymbolExpr::Unary {
                op: UnaryOp::Conj,
                expr: Arc::new(self.clone()),
            },
            SymbolExpr::Value(e) => match e {
                Value::Complex(c) => SymbolExpr::Value(Value::Complex(c.conj())),
                _ => SymbolExpr::Value(*e),
            },
            SymbolExpr::Unary { op, expr } => SymbolExpr::Unary {
                op: op.clone(),
                expr: Arc::new(expr.conjugate()),
            },
            SymbolExpr::Binary { op, lhs, rhs } => {
                if self.is_rational() {
                    self.clone()
                } else {
                    SymbolExpr::Binary {
                        op: op.clone(),
                        lhs: Arc::new(lhs.conjugate()),
                        rhs: Arc::new(rhs.conjugate()),
                    }
                }
            }
        }
    }

    /// check if complex number or not
    pub fn is_complex(&self) -> Option<bool> {
        match self.eval(true) {
            Some(v) => match v {
                Value::Complex(c) => Some(!(-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.im)),
                _ => Some(false),
            },
            None => None,
        }
    }

    /// check if real number or not
    pub fn is_real(&self) -> Option<bool> {
        match self.eval(true) {
            Some(v) => match v {
                Value::Real(_) | Value::Int(_) | Value::Rational { .. } => Some(true),
                Value::Complex(c) => Some((-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.im)),
            },
            None => None,
        }
    }

    /// check if integer or not
    pub fn is_int(&self) -> Option<bool> {
        match self.eval(true) {
            Some(v) => match v {
                Value::Int(_) => Some(true),
                _ => Some(false),
            },
            None => None,
        }
    }

    /// check if the type of the node is a value; that is it matches Value or is an integer fraction
    #[inline(always)]
    pub fn is_value(&self) -> bool {
        matches!(self, SymbolExpr::Value(_))
    }

    /// check if integer rational number or not
    #[inline(always)]
    pub fn is_rational(&self) -> bool {
        matches!(self, SymbolExpr::Value(Value::Rational { .. }))
    }

    /// check if evaluated result is 0
    pub fn is_zero(&self, recursive: bool) -> bool {
        match self.eval(recursive) {
            Some(v) => v.is_zero(),
            None => false,
        }
    }

    /// check if evaluated result is 1
    pub fn is_one(&self, recursive: bool) -> bool {
        match self.eval(recursive) {
            Some(v) => v.is_one(),
            None => false,
        }
    }

    /// check if evaluated result is -1
    pub fn is_minus_one(&self, recursive: bool) -> bool {
        match self.eval(recursive) {
            Some(v) => v.is_minus_one(),
            None => false,
        }
    }

    /// check if evaluated result is negative
    fn is_negative(&self) -> bool {
        match self {
            SymbolExpr::Value(v) => v.is_negative(),
            SymbolExpr::Symbol(_) => false,
            SymbolExpr::Unary { op, .. } => match op {
                UnaryOp::Abs => false,
                UnaryOp::Neg => true,
                _ => false, // TO DO add heuristic determination
            },
            SymbolExpr::Binary { .. } => false,
        }
    }

    /// unary operations
    pub fn abs(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.abs()),
            SymbolExpr::Unary {
                op: UnaryOp::Abs | UnaryOp::Neg,
                expr,
            } => expr.abs(),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Abs,
                expr: Arc::new(self.clone()),
            },
        }
    }
    pub fn sin(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.sin()),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Sin,
                expr: Arc::new(self.clone()),
            },
        }
    }
    pub fn asin(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.asin()),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Asin,
                expr: Arc::new(self.clone()),
            },
        }
    }
    pub fn cos(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.cos()),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Cos,
                expr: Arc::new(self.clone()),
            },
        }
    }
    pub fn acos(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.acos()),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Acos,
                expr: Arc::new(self.clone()),
            },
        }
    }
    pub fn tan(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.tan()),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Tan,
                expr: Arc::new(self.clone()),
            },
        }
    }
    pub fn atan(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.atan()),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Atan,
                expr: Arc::new(self.clone()),
            },
        }
    }
    pub fn exp(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.exp()),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Exp,
                expr: Arc::new(self.clone()),
            },
        }
    }
    pub fn log(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.log()),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Log,
                expr: Arc::new(self.clone()),
            },
        }
    }
    pub fn pow(&self, rhs: &SymbolExpr) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => match rhs {
                SymbolExpr::Value(r) => SymbolExpr::Value(l.pow(r)),
                _ => SymbolExpr::Binary {
                    op: BinaryOp::Pow,
                    lhs: Arc::new(SymbolExpr::Value(*l)),
                    rhs: Arc::new(rhs.clone()),
                },
            },
            _ => SymbolExpr::Binary {
                op: BinaryOp::Pow,
                lhs: Arc::new(self.clone()),
                rhs: Arc::new(rhs.clone()),
            },
        }
    }

    // add values
    #[inline(always)]
    fn add_values(&self, rhs: &SymbolExpr, recursive: bool) -> Option<SymbolExpr> {
        if rhs.is_value() {
            if let SymbolExpr::Value(l) = self {
                if let SymbolExpr::Value(r) = rhs {
                    return Some(SymbolExpr::Value(l + r));
                }
            } else if let SymbolExpr::Binary {
                op,
                lhs: l_lhs,
                rhs: l_rhs,
            } = self
            {
                match op {
                    BinaryOp::Add => {
                        if l_lhs.is_value() || recursive {
                            if let Some(e) = l_lhs.add_values(rhs, recursive) {
                                if e.is_zero(recursive) {
                                    return Some(l_rhs.as_ref().clone());
                                } else {
                                    return Some(_add(e, l_rhs.as_ref().clone()));
                                }
                            }
                        }
                        if l_rhs.is_value() || recursive {
                            if let Some(e) = l_rhs.add_values(rhs, recursive) {
                                if e.is_zero(recursive) {
                                    return Some(l_lhs.as_ref().clone());
                                } else {
                                    return Some(_add(e, l_lhs.as_ref().clone()));
                                }
                            }
                        }
                    }
                    BinaryOp::Sub => {
                        if l_lhs.is_value() || recursive {
                            if let Some(e) = l_lhs.add_values(rhs, recursive) {
                                if e.is_zero(recursive) {
                                    return Some(_neg(l_rhs.as_ref().clone()));
                                } else {
                                    return Some(_sub(e, l_rhs.as_ref().clone()));
                                }
                            }
                        }
                        if l_rhs.is_value() || recursive {
                            if let Some(e) = rhs.sub_values(l_rhs, recursive) {
                                if e.is_zero(recursive) {
                                    return Some(l_lhs.as_ref().clone());
                                } else {
                                    return Some(_add(e, l_lhs.as_ref().clone()));
                                }
                            }
                        }
                    }
                    _ => (),
                }
            }
        } else if self.is_value() {
            if let SymbolExpr::Binary {
                op,
                lhs: r_lhs,
                rhs: r_rhs,
            } = rhs
            {
                match op {
                    BinaryOp::Add => {
                        if r_lhs.is_value() || recursive {
                            if let Some(e) = self.add_values(r_lhs, recursive) {
                                if e.is_zero(recursive) {
                                    return Some(r_rhs.as_ref().clone());
                                } else {
                                    return Some(_add(e, r_rhs.as_ref().clone()));
                                }
                            }
                        }
                        if r_rhs.is_value() || recursive {
                            if let Some(e) = self.add_values(r_rhs, recursive) {
                                if e.is_zero(recursive) {
                                    return Some(r_lhs.as_ref().clone());
                                } else {
                                    return Some(_add(e, r_lhs.as_ref().clone()));
                                }
                            }
                        }
                    }
                    BinaryOp::Sub => {
                        if r_lhs.is_value() || recursive {
                            if let Some(e) = self.add_values(r_lhs, recursive) {
                                if e.is_zero(recursive) {
                                    return Some(_neg(r_rhs.as_ref().clone()));
                                } else {
                                    return Some(_sub(e, r_rhs.as_ref().clone()));
                                }
                            }
                        }
                        if r_rhs.is_value() || recursive {
                            if let Some(e) = self.sub_values(r_rhs, recursive) {
                                if e.is_zero(recursive) {
                                    return Some(r_lhs.as_ref().clone());
                                } else {
                                    return Some(_add(e, r_lhs.as_ref().clone()));
                                }
                            }
                        }
                    }
                    _ => (),
                }
            }
        }
        None
    }

    // add values
    #[inline(always)]
    fn sub_values(&self, rhs: &SymbolExpr, recursive: bool) -> Option<SymbolExpr> {
        if rhs.is_value() {
            if let SymbolExpr::Value(l) = self {
                if let SymbolExpr::Value(r) = rhs {
                    return Some(SymbolExpr::Value(l - r));
                }
            } else if let SymbolExpr::Binary {
                op,
                lhs: l_lhs,
                rhs: l_rhs,
            } = self
            {
                match op {
                    BinaryOp::Add => {
                        if l_lhs.is_value() || recursive {
                            if let Some(e) = l_lhs.sub_values(rhs, recursive) {
                                if e.is_zero(recursive) {
                                    return Some(l_rhs.as_ref().clone());
                                } else {
                                    return Some(_add(e, l_rhs.as_ref().clone()));
                                }
                            }
                        }
                        if l_rhs.is_value() || recursive {
                            if let Some(e) = l_rhs.sub_values(rhs, recursive) {
                                if e.is_zero(recursive) {
                                    return Some(l_lhs.as_ref().clone());
                                } else {
                                    return Some(_add(e, l_lhs.as_ref().clone()));
                                }
                            }
                        }
                    }
                    BinaryOp::Sub => {
                        if l_lhs.is_value() || recursive {
                            if let Some(e) = l_lhs.sub_values(rhs, recursive) {
                                if e.is_zero(recursive) {
                                    return Some(_neg(l_rhs.as_ref().clone()));
                                } else {
                                    return Some(_sub(e, l_rhs.as_ref().clone()));
                                }
                            }
                        }
                        if l_rhs.is_value() || recursive {
                            if let Some(e) = rhs.add_values(l_rhs, recursive) {
                                if e.is_zero(recursive) {
                                    return Some(l_lhs.as_ref().clone());
                                } else {
                                    return Some(_add(_neg(e), l_lhs.as_ref().clone()));
                                }
                            }
                        }
                    }
                    _ => (),
                }
            }
        } else if self.is_value() {
            if let SymbolExpr::Binary {
                op,
                lhs: r_lhs,
                rhs: r_rhs,
            } = rhs
            {
                match op {
                    BinaryOp::Add => {
                        if r_lhs.is_value() || recursive {
                            if let Some(e) = self.sub_values(r_lhs, recursive) {
                                if e.is_zero(recursive) {
                                    return Some(_neg(r_rhs.as_ref().clone()));
                                } else {
                                    return Some(_sub(e, r_rhs.as_ref().clone()));
                                }
                            }
                        }
                        if r_rhs.is_value() || recursive {
                            if let Some(e) = self.sub_values(r_rhs, recursive) {
                                if e.is_zero(recursive) {
                                    return Some(_neg(r_lhs.as_ref().clone()));
                                } else {
                                    return Some(_sub(e, r_lhs.as_ref().clone()));
                                }
                            }
                        }
                    }
                    BinaryOp::Sub => {
                        if r_lhs.is_value() || recursive {
                            if let Some(e) = self.sub_values(r_lhs, recursive) {
                                if e.is_zero(recursive) {
                                    return Some(r_rhs.as_ref().clone());
                                } else {
                                    return Some(_add(e, r_rhs.as_ref().clone()));
                                }
                            }
                        }
                        if r_rhs.is_value() || recursive {
                            if let Some(e) = self.add_values(r_rhs, recursive) {
                                if e.is_zero(recursive) {
                                    return Some(_neg(r_lhs.as_ref().clone()));
                                } else {
                                    return Some(_sub(e, r_lhs.as_ref().clone()));
                                }
                            }
                        }
                    }
                    _ => (),
                }
            }
        }
        None
    }

    // mul values
    #[inline(always)]
    fn mul_values(&self, rhs: &SymbolExpr, recursive: bool) -> Option<SymbolExpr> {
        if rhs.is_value() {
            if let SymbolExpr::Value(l) = self {
                if let SymbolExpr::Value(r) = rhs {
                    return Some(SymbolExpr::Value(l * r));
                }
            } else if let SymbolExpr::Binary {
                op,
                lhs: l_lhs,
                rhs: l_rhs,
            } = self
            {
                match op {
                    BinaryOp::Mul => {
                        if l_lhs.is_value() || recursive {
                            if let Some(e) = l_lhs.mul_values(rhs, recursive) {
                                if e.is_one(recursive) {
                                    return Some(l_rhs.as_ref().clone());
                                } else if e.is_minus_one(recursive) {
                                    return l_rhs.neg_opt();
                                }
                                return Some(_mul(e, l_rhs.as_ref().clone()));
                            }
                        }
                        if l_rhs.is_value() || recursive {
                            if let Some(e) = l_rhs.mul_values(rhs, recursive) {
                                if e.is_one(recursive) {
                                    return Some(l_lhs.as_ref().clone());
                                } else if e.is_minus_one(recursive) {
                                    return l_lhs.neg_opt();
                                }
                                return Some(_mul(e, l_lhs.as_ref().clone()));
                            }
                        }
                    }
                    BinaryOp::Div => {
                        if l_lhs.is_value() || recursive {
                            if let Some(e) = l_lhs.mul_values(rhs, recursive) {
                                return Some(_div(e, l_rhs.as_ref().clone()));
                            }
                        }
                        if l_rhs.is_value() || recursive {
                            if let Some(e) = rhs.div_values(l_rhs, recursive) {
                                if e.is_one(recursive) {
                                    return Some(l_lhs.as_ref().clone());
                                } else if e.is_minus_one(recursive) {
                                    return l_lhs.neg_opt();
                                }
                                return Some(_mul(e, l_lhs.as_ref().clone()));
                            }
                        }
                    }
                    _ => (),
                }
            }
        } else if self.is_value() {
            if let SymbolExpr::Binary {
                op,
                lhs: r_lhs,
                rhs: r_rhs,
            } = rhs
            {
                match op {
                    BinaryOp::Mul => {
                        if r_lhs.is_value() || recursive {
                            if let Some(e) = self.mul_values(r_lhs, recursive) {
                                if e.is_one(recursive) {
                                    return Some(r_rhs.as_ref().clone());
                                } else if e.is_minus_one(recursive) {
                                    return r_rhs.neg_opt();
                                }
                                return Some(_mul(e, r_rhs.as_ref().clone()));
                            }
                        }
                        if r_rhs.is_value() || recursive {
                            if let Some(e) = self.mul_values(r_rhs, recursive) {
                                if e.is_one(recursive) {
                                    return Some(r_lhs.as_ref().clone());
                                } else if e.is_minus_one(recursive) {
                                    return r_lhs.neg_opt();
                                }
                                return Some(_mul(e, r_lhs.as_ref().clone()));
                            }
                        }
                    }
                    BinaryOp::Div => {
                        if r_lhs.is_value() || recursive {
                            if let Some(e) = self.mul_values(r_lhs, recursive) {
                                return Some(_div(e, r_rhs.as_ref().clone()));
                            }
                        }
                        if r_rhs.is_value() || recursive {
                            if let Some(e) = self.div_values(r_rhs, recursive) {
                                if e.is_one(recursive) {
                                    return Some(r_lhs.as_ref().clone());
                                } else if e.is_minus_one(recursive) {
                                    return r_lhs.neg_opt();
                                }
                                return Some(_mul(e, r_lhs.as_ref().clone()));
                            }
                        }
                    }
                    _ => (),
                }
            }
        }
        None
    }

    // divide values
    #[inline(always)]
    fn div_values(&self, rhs: &SymbolExpr, recursive: bool) -> Option<SymbolExpr> {
        if rhs.is_value() {
            if let SymbolExpr::Value(l) = self {
                if let SymbolExpr::Value(r) = rhs {
                    return Some(SymbolExpr::Value(l / r));
                }
            } else if let SymbolExpr::Binary {
                op,
                lhs: l_lhs,
                rhs: l_rhs,
            } = self
            {
                match op {
                    BinaryOp::Mul => {
                        if l_lhs.is_value() || recursive {
                            if let Some(e) = l_lhs.div_values(rhs, recursive) {
                                if e.is_one(recursive) {
                                    return Some(l_rhs.as_ref().clone());
                                } else if e.is_minus_one(recursive) {
                                    return l_rhs.neg_opt();
                                }
                                return Some(_mul(e, l_rhs.as_ref().clone()));
                            }
                        }
                        if l_rhs.is_value() || recursive {
                            if let Some(e) = l_rhs.div_values(rhs, recursive) {
                                if e.is_one(recursive) {
                                    return Some(l_lhs.as_ref().clone());
                                } else if e.is_minus_one(recursive) {
                                    return l_lhs.neg_opt();
                                }
                                return Some(_mul(e, l_lhs.as_ref().clone()));
                            }
                        }
                    }
                    BinaryOp::Div => {
                        if l_lhs.is_value() || recursive {
                            if let Some(e) = l_lhs.div_values(rhs, recursive) {
                                return Some(_div(e, l_rhs.as_ref().clone()));
                            }
                        }
                        if l_rhs.is_value() || recursive {
                            if let Some(e) = rhs.mul_values(l_rhs, recursive) {
                                return Some(_div(l_lhs.as_ref().clone(), e));
                            }
                        }
                    }
                    _ => (),
                }
            }
        } else if self.is_value() {
            if let SymbolExpr::Binary {
                op,
                lhs: r_lhs,
                rhs: r_rhs,
            } = rhs
            {
                match op {
                    BinaryOp::Mul => {
                        if r_lhs.is_value() || recursive {
                            if let Some(e) = self.div_values(r_lhs, recursive) {
                                return Some(_div(e, r_rhs.as_ref().clone()));
                            }
                        }
                        if r_rhs.is_value() || recursive {
                            if let Some(e) = self.div_values(r_rhs, recursive) {
                                return Some(_div(e, r_lhs.as_ref().clone()));
                            }
                        }
                    }
                    BinaryOp::Div => {
                        if r_lhs.is_value() || recursive {
                            if let Some(e) = self.div_values(r_lhs, recursive) {
                                if e.is_one(recursive) {
                                    return Some(r_rhs.as_ref().clone());
                                } else if e.is_minus_one(recursive) {
                                    return r_rhs.neg_opt();
                                }
                                return Some(_mul(e, r_rhs.as_ref().clone()));
                            }
                        }
                        if r_rhs.is_value() || recursive {
                            if let Some(e) = self.mul_values(r_rhs, recursive) {
                                return Some(_div(e, r_lhs.as_ref().clone()));
                            }
                        }
                    }
                    _ => (),
                }
            }
        }
        None
    }

    pub fn string_id(&self) -> String {
        self.repr(true)
    }

    // Add with heuristic optimization
    fn add_opt(&self, rhs: &SymbolExpr, recursive: bool) -> Option<SymbolExpr> {
        if self.is_zero(recursive) {
            Some(rhs.clone())
        } else if rhs.is_zero(recursive) {
            Some(self.clone())
        } else {
            // if neg operation, call sub_opt
            if let SymbolExpr::Unary {
                op: UnaryOp::Neg,
                expr,
            } = rhs
            {
                return self.sub_opt(expr, recursive);
            }
            if self.is_value() || rhs.is_value() {
                if let Some(e) = self.add_values(rhs, recursive) {
                    return Some(e);
                }
            }
            if recursive {
                if let SymbolExpr::Binary {
                    op,
                    lhs: r_lhs,
                    rhs: r_rhs,
                } = rhs
                {
                    // recursive optimization for add and sub
                    if let BinaryOp::Add = &op {
                        if let Some(e) = self.add_opt(r_lhs, true) {
                            return match e.add_opt(r_rhs, true) {
                                Some(ee) => Some(ee),
                                None => Some(_add(e, r_rhs.as_ref().clone())),
                            };
                        }
                        if let Some(e) = self.add_opt(r_rhs, true) {
                            return match e.add_opt(r_lhs, true) {
                                Some(ee) => Some(ee),
                                None => Some(_add(e, r_lhs.as_ref().clone())),
                            };
                        }
                    }
                    if let BinaryOp::Sub = &op {
                        if let Some(e) = self.add_opt(r_lhs, true) {
                            return match e.sub_opt(r_rhs, true) {
                                Some(ee) => Some(ee),
                                None => Some(_sub(e, r_rhs.as_ref().clone())),
                            };
                        }
                        if let Some(e) = self.sub_opt(r_rhs, true) {
                            return match e.add_opt(r_lhs, true) {
                                Some(ee) => Some(ee),
                                None => Some(_add(e, r_lhs.as_ref().clone())),
                            };
                        }
                    }
                }
            }

            // optimization for each node type
            match self {
                SymbolExpr::Value(_) => None, // already optimized above
                SymbolExpr::Symbol(l) => match rhs {
                    SymbolExpr::Value(_) => Some(_add(rhs.clone(), self.clone())),
                    SymbolExpr::Symbol(r) => {
                        if r == l {
                            Some(_mul(SymbolExpr::Value(Value::Int(2)), self.clone()))
                        } else if r < l {
                            Some(_add(rhs.clone(), self.clone()))
                        } else {
                            None
                        }
                    }
                    SymbolExpr::Binary {
                        op: rop,
                        lhs: r_lhs,
                        rhs: r_rhs,
                    } => {
                        if rhs.is_value() {
                            return Some(_add(rhs.clone(), self.clone()));
                        }
                        match rop {
                            BinaryOp::Add => {
                                if let SymbolExpr::Symbol(s) = r_lhs.as_ref() {
                                    if l == s {
                                        return Some(_add(
                                            _mul(SymbolExpr::Value(Value::Int(2)), self.clone()),
                                            r_rhs.as_ref().clone(),
                                        ));
                                    }
                                }
                                if let SymbolExpr::Symbol(s) = r_rhs.as_ref() {
                                    if l == s {
                                        return Some(_add(
                                            _mul(SymbolExpr::Value(Value::Int(2)), self.clone()),
                                            r_lhs.as_ref().clone(),
                                        ));
                                    }
                                }
                                None
                            }
                            BinaryOp::Sub => {
                                if let SymbolExpr::Symbol(s) = r_lhs.as_ref() {
                                    if l == s {
                                        return Some(_add(
                                            _mul(SymbolExpr::Value(Value::Int(2)), self.clone()),
                                            r_rhs.as_ref().clone(),
                                        ));
                                    }
                                }
                                if let SymbolExpr::Symbol(s) = r_rhs.as_ref() {
                                    if l == s {
                                        return Some(r_lhs.as_ref().clone());
                                    }
                                }
                                None
                            }
                            BinaryOp::Mul => {
                                if let SymbolExpr::Symbol(s) = r_rhs.as_ref() {
                                    if l == s {
                                        if let Some(v) = SymbolExpr::Value(Value::Int(1))
                                            .add_values(r_lhs, recursive)
                                        {
                                            if v.is_zero(recursive) {
                                                return Some(SymbolExpr::Value(Value::Int(0)));
                                            } else {
                                                return Some(_mul(v, self.clone()));
                                            }
                                        }
                                    }
                                }
                                None
                            }
                            _ => None,
                        }
                    }
                    _ => None,
                },
                SymbolExpr::Unary { op, expr } => {
                    if let UnaryOp::Neg = op {
                        if let Some(e) = expr.sub_opt(rhs, recursive) {
                            return match e.neg_opt() {
                                Some(ee) => Some(ee),
                                None => Some(_neg(e)),
                            };
                        }
                    } else if let SymbolExpr::Unary {
                        op: rop,
                        expr: rexpr,
                    } = rhs
                    {
                        if op == rop {
                            if let Some(t) = expr.expand().add_opt(&rexpr.expand(), recursive) {
                                if t.is_zero(recursive) {
                                    return Some(SymbolExpr::Value(Value::Int(0)));
                                }
                            }
                        }
                    }

                    // swap nodes by sorting rule
                    if recursive {
                        match rhs {
                            SymbolExpr::Binary { op: rop, .. } => {
                                if let BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow = rop {
                                    if self > rhs {
                                        Some(_add(rhs.clone(), self.clone()))
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            }
                            _ => {
                                if self > rhs {
                                    Some(_add(rhs.clone(), self.clone()))
                                } else {
                                    None
                                }
                            }
                        }
                    } else {
                        None
                    }
                }
                SymbolExpr::Binary {
                    op,
                    lhs: l_lhs,
                    rhs: l_rhs,
                } => {
                    if let SymbolExpr::Binary {
                        op: rop,
                        lhs: r_lhs,
                        rhs: r_rhs,
                    } = rhs
                    {
                        if op == rop {
                            if let BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow = op {
                                if let Some(v) = l_lhs.add_values(r_lhs, false) {
                                    // check equality only for symbols if recursive is not true for performance
                                    let equal = if let (
                                        SymbolExpr::Symbol(l_rhs),
                                        SymbolExpr::Symbol(r_rhs),
                                    ) = (l_rhs.as_ref(), r_rhs.as_ref())
                                    {
                                        l_rhs == r_rhs
                                    } else if recursive {
                                        l_rhs.expand().string_id() == r_rhs.expand().string_id()
                                    } else {
                                        false
                                    };
                                    if equal {
                                        if v.is_zero(recursive) {
                                            return Some(SymbolExpr::Value(Value::Int(0)));
                                        }
                                        match op {
                                            BinaryOp::Mul => {
                                                return match v.mul_opt(l_rhs, recursive) {
                                                    Some(e) => Some(e),
                                                    None => Some(_mul(v, l_rhs.as_ref().clone())),
                                                };
                                            }
                                            BinaryOp::Div => {
                                                return match v.div_opt(l_rhs, recursive) {
                                                    Some(e) => Some(e),
                                                    None => Some(_div(v, l_rhs.as_ref().clone())),
                                                };
                                            }
                                            BinaryOp::Pow => {
                                                return Some(_pow(v, l_rhs.as_ref().clone()));
                                            }
                                            _ => (),
                                        }
                                    }
                                }
                            }
                            if recursive {
                                if let BinaryOp::Div = op {
                                    if l_rhs == r_rhs {
                                        if let Some(e) = l_lhs.add_opt(r_lhs, true) {
                                            if e.is_zero(recursive) {
                                                return Some(SymbolExpr::Value(Value::Int(0)));
                                            } else {
                                                return Some(_div(e, l_rhs.as_ref().clone()));
                                            }
                                        }
                                    } else {
                                        let l = l_lhs.as_ref() * r_rhs.as_ref();
                                        let r = r_lhs.as_ref() * l_rhs.as_ref();
                                        if let Some(e) = l.add_opt(&r, true) {
                                            if e.is_zero(recursive) {
                                                return Some(SymbolExpr::Value(Value::Int(0)));
                                            } else if let Some(d) = l_rhs.mul_opt(r_rhs, true) {
                                                return Some(_div(e, d));
                                            }
                                        }
                                    }
                                } else if let Some(e) = rhs.neg_opt() {
                                    if self.expand().string_id() == e.expand().string_id() {
                                        return Some(SymbolExpr::Value(Value::Int(0)));
                                    }
                                }
                            }
                        }
                    } else if let SymbolExpr::Symbol(r) = rhs {
                        if let (BinaryOp::Mul, SymbolExpr::Symbol(s)) = (op, l_rhs.as_ref()) {
                            if s == r {
                                if let Some(t) =
                                    l_lhs.add_values(&SymbolExpr::Value(Value::Int(1)), false)
                                {
                                    if t.is_zero(recursive) {
                                        return Some(SymbolExpr::Value(Value::Int(0)));
                                    } else if t.is_one(recursive) {
                                        return Some(l_rhs.as_ref().clone());
                                    } else if t.is_minus_one(recursive) {
                                        return Some(_neg(l_rhs.as_ref().clone()));
                                    } else {
                                        return Some(_mul(t, l_rhs.as_ref().clone()));
                                    }
                                }
                            }
                        }
                    }
                    if recursive {
                        if let BinaryOp::Add = op {
                            if let Some(e) = l_lhs.add_opt(rhs, true) {
                                return match e.add_opt(l_rhs, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_add(e, l_rhs.as_ref().clone())),
                                };
                            }
                            if let Some(e) = l_rhs.add_opt(rhs, true) {
                                return match l_lhs.add_opt(&e, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_add(l_lhs.as_ref().clone(), e)),
                                };
                            }
                        } else if let BinaryOp::Sub = op {
                            if let Some(e) = l_lhs.add_opt(rhs, true) {
                                return match e.sub_opt(l_rhs, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_sub(e, l_rhs.as_ref().clone())),
                                };
                            }
                            if let Some(e) = l_rhs.sub_opt(rhs, true) {
                                return match l_lhs.sub_opt(&e, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_sub(l_lhs.as_ref().clone(), e)),
                                };
                            }
                        }

                        // swap nodes by sorting rule
                        if let BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow = op {
                            return match rhs {
                                SymbolExpr::Binary { op: rop, .. } => {
                                    if let BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow = rop {
                                        if self > rhs {
                                            Some(_add(rhs.clone(), self.clone()))
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    }
                                }
                                _ => {
                                    if self > rhs {
                                        Some(_add(rhs.clone(), self.clone()))
                                    } else {
                                        None
                                    }
                                }
                            };
                        }
                    }
                    // put value on the top
                    if l_lhs.is_value() {
                        return match op {
                            BinaryOp::Add => match l_rhs.add_opt(rhs, recursive) {
                                Some(e) => Some(_add(l_lhs.as_ref().clone(), e)),
                                None => Some(_add(
                                    l_lhs.as_ref().clone(),
                                    _add(l_rhs.as_ref().clone(), rhs.clone()),
                                )),
                            },
                            BinaryOp::Sub => match l_rhs.sub_opt(rhs, recursive) {
                                Some(e) => Some(_sub(l_lhs.as_ref().clone(), e)),
                                None => Some(_sub(
                                    l_lhs.as_ref().clone(),
                                    _sub(l_rhs.as_ref().clone(), rhs.clone()),
                                )),
                            },
                            _ => None,
                        };
                    }
                    None
                }
            }
        }
    }

    /// Sub with heuristic optimization
    fn sub_opt(&self, rhs: &SymbolExpr, recursive: bool) -> Option<SymbolExpr> {
        if self.is_zero(recursive) {
            match rhs.neg_opt() {
                Some(e) => Some(e),
                None => Some(_neg(rhs.clone())),
            }
        } else if rhs.is_zero(recursive) {
            Some(self.clone())
        } else {
            // if neg, call add_opt
            if let SymbolExpr::Unary {
                op: UnaryOp::Neg,
                expr,
            } = rhs
            {
                return self.add_opt(expr, recursive);
            }
            if self.is_value() || rhs.is_value() {
                if let Some(e) = self.sub_values(rhs, recursive) {
                    return Some(e);
                }
            }
            if recursive {
                if let SymbolExpr::Binary {
                    op,
                    lhs: r_lhs,
                    rhs: r_rhs,
                } = rhs
                {
                    // recursive optimization for add and sub
                    if let BinaryOp::Add = &op {
                        if let Some(e) = self.sub_opt(r_lhs, true) {
                            return match e.sub_opt(r_rhs, true) {
                                Some(ee) => Some(ee),
                                None => Some(_sub(e, r_rhs.as_ref().clone())),
                            };
                        }
                        if let Some(e) = self.sub_opt(r_rhs, true) {
                            return match e.sub_opt(r_lhs, true) {
                                Some(ee) => Some(ee),
                                None => Some(_sub(e, r_lhs.as_ref().clone())),
                            };
                        }
                    }
                    if let BinaryOp::Sub = &op {
                        if let Some(e) = self.sub_opt(r_lhs, true) {
                            return match e.add_opt(r_rhs, true) {
                                Some(ee) => Some(ee),
                                None => Some(_add(e, r_rhs.as_ref().clone())),
                            };
                        }
                        if let Some(e) = self.add_opt(r_rhs, true) {
                            return match e.sub_opt(r_lhs, true) {
                                Some(ee) => Some(ee),
                                None => Some(_sub(e, r_lhs.as_ref().clone())),
                            };
                        }
                    }
                }
            }

            // optimization for each type
            match self {
                SymbolExpr::Value(_) => None, // already optimized above
                SymbolExpr::Symbol(l) => match &rhs {
                    SymbolExpr::Value(r) => Some(_add(SymbolExpr::Value(-r), self.clone())),
                    SymbolExpr::Symbol(r) => {
                        if r == l {
                            Some(SymbolExpr::Value(Value::Int(0)))
                        } else if r < l {
                            Some(_add(_neg(rhs.clone()), self.clone()))
                        } else {
                            None
                        }
                    }
                    SymbolExpr::Binary {
                        op: rop,
                        lhs: r_lhs,
                        rhs: r_rhs,
                    } => match rop {
                        BinaryOp::Add => {
                            if let SymbolExpr::Symbol(s) = r_lhs.as_ref() {
                                if l == s {
                                    return Some(_neg(r_rhs.as_ref().clone()));
                                }
                            }
                            if let SymbolExpr::Symbol(s) = r_rhs.as_ref() {
                                if l == s {
                                    return Some(_neg(r_lhs.as_ref().clone()));
                                }
                            }
                            None
                        }
                        BinaryOp::Sub => {
                            if let SymbolExpr::Symbol(s) = r_lhs.as_ref() {
                                if l == s {
                                    return Some(_neg(r_rhs.as_ref().clone()));
                                }
                            }
                            if let SymbolExpr::Symbol(s) = r_rhs.as_ref() {
                                if l == s {
                                    return Some(_sub(
                                        _mul(SymbolExpr::Value(Value::Int(2)), self.clone()),
                                        r_lhs.as_ref().clone(),
                                    ));
                                }
                            }
                            None
                        }
                        BinaryOp::Mul => {
                            if let SymbolExpr::Symbol(s) = r_rhs.as_ref() {
                                if l == s {
                                    if let Some(v) =
                                        SymbolExpr::Value(Value::Int(1)).sub_values(r_lhs, false)
                                    {
                                        if v.is_zero(recursive) {
                                            return Some(SymbolExpr::Value(Value::Int(0)));
                                        } else {
                                            return Some(_mul(v, self.clone()));
                                        }
                                    }
                                }
                            }
                            None
                        }
                        _ => None,
                    },
                    _ => None,
                },
                SymbolExpr::Unary { op, expr } => {
                    if let UnaryOp::Neg = op {
                        if let Some(e) = expr.add_opt(rhs, recursive) {
                            return match e.neg_opt() {
                                Some(ee) => Some(ee),
                                None => Some(_neg(e)),
                            };
                        }
                    }
                    if let SymbolExpr::Unary {
                        op: rop,
                        expr: rexpr,
                    } = rhs
                    {
                        if op == rop {
                            if let Some(t) = expr.expand().sub_opt(&rexpr.expand(), true) {
                                if t.is_zero(recursive) {
                                    return Some(SymbolExpr::Value(Value::Int(0)));
                                }
                            }
                        }
                    }

                    // swap nodes by sorting rule
                    if recursive {
                        match rhs {
                            SymbolExpr::Binary { op: rop, .. } => {
                                if let BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow = rop {
                                    if self > rhs {
                                        rhs.neg_opt().map(|e| _add(e, self.clone()))
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            }
                            _ => {
                                if self > rhs {
                                    rhs.neg_opt().map(|e| _add(e, self.clone()))
                                } else {
                                    None
                                }
                            }
                        }
                    } else {
                        None
                    }
                }
                SymbolExpr::Binary {
                    op,
                    lhs: l_lhs,
                    rhs: l_rhs,
                } => {
                    if let SymbolExpr::Binary {
                        op: rop,
                        lhs: r_lhs,
                        rhs: r_rhs,
                    } = rhs
                    {
                        if op == rop {
                            if let BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow = op {
                                if let Some(v) = l_lhs.sub_values(r_lhs, false) {
                                    // check equality only for symbols if recursive is not true for performance
                                    let equal = if let (
                                        SymbolExpr::Symbol(l_rhs),
                                        SymbolExpr::Symbol(r_rhs),
                                    ) = (l_rhs.as_ref(), r_rhs.as_ref())
                                    {
                                        l_rhs == r_rhs
                                    } else if recursive {
                                        l_rhs.expand().string_id() == r_rhs.expand().string_id()
                                    } else {
                                        false
                                    };
                                    if equal {
                                        if v.is_zero(recursive) {
                                            return Some(SymbolExpr::Value(Value::Int(0)));
                                        }
                                        match op {
                                            BinaryOp::Mul => {
                                                return match v.mul_opt(l_rhs, recursive) {
                                                    Some(e) => Some(e),
                                                    None => Some(_mul(v, l_rhs.as_ref().clone())),
                                                };
                                            }
                                            BinaryOp::Div => {
                                                return match v.div_opt(l_rhs, recursive) {
                                                    Some(e) => Some(e),
                                                    None => Some(_div(v, l_rhs.as_ref().clone())),
                                                };
                                            }
                                            BinaryOp::Pow => {
                                                return Some(_pow(v, l_rhs.as_ref().clone()));
                                            }
                                            _ => (),
                                        }
                                    }
                                }
                            }
                            if recursive {
                                if let BinaryOp::Div = op {
                                    if l_rhs == r_rhs {
                                        if let Some(e) = l_lhs.sub_opt(r_lhs, true) {
                                            if e.is_zero(recursive) {
                                                return Some(SymbolExpr::Value(Value::Int(0)));
                                            } else {
                                                return Some(_div(e, l_rhs.as_ref().clone()));
                                            }
                                        }
                                    } else {
                                        let l = l_lhs.as_ref() * r_rhs.as_ref();
                                        let r = r_lhs.as_ref() * l_rhs.as_ref();
                                        if let Some(e) = l.sub_opt(&r, true) {
                                            if e.is_zero(recursive) {
                                                return Some(SymbolExpr::Value(Value::Int(0)));
                                            } else if let Some(d) = l_rhs.mul_opt(r_rhs, true) {
                                                return Some(_div(e, d));
                                            }
                                        }
                                    }
                                } else if self.expand().string_id() == rhs.expand().string_id() {
                                    return Some(SymbolExpr::Value(Value::Int(0)));
                                }
                            }
                        }
                    } else if let SymbolExpr::Symbol(r) = rhs {
                        if let (BinaryOp::Mul, SymbolExpr::Symbol(s)) = (op, l_rhs.as_ref()) {
                            if s == r {
                                if let Some(t) =
                                    l_lhs.sub_values(&SymbolExpr::Value(Value::Int(1)), false)
                                {
                                    if t.is_zero(recursive) {
                                        return Some(SymbolExpr::Value(Value::Int(0)));
                                    } else if t.is_one(recursive) {
                                        return Some(l_rhs.as_ref().clone());
                                    } else if t.is_minus_one(recursive) {
                                        return Some(_neg(l_rhs.as_ref().clone()));
                                    } else {
                                        return Some(_mul(t, l_rhs.as_ref().clone()));
                                    }
                                }
                            }
                        }
                    }
                    if recursive {
                        if let BinaryOp::Add = op {
                            if let Some(e) = l_lhs.sub_opt(rhs, true) {
                                return match e.add_opt(l_rhs, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_add(e, l_rhs.as_ref().clone())),
                                };
                            }
                            if let Some(e) = l_rhs.sub_opt(rhs, true) {
                                return match l_lhs.add_opt(&e, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_add(l_lhs.as_ref().clone(), e)),
                                };
                            }
                        }
                        if let BinaryOp::Sub = op {
                            if let Some(e) = l_lhs.sub_opt(rhs, true) {
                                return match e.sub_opt(l_rhs, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_sub(e, l_rhs.as_ref().clone())),
                                };
                            }
                            if let Some(e) = l_rhs.add_opt(rhs, true) {
                                return match l_lhs.sub_opt(&e, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_sub(l_lhs.as_ref().clone(), e)),
                                };
                            }
                        }
                        // swap nodes by sorting rule
                        if let BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow = op {
                            return match rhs {
                                SymbolExpr::Binary { op: rop, .. } => {
                                    if let BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow = rop {
                                        if self > rhs {
                                            rhs.neg_opt().map(|e| _add(e, self.clone()))
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    }
                                }
                                _ => {
                                    if self > rhs {
                                        rhs.neg_opt().map(|e| _add(e, self.clone()))
                                    } else {
                                        None
                                    }
                                }
                            };
                        }
                    }
                    // put value on the top
                    if l_lhs.is_value() {
                        return match op {
                            BinaryOp::Add => match l_rhs.sub_opt(rhs, recursive) {
                                Some(e) => Some(_add(l_lhs.as_ref().clone(), e)),
                                None => Some(_add(
                                    l_lhs.as_ref().clone(),
                                    _sub(l_rhs.as_ref().clone(), rhs.clone()),
                                )),
                            },
                            BinaryOp::Sub => match l_rhs.add_opt(rhs, recursive) {
                                Some(e) => Some(_sub(l_lhs.as_ref().clone(), e)),
                                None => Some(_sub(
                                    l_lhs.as_ref().clone(),
                                    _add(l_rhs.as_ref().clone(), rhs.clone()),
                                )),
                            },
                            _ => None,
                        };
                    }
                    None
                }
            }
        }
    }

    /// Mul with heuristic optimization
    fn mul_opt(&self, rhs: &SymbolExpr, recursive: bool) -> Option<SymbolExpr> {
        if self.is_zero(recursive) || rhs.is_one(recursive) {
            match self.mul_values(rhs, recursive) {
                Some(e) => Some(e),
                None => Some(self.clone()),
            }
        } else if rhs.is_zero(recursive) || self.is_one(recursive) {
            match self.mul_values(rhs, recursive) {
                Some(e) => Some(e),
                None => Some(rhs.clone()),
            }
        } else if self.is_minus_one(recursive) {
            match self.mul_values(rhs, recursive) {
                Some(e) => Some(e),
                None => match rhs.neg_opt() {
                    Some(e) => Some(e),
                    None => Some(_neg(rhs.clone())),
                },
            }
        } else if rhs.is_minus_one(recursive) {
            match self.mul_values(rhs, recursive) {
                Some(e) => Some(e),
                None => match self.neg_opt() {
                    Some(e) => Some(e),
                    None => Some(_neg(self.clone())),
                },
            }
        } else {
            if let Some(v) = self.mul_values(rhs, recursive) {
                return Some(v);
            }

            if matches!(rhs, SymbolExpr::Value(_) | SymbolExpr::Symbol(_)) {
                if let SymbolExpr::Unary { .. } = self {
                    return match rhs.mul_opt(self, recursive) {
                        Some(e) => Some(e),
                        None => Some(_mul(rhs.clone(), self.clone())),
                    };
                }
            }

            if self.is_value() {
                return match rhs {
                    SymbolExpr::Value(_) => self.mul_values(rhs, recursive),
                    SymbolExpr::Unary {
                        op: UnaryOp::Neg,
                        expr,
                    } => {
                        let l = -self;
                        match l.mul_opt(expr, recursive) {
                            Some(e) => Some(e),
                            None => Some(_mul(l, expr.as_ref().clone())),
                        }
                    }
                    SymbolExpr::Binary { op, lhs: l, rhs: r } => {
                        if recursive {
                            match op {
                                BinaryOp::Mul => match self.mul_opt(l, recursive) {
                                    Some(e) => match e.mul_opt(r, recursive) {
                                        Some(ee) => Some(ee),
                                        None => Some(_mul(e, r.as_ref().clone())),
                                    },
                                    None => self
                                        .mul_opt(r, recursive)
                                        .map(|e| _mul(e, l.as_ref().clone())),
                                },
                                BinaryOp::Div => match self.mul_opt(l, recursive) {
                                    Some(e) => Some(_div(e, r.as_ref().clone())),
                                    None => self
                                        .div_opt(r, recursive)
                                        .map(|e| _mul(e, l.as_ref().clone())),
                                },
                                _ => None,
                            }
                        } else {
                            match op {
                                BinaryOp::Mul => {
                                    if let Some(v) = self.mul_values(l, recursive) {
                                        Some(_mul(v, r.as_ref().clone()))
                                    } else {
                                        self.mul_values(r, recursive)
                                            .map(|v| _mul(v, l.as_ref().clone()))
                                    }
                                }
                                BinaryOp::Div => {
                                    if let Some(v) = self.mul_values(l, recursive) {
                                        Some(_div(v, r.as_ref().clone()))
                                    } else {
                                        self.div_values(r, recursive)
                                            .map(|v| _mul(v, l.as_ref().clone()))
                                    }
                                }
                                _ => None,
                            }
                        }
                    }
                    _ => None,
                };
            }

            match self {
                SymbolExpr::Value(_) => None, // already optimized above
                SymbolExpr::Symbol(e) => match rhs {
                    SymbolExpr::Value(_) => Some(_mul(rhs.clone(), self.clone())),
                    SymbolExpr::Symbol(r) => {
                        if r < e {
                            Some(_mul(rhs.clone(), self.clone()))
                        } else {
                            None
                        }
                    }
                    SymbolExpr::Unary {
                        op: UnaryOp::Neg,
                        expr,
                    } => match expr.as_ref() {
                        SymbolExpr::Value(v) => Some(_mul(SymbolExpr::Value(-v), self.clone())),
                        SymbolExpr::Symbol(s) => {
                            if s < e {
                                Some(_neg(_mul(expr.as_ref().clone(), self.clone())))
                            } else {
                                Some(_neg(_mul(self.clone(), expr.as_ref().clone())))
                            }
                        }
                        SymbolExpr::Binary { .. } => match self.mul_opt(expr, recursive) {
                            Some(e) => match e.neg_opt() {
                                Some(ee) => Some(ee),
                                None => Some(_neg(e)),
                            },
                            None => None,
                        },
                        _ => None,
                    },
                    _ => None,
                },
                SymbolExpr::Unary { op, expr } => match op {
                    UnaryOp::Neg => match expr.mul_opt(rhs, recursive) {
                        Some(e) => match e.neg_opt() {
                            Some(ee) => Some(ee),
                            None => Some(_neg(e)),
                        },
                        None => Some(_neg(_mul(expr.as_ref().clone(), rhs.clone()))),
                    },
                    UnaryOp::Abs => match rhs {
                        SymbolExpr::Unary {
                            op: UnaryOp::Abs,
                            expr: rexpr,
                        } => match expr.mul_opt(rexpr, recursive) {
                            Some(e) => Some(SymbolExpr::Unary {
                                op: UnaryOp::Abs,
                                expr: Arc::new(e),
                            }),
                            None => Some(SymbolExpr::Unary {
                                op: UnaryOp::Abs,
                                expr: Arc::new(_mul(expr.as_ref().clone(), rexpr.as_ref().clone())),
                            }),
                        },
                        _ => None,
                    },
                    _ => None,
                },
                SymbolExpr::Binary {
                    op,
                    lhs: l_lhs,
                    rhs: l_rhs,
                } => {
                    if recursive {
                        if let SymbolExpr::Binary {
                            op: rop,
                            lhs: r_lhs,
                            rhs: r_rhs,
                        } = rhs
                        {
                            if let BinaryOp::Mul = &rop {
                                if let Some(e) = self.mul_opt(r_lhs, true) {
                                    return match e.mul_opt(r_rhs, true) {
                                        Some(ee) => Some(ee),
                                        None => Some(_mul(e, r_rhs.as_ref().clone())),
                                    };
                                }
                                if let Some(e) = self.mul_opt(r_rhs, true) {
                                    return match e.mul_opt(r_lhs, true) {
                                        Some(ee) => Some(ee),
                                        None => Some(_mul(e, r_lhs.as_ref().clone())),
                                    };
                                }
                            }
                            if let BinaryOp::Div = &rop {
                                if let Some(e) = self.mul_opt(r_lhs, true) {
                                    return match e.div_opt(r_rhs, true) {
                                        Some(ee) => Some(ee),
                                        None => Some(_div(e, r_rhs.as_ref().clone())),
                                    };
                                }
                                if let Some(e) = self.div_opt(r_rhs, true) {
                                    return match e.mul_opt(r_lhs, true) {
                                        Some(ee) => Some(ee),
                                        None => Some(_mul(e, r_lhs.as_ref().clone())),
                                    };
                                }
                            }
                        }

                        if let BinaryOp::Mul = &op {
                            if let Some(e) = l_lhs.mul_opt(rhs, true) {
                                return match e.mul_opt(l_rhs, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_mul(e, l_rhs.as_ref().clone())),
                                };
                            }
                            if let Some(e) = l_rhs.mul_opt(rhs, true) {
                                return match l_lhs.mul_opt(&e, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_mul(l_lhs.as_ref().clone(), e)),
                                };
                            }
                        } else if let BinaryOp::Div = &op {
                            if let Some(e) = l_lhs.mul_opt(rhs, true) {
                                return match e.div_opt(l_rhs, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_div(e, l_rhs.as_ref().clone())),
                                };
                            }
                            if let Some(e) = rhs.div_opt(l_rhs, true) {
                                return match l_lhs.mul_opt(&e, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_mul(l_lhs.as_ref().clone(), e)),
                                };
                            }
                        }
                        None
                    } else {
                        if rhs.is_value() {
                            if let BinaryOp::Mul = op {
                                if let Some(v) = l_lhs.mul_values(rhs, recursive) {
                                    return Some(_mul(v, l_rhs.as_ref().clone()));
                                } else if let Some(v) = l_rhs.mul_values(rhs, recursive) {
                                    return Some(_mul(v, l_lhs.as_ref().clone()));
                                }
                            } else if let BinaryOp::Div = op {
                                if let Some(v) = l_lhs.mul_values(rhs, recursive) {
                                    return Some(_div(v, l_rhs.as_ref().clone()));
                                } else if let Some(v) = l_rhs.div_values(rhs, recursive) {
                                    return Some(_mul(v, l_lhs.as_ref().clone()));
                                }
                            }
                            return Some(_mul(rhs.clone(), self.clone()));
                        }
                        match rhs {
                            SymbolExpr::Value(_) => None, // already optimized avobe
                            SymbolExpr::Binary {
                                op: rop,
                                lhs: r_lhs,
                                rhs: r_rhs,
                            } => match (op, rop) {
                                (BinaryOp::Mul, BinaryOp::Mul) => {
                                    if let Some(v) = l_lhs.mul_values(r_lhs, recursive) {
                                        Some(_mul(
                                            v,
                                            _mul(l_rhs.as_ref().clone(), r_rhs.as_ref().clone()),
                                        ))
                                    } else if let Some(v) = l_lhs.mul_values(r_rhs, recursive) {
                                        Some(_mul(
                                            v,
                                            _mul(l_rhs.as_ref().clone(), r_lhs.as_ref().clone()),
                                        ))
                                    } else if let Some(v) = l_rhs.mul_values(r_lhs, recursive) {
                                        Some(_mul(
                                            v,
                                            _mul(l_lhs.as_ref().clone(), r_rhs.as_ref().clone()),
                                        ))
                                    } else {
                                        l_rhs.mul_values(r_rhs, recursive).map(|v| {
                                            _mul(
                                                v,
                                                _mul(
                                                    l_lhs.as_ref().clone(),
                                                    r_lhs.as_ref().clone(),
                                                ),
                                            )
                                        })
                                    }
                                }
                                (BinaryOp::Mul, BinaryOp::Div) => {
                                    if let Some(v) = l_lhs.mul_values(r_lhs, recursive) {
                                        Some(_mul(
                                            v,
                                            _div(l_rhs.as_ref().clone(), r_rhs.as_ref().clone()),
                                        ))
                                    } else if let Some(v) = l_lhs.div_values(r_rhs, recursive) {
                                        Some(_mul(
                                            v,
                                            _mul(l_rhs.as_ref().clone(), r_lhs.as_ref().clone()),
                                        ))
                                    } else if let Some(v) = l_rhs.mul_values(r_lhs, recursive) {
                                        Some(_mul(
                                            v,
                                            _div(l_lhs.as_ref().clone(), r_rhs.as_ref().clone()),
                                        ))
                                    } else {
                                        l_rhs.div_values(r_rhs, recursive).map(|v| {
                                            _mul(
                                                v,
                                                _mul(
                                                    l_lhs.as_ref().clone(),
                                                    r_lhs.as_ref().clone(),
                                                ),
                                            )
                                        })
                                    }
                                }
                                (BinaryOp::Div, BinaryOp::Mul) => {
                                    if let Some(v) = l_lhs.mul_values(r_lhs, recursive) {
                                        Some(_mul(
                                            v,
                                            _div(r_rhs.as_ref().clone(), l_rhs.as_ref().clone()),
                                        ))
                                    } else if let Some(v) = l_lhs.mul_values(r_rhs, recursive) {
                                        Some(_mul(
                                            v,
                                            _div(r_lhs.as_ref().clone(), l_rhs.as_ref().clone()),
                                        ))
                                    } else if let Some(v) = r_lhs.div_values(l_rhs, recursive) {
                                        Some(_mul(
                                            v,
                                            _mul(l_lhs.as_ref().clone(), r_rhs.as_ref().clone()),
                                        ))
                                    } else {
                                        r_rhs.mul_values(l_rhs, recursive).map(|v| {
                                            _mul(
                                                v,
                                                _mul(
                                                    l_lhs.as_ref().clone(),
                                                    r_lhs.as_ref().clone(),
                                                ),
                                            )
                                        })
                                    }
                                }
                                (BinaryOp::Div, BinaryOp::Div) => {
                                    if let Some(v) = l_lhs.mul_values(r_lhs, recursive) {
                                        Some(_div(
                                            v,
                                            _mul(l_rhs.as_ref().clone(), r_rhs.as_ref().clone()),
                                        ))
                                    } else if let Some(v) = l_lhs.div_values(r_rhs, recursive) {
                                        Some(_mul(
                                            v,
                                            _div(r_lhs.as_ref().clone(), l_rhs.as_ref().clone()),
                                        ))
                                    } else if let Some(v) = r_lhs.div_values(l_rhs, recursive) {
                                        Some(_mul(
                                            v,
                                            _div(r_lhs.as_ref().clone(), l_rhs.as_ref().clone()),
                                        ))
                                    } else {
                                        l_rhs.mul_values(r_rhs, recursive).map(|v| {
                                            _mul(
                                                v.rcp(),
                                                _mul(
                                                    l_lhs.as_ref().clone(),
                                                    r_lhs.as_ref().clone(),
                                                ),
                                            )
                                        })
                                    }
                                }
                                (_, _) => None,
                            },
                            _ => None,
                        }
                    }
                }
            }
        }
    }
    /// expand with optimization for mul operation
    fn mul_expand(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        if let SymbolExpr::Binary {
            op: rop,
            lhs: r_lhs,
            rhs: r_rhs,
        } = rhs
        {
            if let BinaryOp::Add | BinaryOp::Sub = &rop {
                let el = match self.mul_expand(r_lhs) {
                    Some(e) => e,
                    None => match self.mul_opt(r_lhs, true) {
                        Some(e) => e,
                        None => _mul(self.clone(), r_lhs.as_ref().clone()),
                    },
                };
                let er = match self.mul_expand(r_rhs) {
                    Some(e) => e,
                    None => match self.mul_opt(r_rhs, true) {
                        Some(e) => e,
                        None => _mul(self.clone(), r_rhs.as_ref().clone()),
                    },
                };
                return match &rop {
                    BinaryOp::Sub => match el.sub_opt(&er, true) {
                        Some(e) => Some(e),
                        None => Some(_sub(el, er)),
                    },
                    _ => match el.add_opt(&er, true) {
                        Some(e) => Some(e),
                        None => Some(_add(el, er)),
                    },
                };
            }
            if let BinaryOp::Mul = &rop {
                return match self.mul_expand(r_lhs) {
                    Some(e) => match e.mul_expand(r_rhs) {
                        Some(ee) => Some(ee),
                        None => Some(_mul(e, r_rhs.as_ref().clone())),
                    },
                    None => self
                        .mul_expand(r_rhs)
                        .map(|e| _mul(e, r_lhs.as_ref().clone())),
                };
            }
            if let BinaryOp::Div = &rop {
                return match self.mul_expand(r_lhs) {
                    Some(e) => match e.mul_expand(r_rhs) {
                        Some(ee) => Some(ee),
                        None => Some(_mul(e, r_rhs.as_ref().clone())),
                    },
                    None => self
                        .div_expand(r_rhs)
                        .map(|e| _div(e, r_lhs.as_ref().clone())),
                };
            }
        }
        if let SymbolExpr::Unary {
            op: UnaryOp::Neg,
            expr: rexpr,
        } = rhs
        {
            return match self.mul_expand(rexpr) {
                Some(e) => match e.neg_opt() {
                    Some(ee) => Some(ee),
                    None => Some(_neg(e)),
                },
                None => match self.mul_opt(rexpr, true) {
                    Some(e) => match e.neg_opt() {
                        Some(ee) => Some(ee),
                        None => Some(_neg(e)),
                    },
                    None => Some(_neg(_mul(self.clone(), rexpr.as_ref().clone()))),
                },
            };
        }

        match self {
            SymbolExpr::Unary {
                op: UnaryOp::Neg,
                expr,
            } => match expr.mul_expand(rhs) {
                Some(e) => match e.neg_opt() {
                    Some(ee) => Some(ee),
                    None => Some(_neg(e)),
                },
                None => match expr.mul_opt(rhs, true) {
                    Some(e) => match e.neg_opt() {
                        Some(ee) => Some(ee),
                        None => Some(_neg(e)),
                    },
                    None => None,
                },
            },
            SymbolExpr::Binary {
                op,
                lhs: l_lhs,
                rhs: l_rhs,
            } => match &op {
                BinaryOp::Add | BinaryOp::Sub => {
                    let l = match l_lhs.mul_expand(rhs) {
                        Some(e) => e,
                        None => match l_lhs.mul_opt(rhs, true) {
                            Some(e) => e,
                            None => _mul(l_lhs.as_ref().clone(), rhs.clone()),
                        },
                    };
                    let r = match l_rhs.mul_expand(rhs) {
                        Some(e) => e,
                        None => match l_rhs.mul_opt(rhs, true) {
                            Some(e) => e,
                            None => _mul(l_rhs.as_ref().clone(), rhs.clone()),
                        },
                    };
                    match &op {
                        BinaryOp::Sub => match l.sub_opt(&r, true) {
                            Some(e) => Some(e),
                            None => Some(_sub(l, r)),
                        },
                        _ => match l.add_opt(&r, true) {
                            Some(e) => Some(e),
                            None => Some(_add(l, r)),
                        },
                    }
                }
                BinaryOp::Mul => match l_lhs.mul_expand(rhs) {
                    Some(e) => match e.mul_expand(l_rhs) {
                        Some(ee) => Some(ee),
                        None => match e.mul_opt(l_rhs, true) {
                            Some(ee) => Some(ee),
                            None => Some(_mul(e, l_rhs.as_ref().clone())),
                        },
                    },
                    None => match l_rhs.mul_expand(rhs) {
                        Some(e) => match l_lhs.mul_expand(&e) {
                            Some(ee) => Some(ee),
                            None => match l_lhs.mul_opt(&e, true) {
                                Some(ee) => Some(ee),
                                None => Some(_mul(l_lhs.as_ref().clone(), e)),
                            },
                        },
                        None => None,
                    },
                },
                BinaryOp::Div => match l_lhs.div_expand(rhs) {
                    Some(e) => Some(_div(e, l_rhs.as_ref().clone())),
                    None => l_rhs
                        .div_expand(rhs)
                        .map(|e| _div(l_lhs.as_ref().clone(), e)),
                },
                _ => None,
            },
            _ => None,
        }
    }

    /// Div with heuristic optimization
    fn div_opt(&self, rhs: &SymbolExpr, recursive: bool) -> Option<SymbolExpr> {
        if rhs.is_zero(recursive) {
            // return inf to detect divide by zero without panic
            Some(SymbolExpr::Value(Value::Real(f64::INFINITY)))
        } else if rhs.is_one(recursive) {
            match self.div_values(rhs, recursive) {
                Some(e) => Some(e),
                None => Some(self.clone()),
            }
        } else if rhs.is_minus_one(recursive) {
            match self.div_values(rhs, recursive) {
                Some(e) => Some(e),
                None => match self.neg_opt() {
                    Some(e) => Some(e),
                    None => Some(_neg(self.clone())),
                },
            }
        } else {
            if let SymbolExpr::Value(v) = rhs {
                if let Value::Int(_) | Value::Rational { .. } = v {
                    let t = v.rcp();
                    return match self.mul_opt(&SymbolExpr::Value(t), recursive) {
                        Some(e) => Some(e),
                        None => Some(_mul(SymbolExpr::Value(t), self.clone())),
                    };
                } else if let Value::Real(r) = v {
                    let t = 1.0 / r;
                    if &(1.0 / t) == r {
                        return match self.mul_opt(&SymbolExpr::Value(Value::Real(t)), recursive) {
                            Some(e) => Some(e),
                            None => Some(_mul(SymbolExpr::Value(Value::Real(t)), self.clone())),
                        };
                    }
                }
            }
            if let Some(v) = self.div_values(rhs, recursive) {
                return Some(v);
            }

            if self.is_value() {
                return match rhs {
                    SymbolExpr::Value(_) => self.div_values(rhs, recursive),
                    SymbolExpr::Unary {
                        op: UnaryOp::Neg,
                        expr,
                    } => {
                        if recursive {
                            self.div_opt(expr, recursive).map(_neg)
                        } else {
                            None
                        }
                    }
                    SymbolExpr::Binary { op, lhs: l, rhs: r } => match op {
                        BinaryOp::Mul => {
                            if let Some(v) = self.div_values(l, recursive) {
                                Some(_div(v, r.as_ref().clone()))
                            } else {
                                self.div_values(r, recursive)
                                    .map(|v| _div(v, l.as_ref().clone()))
                            }
                        }
                        BinaryOp::Div => {
                            if let Some(v) = self.div_values(l, recursive) {
                                Some(_mul(v, r.as_ref().clone()))
                            } else {
                                self.mul_values(r, recursive)
                                    .map(|v| _div(v, l.as_ref().clone()))
                            }
                        }
                        _ => None,
                    },
                    _ => None,
                };
            }

            match self {
                SymbolExpr::Value(_) => None, // already optimized above
                SymbolExpr::Symbol(_) => None,
                SymbolExpr::Unary { op, expr } => match op {
                    UnaryOp::Neg => match expr.div_opt(rhs, recursive) {
                        Some(e) => match e.neg_opt() {
                            Some(ee) => Some(ee),
                            None => Some(_neg(e)),
                        },
                        None => Some(_neg(_div(expr.as_ref().clone(), rhs.clone()))),
                    },
                    UnaryOp::Abs => match rhs {
                        SymbolExpr::Unary {
                            op: UnaryOp::Abs,
                            expr: rexpr,
                        } => match expr.div_opt(rexpr, recursive) {
                            Some(e) => Some(SymbolExpr::Unary {
                                op: UnaryOp::Abs,
                                expr: Arc::new(e),
                            }),
                            None => Some(SymbolExpr::Unary {
                                op: UnaryOp::Abs,
                                expr: Arc::new(_div(expr.as_ref().clone(), rexpr.as_ref().clone())),
                            }),
                        },
                        _ => None,
                    },
                    _ => None,
                },
                SymbolExpr::Binary {
                    op,
                    lhs: l_lhs,
                    rhs: l_rhs,
                } => {
                    if recursive {
                        if let SymbolExpr::Binary {
                            op: rop,
                            lhs: r_lhs,
                            rhs: r_rhs,
                        } = rhs
                        {
                            if let BinaryOp::Mul = &rop {
                                if let Some(e) = self.div_opt(r_lhs, true) {
                                    return match e.div_opt(r_rhs, true) {
                                        Some(ee) => Some(ee),
                                        None => Some(_div(e, r_rhs.as_ref().clone())),
                                    };
                                }
                                if let Some(e) = self.div_opt(r_rhs, true) {
                                    return match e.div_opt(r_lhs, true) {
                                        Some(ee) => Some(ee),
                                        None => Some(_div(e, r_lhs.as_ref().clone())),
                                    };
                                }
                            }
                            if let BinaryOp::Div = &rop {
                                if let Some(e) = self.mul_opt(r_rhs, true) {
                                    return match e.div_opt(r_lhs, true) {
                                        Some(ee) => Some(ee),
                                        None => Some(_div(e, r_lhs.as_ref().clone())),
                                    };
                                }
                                if let Some(e) = self.div_opt(r_lhs, true) {
                                    return match e.mul_opt(r_rhs, true) {
                                        Some(ee) => Some(ee),
                                        None => Some(_mul(e, r_rhs.as_ref().clone())),
                                    };
                                }
                            }
                        }

                        if let BinaryOp::Mul = &op {
                            if let Some(e) = l_lhs.div_opt(rhs, true) {
                                return match e.mul_opt(l_rhs, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_mul(e, l_rhs.as_ref().clone())),
                                };
                            }
                            if let Some(e) = l_rhs.div_opt(rhs, true) {
                                return match l_lhs.mul_opt(&e, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_mul(l_lhs.as_ref().clone(), e)),
                                };
                            }
                        } else if let BinaryOp::Div = &op {
                            if let Some(e) = l_rhs.mul_opt(rhs, true) {
                                return match l_lhs.div_opt(&e, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_div(l_lhs.as_ref().clone(), e)),
                                };
                            }
                            if let Some(e) = l_lhs.div_opt(rhs, true) {
                                return match e.div_opt(l_rhs, true) {
                                    Some(ee) => Some(ee),
                                    None => Some(_div(e, l_rhs.as_ref().clone())),
                                };
                            }
                        }
                        None
                    } else {
                        if rhs.is_value() {
                            if let BinaryOp::Mul = op {
                                if let Some(v) = l_lhs.div_values(rhs, recursive) {
                                    return Some(_mul(v, l_rhs.as_ref().clone()));
                                } else if let Some(v) = l_rhs.div_values(rhs, recursive) {
                                    return Some(_mul(v, l_lhs.as_ref().clone()));
                                }
                            } else if let BinaryOp::Div = op {
                                if let Some(v) = l_lhs.div_values(rhs, recursive) {
                                    return Some(_mul(v, l_rhs.as_ref().clone()));
                                } else if let Some(v) = l_rhs.mul_values(rhs, recursive) {
                                    return Some(_div(v, l_lhs.as_ref().clone()));
                                }
                            }
                            return Some(_mul(rhs.rcp(), self.clone()));
                        }
                        match rhs {
                            SymbolExpr::Value(_) => None, // already optimized avobe
                            SymbolExpr::Binary {
                                op: rop,
                                lhs: r_lhs,
                                rhs: r_rhs,
                            } => {
                                if let Some(v) = l_lhs.div_values(r_lhs, recursive) {
                                    match (op, rop) {
                                        (BinaryOp::Mul, BinaryOp::Mul) => Some(_mul(
                                            v,
                                            _div(l_rhs.as_ref().clone(), r_rhs.as_ref().clone()),
                                        )),
                                        (BinaryOp::Mul, BinaryOp::Div) => Some(_mul(
                                            v,
                                            _div(r_rhs.as_ref().clone(), l_rhs.as_ref().clone()),
                                        )),
                                        (BinaryOp::Div, BinaryOp::Mul) => Some(_div(
                                            v,
                                            _div(r_rhs.as_ref().clone(), l_rhs.as_ref().clone()),
                                        )),
                                        (BinaryOp::Div, BinaryOp::Div) => Some(_div(
                                            v,
                                            _mul(l_rhs.as_ref().clone(), r_rhs.as_ref().clone()),
                                        )),
                                        (_, _) => None,
                                    }
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        }
                    }
                }
            }
        }
    }

    /// expand with optimization for div operation
    fn div_expand(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match self {
            SymbolExpr::Unary { op, expr } => match op {
                UnaryOp::Neg => match expr.div_expand(rhs) {
                    Some(e) => match e.neg_opt() {
                        Some(ee) => Some(ee),
                        None => Some(_neg(e)),
                    },
                    None => match expr.div_opt(rhs, true) {
                        Some(e) => match e.neg_opt() {
                            Some(ee) => Some(ee),
                            None => Some(_neg(e)),
                        },
                        None => None,
                    },
                },
                _ => None,
            },
            SymbolExpr::Binary {
                op,
                lhs: l_lhs,
                rhs: l_rhs,
            } => match &op {
                BinaryOp::Add | BinaryOp::Sub => {
                    let l = match l_lhs.div_expand(rhs) {
                        Some(e) => e,
                        None => match l_lhs.div_opt(rhs, true) {
                            Some(e) => e,
                            None => _div(l_lhs.as_ref().clone(), rhs.clone()),
                        },
                    };
                    let r = match l_rhs.div_expand(rhs) {
                        Some(e) => e,
                        None => match l_rhs.div_opt(rhs, true) {
                            Some(e) => e,
                            None => _div(l_rhs.as_ref().clone(), rhs.clone()),
                        },
                    };
                    match &op {
                        BinaryOp::Sub => match l.sub_opt(&r, true) {
                            Some(e) => Some(e),
                            None => Some(_sub(l, r)),
                        },
                        _ => match l.add_opt(&r, true) {
                            Some(e) => Some(e),
                            None => Some(_add(l, r)),
                        },
                    }
                }
                _ => None,
            },
            _ => self.div_opt(rhs, true),
        }
    }

    /// optimization for neg
    fn neg_opt(&self) -> Option<SymbolExpr> {
        match self {
            SymbolExpr::Value(v) => Some(SymbolExpr::Value(-v)),
            SymbolExpr::Unary {
                op: UnaryOp::Neg,
                expr,
            } => Some(expr.as_ref().clone()),
            SymbolExpr::Binary { op, lhs, rhs } => match &op {
                BinaryOp::Add => match lhs.neg_opt() {
                    Some(ln) => match rhs.neg_opt() {
                        Some(rn) => Some(_add(ln, rn)),
                        None => Some(_sub(ln, rhs.as_ref().clone())),
                    },
                    None => match rhs.neg_opt() {
                        Some(rn) => Some(_add(_neg(lhs.as_ref().clone()), rn)),
                        None => Some(_sub(_neg(lhs.as_ref().clone()), rhs.as_ref().clone())),
                    },
                },
                BinaryOp::Sub => match lhs.neg_opt() {
                    Some(ln) => Some(_add(ln, rhs.as_ref().clone())),
                    None => Some(_add(_neg(lhs.as_ref().clone()), rhs.as_ref().clone())),
                },
                BinaryOp::Mul => match lhs.neg_opt() {
                    Some(ln) => Some(_mul(ln, rhs.as_ref().clone())),
                    None => rhs.neg_opt().map(|rn| _mul(lhs.as_ref().clone(), rn)),
                },
                BinaryOp::Div => match lhs.neg_opt() {
                    Some(ln) => Some(_div(ln, rhs.as_ref().clone())),
                    None => rhs.neg_opt().map(|rn| _div(lhs.as_ref().clone(), rn)),
                },
                _ => None,
            },
            _ => None,
        }
    }

    /// optimize the equation
    pub fn optimize(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(_) => self.clone(),
            SymbolExpr::Symbol(_) => self.clone(),
            SymbolExpr::Unary { op, expr } => {
                let opt = expr.optimize();
                match op {
                    UnaryOp::Neg => match opt.neg_opt() {
                        Some(e) => e,
                        None => _neg(opt),
                    },
                    _ => SymbolExpr::Unary {
                        op: op.clone(),
                        expr: Arc::new(opt),
                    },
                }
            }
            SymbolExpr::Binary { op, lhs, rhs } => {
                let opt_lhs = lhs.optimize();
                let opt_rhs = rhs.optimize();
                match op {
                    BinaryOp::Add => match opt_lhs.add_opt(&opt_rhs, true) {
                        Some(e) => e,
                        None => _add(opt_lhs, opt_rhs),
                    },
                    BinaryOp::Sub => match opt_lhs.sub_opt(&opt_rhs, true) {
                        Some(e) => e,
                        None => _sub(opt_lhs, opt_rhs),
                    },
                    BinaryOp::Mul => match opt_lhs.mul_opt(&opt_rhs, true) {
                        Some(e) => e,
                        None => _mul(opt_lhs, opt_rhs),
                    },
                    BinaryOp::Div => match opt_lhs.div_opt(&opt_rhs, true) {
                        Some(e) => e,
                        None => _div(opt_lhs, opt_rhs),
                    },
                    BinaryOp::Pow => _pow(opt_lhs, opt_rhs),
                }
            }
        }
    }

    // convert sympy compatible format
    pub fn sympify(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Symbol { .. } => self.clone(),
            SymbolExpr::Value(e) => e.sympify(),
            SymbolExpr::Unary { op, expr } => SymbolExpr::Unary {
                op: op.clone(),
                expr: Arc::new(expr.sympify()),
            },
            SymbolExpr::Binary { op, lhs, rhs } => SymbolExpr::Binary {
                op: op.clone(),
                lhs: Arc::new(lhs.sympify()),
                rhs: Arc::new(rhs.sympify()),
            },
        }
    }

    fn repr(&self, with_uuid: bool) -> String {
        match self {
            SymbolExpr::Symbol(e) => e.repr(with_uuid),
            SymbolExpr::Value(e) => e.to_string(),
            SymbolExpr::Unary { op, expr } => {
                let s = expr.repr(with_uuid);
                match op {
                    UnaryOp::Abs => format!("abs({s})"),
                    UnaryOp::Neg => match expr.as_ref() {
                        SymbolExpr::Value(e) => (-e).to_string(),
                        SymbolExpr::Binary {
                            op: BinaryOp::Add | BinaryOp::Sub,
                            ..
                        } => format!("-({s})"),
                        _ => format!("-{s}"),
                    },
                    UnaryOp::Sin => format!("sin({s})"),
                    UnaryOp::Asin => format!("asin({s})"),
                    UnaryOp::Cos => format!("cos({s})"),
                    UnaryOp::Acos => format!("acos({s})"),
                    UnaryOp::Tan => format!("tan({s})"),
                    UnaryOp::Atan => format!("atan({s})"),
                    UnaryOp::Exp => format!("exp({s})"),
                    UnaryOp::Log => format!("log({s})"),
                    UnaryOp::Sign => format!("sign({s})"),
                    UnaryOp::Conj => format!("conj({s})"),
                }
            }
            SymbolExpr::Binary { op, lhs, rhs } => {
                let s_lhs = lhs.repr(with_uuid);
                let s_rhs = rhs.repr(with_uuid);
                let op_lhs = match lhs.as_ref() {
                    SymbolExpr::Binary { op: lop, .. } => {
                        matches!(lop, BinaryOp::Add | BinaryOp::Sub)
                    }
                    SymbolExpr::Value(e) => match e {
                        Value::Real(v) => *v < 0.0,
                        Value::Int(v) => *v < 0,
                        Value::Complex(_) => true,
                        Value::Rational {
                            numerator,
                            denominator,
                        } => numerator * denominator < 0,
                    },
                    _ => false,
                };
                let op_rhs = match rhs.as_ref() {
                    SymbolExpr::Binary { op: rop, .. } => match rop {
                        BinaryOp::Add | BinaryOp::Sub => true,
                        _ => matches!(op, BinaryOp::Div),
                    },
                    SymbolExpr::Value(e) => match e {
                        Value::Real(v) => *v < 0.0,
                        Value::Int(v) => *v < 0,
                        Value::Complex(_) => true,
                        Value::Rational {
                            numerator,
                            denominator,
                        } => numerator * denominator < 0,
                    },
                    _ => false,
                };

                match op {
                    BinaryOp::Add => match rhs.as_ref() {
                        SymbolExpr::Unary {
                            op: UnaryOp::Neg,
                            expr: _,
                        } => {
                            if s_rhs.as_str().char_indices().nth(0).unwrap().1 == '-' {
                                format!("{s_lhs} {s_rhs}")
                            } else {
                                format!("{s_lhs} + {s_rhs}")
                            }
                        }
                        _ => format!("{s_lhs} + {s_rhs}"),
                    },
                    BinaryOp::Sub => match rhs.as_ref() {
                        SymbolExpr::Unary {
                            op: UnaryOp::Neg,
                            expr: _,
                        } => {
                            if s_rhs.as_str().char_indices().nth(0).unwrap().1 == '-' {
                                let st = s_rhs.char_indices().nth(0).unwrap().0;
                                let ed = s_rhs.char_indices().nth(1).unwrap().0;
                                let s_rhs_new: &str = &s_rhs.as_str()[st..ed];
                                format!("{s_lhs} + {s_rhs_new}")
                            } else if op_rhs {
                                format!("{s_lhs} -({s_rhs})")
                            } else {
                                format!("{s_lhs} - {s_rhs}")
                            }
                        }
                        _ => {
                            if op_rhs {
                                format!("{s_lhs} -({s_rhs})")
                            } else {
                                format!("{s_lhs} - {s_rhs}")
                            }
                        }
                    },
                    BinaryOp::Mul => {
                        if let Some((numerator, denominator)) = lhs.rational() {
                            if numerator == 1 {
                                if op_rhs {
                                    if denominator == 1 {
                                        format!("({s_rhs})")
                                    } else if denominator < 0 {
                                        format!("({s_rhs})/({denominator})")
                                    } else {
                                        format!("({s_rhs})/{denominator}")
                                    }
                                } else if denominator == 1 {
                                    s_rhs.to_string()
                                } else if denominator < 0 {
                                    format!("{s_rhs}/({denominator})")
                                } else {
                                    format!("{s_rhs}/{denominator}")
                                }
                            } else if numerator < 0 {
                                if op_rhs {
                                    if denominator == 1 {
                                        format!("({numerator})*({s_rhs})")
                                    } else if denominator < 0 {
                                        format!("({numerator})*({s_rhs})/({denominator})")
                                    } else {
                                        format!("({numerator})*({s_rhs})/{denominator}")
                                    }
                                } else if denominator == 1 {
                                    format!("({numerator})*{s_rhs}")
                                } else if denominator < 0 {
                                    format!("({numerator})*{s_rhs}/({denominator})")
                                } else {
                                    format!("({numerator})*{s_rhs}/{denominator}")
                                }
                            } else if op_rhs {
                                if denominator == 1 {
                                    format!("{numerator}*({s_rhs})")
                                } else if denominator < 0 {
                                    format!("{numerator}*({s_rhs})/({denominator})")
                                } else {
                                    format!("{numerator}*({s_rhs})/{denominator}")
                                }
                            } else if denominator == 1 {
                                format!("{numerator}*{s_rhs}")
                            } else if denominator < 0 {
                                format!("{numerator}*{s_rhs}/({denominator})")
                            } else {
                                format!("{numerator}*{s_rhs}/{denominator}")
                            }
                        } else if op_lhs {
                            if op_rhs {
                                format!("({s_lhs})*({s_rhs})")
                            } else {
                                format!("({s_lhs})*{s_rhs}")
                            }
                        } else if op_rhs {
                            format!("{s_lhs}*({s_rhs})")
                        } else {
                            format!("{s_lhs}*{s_rhs}")
                        }
                    }
                    BinaryOp::Div => {
                        if let Some((numerator, denominator)) = lhs.rational() {
                            if numerator < 0 {
                                if op_rhs {
                                    if denominator == 1 {
                                        format!("({numerator})/({s_rhs})")
                                    } else if denominator < 0 {
                                        format!("({numerator})/({s_rhs})/({denominator})")
                                    } else {
                                        format!("({numerator})/({s_rhs})/{denominator}")
                                    }
                                } else if denominator == 1 {
                                    format!("({numerator})/{s_rhs}")
                                } else if denominator < 0 {
                                    format!("({numerator})/{s_rhs}/({denominator})")
                                } else {
                                    format!("({numerator})/{s_rhs}/{denominator}")
                                }
                            } else if op_rhs {
                                if denominator == 1 {
                                    format!("{numerator}/({s_rhs})")
                                } else if denominator < 0 {
                                    format!("{numerator}/({s_rhs})/({denominator})")
                                } else {
                                    format!("{numerator}/({s_rhs})/{denominator}")
                                }
                            } else if denominator == 1 {
                                format!("{numerator}/{s_rhs}")
                            } else if denominator < 0 {
                                format!("{numerator}/{s_rhs}/({denominator})")
                            } else {
                                format!("{numerator}/{s_rhs}/{denominator}")
                            }
                        } else if op_lhs {
                            if op_rhs {
                                format!("({s_lhs})/({s_rhs})")
                            } else {
                                format!("({s_lhs})/{s_rhs}")
                            }
                        } else if op_rhs {
                            format!("{s_lhs}/({s_rhs})")
                        } else {
                            format!("{s_lhs}/{s_rhs}")
                        }
                    }
                    BinaryOp::Pow => match lhs.as_ref() {
                        SymbolExpr::Binary { .. } | SymbolExpr::Unary { .. } => {
                            match rhs.as_ref() {
                                SymbolExpr::Binary { .. } | SymbolExpr::Unary { .. } => {
                                    format!("({s_lhs})**({s_rhs})")
                                }
                                SymbolExpr::Value(r) => {
                                    if r.as_real() < 0.0 {
                                        format!("({s_lhs})**({s_rhs})")
                                    } else {
                                        format!("({s_lhs})**{s_rhs}")
                                    }
                                }
                                _ => format!("({s_lhs})**{s_rhs}"),
                            }
                        }
                        SymbolExpr::Value(l) => {
                            if l.as_real() < 0.0 {
                                match rhs.as_ref() {
                                    SymbolExpr::Binary { .. } | SymbolExpr::Unary { .. } => {
                                        format!("({s_lhs})**({s_rhs})")
                                    }
                                    _ => format!("({s_lhs})**{s_rhs}"),
                                }
                            } else {
                                match rhs.as_ref() {
                                    SymbolExpr::Binary { .. } | SymbolExpr::Unary { .. } => {
                                        format!("{s_lhs}**({s_rhs})")
                                    }
                                    _ => format!("{s_lhs}**{s_rhs}"),
                                }
                            }
                        }
                        _ => match rhs.as_ref() {
                            SymbolExpr::Binary { .. } | SymbolExpr::Unary { .. } => {
                                format!("{s_lhs}**({s_rhs})")
                            }
                            SymbolExpr::Value(r) => {
                                if r.as_real() < 0.0 {
                                    format!("{s_lhs}**({s_rhs})")
                                } else {
                                    format!("{s_lhs}**{s_rhs}")
                                }
                            }
                            _ => format!("{s_lhs}**{s_rhs}"),
                        },
                    },
                }
            }
        }
    }
}

impl Add for SymbolExpr {
    type Output = SymbolExpr;
    fn add(self, rhs: Self) -> SymbolExpr {
        match self.add_opt(&rhs, false) {
            Some(e) => e,
            None => _add(self, rhs),
        }
    }
}

impl Add for &SymbolExpr {
    type Output = SymbolExpr;
    fn add(self, rhs: Self) -> SymbolExpr {
        match self.add_opt(rhs, false) {
            Some(e) => e,
            None => _add(self.clone(), rhs.clone()),
        }
    }
}

impl Sub for SymbolExpr {
    type Output = SymbolExpr;
    fn sub(self, rhs: Self) -> SymbolExpr {
        match self.sub_opt(&rhs, false) {
            Some(e) => e,
            None => _sub(self, rhs),
        }
    }
}

impl Sub for &SymbolExpr {
    type Output = SymbolExpr;
    fn sub(self, rhs: Self) -> SymbolExpr {
        match self.sub_opt(rhs, false) {
            Some(e) => e,
            None => _sub(self.clone(), rhs.clone()),
        }
    }
}

impl Mul for SymbolExpr {
    type Output = SymbolExpr;
    fn mul(self, rhs: Self) -> SymbolExpr {
        match self.mul_opt(&rhs, false) {
            Some(e) => e,
            None => _mul(self, rhs),
        }
    }
}

impl Mul for &SymbolExpr {
    type Output = SymbolExpr;
    fn mul(self, rhs: Self) -> SymbolExpr {
        match self.mul_opt(rhs, false) {
            Some(e) => e,
            None => _mul(self.clone(), rhs.clone()),
        }
    }
}

impl Div for SymbolExpr {
    type Output = SymbolExpr;
    fn div(self, rhs: Self) -> SymbolExpr {
        match self.div_opt(&rhs, false) {
            Some(e) => e,
            None => _div(self, rhs),
        }
    }
}

impl Div for &SymbolExpr {
    type Output = SymbolExpr;
    fn div(self, rhs: Self) -> SymbolExpr {
        match self.div_opt(rhs, false) {
            Some(e) => e,
            None => _div(self.clone(), rhs.clone()),
        }
    }
}

impl Neg for SymbolExpr {
    type Output = SymbolExpr;
    fn neg(self) -> SymbolExpr {
        match self.neg_opt() {
            Some(e) => e,
            None => _neg(self),
        }
    }
}

impl Neg for &SymbolExpr {
    type Output = SymbolExpr;
    fn neg(self) -> SymbolExpr {
        match self.neg_opt() {
            Some(e) => e,
            None => _neg(self.clone()),
        }
    }
}

impl PartialEq for SymbolExpr {
    fn eq(&self, rexpr: &Self) -> bool {
        if let Some(v) = self.sub_values(rexpr, false) {
            return v.is_zero(true);
        }
        if let (Some(l), Some(r)) = (self.eval(true), rexpr.eval(true)) {
            return l == r;
        }
        if self.string_id() == rexpr.string_id() {
            return true;
        }

        match (self, rexpr) {
            (SymbolExpr::Symbol(l), SymbolExpr::Symbol(r)) => l == r,
            (SymbolExpr::Value(l), SymbolExpr::Value(r)) => l == r,
            (
                SymbolExpr::Binary { .. } | SymbolExpr::Unary { .. },
                SymbolExpr::Binary { .. } | SymbolExpr::Unary { .. },
            ) => {
                let ex_lhs = self.expand();
                let ex_rhs = rexpr.expand();
                if ex_lhs.string_id() == ex_rhs.string_id() {
                    return true;
                }
                match ex_lhs.sub_opt(&ex_rhs, true) {
                    Some(e) => e.is_zero(true),
                    None => {
                        let t = &ex_lhs - &ex_rhs;
                        t.is_zero(true)
                    }
                }
            }
            (SymbolExpr::Binary { .. }, _) => {
                let ex_lhs = self.expand();
                match ex_lhs.sub_opt(rexpr, true) {
                    Some(e) => e.is_zero(true),
                    None => {
                        let t = &ex_lhs - rexpr;
                        t.is_zero(true)
                    }
                }
            }
            (_, SymbolExpr::Binary { .. }) => {
                let ex_rhs = rexpr.expand();
                match self.sub_opt(&ex_rhs, true) {
                    Some(e) => e.is_zero(true),
                    None => {
                        let t = self - &ex_rhs;
                        t.is_zero(true)
                    }
                }
            }
            (_, _) => false,
        }
    }
}

impl Eq for SymbolExpr {}

impl PartialEq<f64> for SymbolExpr {
    fn eq(&self, r: &f64) -> bool {
        match self.eval(true) {
            Some(v) => v == *r,
            None => false,
        }
    }
}

impl PartialEq<Complex64> for SymbolExpr {
    fn eq(&self, r: &Complex64) -> bool {
        match self.eval(true) {
            Some(v) => v == *r,
            None => false,
        }
    }
}

// comparison rules for sorting equation
impl PartialOrd for SymbolExpr {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        if let Some(v) = self.sub_values(rhs, false) {
            if v.is_zero(true) {
                return Some(Ordering::Equal);
            } else if v.is_negative() {
                return Some(Ordering::Less);
            } else {
                return Some(Ordering::Greater);
            }
        }
        match self {
            SymbolExpr::Value(l) => match rhs {
                SymbolExpr::Value(r) => l.partial_cmp(r),
                _ => Some(Ordering::Less),
            },
            SymbolExpr::Symbol(l) => match rhs {
                SymbolExpr::Value(_) => Some(Ordering::Greater),
                SymbolExpr::Symbol(r) => l.partial_cmp(r),
                SymbolExpr::Unary { op: _, expr } => self.partial_cmp(expr),
                _ => Some(Ordering::Less),
            },
            SymbolExpr::Unary { op: _, expr } => match rhs {
                SymbolExpr::Value(_) => Some(Ordering::Greater),
                SymbolExpr::Unary { op: _, expr: rexpr } => expr.partial_cmp(rexpr),
                _ => (expr.as_ref()).partial_cmp(rhs),
            },
            SymbolExpr::Binary {
                op,
                lhs: ll,
                rhs: lr,
            } => match rhs {
                SymbolExpr::Value(_) | SymbolExpr::Symbol(_) => match op {
                    BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow => Some(Ordering::Greater),
                    _ => Some(Ordering::Equal),
                },
                SymbolExpr::Unary { op: _, expr } => self.partial_cmp(expr),
                SymbolExpr::Binary {
                    op: _,
                    lhs: rl,
                    rhs: rr,
                } => {
                    let ls = match ll.as_ref() {
                        SymbolExpr::Value(_) => lr.string_id(),
                        _ => self.string_id(),
                    };
                    let rs = match rl.as_ref() {
                        SymbolExpr::Value(_) => rr.string_id(),
                        _ => rhs.string_id(),
                    };
                    if rs > ls && rs.len() > ls.len() {
                        Some(Ordering::Less)
                    } else if rs < ls && rs.len() < ls.len() {
                        Some(Ordering::Greater)
                    } else {
                        Some(Ordering::Equal)
                    }
                }
            },
        }
    }
}

impl From<&str> for SymbolExpr {
    fn from(v: &str) -> Self {
        SymbolExpr::Symbol(Arc::new(Symbol::new(v, None, None)))
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Value::Real(e) => e.to_string(),
                Value::Int(e) => e.to_string(),
                Value::Complex(e) => {
                    if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&e.re) {
                        if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&e.im) {
                            0.to_string()
                        } else {
                            format!("{}i", e.im)
                        }
                    } else if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&e.im) {
                        e.re.to_string()
                    } else {
                        e.to_string()
                    }
                }
                Value::Rational {
                    numerator,
                    denominator,
                } => format!("{numerator}/{denominator}"),
            }
        )
    }
}

// ===============================================================
//  implementations for Value
// ===============================================================
impl Value {
    pub fn as_real(&self) -> f64 {
        match self {
            Value::Real(e) => *e,
            Value::Int(e) => *e as f64,
            Value::Complex(e) => e.re,
            Value::Rational {
                numerator,
                denominator,
            } => *numerator as f64 / *denominator as f64,
        }
    }

    pub fn is_real(&self) -> bool {
        match self {
            Value::Real(_) | Value::Int(_) | Value::Rational { .. } => true,
            Value::Complex(c) => (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.im),
        }
    }

    pub fn abs(&self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.abs()),
            Value::Int(e) => Value::Int(e.abs()),
            Value::Complex(e) => Value::Real((e.re * e.re + e.im * e.im).sqrt()),
            Value::Rational {
                numerator,
                denominator,
            } => _rational(numerator.abs(), denominator.abs()),
        }
    }

    pub fn sin(&self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.sin()),
            Value::Int(e) => Value::Real((*e as f64).sin()),
            Value::Complex(e) => {
                let t = Value::Complex(e.sin());
                match t.opt_complex() {
                    Some(v) => v,
                    None => t,
                }
            }
            Value::Rational { .. } => Value::Real(self.as_real().sin()),
        }
    }
    pub fn asin(&self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.asin()),
            Value::Int(e) => Value::Real((*e as f64).asin()),
            Value::Complex(e) => {
                let t = Value::Complex(e.asin());
                match t.opt_complex() {
                    Some(v) => v,
                    None => t,
                }
            }
            Value::Rational { .. } => Value::Real(self.as_real().asin()),
        }
    }
    pub fn cos(&self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.cos()),
            Value::Int(e) => Value::Real((*e as f64).cos()),
            Value::Complex(e) => {
                let t = Value::Complex(e.cos());
                match t.opt_complex() {
                    Some(v) => v,
                    None => t,
                }
            }
            Value::Rational { .. } => Value::Real(self.as_real().cos()),
        }
    }
    pub fn acos(&self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.acos()),
            Value::Int(e) => Value::Real((*e as f64).acos()),
            Value::Complex(e) => {
                let t = Value::Complex(e.acos());
                match t.opt_complex() {
                    Some(v) => v,
                    None => t,
                }
            }
            Value::Rational { .. } => Value::Real(self.as_real().acos()),
        }
    }
    pub fn tan(&self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.tan()),
            Value::Int(e) => Value::Real((*e as f64).tan()),
            Value::Complex(e) => {
                let t = Value::Complex(e.tan());
                match t.opt_complex() {
                    Some(v) => v,
                    None => t,
                }
            }
            Value::Rational { .. } => Value::Real(self.as_real().tan()),
        }
    }
    pub fn atan(&self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.atan()),
            Value::Int(e) => Value::Real((*e as f64).atan()),
            Value::Complex(e) => {
                let t = Value::Complex(e.atan());
                match t.opt_complex() {
                    Some(v) => v,
                    None => t,
                }
            }
            Value::Rational { .. } => Value::Real(self.as_real().atan()),
        }
    }
    pub fn exp(&self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.exp()),
            Value::Int(e) => Value::Real((*e as f64).exp()),
            Value::Complex(e) => {
                let t = Value::Complex(e.exp());
                match t.opt_complex() {
                    Some(v) => v,
                    None => t,
                }
            }
            Value::Rational { .. } => Value::Real(self.as_real().exp()),
        }
    }
    pub fn log(&self) -> Value {
        match self {
            Value::Real(e) => {
                if *e < 0.0 {
                    Value::Complex(Complex64::from(e)).log()
                } else {
                    Value::Real(e.ln())
                }
            }
            Value::Int(e) => Value::Real(*e as f64).log(),
            Value::Complex(e) => {
                let t = Value::Complex(e.ln());
                match t.opt_complex() {
                    Some(v) => v,
                    None => t,
                }
            }
            Value::Rational { .. } => Value::Real(self.as_real()).log(),
        }
    }
    pub fn sqrt(&self) -> Value {
        match self {
            Value::Real(e) => {
                if *e < 0.0 {
                    Value::Complex(Complex64::from(e)).sqrt()
                } else {
                    Value::Real(e.sqrt())
                }
            }
            Value::Int(e) => {
                if *e < 0 {
                    Value::Complex(Complex64::from(*e as f64)).pow(&Value::Real(0.5))
                } else {
                    let t = (*e as f64).sqrt();
                    let d = t.floor() - t;
                    if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&d) {
                        Value::Int(t as i64)
                    } else {
                        Value::Real(t)
                    }
                }
            }
            Value::Complex(e) => {
                let t = Value::Complex(e.sqrt());
                match t.opt_complex() {
                    Some(v) => v,
                    None => t,
                }
            }
            Value::Rational { .. } => Value::Real(self.as_real().sqrt()),
        }
    }
    pub fn pow(&self, p: &Value) -> Value {
        match self {
            Value::Real(e) => match p {
                Value::Real(r) => {
                    if *e < 0.0 && r.fract() != 0. {
                        Value::Complex(Complex64::from(e)).pow(p)
                    } else {
                        Value::Real(e.powf(*r))
                    }
                }
                Value::Int(i) => Value::Real(e.powf(*i as f64)),
                Value::Complex(_) => Value::Complex(Complex64::from(e)).pow(p),
                Value::Rational { .. } => self.pow(&Value::Real(p.as_real())),
            },
            Value::Int(e) => match p {
                Value::Real(r) => {
                    if *e < 0 && r.fract() != 0. {
                        Value::Complex(Complex64::from(*e as f64)).pow(p)
                    } else {
                        let t = (*e as f64).powf(*r);
                        let d = t.floor() - t;
                        if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&d) {
                            Value::Int(t as i64)
                        } else {
                            Value::Real(t)
                        }
                    }
                }
                Value::Int(r) => {
                    if *r < 0 {
                        Value::Real(*e as f64).pow(p)
                    } else {
                        Value::Int(e.pow(*r as u32))
                    }
                }
                Value::Complex(_) => Value::Complex(Complex64::from(*e as f64)).pow(p),
                Value::Rational { .. } => self.pow(&Value::Real(p.as_real())),
            },
            Value::Complex(e) => {
                let t = match p {
                    Value::Real(r) => Value::Complex(e.powf(*r)),
                    Value::Int(r) => Value::Complex(e.powf(*r as f64)),
                    Value::Complex(r) => Value::Complex(e.powc(*r)),
                    Value::Rational { .. } => self.pow(&Value::Real(p.as_real())),
                };
                match t.opt_complex() {
                    Some(v) => v,
                    None => t,
                }
            }
            Value::Rational {
                numerator,
                denominator,
            } => {
                if let Value::Int(i) = p {
                    if *i >= 0 {
                        return _rational(numerator.pow(*i as u32), denominator.pow(*i as u32));
                    }
                }
                Value::Real(self.as_real()).pow(p)
            }
        }
    }
    pub fn rcp(&self) -> Value {
        match self {
            Value::Real(e) => Value::Real(1.0 / e),
            Value::Int(e) => _rational(1, *e),
            Value::Complex(e) => Value::Complex(1.0 / e),
            Value::Rational {
                numerator,
                denominator,
            } => _rational(*denominator, *numerator),
        }
    }
    pub fn sign(&self) -> Value {
        match self {
            Value::Real(e) => {
                if *e > SYMEXPR_EPSILON {
                    Value::Real(1.0)
                } else if *e < -SYMEXPR_EPSILON {
                    Value::Real(-1.0)
                } else {
                    Value::Real(0.0)
                }
            }
            Value::Int(e) => {
                if *e > 0 {
                    Value::Int(1)
                } else if *e < 0 {
                    Value::Int(-1)
                } else {
                    Value::Int(0)
                }
            }
            Value::Complex(_) => *self,
            Value::Rational {
                numerator,
                denominator,
            } => {
                if *numerator > 0 {
                    if *denominator > 0 {
                        Value::Int(1)
                    } else {
                        Value::Int(-1)
                    }
                } else if *numerator == 0 {
                    Value::Int(0)
                } else if *denominator > 0 {
                    Value::Int(-1)
                } else {
                    Value::Int(1)
                }
            }
        }
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Value::Real(r) => (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(r),
            Value::Int(i) => *i == 0,
            Value::Complex(c) => {
                (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.re)
                    && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.im)
            }
            Value::Rational { numerator, .. } => *numerator == 0,
        }
    }
    pub fn is_one(&self) -> bool {
        match self {
            Value::Real(r) => (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&(*r - 1.0)),
            Value::Int(i) => *i == 1,
            Value::Complex(c) => {
                (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&(c.re - 1.0))
                    && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.im)
            }
            Value::Rational {
                numerator,
                denominator,
            } => *numerator == *denominator,
        }
    }
    pub fn is_minus_one(&self) -> bool {
        match self {
            Value::Real(r) => (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&(*r + 1.0)),
            Value::Int(i) => *i == -1,
            Value::Complex(c) => {
                (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&(c.re + 1.0))
                    && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.im)
            }
            Value::Rational {
                numerator,
                denominator,
            } => *numerator == -denominator,
        }
    }

    pub fn is_negative(&self) -> bool {
        match self {
            Value::Real(r) => *r < 0.0,
            Value::Int(i) => *i < 0,
            Value::Complex(c) => {
                (c.re < 0.0 && c.im < SYMEXPR_EPSILON && c.im > -SYMEXPR_EPSILON)
                    || (c.im < 0.0 && c.re < SYMEXPR_EPSILON && c.re > -SYMEXPR_EPSILON)
            }
            Value::Rational { .. } => self.sign() == Value::Int(-1),
        }
    }

    pub fn opt_complex(&self) -> Option<Value> {
        match self {
            Value::Complex(c) => {
                if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.im) {
                    Some(Value::Real(c.re))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    // convert sympy compatible format
    pub fn sympify(&self) -> SymbolExpr {
        match self {
            // imaginary number is comverted to value * symbol 'I'
            Value::Complex(c) => _add(
                SymbolExpr::Value(Value::Real(c.re)),
                _mul(
                    SymbolExpr::Value(Value::Real(c.im)),
                    SymbolExpr::Symbol(Arc::new(Symbol::new("I", None, None))),
                ),
            ),
            _ => SymbolExpr::Value(*self),
        }
    }
}

impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::Real(v)
    }
}

impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::Int(v)
    }
}

impl From<Complex64> for Value {
    fn from(v: Complex64) -> Self {
        if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&v.im) {
            Value::Real(v.re)
        } else {
            Value::Complex(v)
        }
    }
}

impl Add for &Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Value {
        *self + *rhs
    }
}

impl Add for Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Value {
        let t = match self {
            Value::Real(l) => match rhs {
                Value::Real(r) => Value::Real(l + r),
                Value::Int(r) => Value::Real(l + r as f64),
                Value::Complex(r) => Value::Complex(l + r),
                Value::Rational {
                    numerator,
                    denominator,
                } => Value::Real((l * denominator as f64 + numerator as f64) / denominator as f64),
            },
            Value::Int(l) => match rhs {
                Value::Real(r) => Value::Real(l as f64 + r),
                Value::Int(r) => Value::Int(l + r),
                Value::Complex(r) => Value::Complex(l as f64 + r),
                Value::Rational {
                    numerator,
                    denominator,
                } => _rational(l * denominator + numerator, denominator),
            },
            Value::Complex(l) => match rhs {
                Value::Real(r) => Value::Complex(l + r),
                Value::Int(r) => Value::Complex(l + r as f64),
                Value::Complex(r) => Value::Complex(l + r),
                Value::Rational {
                    numerator,
                    denominator,
                } => {
                    Value::Complex((l * denominator as f64 + numerator as f64) / denominator as f64)
                }
            },
            Value::Rational {
                numerator,
                denominator,
            } => match rhs {
                Value::Real(r) => {
                    Value::Real((numerator as f64 + r * denominator as f64) / denominator as f64)
                }
                Value::Int(r) => _rational(numerator + r * denominator, denominator),
                Value::Complex(r) => {
                    Value::Complex((numerator as f64 + r * denominator as f64) / denominator as f64)
                }
                Value::Rational {
                    numerator: rn,
                    denominator: rd,
                } => _rational(rd * numerator + denominator * rn, denominator * rd),
            },
        };
        match t.opt_complex() {
            Some(v) => v,
            None => t,
        }
    }
}

impl Sub for &Value {
    type Output = Value;
    fn sub(self, rhs: Self) -> Value {
        *self - *rhs
    }
}

impl Sub for Value {
    type Output = Value;
    fn sub(self, rhs: Self) -> Value {
        let t = match self {
            Value::Real(l) => match rhs {
                Value::Real(r) => Value::Real(l - r),
                Value::Int(r) => Value::Real(l - r as f64),
                Value::Complex(r) => Value::Complex(l - r),
                Value::Rational {
                    numerator,
                    denominator,
                } => Value::Real((l * denominator as f64 - numerator as f64) / denominator as f64),
            },
            Value::Int(l) => match rhs {
                Value::Real(r) => Value::Real(l as f64 - r),
                Value::Int(r) => Value::Int(l - r),
                Value::Complex(r) => Value::Complex(l as f64 - r),
                Value::Rational {
                    numerator,
                    denominator,
                } => _rational(l * denominator - numerator, denominator),
            },
            Value::Complex(l) => match rhs {
                Value::Real(r) => Value::Complex(l - r),
                Value::Int(r) => Value::Complex(l - r as f64),
                Value::Complex(r) => Value::Complex(l - r),
                Value::Rational {
                    numerator,
                    denominator,
                } => {
                    Value::Complex((l * denominator as f64 - numerator as f64) / denominator as f64)
                }
            },
            Value::Rational {
                numerator,
                denominator,
            } => match rhs {
                Value::Real(r) => {
                    Value::Real((numerator as f64 - r * denominator as f64) / denominator as f64)
                }
                Value::Int(r) => _rational(numerator - r * denominator, denominator),
                Value::Complex(r) => {
                    Value::Complex((numerator as f64 - r * denominator as f64) / denominator as f64)
                }
                Value::Rational {
                    numerator: rn,
                    denominator: rd,
                } => _rational(rd * numerator - denominator * rn, denominator * rd),
            },
        };
        match t.opt_complex() {
            Some(v) => v,
            None => t,
        }
    }
}

impl Mul for &Value {
    type Output = Value;
    fn mul(self, rhs: Self) -> Value {
        *self * *rhs
    }
}

impl Mul for Value {
    type Output = Value;
    fn mul(self, rhs: Self) -> Value {
        let t = match self {
            Value::Real(l) => match rhs {
                Value::Real(r) => Value::Real(l * r),
                Value::Int(r) => Value::Real(l * r as f64),
                Value::Complex(r) => Value::Complex(l * r),
                Value::Rational {
                    numerator,
                    denominator,
                } => Value::Real((l * numerator as f64) / denominator as f64),
            },
            Value::Int(l) => match rhs {
                Value::Real(r) => Value::Real(l as f64 * r),
                Value::Int(r) => Value::Int(l * r),
                Value::Complex(r) => Value::Complex(l as f64 * r),
                Value::Rational {
                    numerator,
                    denominator,
                } => _rational(l * numerator, denominator),
            },
            Value::Complex(l) => match rhs {
                Value::Real(r) => Value::Complex(l * r),
                Value::Int(r) => Value::Complex(l * r as f64),
                Value::Complex(r) => Value::Complex(l * r),
                Value::Rational {
                    numerator,
                    denominator,
                } => Value::Complex((l * numerator as f64) / denominator as f64),
            },
            Value::Rational {
                numerator,
                denominator,
            } => match rhs {
                Value::Real(r) => Value::Real((numerator as f64 * r) / denominator as f64),
                Value::Int(r) => _rational(numerator * r, denominator),
                Value::Complex(r) => Value::Complex((numerator as f64 * r) / denominator as f64),
                Value::Rational {
                    numerator: rn,
                    denominator: rd,
                } => _rational(numerator * rn, denominator * rd),
            },
        };
        match t.opt_complex() {
            Some(v) => v,
            None => t,
        }
    }
}

impl Div for &Value {
    type Output = Value;
    fn div(self, rhs: Self) -> Value {
        *self / *rhs
    }
}

impl Div for Value {
    type Output = Value;
    fn div(self, rhs: Self) -> Value {
        let t = match self {
            Value::Real(l) => match rhs {
                Value::Real(r) => Value::Real(l / r),
                Value::Int(r) => Value::Real(l / r as f64),
                Value::Complex(r) => Value::Complex(l / r),
                Value::Rational {
                    numerator,
                    denominator,
                } => Value::Real(l * denominator as f64 / numerator as f64),
            },
            Value::Int(l) => {
                if rhs.is_zero() {
                    return Value::Real(f64::INFINITY);
                }
                match rhs {
                    Value::Real(r) => Value::Real(l as f64 / r),
                    Value::Int(r) => _rational(l, r),
                    Value::Complex(r) => Value::Complex(l as f64 / r),
                    Value::Rational {
                        numerator,
                        denominator,
                    } => _rational(l * denominator, numerator),
                }
            }
            Value::Complex(l) => match rhs {
                Value::Real(r) => Value::Complex(l / r),
                Value::Int(r) => Value::Complex(l / r as f64),
                Value::Complex(r) => Value::Complex(l / r),
                Value::Rational {
                    numerator,
                    denominator,
                } => Value::Complex((l * denominator as f64) / numerator as f64),
            },
            Value::Rational {
                numerator,
                denominator,
            } => match rhs {
                Value::Real(r) => Value::Real(numerator as f64 / (denominator as f64 * r)),
                Value::Int(r) => _rational(numerator, denominator * r),
                Value::Complex(r) => Value::Complex(numerator as f64 / (denominator as f64 * r)),
                Value::Rational {
                    numerator: rn,
                    denominator: rd,
                } => _rational(numerator * rd, denominator * rn),
            },
        };
        match t.opt_complex() {
            Some(v) => v,
            None => t,
        }
    }
}

impl Neg for &Value {
    type Output = Value;
    fn neg(self) -> Value {
        -*self
    }
}

impl Neg for Value {
    type Output = Value;
    fn neg(self) -> Value {
        match self {
            Value::Real(v) => Value::Real(-v),
            Value::Int(v) => Value::Int(-v),
            Value::Complex(v) => Value::Complex(-v),
            Value::Rational {
                numerator,
                denominator,
            } => _rational(-numerator, denominator),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, r: &Self) -> bool {
        match self {
            Value::Real(e) => match r {
                Value::Real(rv) => (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&(e - rv)),
                Value::Int(rv) => (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&(e - *rv as f64)),
                Value::Complex(rv) => {
                    let t = Complex64::from(*e) - rv;
                    (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.re)
                        && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.im)
                }
                Value::Rational {
                    numerator,
                    denominator,
                } => (-SYMEXPR_EPSILON..SYMEXPR_EPSILON)
                    .contains(&(e * *denominator as f64 - *numerator as f64)),
            },
            Value::Int(e) => match r {
                Value::Int(rv) => e == rv,
                Value::Real(rv) => (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&(*e as f64 - rv)),
                Value::Complex(rv) => {
                    let t = Complex64::from(*e as f64) - rv;
                    (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.re)
                        && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.im)
                }
                Value::Rational {
                    numerator,
                    denominator,
                } => e * *denominator == *numerator,
            },
            Value::Complex(e) => match r {
                Value::Real(rv) => {
                    let t = *e - Complex64::from(rv);
                    (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.re)
                        && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.im)
                }
                Value::Int(rv) => {
                    let t = *e - Complex64::from(*rv as f64);
                    (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.re)
                        && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.im)
                }
                Value::Complex(rv) => {
                    let t = *e - rv;
                    (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.re)
                        && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.im)
                }
                Value::Rational {
                    numerator,
                    denominator,
                } => {
                    let t = *e * *denominator as f64 - *numerator as f64;
                    (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.re)
                        && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.im)
                }
            },
            Value::Rational {
                numerator,
                denominator,
            } => match r {
                Value::Int(rv) => *numerator == rv * *denominator,
                Value::Real(rv) => (-SYMEXPR_EPSILON..SYMEXPR_EPSILON)
                    .contains(&(*numerator as f64 - rv * *denominator as f64)),
                Value::Complex(rv) => {
                    let t = rv * *denominator as f64 - *numerator as f64;
                    (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.re)
                        && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.im)
                }
                Value::Rational {
                    numerator: rn,
                    denominator: rd,
                } => *denominator * *rn == *numerator * *rd,
            },
        }
    }
}

impl PartialEq<f64> for Value {
    fn eq(&self, r: &f64) -> bool {
        match self {
            Value::Real(e) => (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&(e - r)),
            Value::Int(e) => (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&(*e as f64 - r)),
            Value::Complex(e) => {
                let t = *e - Complex64::from(r);
                (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.re)
                    && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.im)
            }
            Value::Rational {
                numerator,
                denominator,
            } => (-SYMEXPR_EPSILON..SYMEXPR_EPSILON)
                .contains(&(*numerator as f64 - r * *denominator as f64)),
        }
    }
}

impl PartialEq<Complex64> for Value {
    fn eq(&self, r: &Complex64) -> bool {
        match self {
            Value::Real(e) => {
                let t = Complex64::from(*e) - r;
                (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.re)
                    && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.im)
            }
            Value::Int(e) => {
                let t = Complex64::from(*e as f64) - r;
                (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.re)
                    && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.im)
            }
            Value::Complex(e) => {
                let t = *e - r;
                (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.re)
                    && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.im)
            }
            Value::Rational {
                numerator,
                denominator,
            } => {
                let t = r * *denominator as f64 - *numerator as f64;
                (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.re)
                    && (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&t.im)
            }
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        match self {
            Value::Real(l) => match rhs {
                Value::Real(r) => l.partial_cmp(r),
                Value::Int(r) => l.partial_cmp(&(*r as f64)),
                Value::Complex(_) => None,
                Value::Rational {
                    numerator,
                    denominator,
                } => (l * *denominator as f64).partial_cmp(&(*numerator as f64)),
            },
            Value::Int(l) => match rhs {
                Value::Real(r) => (*l as f64).partial_cmp(r),
                Value::Int(r) => l.partial_cmp(r),
                Value::Complex(_) => None,
                Value::Rational {
                    numerator,
                    denominator,
                } => (l * *denominator).partial_cmp(numerator),
            },
            Value::Complex(_) => None,
            Value::Rational {
                numerator,
                denominator,
            } => match rhs {
                Value::Real(r) => (*numerator as f64).partial_cmp(&(r * *denominator as f64)),
                Value::Int(r) => numerator.partial_cmp(&(r * *denominator)),
                Value::Complex(_) => None,
                Value::Rational {
                    numerator: rn,
                    denominator: rd,
                } => (rd * *numerator).partial_cmp(&(rn * *denominator)),
            },
        }
    }
}

/// Replace [Symbol]s in a [SymbolExpr] according to the name map. This
/// is used to reconstruct a parameter expression from a string.
pub fn replace_symbol(symbol_expr: &SymbolExpr, name_map: &HashMap<String, Symbol>) -> SymbolExpr {
    match symbol_expr {
        SymbolExpr::Symbol(existing_symbol) => {
            let name = existing_symbol.repr(false);
            if let Some(new_symbol) = name_map.get(&name) {
                SymbolExpr::Symbol(Arc::new(new_symbol.clone()))
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

// calculate greatest common divider
fn _gcd(a: u64, b: u64) -> u64 {
    if b > a {
        return _gcd(b, a);
    }
    if b == 0 { a } else { _gcd(b, a % b) }
}

// make new integer rational number as Binary div
fn _rational(numerator: i64, denominator: i64) -> Value {
    if numerator == 0 {
        return Value::Int(0);
    }
    let mut ret_n = numerator;
    let mut ret_d = denominator;
    let gcd = _gcd(numerator.unsigned_abs(), denominator.unsigned_abs());
    if gcd > 1 {
        ret_n /= gcd as i64;
        ret_d /= gcd as i64;
    }
    if numerator != 0 && denominator < 0 {
        ret_n = -ret_n;
        ret_d = -ret_d;
    }
    Value::Rational {
        numerator: ret_n,
        denominator: ret_d,
    }
}
