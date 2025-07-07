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

// parameterexpression.rs
// rust implementation of Parameter / ParameterVectorElement / ParameterExpression
use crate::symbol_expr::SymbolExpr;
use crate::symbol_expr::{self, Value};
use crate::symbol_parser::parse_expression;

use hashbrown::{HashMap, HashSet};
use num_complex::Complex64;
use uuid::Uuid;

use core::f64;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

use pyo3::import_exception;
use pyo3::prelude::*;
use pyo3::types::PySet;
use pyo3::IntoPyObjectExt;

import_exception!(qiskit.circuit.exceptions, CircuitError);

// enum for acceptable types for parameter
#[derive(IntoPyObject, FromPyObject, Clone, Debug)]
pub enum ParameterValueType {
    Int(i64),
    Float(f64),
    Complex(Complex64),
    Parameter(ParameterExpression),
}

impl ParameterValueType {
    fn clone_expr_for_replay(expr: &ParameterExpression) -> ParameterValueType {
        if expr.is_numeric() {
            if let Some(v) = expr.expr.eval(true) {
                return match v {
                    symbol_expr::Value::Int(i) => ParameterValueType::Int(i),
                    symbol_expr::Value::Real(r) => ParameterValueType::Float(r),
                    symbol_expr::Value::Complex(c) => ParameterValueType::Complex(c),
                };
            }
        }
        ParameterValueType::Parameter(expr.clone())
    }
}

#[pyfunction]
fn _extract_value(value: &Bound<PyAny>) -> Option<ParameterExpression> {
    if let Ok(e) = value.extract::<ParameterExpression>() {
        Some(e)
    } else if value.extract::<String>().is_ok() {
        None
    } else if let Ok(i) = value.extract::<i64>() {
        Some(ParameterExpression {
            expr: SymbolExpr::Value(symbol_expr::Value::from(i)),
            parameter_symbols: None,
        })
    } else if let Ok(c) = value.extract::<Complex64>() {
        if c.is_infinite() || c.is_nan() {
            return None;
        }
        Some(ParameterExpression {
            expr: SymbolExpr::Value(symbol_expr::Value::from(c)),
            parameter_symbols: None,
        })
    } else if let Ok(r) = value.extract::<f64>() {
        if r.is_infinite() || r.is_nan() {
            return None;
        }
        Some(ParameterExpression {
            expr: SymbolExpr::Value(symbol_expr::Value::from(r)),
            parameter_symbols: None,
        })
    } else {
        None
    }
}

// OP codes for QPY replay
#[pyclass(sequence, module = "qiskit._accelerate.circuit")]
#[derive(Clone, Debug)]
pub enum _OPCode {
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
    GRAD = 13,
    CONJ = 14,
    SUBSTITUTE = 15,
    ABS = 16,
    ATAN = 17,
    RSUB = 18,
    RDIV = 19,
    RPOW = 20,
}

// enum for QPY replay
#[pyclass(sequence, module = "qiskit._accelerate.circuit")]
#[derive(Clone, Debug)]
pub enum OPReplay {
    _INSTRUCTION {
        op: _OPCode,
        lhs: Option<ParameterValueType>,
        rhs: Option<ParameterValueType>,
    },
    _SUBS {
        binds: HashMap<ParameterExpression, ParameterValueType>,
        op: _OPCode,
    },
}

impl OPReplay {
    pub fn new_instruction(
        op: _OPCode,
        lhs: Option<ParameterValueType>,
        rhs: Option<ParameterValueType>,
    ) -> OPReplay {
        OPReplay::_INSTRUCTION { op, lhs, rhs }
    }
}

// ===========================================
// ParameterExpression class
// ===========================================
#[pyclass(sequence, subclass, module = "qiskit._accelerate.circuit")]
#[derive(Clone, Debug)]
pub struct ParameterExpression {
    expr: SymbolExpr,                                      // expression
    parameter_symbols: Option<HashMap<Arc<String>, u128>>, // symbols with UUID
}

impl Hash for ParameterExpression {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.expr.to_string().hash(state);
    }
}

impl Default for ParameterExpression {
    // default constructor returns zero
    fn default() -> Self {
        ParameterExpression {
            expr: SymbolExpr::Value(symbol_expr::Value::Int(0)),
            parameter_symbols: None,
        }
    }
}

impl fmt::Display for ParameterExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expr)
    }
}

// Rust side ParameterExpression implementation
impl ParameterExpression {
    /// new for ParameterExpression
    /// make new symbol with its name and uuid
    pub fn new(name: String, uuid: Option<u128>) -> ParameterExpression {
        let uuid = match uuid {
            Some(u) => u,
            None => Uuid::new_v4().as_u128(),
        };
        let symbol = Arc::new(name);
        let mut map = HashMap::<Arc<String>, u128>::new();
        map.insert(Arc::clone(&symbol), uuid);
        ParameterExpression {
            expr: SymbolExpr::Symbol(symbol),
            parameter_symbols: Some(map),
        }
    }

    /// get uuid for this symbol
    pub fn uuid(&self) -> &u128 {
        if let Some(map) = &self.parameter_symbols {
            if let Some(uuid) = map.get(&self.expr.to_string()) {
                return uuid;
            }
        }
        &0_u128
    }

    // clone SymbolExpr
    fn expr(&self) -> SymbolExpr {
        self.expr.clone()
    }

    /// check if this is symbol
    pub fn is_symbol(&self) -> bool {
        matches!(self.expr, SymbolExpr::Symbol(_))
    }

    /// check if this is numeric
    pub fn is_numeric(&self) -> bool {
        matches!(self.expr, SymbolExpr::Value(_))
    }

    /// return number of symbols in this expression
    pub fn num_symbols(&self) -> usize {
        if let Some(map) = &self.parameter_symbols {
            return map.len();
        }
        0
    }

    /// check if the symbol is used in this expression
    pub fn has_symbol(&self, symbol: String) -> bool {
        if let Some(map) = &self.parameter_symbols {
            for (s, _) in map.iter() {
                if s.as_ref() == &symbol {
                    return true;
                }
            }
        }
        false
    }

    /// return true if this is not complex number
    pub fn is_real(&self) -> Option<bool> {
        self.expr.is_complex().map(|b| !b)
    }

    // return merged set of parameter symbils in 2 parameters
    fn merge_parameter_symbols(
        &self,
        other: &ParameterExpression,
    ) -> Option<HashMap<Arc<String>, u128>> {
        let mut ret: HashMap<Arc<String>, u128> = match &self.parameter_symbols {
            Some(map) => map.clone(),
            None => HashMap::new(),
        };
        if let Some(symbols) = &other.parameter_symbols {
            for (s, u) in symbols {
                ret.insert(Arc::clone(s), *u);
            }
        }
        if !ret.is_empty() {
            Some(ret)
        } else {
            None
        }
    }

    // get conflict parameters
    fn get_conflict_parameters(&self, other: &ParameterExpression) -> HashSet<String> {
        let mut conflicts = HashSet::<String>::new();
        let my_symbols = match &self.parameter_symbols {
            Some(map) => map,
            None => &HashMap::new(),
        };
        let other_symbols = match &other.parameter_symbols {
            Some(map) => map,
            None => &HashMap::new(),
        };
        for (o, u) in other_symbols {
            // find symbol with different uuid
            if let Some(m) = my_symbols.get(o) {
                if u != m {
                    conflicts.insert(o.as_ref().clone());
                }
            }
        }
        conflicts
    }

    // default functions for unary operations
    pub fn neg(&self) -> ParameterExpression {
        ParameterExpression {
            expr: -&self.expr,
            parameter_symbols: self.parameter_symbols.clone(),
        }
    }
    pub fn pos(&self) -> ParameterExpression {
        ParameterExpression {
            expr: self.expr.clone(),
            parameter_symbols: self.parameter_symbols.clone(),
        }
    }
    pub fn sin(&self) -> ParameterExpression {
        ParameterExpression {
            expr: self.expr.sin(),
            parameter_symbols: self.parameter_symbols.clone(),
        }
    }
    pub fn cos(&self) -> ParameterExpression {
        ParameterExpression {
            expr: self.expr.cos(),
            parameter_symbols: self.parameter_symbols.clone(),
        }
    }
    pub fn tan(&self) -> ParameterExpression {
        ParameterExpression {
            expr: self.expr.tan(),
            parameter_symbols: self.parameter_symbols.clone(),
        }
    }
    pub fn asin(&self) -> ParameterExpression {
        ParameterExpression {
            expr: self.expr.asin(),
            parameter_symbols: self.parameter_symbols.clone(),
        }
    }
    pub fn acos(&self) -> ParameterExpression {
        ParameterExpression {
            expr: self.expr.acos(),
            parameter_symbols: self.parameter_symbols.clone(),
        }
    }
    pub fn atan(&self) -> ParameterExpression {
        ParameterExpression {
            expr: self.expr.atan(),
            parameter_symbols: self.parameter_symbols.clone(),
        }
    }
    pub fn exp(&self) -> ParameterExpression {
        ParameterExpression {
            expr: self.expr.exp(),
            parameter_symbols: self.parameter_symbols.clone(),
        }
    }
    pub fn log(&self) -> ParameterExpression {
        ParameterExpression {
            expr: self.expr.log(),
            parameter_symbols: self.parameter_symbols.clone(),
        }
    }
    pub fn abs(&self) -> ParameterExpression {
        ParameterExpression {
            expr: self.expr.abs(),
            parameter_symbols: self.parameter_symbols.clone(),
        }
    }
    pub fn sign(&self) -> ParameterExpression {
        ParameterExpression {
            expr: self.expr.sign(),
            parameter_symbols: self.parameter_symbols.clone(),
        }
    }

    /// return conjugate of expression
    pub fn conjugate(&self) -> ParameterExpression {
        ParameterExpression {
            expr: self.expr.conjugate(),
            parameter_symbols: self.parameter_symbols.clone(),
        }
    }

    // default functions for binary operations
    pub fn add(&self, rhs: &ParameterExpression) -> ParameterExpression {
        ParameterExpression {
            expr: &self.expr + &rhs.expr,
            parameter_symbols: self.merge_parameter_symbols(rhs),
        }
    }
    pub fn radd(&self, lhs: &ParameterExpression) -> ParameterExpression {
        ParameterExpression {
            expr: &lhs.expr + &self.expr,
            parameter_symbols: self.merge_parameter_symbols(lhs),
        }
    }
    pub fn sub(&self, rhs: &ParameterExpression) -> ParameterExpression {
        ParameterExpression {
            expr: &self.expr - &rhs.expr,
            parameter_symbols: self.merge_parameter_symbols(rhs),
        }
    }
    pub fn rsub(&self, lhs: &ParameterExpression) -> ParameterExpression {
        ParameterExpression {
            expr: &lhs.expr - &self.expr,
            parameter_symbols: self.merge_parameter_symbols(lhs),
        }
    }
    pub fn mul(&self, rhs: &ParameterExpression) -> ParameterExpression {
        ParameterExpression {
            expr: &self.expr * &rhs.expr,
            parameter_symbols: self.merge_parameter_symbols(rhs),
        }
    }
    pub fn rmul(&self, lhs: &ParameterExpression) -> ParameterExpression {
        ParameterExpression {
            expr: &lhs.expr * &self.expr,
            parameter_symbols: self.merge_parameter_symbols(lhs),
        }
    }
    pub fn div(&self, rhs: &ParameterExpression) -> ParameterExpression {
        ParameterExpression {
            expr: &self.expr / &rhs.expr,
            parameter_symbols: self.merge_parameter_symbols(rhs),
        }
    }
    pub fn rdiv(&self, lhs: &ParameterExpression) -> ParameterExpression {
        ParameterExpression {
            expr: &lhs.expr / &self.expr,
            parameter_symbols: self.merge_parameter_symbols(lhs),
        }
    }
    pub fn pow(&self, rhs: &ParameterExpression) -> ParameterExpression {
        ParameterExpression {
            expr: self.expr.pow(&rhs.expr),
            parameter_symbols: self.merge_parameter_symbols(rhs),
        }
    }
    pub fn rpow(&self, lhs: &ParameterExpression) -> ParameterExpression {
        ParameterExpression {
            expr: lhs.expr.pow(&self.expr),
            parameter_symbols: self.merge_parameter_symbols(lhs),
        }
    }

    pub fn substitute(
        &self,
        in_map: HashMap<ParameterExpression, ParameterExpression>,
        eval: bool,
        allow_unknown_parameters: bool,
    ) -> PyResult<ParameterExpression> {
        let mut map: HashMap<String, SymbolExpr> = HashMap::new();
        let mut subs_map: HashMap<ParameterExpression, ParameterValueType> = HashMap::new();
        let mut unknown_params: HashSet<String> = HashSet::new();
        let mut symbols = match &self.parameter_symbols {
            Some(s) => s.clone(),
            None => HashMap::new(),
        };

        for (key, param) in &in_map {
            // check if value in map is valid
            if let SymbolExpr::Value(Value::Real(r)) = &param.expr {
                if r.is_nan() || r.is_infinite() {
                    return Err(CircuitError::new_err(
                        "Expression cannot bind non-numeric values",
                    ));
                }
            } else if let SymbolExpr::Value(Value::Complex(c)) = &param.expr {
                if c.is_nan() || c.is_infinite() {
                    return Err(CircuitError::new_err(
                        "Expression cannot bind non-numeric values",
                    ));
                }
            }

            if key != param {
                self._raise_if_parameter_conflict(param)?;
                map.insert(key.to_string(), param.expr());
            }
            subs_map.insert(
                key.clone(),
                ParameterValueType::clone_expr_for_replay(param),
            );
            if symbols.contains_key(&key.to_string()) {
                symbols.remove(&key.to_string());
            } else if !allow_unknown_parameters {
                unknown_params.insert(key.to_string());
            }

            if let Some(map) = &param.parameter_symbols {
                for (s, u) in map {
                    symbols.insert(s.clone(), *u);
                }
            }
        }
        if !allow_unknown_parameters && !unknown_params.is_empty() {
            return Err(CircuitError::new_err(format!(
                "Cannot bind Parameters ({:?}) not present in expression.",
                unknown_params
            )));
        }

        let bound = self.expr.subs(&map);

        if eval && symbols.is_empty() {
            let ret = match bound.eval(true) {
                Some(v) => match &v {
                    symbol_expr::Value::Real(r) => {
                        if r.is_infinite() {
                            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                                "zero division occurs while binding parameter",
                            ));
                        } else if r.is_nan() {
                            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                                "NAN detected while binding parameter",
                            ));
                        } else {
                            SymbolExpr::Value(v)
                        }
                    }
                    symbol_expr::Value::Complex(c) => {
                        if c.re.is_infinite() || c.im.is_infinite() {
                            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                                "zero division occurs while binding parameter",
                            ));
                        } else if c.re.is_nan() || c.im.is_nan() {
                            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                                "NAN detected while binding parameter",
                            ));
                        } else if (-symbol_expr::SYMEXPR_EPSILON..symbol_expr::SYMEXPR_EPSILON)
                            .contains(&c.im)
                        {
                            SymbolExpr::Value(symbol_expr::Value::Real(c.re))
                        } else {
                            SymbolExpr::Value(v)
                        }
                    }
                    _ => SymbolExpr::Value(v),
                },
                None => bound,
            };
            Ok(ParameterExpression {
                expr: ret,
                parameter_symbols: if !symbols.is_empty() {
                    Some(symbols)
                } else {
                    None
                },
            })
        } else {
            Ok(ParameterExpression {
                expr: bound,
                parameter_symbols: if !symbols.is_empty() {
                    Some(symbols)
                } else {
                    None
                },
            })
        }
    }

    // compare 2 expression
    // check_uuid = true also compares uuid for equality
    pub fn compare_eq(&self, other: &ParameterExpression, check_uuid: bool) -> bool {
        match (&self.expr, &other.expr) {
            (SymbolExpr::Symbol(lhs), SymbolExpr::Symbol(rhs)) => {
                if lhs == rhs {
                    if check_uuid {
                        self.uuid() == other.uuid()
                    } else {
                        true
                    }
                } else {
                    false
                }
            }
            (_, _) => {
                if self.expr == other.expr {
                    if check_uuid {
                        // if there are some conflicts, the equation is not equal
                        let conflicts = self.get_conflict_parameters(other);
                        conflicts.is_empty()
                    } else {
                        true
                    }
                } else {
                    false
                }
            }
        }
    }

    /// expand expression
    pub fn expand(&self) -> Self {
        ParameterExpression {
            expr: self.expr.expand(),
            parameter_symbols: self.parameter_symbols.clone(),
        }
    }

    /// calculate gradient
    pub fn gradient(&self, param: &Self) -> Result<Self, String> {
        if let Some(parameter_symbols) = &self.parameter_symbols {
            if parameter_symbols.is_empty() {
                return Ok(ParameterExpression {
                    expr: SymbolExpr::Value(Value::Int(0)),
                    parameter_symbols: None,
                });
            }

            let expr_grad = match self.expr.derivative(&param.expr) {
                Ok(expr) => expr,
                Err(e) => return Err(e),
            };
            match expr_grad.eval(true) {
                Some(v) => Ok(ParameterExpression {
                    expr: SymbolExpr::Value(v),
                    parameter_symbols: None,
                }),
                None => {
                    // update parameter symbols
                    let symbols = expr_grad.symbols();
                    let mut new_map = HashMap::<Arc<String>, u128>::new();
                    for (s, u) in parameter_symbols {
                        if symbols.contains(s.as_ref()) {
                            new_map.insert(s.clone(), *u);
                        }
                    }
                    Ok(ParameterExpression {
                        expr: expr_grad,
                        parameter_symbols: Some(new_map),
                    })
                }
            }
        } else {
            Ok(ParameterExpression {
                expr: SymbolExpr::Value(Value::Int(0)),
                parameter_symbols: None,
            })
        }
    }

    // make replay recursive function
    fn _make_qpy_replay(
        &self,
        param: &SymbolExpr,
        replay: &mut Vec<OPReplay>,
    ) -> Option<ParameterValueType> {
        match param {
            SymbolExpr::Value(v) => match v {
                symbol_expr::Value::Int(i) => Some(ParameterValueType::Int(*i)),
                symbol_expr::Value::Real(r) => Some(ParameterValueType::Float(*r)),
                symbol_expr::Value::Complex(c) => Some(ParameterValueType::Complex(*c)),
            },
            SymbolExpr::Symbol(s) => match &self.parameter_symbols {
                Some(map) => {
                    for (k, u) in map {
                        if k.as_ref() == s.as_ref() {
                            return Some(ParameterValueType::Parameter(ParameterExpression::new(
                                k.as_ref().clone(),
                                Some(*u),
                            )));
                        }
                    }
                    None
                }
                None => None,
            },
            SymbolExpr::Unary { op, expr } => {
                match self._make_qpy_replay(expr, replay) {
                    Some(lhs) => {
                        let op = match op {
                            symbol_expr::UnaryOp::Abs => _OPCode::ABS,
                            symbol_expr::UnaryOp::Acos => _OPCode::ACOS,
                            symbol_expr::UnaryOp::Asin => _OPCode::ASIN,
                            symbol_expr::UnaryOp::Atan => _OPCode::ATAN,
                            symbol_expr::UnaryOp::Conj => _OPCode::CONJ,
                            symbol_expr::UnaryOp::Cos => _OPCode::COS,
                            symbol_expr::UnaryOp::Exp => _OPCode::EXP,
                            symbol_expr::UnaryOp::Log => _OPCode::LOG,
                            symbol_expr::UnaryOp::Neg => _OPCode::MUL,
                            symbol_expr::UnaryOp::Sign => _OPCode::SIGN,
                            symbol_expr::UnaryOp::Sin => _OPCode::SIN,
                            symbol_expr::UnaryOp::Tan => _OPCode::TAN,
                        };
                        if let _OPCode::MUL = &op {
                            // make neg as multiply -1
                            replay.push(OPReplay::_INSTRUCTION {
                                op,
                                lhs: Some(lhs),
                                rhs: Some(ParameterValueType::Int(-1)),
                            });
                        } else {
                            replay.push(OPReplay::_INSTRUCTION {
                                op,
                                lhs: Some(lhs),
                                rhs: None,
                            });
                        }
                        Some(ParameterValueType::Parameter(ParameterExpression {
                            expr: param.clone(),
                            parameter_symbols: None,
                        }))
                    }
                    None => None,
                }
            }
            SymbolExpr::Binary { op, lhs, rhs } => {
                let lhs = self._make_qpy_replay(lhs.as_ref(), replay);
                let rhs = self._make_qpy_replay(rhs.as_ref(), replay);
                if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
                    match lhs {
                        ParameterValueType::Parameter(_) => {
                            let op = match op {
                                symbol_expr::BinaryOp::Add => _OPCode::ADD,
                                symbol_expr::BinaryOp::Sub => _OPCode::SUB,
                                symbol_expr::BinaryOp::Mul => _OPCode::MUL,
                                symbol_expr::BinaryOp::Div => _OPCode::DIV,
                                symbol_expr::BinaryOp::Pow => _OPCode::POW,
                            };
                            replay.push(OPReplay::_INSTRUCTION {
                                op,
                                lhs: Some(lhs),
                                rhs: Some(rhs),
                            });
                        }
                        _ => {
                            let op = match op {
                                symbol_expr::BinaryOp::Add => _OPCode::ADD,
                                symbol_expr::BinaryOp::Sub => _OPCode::RSUB,
                                symbol_expr::BinaryOp::Mul => _OPCode::MUL,
                                symbol_expr::BinaryOp::Div => _OPCode::RDIV,
                                symbol_expr::BinaryOp::Pow => _OPCode::RPOW,
                            };
                            replay.push(OPReplay::_INSTRUCTION {
                                op,
                                lhs: Some(rhs),
                                rhs: Some(lhs),
                            });
                        }
                    }
                    return Some(ParameterValueType::Parameter(ParameterExpression {
                        expr: param.clone(),
                        parameter_symbols: None,
                    }));
                }
                None
            }
        }
    }

    pub fn make_qpy_replay(&self) -> Option<Vec<OPReplay>> {
        match self.expr {
            SymbolExpr::Binary { .. } | SymbolExpr::Unary { .. } => {
                let mut replay = Vec::<OPReplay>::new();
                self._make_qpy_replay(&self.expr, &mut replay)
                    .map(|_| replay)
            }
            _ => None,
        }
    }
}

impl PartialEq for ParameterExpression {
    fn eq(&self, rhs: &Self) -> bool {
        self.compare_eq(rhs, false)
    }
}

impl Eq for ParameterExpression {}

impl PartialEq<f64> for ParameterExpression {
    fn eq(&self, r: &f64) -> bool {
        &self.expr == r
    }
}

impl PartialEq<Complex64> for ParameterExpression {
    fn eq(&self, c: &Complex64) -> bool {
        &self.expr == c
    }
}

// =============================
// Make from Rust native types
// =============================

impl From<i32> for ParameterExpression {
    fn from(v: i32) -> Self {
        ParameterExpression {
            expr: SymbolExpr::Value(symbol_expr::Value::Real(v as f64)),
            parameter_symbols: None,
        }
    }
}
impl From<i64> for ParameterExpression {
    fn from(v: i64) -> Self {
        ParameterExpression {
            expr: SymbolExpr::Value(symbol_expr::Value::Real(v as f64)),
            parameter_symbols: None,
        }
    }
}

impl From<u32> for ParameterExpression {
    fn from(v: u32) -> Self {
        ParameterExpression {
            expr: SymbolExpr::Value(symbol_expr::Value::Real(v as f64)),
            parameter_symbols: None,
        }
    }
}

impl From<f64> for ParameterExpression {
    fn from(v: f64) -> Self {
        ParameterExpression {
            expr: SymbolExpr::Value(symbol_expr::Value::Real(v)),
            parameter_symbols: None,
        }
    }
}

impl From<Complex64> for ParameterExpression {
    fn from(v: Complex64) -> Self {
        ParameterExpression {
            expr: SymbolExpr::Value(symbol_expr::Value::Complex(v)),
            parameter_symbols: None,
        }
    }
}

impl From<&str> for ParameterExpression {
    fn from(s: &str) -> Self {
        ParameterExpression::new(s.to_string(), None)
    }
}

// =============================
// Unary operations
// =============================
impl Neg for &ParameterExpression {
    type Output = ParameterExpression;
    fn neg(self) -> Self::Output {
        self.neg()
    }
}

// =============================
// Add
// =============================
impl Add for &ParameterExpression {
    type Output = ParameterExpression;
    #[inline]
    fn add(self, other: &ParameterExpression) -> Self::Output {
        self.add(other)
    }
}

macro_rules! add_impl_expr {
    ($($t:ty)*) => ($(
        impl Add<$t> for &ParameterExpression {
            type Output = ParameterExpression;

            #[inline]
            #[track_caller]
            fn add(self, other: $t) -> Self::Output {
                self.add(&other.into())
            }
        }
    )*)
}

add_impl_expr! {f64 i32 u32}

// =============================
// Sub
// =============================
impl Sub for &ParameterExpression {
    type Output = ParameterExpression;
    #[inline]
    fn sub(self, other: &ParameterExpression) -> Self::Output {
        self.sub(other)
    }
}

macro_rules! sub_impl_expr {
    ($($t:ty)*) => ($(
        impl Sub<$t> for &ParameterExpression {
            type Output = ParameterExpression;

            #[inline]
            #[track_caller]
            fn sub(self, other: $t) -> Self::Output {
                self.sub(&other.into())
            }
        }
    )*)
}

sub_impl_expr! {f64 i32 u32}

// =============================
// Mul
// =============================
impl Mul for &ParameterExpression {
    type Output = ParameterExpression;
    #[inline]
    fn mul(self, other: &ParameterExpression) -> Self::Output {
        self.mul(other)
    }
}

macro_rules! mul_impl_expr {
    ($($t:ty)*) => ($(
        impl Mul<$t> for &ParameterExpression {
            type Output = ParameterExpression;

            #[inline]
            #[track_caller]
            fn mul(self, other: $t) -> Self::Output {
                self.mul(&other.into())
            }
        }
    )*)
}

mul_impl_expr! {f64 i32 u32}

// =============================
// Div
// =============================
impl Div for &ParameterExpression {
    type Output = ParameterExpression;
    #[inline]
    fn div(self, other: &ParameterExpression) -> Self::Output {
        self.div(other)
    }
}

macro_rules! div_impl_expr {
    ($($t:ty)*) => ($(
        impl Div<$t> for &ParameterExpression {
            type Output = ParameterExpression;

            #[inline]
            #[track_caller]
            fn div(self, other: $t) -> Self::Output {
                self.div(&other.into())
            }
        }
    )*)
}
div_impl_expr! {f64 i32 u32}

// =================================================================
//   Python interface to ParameterExpression/Parameter/ParameterVectorElement
// =================================================================
#[pymethods]
impl ParameterExpression {
    /// ParameterExpression::__init__
    /// initialize ParameterExpression from the equation stored in string
    #[new]
    #[pyo3(signature = (symbol_map = None, expr = None, _qpy_replay = None))]
    pub fn __new__(
        symbol_map: Option<HashSet<ParameterExpression>>,
        expr: Option<Bound<PyAny>>,
        _qpy_replay: Option<Vec<OPReplay>>,
    ) -> PyResult<Self> {
        let Some(expr) = expr else {
            return Ok(ParameterExpression::default());
        };

        if let Ok(expr) = expr.extract::<ParameterExpression>() {
            Ok(expr)
        } else if let Ok(expr) = expr.extract::<String>() {
            // check if expr contains replacements for sympy
            let expr = expr
                .replace("__begin_sympy_replace__", "$\\")
                .replace("__end_sympy_replace__", "$");
            match parse_expression(&expr) {
                Ok(expr) => {
                    let mut parameter_symbols = HashMap::<Arc<String>, u128>::new();
                    if let Some(symbol_map) = symbol_map {
                        for param in symbol_map {
                            let u = param.uuid();
                            if *u == 0_u128 {
                                parameter_symbols
                                    .insert(Arc::new(param.to_string()), Uuid::new_v4().as_u128());
                            } else {
                                parameter_symbols.insert(Arc::new(param.to_string()), *u);
                            }
                        }
                    } else {
                        for symbol in expr.symbols() {
                            parameter_symbols
                                .insert(Arc::new(symbol.clone()), Uuid::new_v4().as_u128());
                        }
                    }
                    // substitute 'I' to imaginary number i before returning expression
                    Ok(ParameterExpression {
                        expr: expr.bind(&HashMap::from([(
                            "I".to_string(),
                            symbol_expr::Value::from(Complex64::i()),
                        )])),
                        parameter_symbols: Some(parameter_symbols),
                    })
                }
                Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
            }
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Unsupported data type to initialize ParameterExpression.",
            ))
        }
    }

    /// create new expression as symbol
    #[allow(non_snake_case)]
    #[staticmethod]
    #[pyo3(signature = (name, uuid = None))]
    pub fn Symbol(name: String, uuid: Option<u128>) -> PyResult<Self> {
        // check if expr contains replacements for sympy
        let name = name
            .replace("__begin_sympy_replace__", "$\\")
            .replace("__end_sympy_replace__", "$");

        Ok(ParameterExpression::new(name, uuid))
    }

    /// create new expression as a value
    /// input value should be one of int/real/complex data types
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

    // return string to pass to sympify
    #[pyo3(name = "sympify")]
    pub fn py_sympify(&self) -> String {
        let ret = self.expr.optimize().sympify().to_string();
        ret.replace("$\\", "__begin_sympy_replace__")
            .replace('$', "__end_sympy_replace__")
    }

    // for backward compatibility
    // just return clone of myself
    #[getter("_symbol_expr")]
    pub fn py_get_symbol_expr(&self) -> ParameterExpression {
        self.clone()
    }

    /// return numeric value if expression does not contain any symbols
    #[pyo3(name = "numeric")]
    pub fn py_get_numeric_value(&self, py: Python) -> PyResult<PyObject> {
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
            None => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Expression with unbound parameters '{:?}' is not numeric",
                self.parameter_symbols
            ))),
        }
    }

    /// check if this is symbol
    #[getter("is_symbol")]
    pub fn py_is_symbol(&self) -> bool {
        self.is_symbol()
    }

    /// check if this is numeric
    #[getter("is_numeric")]
    pub fn py_is_numeric(&self) -> bool {
        self.is_numeric()
    }

    /// check if this is not complex
    #[pyo3(name = "is_real")]
    pub fn py_is_real(&self) -> Option<bool> {
        self.is_real()
    }

    /// get uuid for this symbol
    #[pyo3(name = "get_uuid")]
    pub fn py_get_uuid(&self) -> u128 {
        *self.uuid()
    }

    /// return this as complex if this is numeric
    pub fn __complex__(&self) -> PyResult<Complex64> {
        match self.expr.eval(true) {
            Some(v) => match v {
                symbol_expr::Value::Real(r) => Ok(Complex64::from(r)),
                symbol_expr::Value::Int(i) => Ok(Complex64::from(i as f64)),
                symbol_expr::Value::Complex(c) => Ok(c),
            },
            None => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "ParameterExpression with unbound parameters ({:?}) cannot be cast to a complex.",
                self.parameter_symbols
            ))),
        }
    }

    /// return this as real if this is numeric
    pub fn __float__(&self) -> PyResult<f64> {
        match self.expr.eval(true) {
            Some(v) => match v {
                symbol_expr::Value::Real(r) => Ok(r),
                symbol_expr::Value::Int(i) => Ok(i as f64),
                symbol_expr::Value::Complex(c) => {
                    if (-symbol_expr::SYMEXPR_EPSILON..symbol_expr::SYMEXPR_EPSILON).contains(&c.im)
                    {
                        Ok(c.re)
                    } else {
                        Err(pyo3::exceptions::PyTypeError::new_err(
                            "could not cast expression to float",
                        ))
                    }
                }
            },
            None => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "ParameterExpression with unbound parameters ({:?}) cannot be cast to a float.",
                self.parameter_symbols
            ))),
        }
    }

    /// return this as int if this is numeric
    pub fn __int__(&self) -> PyResult<i64> {
        match self.expr.eval(true) {
            Some(v) => match v {
                symbol_expr::Value::Real(r) => Ok(r as i64),
                symbol_expr::Value::Int(i) => Ok(i),
                symbol_expr::Value::Complex(c) => {
                    if (-symbol_expr::SYMEXPR_EPSILON..symbol_expr::SYMEXPR_EPSILON).contains(&c.im)
                    {
                        Ok(c.re as i64)
                    } else {
                        Err(pyo3::exceptions::PyTypeError::new_err(
                            "could not cast expression to int",
                        ))
                    }
                }
            },
            None => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "ParameterExpression with unbound parameters ({:?}) cannot be cast to int.",
                self.parameter_symbols
            ))),
        }
    }

    /// clone expression
    pub fn py_copy(&self) -> Self {
        self.clone()
    }

    /// return derivative of this expression for param
    pub fn py_derivative(&self, param: &ParameterExpression) -> PyResult<ParameterExpression> {
        match self.gradient(param) {
            Ok(expr) => Ok(expr),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }

    /// return conjugate of expression
    pub fn py_conjugate(&self) -> ParameterExpression {
        self.conjugate()
    }

    /// Get the derivative of a parameter expression w.r.t. a specified parameter expression.
    /// Args:
    ///     param (Parameter): Parameter w.r.t. which we want to take the derivative
    /// Returns:
    ///     ParameterExpression representing the gradient of param_expr w.r.t. param
    ///     or complex or float number
    pub fn py_gradient(&self, param: &Self, py: Python) -> PyResult<PyObject> {
        match self.gradient(param) {
            Ok(grad) => match &grad.expr {
                SymbolExpr::Value(v) => match v {
                    symbol_expr::Value::Real(r) => r.into_py_any(py),
                    symbol_expr::Value::Int(i) => i.into_py_any(py),
                    symbol_expr::Value::Complex(c) => {
                        if (-symbol_expr::SYMEXPR_EPSILON..symbol_expr::SYMEXPR_EPSILON)
                            .contains(&c.im)
                        {
                            c.re.into_py_any(py)
                        } else {
                            c.into_py_any(py)
                        }
                    }
                },
                _ => grad.into_py_any(py),
            },
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }

    /// get hashset of all the symbols used in this expression
    #[getter("parameters")]
    pub fn py_parameters(&self, py: Python) -> PyResult<Py<PySet>> {
        let out = PySet::empty(py)?;
        match &self.parameter_symbols {
            Some(symbols) => {
                for (s, u) in symbols {
                    out.add(ParameterExpression::new(s.as_ref().clone(), Some(*u)))?;
                }
                Ok(out.unbind())
            }
            None => Ok(out.unbind()),
        }
    }

    /// return all values in this equation
    #[pyo3(name = "values")]
    pub fn py_values(&self, py: Python) -> PyResult<Vec<PyObject>> {
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
    #[getter("name")]
    pub fn py_name(&self) -> String {
        self.__str__()
    }

    pub fn py_assign(&self, param: &ParameterExpression, value: &Bound<PyAny>) -> PyResult<Self> {
        if let Some(e) = _extract_value(value) {
            let eval = matches!(&e.expr, SymbolExpr::Value(_));
            if self == param {
                return Ok(e);
            }
            self.substitute(HashMap::from([(param.clone(), e)]), eval, false)
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "unsupported data type is passed to assign parameter",
            ))
        }
    }

    /// bind values to symbols given by input hashmap
    #[pyo3(signature = (in_map, allow_unknown_parameters = None))]
    pub fn py_bind(
        &self,
        in_map: HashMap<ParameterExpression, Bound<PyAny>>,
        allow_unknown_parameters: Option<bool>,
    ) -> PyResult<Self> {
        let mut map: HashMap<ParameterExpression, ParameterExpression> = HashMap::new();
        for (key, val) in in_map {
            if let Ok(e) = Self::Value(&val) {
                map.insert(key.clone(), e);
            }
        }
        self.substitute(map, true, allow_unknown_parameters.unwrap_or(false))
    }

    /// substitute symbols to expressions (or values) given by hash map
    #[pyo3(signature = (map, allow_unknown_parameters = None))]
    pub fn py_subs(
        &self,
        map: HashMap<ParameterExpression, ParameterExpression>,
        allow_unknown_parameters: Option<bool>,
    ) -> PyResult<Self> {
        self.substitute(map, false, allow_unknown_parameters.unwrap_or(false))
    }

    fn _raise_if_parameter_conflict(&self, other: &ParameterExpression) -> PyResult<bool> {
        let conflicts = self.get_conflict_parameters(other);
        if !conflicts.is_empty() {
            Err(CircuitError::new_err(format!(
                "Name conflict applying operation for parameters: {:?}",
                conflicts
            )))
        } else {
            Ok(true)
        }
    }

    // ====================================
    // operator overrides
    // ====================================
    pub fn __eq__(&self, rhs: &Bound<PyAny>) -> bool {
        match _extract_value(rhs) {
            Some(rhs) => self.compare_eq(&rhs, true),
            None => false,
        }
    }
    pub fn __ne__(&self, rhs: &Bound<PyAny>) -> bool {
        !self.__eq__(rhs)
    }

    pub fn __lt__(&self, rhs: &Bound<PyAny>) -> bool {
        match _extract_value(rhs) {
            Some(rhs) => self.expr < rhs.expr,
            None => false,
        }
    }
    pub fn __gt__(&self, rhs: &Bound<PyAny>) -> bool {
        match _extract_value(rhs) {
            Some(rhs) => self.expr > rhs.expr,
            None => false,
        }
    }

    // unary operators
    pub fn py_neg(&self) -> ParameterExpression {
        self.neg()
    }
    pub fn py_pos(&self) -> ParameterExpression {
        self.pos()
    }
    pub fn py_sin(&self) -> ParameterExpression {
        self.sin()
    }
    pub fn py_cos(&self) -> ParameterExpression {
        self.cos()
    }
    pub fn py_tan(&self) -> ParameterExpression {
        self.tan()
    }
    pub fn py_arcsin(&self) -> ParameterExpression {
        self.asin()
    }
    pub fn py_arccos(&self) -> ParameterExpression {
        self.acos()
    }
    pub fn py_arctan(&self) -> ParameterExpression {
        self.atan()
    }
    pub fn py_exp(&self) -> ParameterExpression {
        self.exp()
    }
    pub fn py_log(&self) -> ParameterExpression {
        self.log()
    }
    pub fn py_abs(&self) -> ParameterExpression {
        self.abs()
    }
    pub fn py_sign(&self) -> ParameterExpression {
        self.sign()
    }

    // binary operators
    pub fn py_add(&self, rhs: &Bound<PyAny>) -> PyResult<ParameterExpression> {
        match _extract_value(rhs) {
            Some(rhs) => {
                self._raise_if_parameter_conflict(&rhs)?;
                Ok(self.add(&rhs))
            }
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __add__",
            )),
        }
    }
    pub fn py_radd(&self, lhs: &Bound<PyAny>) -> PyResult<ParameterExpression> {
        match _extract_value(lhs) {
            Some(lhs) => {
                self._raise_if_parameter_conflict(&lhs)?;
                Ok(self.radd(&lhs))
            }
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __radd__",
            )),
        }
    }
    pub fn py_sub(&self, rhs: &Bound<PyAny>) -> PyResult<ParameterExpression> {
        match _extract_value(rhs) {
            Some(rhs) => {
                self._raise_if_parameter_conflict(&rhs)?;
                Ok(self.sub(&rhs))
            }
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __sub__",
            )),
        }
    }
    pub fn py_rsub(&self, lhs: &Bound<PyAny>) -> PyResult<ParameterExpression> {
        match _extract_value(lhs) {
            Some(lhs) => {
                self._raise_if_parameter_conflict(&lhs)?;
                Ok(self.rsub(&lhs))
            }
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __rsub__",
            )),
        }
    }
    pub fn py_mul(&self, rhs: &Bound<PyAny>) -> PyResult<ParameterExpression> {
        match _extract_value(rhs) {
            Some(rhs) => {
                self._raise_if_parameter_conflict(&rhs)?;
                Ok(self.mul(&rhs))
            }
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __mul__",
            )),
        }
    }

    pub fn py_rmul(&self, lhs: &Bound<PyAny>) -> PyResult<ParameterExpression> {
        match _extract_value(lhs) {
            Some(lhs) => {
                self._raise_if_parameter_conflict(&lhs)?;
                Ok(self.rmul(&lhs))
            }
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __rmul__",
            )),
        }
    }

    pub fn py_div(&self, rhs: &Bound<PyAny>) -> PyResult<ParameterExpression> {
        match _extract_value(rhs) {
            Some(rhs) => {
                if let SymbolExpr::Value(v) = &rhs.expr {
                    let zero = match v {
                        symbol_expr::Value::Int(i) => *i == 0,
                        symbol_expr::Value::Real(r) => *r == 0.0,
                        symbol_expr::Value::Complex(c) => c.re == 0.0 && c.im == 0.0,
                    };
                    if zero {
                        return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                            "Division of a ParameterExpression by zero.",
                        ));
                    }
                }
                self._raise_if_parameter_conflict(&rhs)?;
                Ok(self.div(&rhs))
            }
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __truediv__",
            )),
        }
    }

    pub fn py_rdiv(&self, lhs: &Bound<PyAny>) -> PyResult<ParameterExpression> {
        match _extract_value(lhs) {
            Some(lhs) => {
                self._raise_if_parameter_conflict(&lhs)?;
                Ok(self.rdiv(&lhs))
            }
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __truediv__",
            )),
        }
    }


    #[pyo3(signature = (rhs, _modulo = None))]
    pub fn py_pow(
        &self,
        rhs: &Bound<PyAny>,
        _modulo: Option<i32>,
    ) -> PyResult<ParameterExpression> {
        match _extract_value(rhs) {
            Some(rhs) => {
                self._raise_if_parameter_conflict(&rhs)?;
                Ok(self.pow(&rhs))
            }
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __pow__",
            )),
        }
    }
    #[pyo3(signature = (lhs, _modulo = None))]
    pub fn py_rpow(
        &self,
        lhs: &Bound<PyAny>,
        _modulo: Option<i32>,
    ) -> PyResult<ParameterExpression> {
        match _extract_value(lhs) {
            Some(lhs) => {
                self._raise_if_parameter_conflict(&lhs)?;
                Ok(self.rpow(&lhs))
            }
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __rpow__",
            )),
        }
    }

    pub fn __str__(&self) -> String {
        if let SymbolExpr::Symbol(s) = &self.expr {
            return s.as_ref().clone();
        }
        match self.expr.eval(true) {
            Some(e) => e.to_string(),
            None => self.expr.optimize().to_string(),
        }
    }

    pub fn __hash__(&self, py: Python) -> PyResult<isize> {
        match self.expr.eval(true) {
            Some(v) => match v {
                symbol_expr::Value::Int(i) => i.into_pyobject(py)?.hash(),
                symbol_expr::Value::Real(r) => r.into_pyobject(py)?.hash(),
                symbol_expr::Value::Complex(c) => c.into_pyobject(py)?.hash(),
            },
            None => self.expr.to_string().into_pyobject(py)?.hash(),
        }
    }

    // for pickle, we can reproduce equation from expression string
    fn __getstate__(&self) -> PyResult<(String, Option<HashMap<String, u128>>)> {
        if let Some(symbols) = &self.parameter_symbols {
            let mut ret = HashMap::<String, u128>::new();
            for (s, u) in symbols {
                ret.insert(s.as_ref().clone(), *u);
            }
            return Ok((self.to_string(), Some(ret)));
        }
        Ok((self.to_string(), None))
    }

    fn __setstate__(&mut self, state: (String, Option<HashMap<String, u128>>)) -> PyResult<()> {
        match parse_expression(&state.0) {
            Ok(expr) => {
                if let Some(symbols) = state.1 {
                    let mut parameter_symbols = HashMap::<Arc<String>, u128>::new();
                    for (name, uuid) in symbols {
                        parameter_symbols.insert(Arc::new(name), uuid);
                    }
                    self.expr = expr;
                    self.parameter_symbols = Some(parameter_symbols);
                } else {
                    self.expr = expr;
                    self.parameter_symbols = None;
                }
                Ok(())
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }

    /// return QPY replay
    #[pyo3(name = "replay")]
    pub fn py_get_replay(&self) -> Option<Vec<OPReplay>> {
        self.make_qpy_replay()
    }
    #[getter("_qpy_replay")]
    pub fn py_qpy_replay(&self) -> Option<Vec<OPReplay>> {
        self.make_qpy_replay()
    }
}

/*
#[pyclass]
struct ParameterIter {
    inner: std::vec::IntoIter<ParameterExpression>,
}

#[pymethods]
impl ParameterIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PyObject> {
        // make new vector element object from ParameterExpression for next object
        match slf.inner.next() {
            Some(e) => match e.into_py_any(slf.py()) {
                Ok(e) => Some(e),
                Err(_) => None,
            },
            None => None,
        }
    }
}
*/
// ===========================================
// ParameterVector class
// ===========================================
/*#[pyclass(sequence, subclass, module = "qiskit._accelerate.circuit")]
#[derive(Clone, Debug)]
pub struct ParameterVector {
    name: String,
    root_uuid: u128,
    params: Vec<ParameterExpression>,
}

#[pymethods]
impl ParameterVector {
    #[new]
    #[pyo3(signature = (name="".to_string(), length = 0, uuid = None))]
    pub fn new(name: String, length: usize, uuid: Option<u128>) -> PyResult<Self> {
        // check if expr contains replacements for sympy
        let name = name
            .replace("__begin_sympy_replace__", "$\\")
            .replace("__end_sympy_replace__", "$");

        let root_uuid = match uuid {
            Some(uuid) => uuid,
            None => Uuid::new_v4().as_u128(),
        };
        let mut ret = ParameterVector {
            name: name.clone(),
            root_uuid,
            params: Vec::with_capacity(length),
        };

        for i in 0..length {
            let pe = ParameterExpression {
                inner: ParameterInner::VectorElement {
                    name: Arc::new(name.clone()),
                    index: i,
                    uuid: root_uuid + i as u128,
                    vector: None,
                },
            };
            ret.params.push(pe);
        }
        let t = Arc::<ParameterVector>::new(ret.to_owned());
        for pe in &mut ret.params {
            pe.set_vector(t.clone());
        }
        Ok(ret)
    }

    /// """The name of the :class:`ParameterVector`."""
    #[getter("name")]
    pub fn py_get_name(&self) -> String {
        self.name.clone()
    }
    #[getter("_name")]
    pub fn py_name(&self) -> String {
        self.name.clone()
    }

    #[getter("_root_uuid")]
    pub fn py_root_uuid(&self) -> u128 {
        self.root_uuid
    }

    /// """A list of the contained :class:`ParameterVectorElement` instances.
    ///
    /// It is not safe to mutate this list."""
    #[getter("params")]
    pub fn py_get_params(&self, py: Python) -> PyResult<Py<PyList>> {
        let out = PyList::empty(py);
        for s in &self.params {
            out.append(s.clone())?;
        }
        Ok(out.unbind())
    }

    #[pyo3(name = "index")]
    pub fn py_index(&self, param: &ParameterExpression) -> Option<usize> {
        for (i, p) in self.params.iter().enumerate() {
            if param == p {
                return Some(i);
            }
        }
        None
    }

    pub fn __getitem__(&self, py: Python, index: PySequenceIndex) -> PyResult<PyObject> {
        match index.with_len(self.params.len())? {
            SequenceIndex::Int(index) => self.params[index].clone().into_py_any(py),
            indices => {
                let out = PyList::empty(py);
                for i in indices {
                    out.append(self.params[i].clone())?;
                }
                out.into_py_any(py)
            }
        }
    }

    pub fn __setitem__(&mut self, index: PySequenceIndex, value: &Bound<PyAny>) -> PyResult<()> {
        match index.with_len(self.params.len())? {
            SequenceIndex::Int(index) => {
                if let Ok(e) = value.extract::<ParameterExpression>() {
                    self.params[index] = e;
                    Ok(())
                } else {
                    Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "unsupported data type is passed to ParameterVector.__setitem__",
                    ))
                }
            }
            indices => {
                if let Ok(v) = value.extract::<Vec<ParameterExpression>>() {
                    for (i, index) in indices.iter().enumerate() {
                        self.params[index] = v[i].clone();
                    }
                    Ok(())
                } else {
                    Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "unsupported data type is passed to ParameterVector.__setitem__",
                    ))
                }
            }
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<ParameterIter>> {
        let iter = ParameterIter {
            inner: slf.params.clone().into_iter(),
        };
        Py::new(slf.py(), iter)
    }
    pub fn __len__(&self) -> usize {
        self.params.len()
    }

    pub fn __str__(&self) -> String {
        let params: Vec<String> = self.params.iter().map(|p| p.to_string()).collect();
        let mut out = format!("{}, [", self.name).to_string();
        for (i, p) in params.iter().enumerate() {
            if i != params.len() - 1 {
                out += &format!("'{}', ", p);
            } else {
                out += &format!("'{}'", p);
            }
        }
        out += "]";
        out
    }

    pub fn __repr__(&self) -> String {
        format!(
            "ParameterVector(name='{}', length={})",
            self.name,
            self.params.len()
        )
        .to_string()
    }

    /// """Resize the parameter vector.  If necessary, new elements are generated.
    ///
    /// Note that the UUID of each :class:`.Parameter` element will be generated
    /// deterministically given the root UUID of the ``ParameterVector`` and the index
    /// of the element.  In particular, if a ``ParameterVector`` is resized to
    /// be smaller and then later resized to be larger, the UUID of the later
    /// generated element at a given index will be the same as the UUID of the
    /// previous element at that index.
    /// This is to ensure that the parameter instances do not change.
    ///
    /// >>> from qiskit.circuit import ParameterVector
    /// >>> pv = ParameterVector("theta", 20)
    /// >>> elt_19 = pv[19]
    /// >>> rv.resize(10)
    /// >>> rv.resize(20)
    /// >>> pv[19] == elt_19
    /// True
    /// """
    #[pyo3(name = "resize")]
    pub fn py_resize(&mut self, length: usize) {
        if length > self.params.len() {
            let root_uuid = self.root_uuid;
            for i in self.params.len()..length {
                let pe = ParameterExpression {
                    inner: ParameterInner::VectorElement {
                        name: Arc::new(self.name.clone()),
                        index: i,
                        uuid: root_uuid + i as u128,
                        vector: None,
                    },
                };
                self.params.push(pe);
            }
            let t = Arc::<ParameterVector>::new(self.to_owned());
            for pe in &mut self.params {
                pe.set_vector(t.clone());
            }
        } else {
            self.params.resize(length, ParameterExpression::default());
        }
    }

    fn __getstate__(&self) -> PyResult<(String, usize, u128)> {
        Ok((self.name.clone(), self.params.len(), self.root_uuid))
    }
    fn __setstate__(&mut self, state: (String, usize, u128)) -> PyResult<()> {
        self.name = state.0;
        self.root_uuid = state.2;
        let length = state.1;
        self.params = Vec::with_capacity(length);
        for i in 0..length {
            let pe = ParameterExpression {
                inner: ParameterInner::VectorElement {
                    name: Arc::new(self.name.clone()),
                    index: i,
                    uuid: self.root_uuid + i as u128,
                    vector: None,
                },
            };
            self.params.push(pe);
        }
        let t = Arc::<ParameterVector>::new(self.to_owned());
        for pe in &mut self.params {
            pe.set_vector(t.clone());
        }
        Ok(())
    }
}
*/
