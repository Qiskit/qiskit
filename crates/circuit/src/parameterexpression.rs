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
use crate::slice::{PySequenceIndex, SequenceIndex};
use crate::symbol_expr::{SymbolExpr, Value, SYMEXPR_EPSILON};
use crate::symbol_parser::parse_expression;

use hashbrown::{HashMap, HashSet};
use num_complex::Complex64;
use uuid::Uuid;

use core::f64;
use std::cmp::PartialOrd;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

use pyo3::import_exception;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::IntoPyObjectExt;

import_exception!(qiskit.circuit.exceptions, CircuitError);

// enum for acceptable types for parameter
#[derive(IntoPyObject, FromPyObject, Clone, Debug)]
pub enum ParameterValueType {
    Int(i64),
    Float(f64),
    Complex(Complex64),
    Expression(ParameterExpression),
}

impl ParameterValueType {
    fn clone_expr_for_replay(expr: &ParameterExpression) -> ParameterValueType {
        if expr.is_numeric() {
            if let Some(v) = expr.expr.eval(true) {
                return match v {
                    Value::Int(i) => ParameterValueType::Int(i),
                    Value::Real(r) => ParameterValueType::Float(r),
                    Value::Complex(c) => ParameterValueType::Complex(c),
                };
            }
        }
        ParameterValueType::Expression(ParameterExpression {
            expr: expr.expr.clone(),
            uuid: expr.uuid.clone(),
            qpy_replay: expr.qpy_replay.clone(),
            parameter_symbols: None,
            parameter_vector: expr.parameter_vector.clone(),
        })
    }
}

#[pyfunction]
fn _extract_value(value: &Bound<PyAny>) -> Option<ParameterExpression> {
    if let Ok(e) = value.extract::<ParameterExpression>() {
        Some(e)
    } else if let Ok(_) = value.extract::<String>() {
        None
    } else if let Ok(i) = value.extract::<i64>() {
        Some(ParameterExpression {
            expr: SymbolExpr::Value(Value::from(i)),
            uuid: 0,
            qpy_replay: None,
            parameter_symbols: None,
            parameter_vector: None,
        })
    } else if let Ok(c) = value.extract::<Complex64>() {
        if c.is_infinite() || c.is_nan() {
            return None;
        }
        Some(ParameterExpression {
            expr: SymbolExpr::Value(Value::from(c)),
            uuid: 0,
            qpy_replay: None,
            parameter_symbols: None,
            parameter_vector: None,
        })
    } else if let Ok(r) = value.extract::<f64>() {
        if r.is_infinite() || r.is_nan() {
            return None;
        }
        Some(ParameterExpression {
            expr: SymbolExpr::Value(Value::from(r)),
            uuid: 0,
            qpy_replay: None,
            parameter_symbols: None,
            parameter_vector: None,
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
#[pyclass(sequence, module = "qiskit._accelerate.circuit")]
#[derive(Clone, Debug)]
pub struct ParameterExpression {
    expr: SymbolExpr,
    uuid: u128,
    qpy_replay: Option<Vec<OPReplay>>,
    parameter_symbols: Option<HashSet<Arc<ParameterExpression>>>,
    parameter_vector: Option<Arc<ParameterVector>>,
}

impl Hash for ParameterExpression {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.expr.to_string().hash(state);
    }
}

impl Default for ParameterExpression {
    // default constructor returns zero
    fn default() -> Self {
        Self {
            expr: SymbolExpr::Value(Value::Int(0)),
            uuid: 0,
            qpy_replay: None,
            parameter_symbols: None,
            parameter_vector: None,
        }
    }
}

impl fmt::Display for ParameterExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expr)
    }
}

// ParameterExpression implementation for both Rust and Python
impl ParameterExpression {
    /// return number of symbols in this expression
    pub fn num_symbols(&self) -> usize {
        self.expr.symbols().len()
    }

    /// check if the symbol is used in this expression
    pub fn has_symbol(&self, symbol: String) -> bool {
        self.expr.symbols_in_string().contains(&symbol)
    }

    // return merged set of parameter symbils in 2 parameters
    fn merge_parameter_symbols(
        &self,
        other: &ParameterExpression,
    ) -> Option<HashSet<Arc<ParameterExpression>>> {
        let mut ret: HashSet<Arc<ParameterExpression>> = match &self.parameter_symbols {
            Some(s) => s.clone(),
            None => match self.expr {
                SymbolExpr::Symbol { name: _, index: _ } => {
                    HashSet::from([Arc::new(self.to_owned())])
                }
                _ => HashSet::new(),
            },
        };
        let other_symbols: &HashSet<Arc<ParameterExpression>> = match &other.parameter_symbols {
            Some(s) => s,
            None => match other.expr {
                SymbolExpr::Symbol { name: _, index: _ } => {
                    &HashSet::from([Arc::new(other.to_owned())])
                }
                _ => &HashSet::new(),
            },
        };
        for o in other_symbols {
            ret.insert(o.clone());
        }
        Some(ret)
    }

    // get conflict parameters
    fn get_conflict_parameters(&self, other: &ParameterExpression) -> HashSet<String> {
        let mut conflicts = HashSet::<String>::new();
        let my_symbols: &HashSet<Arc<ParameterExpression>> = match &self.parameter_symbols {
            Some(s) => s,
            None => match self.expr {
                SymbolExpr::Symbol { name: _, index: _ } => {
                    &HashSet::from([Arc::new(self.to_owned())])
                }
                _ => &HashSet::new(),
            },
        };
        let other_symbols: &HashSet<Arc<ParameterExpression>> = match &other.parameter_symbols {
            Some(s) => s,
            None => match other.expr {
                SymbolExpr::Symbol { name: _, index: _ } => {
                    &HashSet::from([Arc::new(other.to_owned())])
                }
                _ => &HashSet::new(),
            },
        };
        for o in other_symbols {
            // find symbol with different uuid
            if let Some(m) = my_symbols.get(o) {
                if m.uuid != o.uuid {
                    conflicts.insert(o.to_string());
                }
            }
        }
        conflicts
    }

    #[inline(always)]
    fn clone_replay(&self) -> Vec<OPReplay> {
        match &self.qpy_replay {
            Some(r) => r.clone(),
            None => Vec::<OPReplay>::new(),
        }
    }

    #[inline(always)]
    fn _my_parameters(&self) -> Option<HashSet<Arc<ParameterExpression>>> {
        match self.parameter_symbols {
            Some(_) => self.parameter_symbols.clone(),
            None => match self.expr {
                SymbolExpr::Symbol { name: _, index: _ } => {
                    Some(HashSet::from([Arc::new(self.to_owned())]))
                }
                _ => None,
            },
        }
    }

    fn _update_uuid(&mut self) {
        if let Some(symbols) = &self.parameter_symbols {
            if symbols.len() == 1 {
                self.uuid = symbols.iter().next().unwrap().uuid;
            } else if let SymbolExpr::Symbol { name: _, index: _ } = self.expr {
                if let Some(s) = symbols.get(self) {
                    self.uuid = s.uuid;
                }
            }
        }
    }

    // default functions for unary operations
    fn _neg(&self) -> ParameterExpression {
        let mut replay = self.clone_replay();
        replay.push(OPReplay::_INSTRUCTION {
            op: _OPCode::MUL,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: Some(ParameterValueType::Int(-1)),
        });
        let mut ret = ParameterExpression {
            expr: -&self.expr,
            uuid: self.uuid.clone(),
            qpy_replay: Some(replay),
            parameter_symbols: self._my_parameters(),
            parameter_vector: None,
        };
        ret._update_uuid();
        ret
    }
    fn _pos(&self) -> ParameterExpression {
        let mut replay = self.clone_replay();
        replay.push(OPReplay::_INSTRUCTION {
            op: _OPCode::MUL,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: Some(ParameterValueType::Int(1)),
        });
        let mut ret = ParameterExpression {
            expr: self.expr.clone(),
            uuid: self.uuid.clone(),
            qpy_replay: Some(replay),
            parameter_symbols: self._my_parameters(),
            parameter_vector: None,
        };
        ret._update_uuid();
        ret
    }
    fn _sin(&self) -> ParameterExpression {
        let mut replay = self.clone_replay();
        replay.push(OPReplay::_INSTRUCTION {
            op: _OPCode::SIN,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: None,
        });
        ParameterExpression {
            expr: self.expr.sin(),
            uuid: self.uuid.clone(),
            qpy_replay: Some(replay),
            parameter_symbols: self._my_parameters(),
            parameter_vector: None,
        }
    }
    fn _cos(&self) -> ParameterExpression {
        let mut replay = self.clone_replay();
        replay.push(OPReplay::_INSTRUCTION {
            op: _OPCode::COS,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: None,
        });
        ParameterExpression {
            expr: self.expr.cos(),
            uuid: self.uuid.clone(),
            qpy_replay: Some(replay),
            parameter_symbols: self._my_parameters(),
            parameter_vector: None,
        }
    }
    fn _tan(&self) -> ParameterExpression {
        let mut replay = self.clone_replay();
        replay.push(OPReplay::_INSTRUCTION {
            op: _OPCode::TAN,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: None,
        });
        ParameterExpression {
            expr: self.expr.tan(),
            uuid: self.uuid.clone(),
            qpy_replay: Some(replay),
            parameter_symbols: self._my_parameters(),
            parameter_vector: None,
        }
    }
    fn _asin(&self) -> ParameterExpression {
        let mut replay = self.clone_replay();
        replay.push(OPReplay::_INSTRUCTION {
            op: _OPCode::ASIN,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: None,
        });
        ParameterExpression {
            expr: self.expr.asin(),
            uuid: self.uuid.clone(),
            qpy_replay: Some(replay),
            parameter_symbols: self._my_parameters(),
            parameter_vector: None,
        }
    }
    fn _acos(&self) -> ParameterExpression {
        let mut replay = self.clone_replay();
        replay.push(OPReplay::_INSTRUCTION {
            op: _OPCode::ACOS,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: None,
        });
        ParameterExpression {
            expr: self.expr.acos(),
            uuid: self.uuid.clone(),
            qpy_replay: Some(replay),
            parameter_symbols: self._my_parameters(),
            parameter_vector: None,
        }
    }
    fn _atan(&self) -> ParameterExpression {
        let mut replay = self.clone_replay();
        replay.push(OPReplay::_INSTRUCTION {
            op: _OPCode::ATAN,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: None,
        });
        ParameterExpression {
            expr: self.expr.atan(),
            uuid: self.uuid.clone(),
            qpy_replay: Some(replay),
            parameter_symbols: self._my_parameters(),
            parameter_vector: None,
        }
    }
    fn _exp(&self) -> ParameterExpression {
        let mut replay = self.clone_replay();
        replay.push(OPReplay::_INSTRUCTION {
            op: _OPCode::EXP,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: None,
        });
        ParameterExpression {
            expr: self.expr.exp(),
            uuid: self.uuid.clone(),
            qpy_replay: Some(replay),
            parameter_symbols: self._my_parameters(),
            parameter_vector: None,
        }
    }
    fn _log(&self) -> ParameterExpression {
        let mut replay = self.clone_replay();
        replay.push(OPReplay::_INSTRUCTION {
            op: _OPCode::LOG,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: None,
        });
        ParameterExpression {
            expr: self.expr.log(),
            uuid: self.uuid.clone(),
            qpy_replay: Some(replay),
            parameter_symbols: self._my_parameters(),
            parameter_vector: None,
        }
    }
    fn _abs(&self) -> ParameterExpression {
        let mut replay = self.clone_replay();
        replay.push(OPReplay::_INSTRUCTION {
            op: _OPCode::ABS,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: None,
        });
        ParameterExpression {
            expr: self.expr.abs(),
            uuid: self.uuid.clone(),
            qpy_replay: Some(replay),
            parameter_symbols: self._my_parameters(),
            parameter_vector: None,
        }
    }
    fn _sign(&self) -> ParameterExpression {
        let mut replay = self.clone_replay();
        replay.push(OPReplay::_INSTRUCTION {
            op: _OPCode::SIGN,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: None,
        });
        ParameterExpression {
            expr: self.expr.sign(),
            uuid: self.uuid.clone(),
            qpy_replay: Some(replay),
            parameter_symbols: self._my_parameters(),
            parameter_vector: None,
        }
    }
    fn _conjugate(&self) -> ParameterExpression {
        let mut replay = self.clone_replay();
        replay.push(OPReplay::_INSTRUCTION {
            op: _OPCode::CONJ,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: None,
        });
        ParameterExpression {
            expr: self.expr.conjugate(),
            uuid: self.uuid.clone(),
            qpy_replay: Some(replay),
            parameter_symbols: self._my_parameters(),
            parameter_vector: None,
        }
    }

    // default functions for binary operations
    fn _add(&self, rhs: &ParameterExpression) -> ParameterExpression {
        let mut replay = self.clone_replay();
        replay.push(OPReplay::_INSTRUCTION {
            op: _OPCode::ADD,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: Some(ParameterValueType::clone_expr_for_replay(rhs)),
        });
        let mut ret = ParameterExpression {
            expr: &self.expr + &rhs.expr,
            uuid: self.uuid.clone(),
            qpy_replay: Some(replay),
            parameter_symbols: self.merge_parameter_symbols(rhs),
            parameter_vector: None,
        };
        ret._update_uuid();
        ret
    }
    fn _radd(&self, lhs: &ParameterExpression) -> ParameterExpression {
        let mut replay = self.clone_replay();
        replay.push(OPReplay::_INSTRUCTION {
            op: _OPCode::ADD,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: Some(ParameterValueType::clone_expr_for_replay(lhs)),
        });
        let mut ret = ParameterExpression {
            expr: &lhs.expr + &self.expr,
            uuid: self.uuid.clone(),
            qpy_replay: Some(replay),
            parameter_symbols: self.merge_parameter_symbols(lhs),
            parameter_vector: None,
        };
        ret._update_uuid();
        ret
    }
    fn _sub(&self, rhs: &ParameterExpression) -> ParameterExpression {
        let mut replay = self.clone_replay();
        replay.push(OPReplay::_INSTRUCTION {
            op: _OPCode::SUB,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: Some(ParameterValueType::clone_expr_for_replay(rhs)),
        });
        let mut ret = ParameterExpression {
            expr: &self.expr - &rhs.expr,
            uuid: self.uuid.clone(),
            qpy_replay: Some(replay),
            parameter_symbols: self.merge_parameter_symbols(rhs),
            parameter_vector: None,
        };
        ret._update_uuid();
        ret
    }
    fn _rsub(&self, lhs: &ParameterExpression) -> ParameterExpression {
        let mut replay = self.clone_replay();
        replay.push(OPReplay::_INSTRUCTION {
            op: _OPCode::RSUB,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: Some(ParameterValueType::clone_expr_for_replay(lhs)),
        });
        let mut ret = ParameterExpression {
            expr: &lhs.expr - &self.expr,
            uuid: self.uuid.clone(),
            qpy_replay: Some(replay),
            parameter_symbols: self.merge_parameter_symbols(lhs),
            parameter_vector: None,
        };
        ret._update_uuid();
        ret
    }
    fn _mul(&self, rhs: &ParameterExpression) -> ParameterExpression {
        let mut replay = self.clone_replay();
        replay.push(OPReplay::_INSTRUCTION {
            op: _OPCode::MUL,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: Some(ParameterValueType::clone_expr_for_replay(rhs)),
        });
        let mut ret = ParameterExpression {
            expr: &self.expr * &rhs.expr,
            uuid: self.uuid.clone(),
            qpy_replay: Some(replay),
            parameter_symbols: self.merge_parameter_symbols(rhs),
            parameter_vector: None,
        };
        ret._update_uuid();
        ret
    }
    fn _rmul(&self, lhs: &ParameterExpression) -> ParameterExpression {
        let mut replay = self.clone_replay();
        replay.push(OPReplay::_INSTRUCTION {
            op: _OPCode::MUL,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: Some(ParameterValueType::clone_expr_for_replay(lhs)),
        });
        let mut ret = ParameterExpression {
            expr: &lhs.expr * &self.expr,
            uuid: self.uuid.clone(),
            qpy_replay: Some(replay),
            parameter_symbols: self.merge_parameter_symbols(lhs),
            parameter_vector: None,
        };
        ret._update_uuid();
        ret
    }
    fn _div(&self, rhs: &ParameterExpression) -> ParameterExpression {
        let mut replay = self.clone_replay();
        replay.push(OPReplay::_INSTRUCTION {
            op: _OPCode::DIV,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: Some(ParameterValueType::clone_expr_for_replay(rhs)),
        });
        let mut ret = ParameterExpression {
            expr: &self.expr / &rhs.expr,
            uuid: self.uuid.clone(),
            qpy_replay: Some(replay),
            parameter_symbols: self.merge_parameter_symbols(rhs),
            parameter_vector: None,
        };
        ret._update_uuid();
        ret
    }
    fn _rdiv(&self, lhs: &ParameterExpression) -> ParameterExpression {
        let mut replay = self.clone_replay();
        replay.push(OPReplay::_INSTRUCTION {
            op: _OPCode::RDIV,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: Some(ParameterValueType::clone_expr_for_replay(lhs)),
        });
        let mut ret = ParameterExpression {
            expr: &lhs.expr / &self.expr,
            uuid: self.uuid.clone(),
            qpy_replay: Some(replay),
            parameter_symbols: self.merge_parameter_symbols(lhs),
            parameter_vector: None,
        };
        ret._update_uuid();
        ret
    }
    fn _pow(&self, rhs: &ParameterExpression) -> ParameterExpression {
        let mut replay = self.clone_replay();
        replay.push(OPReplay::_INSTRUCTION {
            op: _OPCode::POW,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: Some(ParameterValueType::clone_expr_for_replay(rhs)),
        });
        let mut ret = ParameterExpression {
            expr: self.expr.pow(&rhs.expr),
            uuid: self.uuid.clone(),
            qpy_replay: Some(replay),
            parameter_symbols: self.merge_parameter_symbols(rhs),
            parameter_vector: None,
        };
        ret._update_uuid();
        ret
    }
    fn _rpow(&self, lhs: &ParameterExpression) -> ParameterExpression {
        let mut replay = self.clone_replay();
        replay.push(OPReplay::_INSTRUCTION {
            op: _OPCode::RPOW,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: Some(ParameterValueType::clone_expr_for_replay(lhs)),
        });
        let mut ret = ParameterExpression {
            expr: lhs.expr.pow(&self.expr),
            uuid: self.uuid.clone(),
            qpy_replay: Some(replay),
            parameter_symbols: self.merge_parameter_symbols(lhs),
            parameter_vector: None,
        };
        ret._update_uuid();
        ret
    }

    fn _derivative(&self, param: &ParameterExpression) -> ParameterExpression {
        ParameterExpression {
            expr: self.expr.derivative(&param.expr),
            uuid: self.uuid.clone(),
            qpy_replay: self.qpy_replay.clone(),
            parameter_symbols: self.parameter_symbols.clone(),
            parameter_vector: None,
        }
    }
}

impl PartialEq for ParameterExpression {
    fn eq(&self, rprm: &Self) -> bool {
        self.expr == rprm.expr
    }
}

impl Eq for ParameterExpression {}

impl PartialEq<f64> for ParameterExpression {
    fn eq(&self, r: &f64) -> bool {
        &self.expr == r
    }
}

impl PartialEq<Complex64> for ParameterExpression {
    fn eq(&self, r: &Complex64) -> bool {
        &self.expr == r
    }
}

// =============================
// Make from Rust native types
// =============================

impl From<i32> for ParameterExpression {
    fn from(v: i32) -> Self {
        Self {
            expr: SymbolExpr::Value(Value::Real(v as f64)),
            uuid: 0,
            qpy_replay: None,
            parameter_symbols: None,
            parameter_vector: None,
        }
    }
}
impl From<i64> for ParameterExpression {
    fn from(v: i64) -> Self {
        Self {
            expr: SymbolExpr::Value(Value::Real(v as f64)),
            uuid: 0,
            qpy_replay: None,
            parameter_symbols: None,
            parameter_vector: None,
        }
    }
}

impl From<u32> for ParameterExpression {
    fn from(v: u32) -> Self {
        Self {
            expr: SymbolExpr::Value(Value::Real(v as f64)),
            uuid: 0,
            qpy_replay: None,
            parameter_symbols: None,
            parameter_vector: None,
        }
    }
}

impl From<f64> for ParameterExpression {
    fn from(v: f64) -> Self {
        Self {
            expr: SymbolExpr::Value(Value::Real(v)),
            uuid: 0,
            qpy_replay: None,
            parameter_symbols: None,
            parameter_vector: None,
        }
    }
}

impl From<Complex64> for ParameterExpression {
    fn from(v: Complex64) -> Self {
        Self {
            expr: SymbolExpr::Value(Value::Complex(v)),
            uuid: 0,
            qpy_replay: None,
            parameter_symbols: None,
            parameter_vector: None,
        }
    }
}

impl From<&str> for ParameterExpression {
    fn from(s: &str) -> Self {
        Self {
            expr: parse_expression(s),
            uuid: 0,
            qpy_replay: None,
            parameter_symbols: None,
            parameter_vector: None,
        }
    }
}

impl From<&SymbolExpr> for ParameterExpression {
    fn from(expr: &SymbolExpr) -> Self {
        Self {
            expr: expr.clone(),
            uuid: 0,
            qpy_replay: None,
            parameter_symbols: None,
            parameter_vector: None,
        }
    }
}

// =============================
// Unary operations
// =============================
impl Neg for ParameterExpression {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self._neg()
    }
}

// =============================
// Add
// =============================

macro_rules! add_impl_expr {
    ($($t:ty)*) => ($(
        impl Add<$t> for ParameterExpression {
            type Output = Self;

            #[inline]
            #[track_caller]
            fn add(self, other: $t) -> Self::Output {
                self._add(&other.into())
            }
        }
    )*)
}

add_impl_expr! {f64 i32 u32 ParameterExpression}

// =============================
// Sub
// =============================

macro_rules! sub_impl_expr {
    ($($t:ty)*) => ($(
        impl Sub<$t> for ParameterExpression {
            type Output = Self;

            #[inline]
            #[track_caller]
            fn sub(self, other: $t) -> Self::Output {
                self._sub(&other.into())
            }
        }
    )*)
}

sub_impl_expr! {f64 i32 u32 ParameterExpression}

// =============================
// Mul
// =============================

macro_rules! mul_impl_expr {
    ($($t:ty)*) => ($(
        impl Mul<$t> for ParameterExpression {
            type Output = Self;

            #[inline]
            #[track_caller]
            fn mul(self, other: $t) -> Self::Output {
                self._mul(&other.into())
            }
        }
    )*)
}

mul_impl_expr! {f64 i32 u32 ParameterExpression}

// =============================
// Div
// =============================

macro_rules! div_impl_expr {
    ($($t:ty)*) => ($(
        impl Div<$t> for ParameterExpression {
            type Output = Self;

            #[inline]
            #[track_caller]
            fn div(self, other: $t) -> Self::Output {
                self._div(&other.into())
            }
        }
    )*)
}
div_impl_expr! {f64 i32 u32 ParameterExpression}

#[pymethods]
impl ParameterExpression {
    #[new]
    #[pyo3(signature = (name=None, uuid = None, vec_idx = None))]
    pub fn new(name: Option<String>, uuid: Option<u128>, vec_idx: Option<usize>) -> PyResult<Self> {
        match name {
            Some(name) => {
                // check if expr contains replacements for sympy
                let name = name
                    .replace("__begin_sympy_replace__", "$\\")
                    .replace("__end_sympy_replace__", "$");

                let uuid = match uuid {
                    Some(u) => u,
                    None => Uuid::new_v4().as_u128(),
                };
                let ret = ParameterExpression {
                    expr: SymbolExpr::Symbol {
                        name: Box::new(name.clone()),
                        index: vec_idx,
                    },
                    uuid: uuid,
                    qpy_replay: None,
                    parameter_symbols: None,
                    parameter_vector: None,
                };
                Ok(ret)
            }
            None => Ok(ParameterExpression::default()),
        }
    }

    /// create new expression as a symbol
    #[allow(non_snake_case)]
    #[staticmethod]
    #[pyo3(signature = (name, uuid = None))]
    pub fn Symbol(name: String, uuid: Option<u128>) -> PyResult<Self> {
        ParameterExpression::new(Some(name), uuid, None)
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
    /// This method is used to initialize like ParameterExpression.__init__
    /// as implemented in Python
    #[allow(non_snake_case)]
    #[staticmethod]
    pub fn Expression(
        str_expr: String,
        map: Option<HashMap<ParameterExpression, PyObject>>,
    ) -> Self {
        // check if expr contains replacements for sympy
        let str_expr = str_expr
            .replace("__begin_sympy_replace__", "$\\")
            .replace("__end_sympy_replace__", "$");
        let expr = parse_expression(&str_expr);
        if let Some(map) = map {
            let mut parameter_symbols = HashSet::<Arc<ParameterExpression>>::new();
            let mut uuid: u128 = 0;
            for (param, _) in map {
                uuid = param.uuid;
                parameter_symbols.insert(Arc::new(param.to_owned()));
            }
            if parameter_symbols.len() > 1 {
                uuid = 0;
            }
            ParameterExpression {
                expr: expr,
                uuid: uuid,
                qpy_replay: None,
                parameter_symbols: Some(parameter_symbols),
                parameter_vector: None,
            }
        } else {
            let symbols = expr.symbols();
            if symbols.len() > 0 {
                let mut parameter_symbols = HashSet::<Arc<ParameterExpression>>::new();
                let mut uuid: u128 = 0;
                for s in symbols {
                    uuid = Uuid::new_v4().as_u128();
                    let p = ParameterExpression {
                        expr: s.clone(),
                        uuid: uuid.clone(),
                        qpy_replay: None,
                        parameter_symbols: None,
                        parameter_vector: None,
                    };
                    parameter_symbols.insert(Arc::<ParameterExpression>::new(p.to_owned()));
                }
                if parameter_symbols.len() > 1 {
                    uuid = 0;
                }
                ParameterExpression {
                    expr: expr,
                    uuid: uuid,
                    qpy_replay: None,
                    parameter_symbols: Some(parameter_symbols),
                    parameter_vector: None,
                }
            } else {
                ParameterExpression {
                    expr: expr,
                    uuid: 0,
                    qpy_replay: None,
                    parameter_symbols: None,
                    parameter_vector: None,
                }
            }
        }
    }

    // return string to pass to sympify
    pub fn sympify(&self) -> String {
        let ret = self.expr.optimize().to_string();
        ret.replace("$\\", "__begin_sympy_replace__")
            .replace('$', "__end_sympy_replace__")
    }

    // for backward compatibility
    // just return myself
    #[getter]
    pub fn _symbol_expr(&self) -> ParameterExpression {
        self.clone()
    }

    /// return value if expression does not contain any symbols
    pub fn value(&self, py: Python) -> PyResult<PyObject> {
        match self.expr.eval(true) {
            Some(v) => match v {
                Value::Real(r) => r.into_py_any(py),
                Value::Int(i) => i.into_py_any(py),
                Value::Complex(c) => {
                    if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.im) {
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

    pub fn numeric(&self, py: Python) -> PyResult<PyObject> {
        self.value(py)
    }

    /// check if this is symbol
    #[getter]
    pub fn is_symbol(&self) -> bool {
        if let SymbolExpr::Symbol { name: _, index: _ } = self.expr {
            true
        } else {
            false
        }
    }

    /// check if this is numeric
    #[getter]
    pub fn is_numeric(&self) -> bool {
        if let SymbolExpr::Value(_) = self.expr {
            true
        } else {
            false
        }
    }

    /// check if ParameterVectorElement
    #[getter]
    pub fn is_vector_element(&self) -> bool {
        if let SymbolExpr::Symbol { name: _, index } = &self.expr {
            return match index {
                Some(_) => true,
                None => false,
            };
        }
        false
    }

    /// get uuid for this symbol
    #[getter]
    pub fn uuid(&self) -> u128 {
        self.uuid
    }

    #[getter]
    pub fn vector(&self) -> PyResult<ParameterVector> {
        match &self.parameter_vector {
            Some(vec) => Ok(Arc::unwrap_or_clone(vec.clone())),
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Not a vector element",
            )),
        }
    }
    #[getter]
    pub fn _vector(&self) -> PyResult<ParameterVector> {
        self.vector()
    }

    #[getter]
    pub fn index(&self) -> usize {
        match self.expr {
            SymbolExpr::Symbol { name: _, index } => match index {
                Some(index) => index,
                None => 0,
            },
            _ => 0,
        }
    }

    pub fn is_real(&self) -> Option<bool> {
        match self.expr.is_complex() {
            Some(b) => Some(!b),
            None => None,
        }
    }

    pub fn __complex__(&self) -> PyResult<Complex64> {
        match self.expr.eval(true) {
            Some(v) => match v {
                Value::Real(r) => Ok(Complex64::from(r)),
                Value::Int(i) => Ok(Complex64::from(i as f64)),
                Value::Complex(c) => Ok(c),
            },
            None => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "ParameterExpression with unbound parameters ({:?}) cannot be cast to a complex.",
                self.parameters()
            ))),
        }
    }

    pub fn __float__(&self) -> PyResult<f64> {
        match self.expr.eval(true) {
            Some(v) => match v {
                Value::Real(r) => Ok(r),
                Value::Int(i) => Ok(i as f64),
                Value::Complex(c) => {
                    if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.im) {
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
                self.parameters()
            ))),
        }
    }

    pub fn __int__(&self) -> PyResult<i64> {
        match self.expr.eval(true) {
            Some(v) => match v {
                Value::Real(r) => Ok(r as i64),
                Value::Int(i) => Ok(i),
                Value::Complex(c) => {
                    if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.im) {
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
                self.parameters()
            ))),
        }
    }

    /// clone expression
    pub fn __copy__(&self) -> Self {
        Self {
            expr: self.expr.clone(),
            uuid: self.uuid.clone(),
            qpy_replay: self.qpy_replay.clone(),
            parameter_symbols: self.parameter_symbols.clone(),
            parameter_vector: self.parameter_vector.clone(),
        }
    }
    pub fn __deepcopy__(&self, _memo: Option<PyObject>) -> Self {
        Self {
            expr: self.expr.clone(),
            uuid: self.uuid.clone(),
            qpy_replay: self.qpy_replay.clone(),
            parameter_symbols: self.parameter_symbols.clone(),
            parameter_vector: self.parameter_vector.clone(),
        }
    }

    /// return conjugate of expression
    pub fn conjugate(&self) -> ParameterExpression {
        self._conjugate()
    }

    /// return derivative of this expression for param
    pub fn derivative(&self, param: &ParameterExpression) -> ParameterExpression {
        self._derivative(param)
    }

    /// Get the derivative of a parameter expression w.r.t. a specified parameter expression.
    /// Args:
    ///     param (Parameter): Parameter w.r.t. which we want to take the derivative
    /// Returns:
    ///     ParameterExpression representing the gradient of param_expr w.r.t. param
    ///     or complex or float number
    pub fn gradient(&self, param: &Self, py: Python) -> PyResult<PyObject> {
        if let Some(s) = &self.parameter_symbols {
            if s.len() == 0 {
                return 0.into_py_any(py);
            }

            let expr_grad = self.expr.derivative(&param.expr);
            if let Some(v) = expr_grad.eval(true) {
                return match v {
                    Value::Real(r) => r.into_py_any(py),
                    Value::Int(i) => i.into_py_any(py),
                    Value::Complex(c) => {
                        if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.im) {
                            c.re.into_py_any(py)
                        } else {
                            c.into_py_any(py)
                        }
                    }
                };
            }

            // update parameter symbols
            let symbols = expr_grad.symbols();
            let mut new_map = HashSet::<Arc<ParameterExpression>>::new();
            for symbol in s {
                if symbols.contains(&symbol.expr) {
                    new_map.insert(symbol.clone());
                }
            }

            let mut replay = self.clone_replay();
            replay.push(OPReplay::_INSTRUCTION {
                op: _OPCode::GRAD,
                lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
                rhs: Some(ParameterValueType::clone_expr_for_replay(param)),
            });
            let mut ret = Self {
                expr: expr_grad,
                uuid: self.uuid.clone(),
                qpy_replay: Some(replay),
                parameter_symbols: Some(new_map),
                parameter_vector: None,
            };
            ret._update_uuid();
            ret.into_py_any(py)
        } else {
            return 0.into_py_any(py);
        }
    }

    /// expand expression
    pub fn expand(&self) -> Self {
        match self.expr {
            SymbolExpr::Value(_) | SymbolExpr::Symbol { name: _, index: _ } => self.clone(),
            _ => Self {
                expr: self.expr.expand(),
                uuid: self.uuid.clone(),
                qpy_replay: self.qpy_replay.clone(),
                parameter_symbols: self.parameter_symbols.clone(),
                parameter_vector: None,
            },
        }
    }

    /// get hashset of all the symbols used in this expression
    #[getter]
    pub fn parameters(&self) -> HashSet<ParameterExpression> {
        match &self.parameter_symbols {
            Some(symbols) => {
                let mut ret = HashSet::<ParameterExpression>::new();
                for s in symbols {
                    ret.insert(Arc::<ParameterExpression>::unwrap_or_clone(s.clone()));
                }
                ret
            }
            None => match self.expr {
                SymbolExpr::Symbol { name: _, index: _ } => HashSet::from([self.clone()]),
                _ => HashSet::<ParameterExpression>::new(),
            },
        }
    }
    #[getter]
    pub fn _parameter_symbols(&self) -> HashSet<ParameterExpression> {
        self.parameters()
    }

    /// return all values in this equation
    pub fn values(&self, py: Python) -> PyResult<Vec<PyObject>> {
        let ret: Vec<PyObject> = self
            .expr
            .values()
            .iter()
            .map(|val| match val {
                Value::Real(r) => r.into_py_any(py).unwrap(),
                Value::Int(i) => i.into_py_any(py).unwrap(),
                Value::Complex(c) => c.into_py_any(py).unwrap(),
            })
            .collect();
        Ok(ret)
    }

    /// return expression as a string
    #[getter]
    pub fn name(&self) -> String {
        self.__str__()
    }

    pub fn assign(&self, param: &ParameterExpression, value: &Bound<PyAny>) -> PyResult<Self> {
        if let Some(e) = _extract_value(value) {
            let eval = match e.expr {
                SymbolExpr::Value(_) => true,
                _ => false,
            };
            if let SymbolExpr::Symbol { name: _, index: _ } = &self.expr {
                if &self.expr == &param.expr {
                    return Ok(e);
                }
            }
            self._subs(HashMap::from([(param.clone(), e)]), eval, false)
        } else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "unsupported data type is passed to assign parameter",
            ));
        }
    }

    fn _subs(
        &self,
        in_map: HashMap<ParameterExpression, ParameterExpression>,
        eval: bool,
        allow_unknown_parameters: bool,
    ) -> PyResult<ParameterExpression> {
        let mut map: HashMap<String, SymbolExpr> = HashMap::new();
        let mut subs_map: HashMap<ParameterExpression, ParameterValueType> = HashMap::new();
        let mut unknown_params: HashSet<String> = HashSet::new();
        let mut symbols: HashSet<Arc<ParameterExpression>> = match &self.parameter_symbols {
            Some(s) => s.clone(),
            None => match self.expr {
                SymbolExpr::Symbol { name: _, index: _ } => {
                    HashSet::from([Arc::new(self.to_owned())])
                }
                _ => HashSet::new(),
            },
        };

        for (key, expr) in &in_map {
            // check if value in map is valid
            if let SymbolExpr::Value(v) = &expr.expr {
                if let Value::Real(r) = v {
                    if r.is_nan() || r.is_infinite() {
                        return Err(CircuitError::new_err(
                            "Expression cannot bind non-numeric values",
                        ));
                    }
                } else if let Value::Complex(c) = v {
                    if c.is_nan() || c.is_infinite() {
                        return Err(CircuitError::new_err(
                            "Expression cannot bind non-numeric values",
                        ));
                    }
                }
            }

            if key.expr != expr.expr {
                self._raise_if_parameter_conflict(&expr)?;
                map.insert(key.to_string(), expr.expr.clone());
            }
            subs_map.insert(key.clone(), ParameterValueType::clone_expr_for_replay(expr));
            if symbols.contains(key) {
                symbols.remove(key);
            } else if !allow_unknown_parameters {
                unknown_params.insert(key.to_string());
            }

            match &expr.parameter_symbols {
                Some(o) => {
                    for k in o {
                        symbols.insert(k.clone());
                    }
                }
                None => {
                    if let SymbolExpr::Symbol { name: _, index: _ } = expr.expr {
                        symbols.insert(Arc::new(expr.to_owned()));
                    }
                }
            }
        }
        if !allow_unknown_parameters && unknown_params.len() > 0 {
            return Err(CircuitError::new_err(format!(
                "Cannot bind Parameters ({:?}) not present in expression.",
                unknown_params
            )));
        }

        let bound = self.expr.subs(&map);

        if eval && symbols.len() == 0 {
            let ret = match bound.eval(true) {
                Some(v) => match &v {
                    Value::Real(r) => {
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
                    Value::Int(_) => SymbolExpr::Value(v),
                    Value::Complex(c) => {
                        if c.re.is_infinite() || c.im.is_infinite() {
                            return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                                "zero division occurs while binding parameter",
                            ));
                        } else if c.re.is_nan() || c.im.is_nan() {
                            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                                "NAN detected while binding parameter",
                            ));
                        } else if (-SYMEXPR_EPSILON..SYMEXPR_EPSILON).contains(&c.im) {
                            SymbolExpr::Value(Value::Real(c.re))
                        } else {
                            SymbolExpr::Value(v)
                        }
                    }
                },
                None => bound,
            };
            let mut replay = self.clone_replay();
            replay.push(OPReplay::_SUBS {
                binds: subs_map,
                op: _OPCode::SUBSTITUTE,
            });
            Ok(ParameterExpression {
                expr: ret,
                uuid: self.uuid.clone(),
                qpy_replay: Some(replay),
                parameter_symbols: if symbols.len() > 0 {
                    Some(symbols)
                } else {
                    None
                },
                parameter_vector: None,
            })
        } else {
            let mut replay = self.clone_replay();
            replay.push(OPReplay::_SUBS {
                binds: subs_map,
                op: _OPCode::SUBSTITUTE,
            });
            let mut ret = ParameterExpression {
                expr: bound,
                uuid: self.uuid.clone(),
                qpy_replay: Some(replay),
                parameter_symbols: if symbols.len() > 0 {
                    Some(symbols)
                } else {
                    None
                },
                parameter_vector: None,
            };
            ret._update_uuid();
            Ok(ret)
        }
    }

    /// bind values to symbols given by input hashmap
    #[pyo3(signature = (in_map, allow_unknown_parameters = None))]
    pub fn bind(
        &self,
        in_map: HashMap<ParameterExpression, Bound<PyAny>>,
        allow_unknown_parameters: Option<bool>,
    ) -> PyResult<Self> {
        let mut map: HashMap<ParameterExpression, ParameterExpression> = HashMap::new();
        let allow_unknown_parameters = match allow_unknown_parameters {
            Some(b) => b,
            None => false,
        };

        for (key, val) in in_map {
            if let Ok(e) = Self::Value(&val) {
                map.insert(key.clone(), e);
            }
        }
        self._subs(map, true, allow_unknown_parameters)
    }

    /// substitute symbols to expressions (or values) given by hash map
    #[pyo3(signature = (map, allow_unknown_parameters = None))]
    pub fn subs(
        &self,
        map: HashMap<ParameterExpression, Self>,
        allow_unknown_parameters: Option<bool>,
    ) -> PyResult<Self> {
        let allow_unknown_parameters = match allow_unknown_parameters {
            Some(b) => b,
            None => false,
        };

        self._subs(map, false, allow_unknown_parameters)
    }

    fn _raise_if_parameter_conflict(&self, other: &ParameterExpression) -> PyResult<bool> {
        let conflicts = self.get_conflict_parameters(other);
        if conflicts.len() > 0 {
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
            Some(rhs) => match (&self.expr, &rhs.expr) {
                (
                    SymbolExpr::Symbol { name: _, index: _ },
                    SymbolExpr::Symbol { name: _, index: _ },
                ) => {
                    if self.expr.to_string() == rhs.expr.to_string() {
                        if self.uuid == rhs.uuid {
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
                (_, _) => {
                    if self.expr == rhs.expr {
                        // if there are some conflicts, the equation is not equal
                        let conflicts = self.get_conflict_parameters(&rhs);
                        conflicts.len() == 0
                    } else {
                        false
                    }
                }
            },
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
    pub fn __neg__(&self) -> ParameterExpression {
        self._neg()
    }
    pub fn __pos__(&self) -> ParameterExpression {
        self._pos()
    }
    pub fn sin(&self) -> ParameterExpression {
        self._sin()
    }
    pub fn cos(&self) -> ParameterExpression {
        self._cos()
    }
    pub fn tan(&self) -> ParameterExpression {
        self._tan()
    }
    pub fn arcsin(&self) -> ParameterExpression {
        self._asin()
    }
    pub fn arccos(&self) -> ParameterExpression {
        self._acos()
    }
    pub fn arctan(&self) -> ParameterExpression {
        self._atan()
    }
    pub fn exp(&self) -> ParameterExpression {
        self._exp()
    }
    pub fn log(&self) -> ParameterExpression {
        self._log()
    }
    pub fn __abs__(&self) -> ParameterExpression {
        self._abs()
    }
    pub fn abs(&self) -> ParameterExpression {
        self._abs()
    }
    pub fn sign(&self) -> ParameterExpression {
        self._sign()
    }

    // binary operators
    pub fn __add__(&self, rhs: &Bound<PyAny>) -> PyResult<ParameterExpression> {
        match _extract_value(rhs) {
            Some(rhs) => {
                self._raise_if_parameter_conflict(&rhs)?;
                Ok(self._add(&rhs))
            }
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __add__",
            )),
        }
    }
    pub fn __radd__(&self, lhs: &Bound<PyAny>) -> PyResult<ParameterExpression> {
        match _extract_value(lhs) {
            Some(lhs) => {
                self._raise_if_parameter_conflict(&lhs)?;
                Ok(self._radd(&lhs))
            }
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __radd__",
            )),
        }
    }
    pub fn __sub__(&self, rhs: &Bound<PyAny>) -> PyResult<ParameterExpression> {
        match _extract_value(rhs) {
            Some(rhs) => {
                self._raise_if_parameter_conflict(&rhs)?;
                Ok(self._sub(&rhs))
            }
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __sub__",
            )),
        }
    }
    pub fn __rsub__(&self, lhs: &Bound<PyAny>) -> PyResult<ParameterExpression> {
        match _extract_value(lhs) {
            Some(lhs) => {
                self._raise_if_parameter_conflict(&lhs)?;
                Ok(self._rsub(&lhs))
            }
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __rsub__",
            )),
        }
    }
    pub fn __mul__(&self, rhs: &Bound<PyAny>) -> PyResult<ParameterExpression> {
        match _extract_value(rhs) {
            Some(rhs) => {
                self._raise_if_parameter_conflict(&rhs)?;
                Ok(self._mul(&rhs))
            }
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __mul__",
            )),
        }
    }
    pub fn __rmul__(&self, lhs: &Bound<PyAny>) -> PyResult<ParameterExpression> {
        match _extract_value(lhs) {
            Some(lhs) => {
                self._raise_if_parameter_conflict(&lhs)?;
                Ok(self._rmul(&lhs))
            }
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __rmul__",
            )),
        }
    }

    pub fn __truediv__(&self, rhs: &Bound<PyAny>) -> PyResult<ParameterExpression> {
        match _extract_value(rhs) {
            Some(rhs) => {
                if let SymbolExpr::Value(v) = &rhs.expr {
                    let zero = match v {
                        Value::Int(i) => *i == 0,
                        Value::Real(r) => *r == 0.0,
                        Value::Complex(c) => c.re == 0.0 && c.im == 0.0,
                    };
                    if zero {
                        return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                            "Division of a ParameterExpression by zero.",
                        ));
                    }
                }
                self._raise_if_parameter_conflict(&rhs)?;
                Ok(self._div(&rhs))
            }
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __truediv__",
            )),
        }
    }
    pub fn __rtruediv__(&self, lhs: &Bound<PyAny>) -> PyResult<ParameterExpression> {
        match _extract_value(lhs) {
            Some(lhs) => {
                self._raise_if_parameter_conflict(&lhs)?;
                Ok(self._rdiv(&lhs))
            }
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __truediv__",
            )),
        }
    }
    pub fn __pow__(
        &self,
        rhs: &Bound<PyAny>,
        _modulo: Option<i32>,
    ) -> PyResult<ParameterExpression> {
        match _extract_value(rhs) {
            Some(rhs) => {
                self._raise_if_parameter_conflict(&rhs)?;
                Ok(self._pow(&rhs))
            }
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __pow__",
            )),
        }
    }
    pub fn __rpow__(
        &self,
        lhs: &Bound<PyAny>,
        _modulo: Option<i32>,
    ) -> PyResult<ParameterExpression> {
        match _extract_value(lhs) {
            Some(lhs) => {
                self._raise_if_parameter_conflict(&lhs)?;
                Ok(self._rpow(&lhs))
            }
            None => Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported data type for __rpow__",
            )),
        }
    }

    pub fn __str__(&self) -> String {
        if let SymbolExpr::Symbol { name: _, index: _ } = &self.expr {
            return self.expr.to_string();
        }
        match self.expr.eval(true) {
            Some(e) => e.to_string(),
            None => self.expr.optimize().to_string(),
        }
    }

    pub fn __hash__(&self, py: Python) -> PyResult<isize> {
        match self.expr.eval(true) {
            Some(v) => match v {
                Value::Int(i) => i.into_pyobject(py)?.hash(),
                Value::Real(r) => r.into_pyobject(py)?.hash(),
                Value::Complex(c) => c.into_pyobject(py)?.hash(),
            },
            None => self.expr.to_string().into_pyobject(py)?.hash(),
        }
    }

    // for pickle, we can reproduce equation from expression string
    fn __getstate__(&self) -> PyResult<(String, u128, Option<HashMap<String, u128>>)> {
        if let Some(symbols) = &self.parameter_symbols {
            let mut ret = HashMap::<String, u128>::new();
            for s in symbols {
                ret.insert(s.to_string(), s.uuid.clone());
            }
            return Ok((self.to_string(), self.uuid.clone(), Some(ret)));
        }
        Ok((self.to_string(), self.uuid.clone(), None))
    }
    fn __setstate__(
        &mut self,
        state: (String, u128, Option<HashMap<String, u128>>),
    ) -> PyResult<()> {
        self.expr = parse_expression(&state.0);
        self.uuid = state.1;
        if let Some(symbols) = state.2 {
            let mut parameter_symbols = HashSet::<Arc<ParameterExpression>>::new();
            for (name, uuid) in symbols {
                if let Ok(p) = ParameterExpression::Symbol(name, Some(uuid)) {
                    parameter_symbols.insert(Arc::<ParameterExpression>::new(p.to_owned()));
                }
            }
            self.parameter_symbols = Some(parameter_symbols);
        } else {
            self.parameter_symbols = None;
        }
        Ok(())
    }

    pub fn __repr__(&self) -> String {
        if let SymbolExpr::Symbol { name: _, index } = &self.expr {
            return match index {
                Some(_) => format!("ParameterVectorElement({})", self.expr.to_string()),
                None => format!("Parameter({})", self.expr.to_string()),
            };
        }
        format!("ParameterExpression({})", self.expr.optimize().to_string())
    }

    /// return QPY replay
    pub fn replay(&self) -> Option<Vec<OPReplay>> {
        self.qpy_replay.clone()
    }
    #[getter]
    pub fn _qpy_replay(&self) -> Option<Vec<OPReplay>> {
        self.qpy_replay.clone()
    }
}

#[pyclass]
struct ParameterIter {
    inner: std::vec::IntoIter<ParameterExpression>,
}

#[pymethods]
impl ParameterIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<ParameterExpression> {
        slf.inner.next()
    }
}

// ===========================================
// ParameterVector class
// ===========================================
#[pyclass(sequence, module = "qiskit._accelerate.circuit")]
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
            root_uuid: root_uuid,
            params: Vec::with_capacity(length),
        };

        for i in 0..length {
            let pe = ParameterExpression {
                expr: SymbolExpr::Symbol {
                    name: Box::new(name.clone()),
                    index: Some(i),
                },
                uuid: root_uuid + i as u128,
                qpy_replay: None,
                parameter_symbols: None,
                parameter_vector: None,
            };
            ret.params.push(pe);
        }
        let t = Arc::<ParameterVector>::new(ret.to_owned());
        for pe in &mut ret.params {
            pe.parameter_vector = Some(t.clone());
        }

        Ok(ret)
    }

    /// """The name of the :class:`ParameterVector`."""
    #[getter]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    #[getter]
    pub fn _root_uuid(&self) -> u128 {
        self.root_uuid
    }

    /// """A list of the contained :class:`ParameterVectorElement` instances.
    ///
    /// It is not safe to mutate this list."""
    #[getter]
    pub fn params(&self) -> Vec<ParameterExpression> {
        self.params.clone()
    }

    pub fn index(&self, param: &ParameterExpression) -> Option<usize> {
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
            indices => PyList::new(py, indices.iter().map(|index| self.params[index].clone()))?
                .into_py_any(py),
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
    pub fn resize(&mut self, length: usize) {
        if length > self.params.len() {
            let root_uuid = self.root_uuid;
            for i in self.params.len()..length {
                let pe = ParameterExpression {
                    expr: SymbolExpr::Symbol {
                        name: Box::new(self.name.clone()),
                        index: Some(i),
                    },
                    uuid: root_uuid + i as u128,
                    qpy_replay: None,
                    parameter_symbols: None,
                    parameter_vector: None,
                };
                self.params.push(pe);
            }
            let t = Arc::<ParameterVector>::new(self.to_owned());
            for pe in &mut self.params {
                pe.parameter_vector = Some(t.clone());
            }
        } else {
            self.params.resize(length, ParameterExpression::default());
        }
    }

    fn __getstate__(&self) -> PyResult<(String, usize, u128)> {
        Ok((self.name.clone(), self.params.len(), self.root_uuid.clone()))
    }
    fn __setstate__(&mut self, state: (String, usize, u128)) -> PyResult<()> {
        self.name = state.0;
        self.root_uuid = state.2;
        let length = state.1;
        self.params = Vec::with_capacity(length);
        for i in 0..length {
            let pe = ParameterExpression {
                expr: SymbolExpr::Symbol {
                    name: Box::new(self.name.clone()),
                    index: Some(i),
                },
                uuid: self.root_uuid + i as u128,
                qpy_replay: None,
                parameter_symbols: None,
                parameter_vector: None,
            };
            self.params.push(pe);
        }
        let t = Arc::<ParameterVector>::new(self.to_owned());
        for pe in &mut self.params {
            pe.parameter_vector = Some(t.clone());
        }
        Ok(())
    }
}
