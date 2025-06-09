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
use pyo3::types::{PyList, PySet};
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
            if let ParameterInner::Expression { expr, .. } = &expr.inner {
                if let Some(v) = expr.eval(true) {
                    return match v {
                        symbol_expr::Value::Int(i) => ParameterValueType::Int(i),
                        symbol_expr::Value::Real(r) => ParameterValueType::Float(r),
                        symbol_expr::Value::Complex(c) => ParameterValueType::Complex(c),
                    };
                }
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
            inner: ParameterInner::Expression {
                expr: SymbolExpr::Value(symbol_expr::Value::from(i)),
                qpy_replay: Vec::new(),
                parameter_symbols: None,
            },
        })
    } else if let Ok(c) = value.extract::<Complex64>() {
        if c.is_infinite() || c.is_nan() {
            return None;
        }
        Some(ParameterExpression {
            inner: ParameterInner::Expression {
                expr: SymbolExpr::Value(symbol_expr::Value::from(c)),
                qpy_replay: Vec::new(),
                parameter_symbols: None,
            },
        })
    } else if let Ok(r) = value.extract::<f64>() {
        if r.is_infinite() || r.is_nan() {
            return None;
        }
        Some(ParameterExpression {
            inner: ParameterInner::Expression {
                expr: SymbolExpr::Value(symbol_expr::Value::from(r)),
                qpy_replay: Vec::new(),
                parameter_symbols: None,
            },
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
#[derive(Clone, Debug)]
pub enum ParameterInner {
    Symbol {
        name: Box<String>,
        uuid: u128,
    },
    VectorElement {
        name: Box<String>,
        index: usize,
        uuid: u128,
        vector: Option<Arc<ParameterVector>>,
    },
    Expression {
        expr: SymbolExpr,
        qpy_replay: Vec<OPReplay>,
        parameter_symbols: Option<HashSet<Arc<ParameterExpression>>>,
    },
}

#[pyclass(sequence, subclass, module = "qiskit._accelerate.circuit")]
#[derive(Clone, Debug)]
pub struct ParameterExpression {
    inner: ParameterInner,
}

impl Hash for ParameterExpression {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match &self.inner {
            ParameterInner::Symbol { name, .. } => {
                name.hash(state);
            }
            ParameterInner::VectorElement { name, index, .. } => {
                name.hash(state);
                index.hash(state);
            }
            ParameterInner::Expression { expr, .. } => {
                expr.to_string().hash(state);
            }
        }
    }
}

impl Default for ParameterExpression {
    // default constructor returns zero
    fn default() -> Self {
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: SymbolExpr::Value(symbol_expr::Value::Int(0)),
                qpy_replay: Vec::new(),
                parameter_symbols: None,
            },
        }
    }
}

impl fmt::Display for ParameterExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner {
            ParameterInner::Symbol { name, uuid: _ } => {
                write!(f, "{}", name)
            }
            ParameterInner::VectorElement { name, index, .. } => {
                write!(f, "{}[{}]", name, index)
            }
            ParameterInner::Expression { expr, .. } => {
                write!(f, "{}", expr)
            }
        }
    }
}

// ===========================================
// Parameter class
// ===========================================
#[pyclass(sequence, subclass, module = "qiskit._accelerate.circuit", extends=ParameterExpression, name="Parameter")]
#[derive(Clone, Debug)]
pub struct PyParameter {}

#[pymethods]
impl PyParameter {
    #[new]
    #[pyo3(signature = (name = None, uuid = None))]
    pub fn new(name: Option<String>, uuid: Option<u128>) -> PyClassInitializer<Self> {
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
                PyClassInitializer::from(ParameterExpression {
                    inner: ParameterInner::Symbol {
                        name: Box::new(name.clone()),
                        uuid,
                    },
                })
                .add_subclass(Self {})
            }
            None => PyClassInitializer::from(ParameterExpression::default()).add_subclass(Self {}),
        }
    }
}

// ===========================================
// ParameterVectorElement class
// ===========================================
#[pyclass(sequence, module = "qiskit._accelerate.circuit", extends=PyParameter, name="ParameterVectorElement")]
#[derive(Clone, Debug)]
pub struct PyParameterVectorElement {}

#[pymethods]
impl PyParameterVectorElement {
    #[new]
    #[pyo3(signature = (vector, index, uuid = None))]
    pub fn new(
        vector: &ParameterVector,
        index: usize,
        uuid: Option<u128>,
    ) -> PyClassInitializer<Self> {
        let uuid = match uuid {
            Some(u) => u,
            None => Uuid::new_v4().as_u128(),
        };

        PyClassInitializer::from(ParameterExpression {
            inner: ParameterInner::VectorElement {
                name: Box::new(vector.name.clone()),
                index,
                uuid,
                vector: Some(Arc::new(vector.clone())),
            },
        })
        .add_subclass(PyParameter {})
        .add_subclass(Self {})
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
        ParameterExpression {
            inner: ParameterInner::Symbol {
                name: Box::new(name.clone()),
                uuid,
            },
        }
    }

    /// get uuid for this symbol
    pub fn uuid(&self) -> u128 {
        match &self.inner {
            ParameterInner::Symbol { name: _, uuid } => *uuid,
            ParameterInner::VectorElement {
                name: _,
                index: _,
                uuid,
                vector: _,
            } => *uuid,
            ParameterInner::Expression {
                expr: _,
                qpy_replay: _,
                parameter_symbols,
            } => {
                if let Some(symbols) = parameter_symbols {
                    if symbols.len() == 1 {
                        return symbols.iter().next().unwrap().uuid();
                    }
                }
                0_u128
            }
        }
    }

    // clone or new SymbolExpr
    fn expr(&self) -> SymbolExpr {
        match &self.inner {
            ParameterInner::Expression { expr, .. } => expr.clone(),
            ParameterInner::Symbol { .. } | ParameterInner::VectorElement { .. } => {
                SymbolExpr::Symbol(Box::new(self.to_string()))
            }
        }
    }

    /// check if this is symbol
    pub fn is_symbol(&self) -> bool {
        match &self.inner {
            ParameterInner::Symbol { .. } => true,
            ParameterInner::VectorElement { .. } => true,
            ParameterInner::Expression { expr, .. } => {
                matches!(expr, SymbolExpr::Symbol(_))
            }
        }
    }

    /// check if this is numeric
    pub fn is_numeric(&self) -> bool {
        match &self.inner {
            ParameterInner::Symbol { .. } => false,
            ParameterInner::VectorElement { .. } => false,
            ParameterInner::Expression { expr, .. } => {
                matches!(expr, SymbolExpr::Value(_))
            }
        }
    }

    /// check if ParameterVectorElement
    pub fn is_vector_element(&self) -> bool {
        match &self.inner {
            ParameterInner::Symbol { .. } => false,
            ParameterInner::VectorElement { .. } => true,
            ParameterInner::Expression { .. } => false,
        }
    }

    /// return number of symbols in this expression
    pub fn num_symbols(&self) -> usize {
        match &self.inner {
            ParameterInner::Symbol { .. } => 1,
            ParameterInner::VectorElement { .. } => 1,
            ParameterInner::Expression { expr, .. } => expr.symbols().len(),
        }
    }

    /// check if the symbol is used in this expression
    pub fn has_symbol(&self, symbol: String) -> bool {
        match &self.inner {
            ParameterInner::Symbol { name, .. } => symbol == *name.as_ref(),
            ParameterInner::VectorElement { name, .. } => symbol == *name.as_ref(),
            ParameterInner::Expression { expr, .. } => expr.symbols_in_string().contains(&symbol),
        }
    }

    /// return true if this is not complex number
    pub fn is_real(&self) -> Option<bool> {
        match &self.inner {
            ParameterInner::Symbol { .. } => None,
            ParameterInner::VectorElement { .. } => None,
            ParameterInner::Expression { expr, .. } => expr.is_complex().map(|b| !b),
        }
    }

    // return merged set of parameter symbils in 2 parameters
    fn merge_parameter_symbols(
        &self,
        other: &ParameterExpression,
    ) -> Option<HashSet<Arc<ParameterExpression>>> {
        let mut ret: HashSet<Arc<ParameterExpression>> = match &self.inner {
            ParameterInner::Expression {
                expr: _,
                qpy_replay: _,
                parameter_symbols,
            } => {
                if let Some(symbols) = parameter_symbols {
                    symbols.clone()
                } else {
                    HashSet::new()
                }
            }
            ParameterInner::Symbol { .. } | ParameterInner::VectorElement { .. } => {
                HashSet::from([Arc::new(self.clone())])
            }
        };
        match &other.inner {
            ParameterInner::Expression {
                expr: _,
                qpy_replay: _,
                parameter_symbols,
            } => {
                if let Some(symbols) = parameter_symbols {
                    for o in symbols {
                        ret.insert(o.clone());
                    }
                }
            }
            ParameterInner::Symbol { .. } | ParameterInner::VectorElement { .. } => {
                ret.insert(Arc::new(other.clone()));
            }
        };
        if !ret.is_empty() {
            Some(ret)
        } else {
            None
        }
    }

    // get conflict parameters
    fn get_conflict_parameters(&self, other: &ParameterExpression) -> HashSet<String> {
        let mut conflicts = HashSet::<String>::new();
        let my_symbols: &HashSet<Arc<ParameterExpression>> = match &self.inner {
            ParameterInner::Expression {
                expr: _,
                qpy_replay: _,
                parameter_symbols,
            } => {
                if let Some(symbols) = parameter_symbols {
                    symbols
                } else {
                    &HashSet::new()
                }
            }
            ParameterInner::Symbol { .. } | ParameterInner::VectorElement { .. } => {
                &HashSet::from([Arc::new(self.clone())])
            }
        };
        let other_symbols: &HashSet<Arc<ParameterExpression>> = match &other.inner {
            ParameterInner::Expression {
                expr: _,
                qpy_replay: _,
                parameter_symbols,
            } => {
                if let Some(symbols) = parameter_symbols {
                    symbols
                } else {
                    &HashSet::new()
                }
            }
            ParameterInner::Symbol { .. } | ParameterInner::VectorElement { .. } => {
                &HashSet::from([Arc::new(other.clone())])
            }
        };
        for o in other_symbols {
            // find symbol with different uuid
            if let Some(m) = my_symbols.get(o) {
                if m.uuid() != o.uuid() {
                    conflicts.insert(o.to_string());
                }
            }
        }
        conflicts
    }

    #[inline(always)]
    fn _my_parameters(&self) -> Option<HashSet<Arc<ParameterExpression>>> {
        match &self.inner {
            ParameterInner::Expression {
                expr: _,
                qpy_replay: _,
                parameter_symbols,
            } => parameter_symbols.clone(),
            ParameterInner::Symbol { .. } | ParameterInner::VectorElement { .. } => {
                Some(HashSet::from([Arc::new(self.clone())]))
            }
        }
    }

    // default functions for unary operations
    pub fn neg(&self) -> ParameterExpression {
        let replay = OPReplay::_INSTRUCTION {
            op: _OPCode::MUL,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: Some(ParameterValueType::Int(-1)),
        };
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: match &self.inner {
                    ParameterInner::Expression { expr, .. } => -expr,
                    _ => -SymbolExpr::Symbol(Box::new(self.to_string())),
                },
                qpy_replay: Vec::from([replay]),
                parameter_symbols: self._my_parameters(),
            },
        }
    }
    pub fn pos(&self) -> ParameterExpression {
        let replay = OPReplay::_INSTRUCTION {
            op: _OPCode::MUL,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: Some(ParameterValueType::Int(1)),
        };
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: match &self.inner {
                    ParameterInner::Expression { expr, .. } => expr.clone(),
                    _ => SymbolExpr::Symbol(Box::new(self.to_string())),
                },
                qpy_replay: Vec::from([replay]),
                parameter_symbols: self._my_parameters(),
            },
        }
    }
    pub fn sin(&self) -> ParameterExpression {
        let replay = OPReplay::_INSTRUCTION {
            op: _OPCode::SIN,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: None,
        };
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: match &self.inner {
                    ParameterInner::Expression { expr, .. } => expr.sin(),
                    _ => SymbolExpr::Symbol(Box::new(self.to_string())).sin(),
                },
                qpy_replay: Vec::from([replay]),
                parameter_symbols: self._my_parameters(),
            },
        }
    }
    pub fn cos(&self) -> ParameterExpression {
        let replay = OPReplay::_INSTRUCTION {
            op: _OPCode::COS,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: None,
        };
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: match &self.inner {
                    ParameterInner::Expression { expr, .. } => expr.cos(),
                    _ => SymbolExpr::Symbol(Box::new(self.to_string())).cos(),
                },
                qpy_replay: Vec::from([replay]),
                parameter_symbols: self._my_parameters(),
            },
        }
    }
    pub fn tan(&self) -> ParameterExpression {
        let replay = OPReplay::_INSTRUCTION {
            op: _OPCode::TAN,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: None,
        };
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: match &self.inner {
                    ParameterInner::Expression { expr, .. } => expr.tan(),
                    _ => SymbolExpr::Symbol(Box::new(self.to_string())).tan(),
                },
                qpy_replay: Vec::from([replay]),
                parameter_symbols: self._my_parameters(),
            },
        }
    }
    pub fn asin(&self) -> ParameterExpression {
        let replay = OPReplay::_INSTRUCTION {
            op: _OPCode::ASIN,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: None,
        };
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: match &self.inner {
                    ParameterInner::Expression { expr, .. } => expr.asin(),
                    _ => SymbolExpr::Symbol(Box::new(self.to_string())).asin(),
                },
                qpy_replay: Vec::from([replay]),
                parameter_symbols: self._my_parameters(),
            },
        }
    }
    pub fn acos(&self) -> ParameterExpression {
        let replay = OPReplay::_INSTRUCTION {
            op: _OPCode::ACOS,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: None,
        };
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: match &self.inner {
                    ParameterInner::Expression { expr, .. } => expr.acos(),
                    _ => SymbolExpr::Symbol(Box::new(self.to_string())).acos(),
                },
                qpy_replay: Vec::from([replay]),
                parameter_symbols: self._my_parameters(),
            },
        }
    }
    pub fn atan(&self) -> ParameterExpression {
        let replay = OPReplay::_INSTRUCTION {
            op: _OPCode::ATAN,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: None,
        };
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: match &self.inner {
                    ParameterInner::Expression { expr, .. } => expr.atan(),
                    _ => SymbolExpr::Symbol(Box::new(self.to_string())).atan(),
                },
                qpy_replay: Vec::from([replay]),
                parameter_symbols: self._my_parameters(),
            },
        }
    }
    pub fn exp(&self) -> ParameterExpression {
        let replay = OPReplay::_INSTRUCTION {
            op: _OPCode::EXP,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: None,
        };
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: match &self.inner {
                    ParameterInner::Expression { expr, .. } => expr.exp(),
                    _ => SymbolExpr::Symbol(Box::new(self.to_string())).exp(),
                },
                qpy_replay: Vec::from([replay]),
                parameter_symbols: self._my_parameters(),
            },
        }
    }
    pub fn log(&self) -> ParameterExpression {
        let replay = OPReplay::_INSTRUCTION {
            op: _OPCode::LOG,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: None,
        };
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: match &self.inner {
                    ParameterInner::Expression { expr, .. } => expr.log(),
                    _ => SymbolExpr::Symbol(Box::new(self.to_string())).log(),
                },
                qpy_replay: Vec::from([replay]),
                parameter_symbols: self._my_parameters(),
            },
        }
    }
    pub fn abs(&self) -> ParameterExpression {
        let replay = OPReplay::_INSTRUCTION {
            op: _OPCode::ABS,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: None,
        };
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: match &self.inner {
                    ParameterInner::Expression { expr, .. } => expr.abs(),
                    _ => SymbolExpr::Symbol(Box::new(self.to_string())).abs(),
                },
                qpy_replay: Vec::from([replay]),
                parameter_symbols: self._my_parameters(),
            },
        }
    }
    pub fn sign(&self) -> ParameterExpression {
        let replay = OPReplay::_INSTRUCTION {
            op: _OPCode::SIGN,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: None,
        };
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: match &self.inner {
                    ParameterInner::Expression { expr, .. } => expr.sign(),
                    _ => SymbolExpr::Symbol(Box::new(self.to_string())).sign(),
                },
                qpy_replay: Vec::from([replay]),
                parameter_symbols: self._my_parameters(),
            },
        }
    }

    /// return conjugate of expression
    pub fn conjugate(&self) -> ParameterExpression {
        let replay = OPReplay::_INSTRUCTION {
            op: _OPCode::CONJ,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: None,
        };
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: match &self.inner {
                    ParameterInner::Expression { expr, .. } => expr.conjugate(),
                    _ => SymbolExpr::Symbol(Box::new(self.to_string())).conjugate(),
                },
                qpy_replay: Vec::from([replay]),
                parameter_symbols: self._my_parameters(),
            },
        }
    }

    // default functions for binary operations
    pub fn add(&self, rhs: &ParameterExpression) -> ParameterExpression {
        let replay = OPReplay::_INSTRUCTION {
            op: _OPCode::ADD,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: Some(ParameterValueType::clone_expr_for_replay(rhs)),
        };
        let expr_lhs = match &self.inner {
            ParameterInner::Expression { expr, .. } => expr,
            _ => &SymbolExpr::Symbol(Box::new(self.to_string())),
        };
        let expr_rhs = match &rhs.inner {
            ParameterInner::Expression { expr, .. } => expr,
            _ => &SymbolExpr::Symbol(Box::new(rhs.to_string())),
        };
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: expr_lhs + expr_rhs,
                qpy_replay: Vec::from([replay]),
                parameter_symbols: self.merge_parameter_symbols(rhs),
            },
        }
    }
    pub fn radd(&self, lhs: &ParameterExpression) -> ParameterExpression {
        let replay = OPReplay::_INSTRUCTION {
            op: _OPCode::ADD,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: Some(ParameterValueType::clone_expr_for_replay(lhs)),
        };
        let expr_rhs = match &self.inner {
            ParameterInner::Expression { expr, .. } => expr,
            _ => &SymbolExpr::Symbol(Box::new(self.to_string())),
        };
        let expr_lhs = match &lhs.inner {
            ParameterInner::Expression { expr, .. } => expr,
            _ => &SymbolExpr::Symbol(Box::new(lhs.to_string())),
        };
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: expr_lhs + expr_rhs,
                qpy_replay: Vec::from([replay]),
                parameter_symbols: self.merge_parameter_symbols(lhs),
            },
        }
    }
    pub fn sub(&self, rhs: &ParameterExpression) -> ParameterExpression {
        let replay = OPReplay::_INSTRUCTION {
            op: _OPCode::SUB,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: Some(ParameterValueType::clone_expr_for_replay(rhs)),
        };
        let expr_lhs = match &self.inner {
            ParameterInner::Expression { expr, .. } => expr,
            _ => &SymbolExpr::Symbol(Box::new(self.to_string())),
        };
        let expr_rhs = match &rhs.inner {
            ParameterInner::Expression { expr, .. } => expr,
            _ => &SymbolExpr::Symbol(Box::new(rhs.to_string())),
        };
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: expr_lhs - expr_rhs,
                qpy_replay: Vec::from([replay]),
                parameter_symbols: self.merge_parameter_symbols(rhs),
            },
        }
    }
    pub fn rsub(&self, lhs: &ParameterExpression) -> ParameterExpression {
        let replay = OPReplay::_INSTRUCTION {
            op: _OPCode::RSUB,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: Some(ParameterValueType::clone_expr_for_replay(lhs)),
        };
        let expr_rhs = match &self.inner {
            ParameterInner::Expression { expr, .. } => expr,
            _ => &SymbolExpr::Symbol(Box::new(self.to_string())),
        };
        let expr_lhs = match &lhs.inner {
            ParameterInner::Expression { expr, .. } => expr,
            _ => &SymbolExpr::Symbol(Box::new(lhs.to_string())),
        };
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: expr_lhs - expr_rhs,
                qpy_replay: Vec::from([replay]),
                parameter_symbols: self.merge_parameter_symbols(lhs),
            },
        }
    }
    pub fn mul(&self, rhs: &ParameterExpression) -> ParameterExpression {
        let replay = OPReplay::_INSTRUCTION {
            op: _OPCode::MUL,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: Some(ParameterValueType::clone_expr_for_replay(rhs)),
        };
        let expr_lhs = match &self.inner {
            ParameterInner::Expression { expr, .. } => expr,
            _ => &SymbolExpr::Symbol(Box::new(self.to_string())),
        };
        let expr_rhs = match &rhs.inner {
            ParameterInner::Expression { expr, .. } => expr,
            _ => &SymbolExpr::Symbol(Box::new(rhs.to_string())),
        };
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: expr_lhs * expr_rhs,
                qpy_replay: Vec::from([replay]),
                parameter_symbols: self.merge_parameter_symbols(rhs),
            },
        }
    }
    pub fn rmul(&self, lhs: &ParameterExpression) -> ParameterExpression {
        let replay = OPReplay::_INSTRUCTION {
            op: _OPCode::MUL,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: Some(ParameterValueType::clone_expr_for_replay(lhs)),
        };
        let expr_rhs = match &self.inner {
            ParameterInner::Expression { expr, .. } => expr,
            _ => &SymbolExpr::Symbol(Box::new(self.to_string())),
        };
        let expr_lhs = match &lhs.inner {
            ParameterInner::Expression { expr, .. } => expr,
            _ => &SymbolExpr::Symbol(Box::new(lhs.to_string())),
        };
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: expr_lhs * expr_rhs,
                qpy_replay: Vec::from([replay]),
                parameter_symbols: self.merge_parameter_symbols(lhs),
            },
        }
    }
    pub fn div(&self, rhs: &ParameterExpression) -> ParameterExpression {
        let replay = OPReplay::_INSTRUCTION {
            op: _OPCode::DIV,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: Some(ParameterValueType::clone_expr_for_replay(rhs)),
        };
        let expr_lhs = match &self.inner {
            ParameterInner::Expression { expr, .. } => expr,
            _ => &SymbolExpr::Symbol(Box::new(self.to_string())),
        };
        let expr_rhs = match &rhs.inner {
            ParameterInner::Expression { expr, .. } => expr,
            _ => &SymbolExpr::Symbol(Box::new(rhs.to_string())),
        };
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: expr_lhs / expr_rhs,
                qpy_replay: Vec::from([replay]),
                parameter_symbols: self.merge_parameter_symbols(rhs),
            },
        }
    }
    pub fn rdiv(&self, lhs: &ParameterExpression) -> ParameterExpression {
        let replay = OPReplay::_INSTRUCTION {
            op: _OPCode::RDIV,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: Some(ParameterValueType::clone_expr_for_replay(lhs)),
        };
        let expr_rhs = match &self.inner {
            ParameterInner::Expression { expr, .. } => expr,
            _ => &SymbolExpr::Symbol(Box::new(self.to_string())),
        };
        let expr_lhs = match &lhs.inner {
            ParameterInner::Expression { expr, .. } => expr,
            _ => &SymbolExpr::Symbol(Box::new(lhs.to_string())),
        };
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: expr_lhs / expr_rhs,
                qpy_replay: Vec::from([replay]),
                parameter_symbols: self.merge_parameter_symbols(lhs),
            },
        }
    }
    pub fn pow(&self, rhs: &ParameterExpression) -> ParameterExpression {
        let replay = OPReplay::_INSTRUCTION {
            op: _OPCode::POW,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: Some(ParameterValueType::clone_expr_for_replay(rhs)),
        };
        let expr_lhs = match &self.inner {
            ParameterInner::Expression { expr, .. } => expr,
            _ => &SymbolExpr::Symbol(Box::new(self.to_string())),
        };
        let expr_rhs = match &rhs.inner {
            ParameterInner::Expression { expr, .. } => expr,
            _ => &SymbolExpr::Symbol(Box::new(rhs.to_string())),
        };
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: expr_lhs.pow(expr_rhs),
                qpy_replay: Vec::from([replay]),
                parameter_symbols: self.merge_parameter_symbols(rhs),
            },
        }
    }
    pub fn rpow(&self, lhs: &ParameterExpression) -> ParameterExpression {
        let replay = OPReplay::_INSTRUCTION {
            op: _OPCode::RPOW,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: Some(ParameterValueType::clone_expr_for_replay(lhs)),
        };
        let expr_rhs = match &self.inner {
            ParameterInner::Expression { expr, .. } => expr,
            _ => &SymbolExpr::Symbol(Box::new(self.to_string())),
        };
        let expr_lhs = match &lhs.inner {
            ParameterInner::Expression { expr, .. } => expr,
            _ => &SymbolExpr::Symbol(Box::new(lhs.to_string())),
        };
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: expr_lhs.pow(expr_rhs),
                qpy_replay: Vec::from([replay]),
                parameter_symbols: self.merge_parameter_symbols(lhs),
            },
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
        let mut symbols: HashSet<Arc<ParameterExpression>> = match self._my_parameters() {
            Some(s) => s,
            None => HashSet::new(),
        };

        for (key, param) in &in_map {
            // check if value in map is valid
            if let ParameterInner::Expression {
                expr: SymbolExpr::Value(v),
                ..
            } = &param.inner
            {
                if let symbol_expr::Value::Real(r) = v {
                    if r.is_nan() || r.is_infinite() {
                        return Err(CircuitError::new_err(
                            "Expression cannot bind non-numeric values",
                        ));
                    }
                } else if let symbol_expr::Value::Complex(c) = v {
                    if c.is_nan() || c.is_infinite() {
                        return Err(CircuitError::new_err(
                            "Expression cannot bind non-numeric values",
                        ));
                    }
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
            if symbols.contains(key) {
                symbols.remove(key);
            } else if !allow_unknown_parameters {
                unknown_params.insert(key.to_string());
            }

            match &param.inner {
                ParameterInner::Expression {
                    expr,
                    qpy_replay: _,
                    parameter_symbols,
                } => match parameter_symbols {
                    Some(o) => {
                        for k in o {
                            symbols.insert(k.clone());
                        }
                    }
                    None => {
                        if let SymbolExpr::Symbol { .. } = expr {
                            symbols.insert(Arc::new(param.to_owned()));
                        }
                    }
                },
                _ => {
                    symbols.insert(Arc::new(param.to_owned()));
                }
            }
        }
        if !allow_unknown_parameters && !unknown_params.is_empty() {
            return Err(CircuitError::new_err(format!(
                "Cannot bind Parameters ({:?}) not present in expression.",
                unknown_params
            )));
        }

        let bound = match &self.inner {
            ParameterInner::Expression { expr, .. } => expr.subs(&map),
            _ => self.expr().subs(&map),
        };

        let mut replay = match &self.inner {
            ParameterInner::Expression {
                expr: _,
                qpy_replay,
                ..
            } => qpy_replay.clone(),
            _ => Vec::<OPReplay>::new(),
        };
        replay.push(OPReplay::_SUBS {
            binds: subs_map,
            op: _OPCode::SUBSTITUTE,
        });
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
                inner: ParameterInner::Expression {
                    expr: ret,
                    qpy_replay: replay,
                    parameter_symbols: if !symbols.is_empty() {
                        Some(symbols)
                    } else {
                        None
                    },
                },
            })
        } else {
            Ok(ParameterExpression {
                inner: ParameterInner::Expression {
                    expr: bound,
                    qpy_replay: replay,
                    parameter_symbols: if !symbols.is_empty() {
                        Some(symbols)
                    } else {
                        None
                    },
                },
            })
        }
    }

    // compare 2 expression
    // check_uuid = true also compares uuid for equality
    pub fn compare_eq(&self, other: &ParameterExpression, check_uuid: bool) -> bool {
        match (&self.inner, &other.inner) {
            (
                ParameterInner::Symbol { .. } | ParameterInner::VectorElement { .. },
                ParameterInner::Symbol { .. } | ParameterInner::VectorElement { .. },
            ) => {
                if self.to_string() == other.to_string() {
                    if check_uuid {
                        self.uuid() == other.uuid()
                    } else {
                        true
                    }
                } else {
                    false
                }
            }
            (
                ParameterInner::Expression { expr, .. },
                ParameterInner::Expression {
                    expr: other_expr, ..
                },
            ) => {
                if expr == other_expr {
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
            (ParameterInner::Expression { expr, .. }, _) => {
                if expr == &other.expr() {
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
            (
                _,
                ParameterInner::Expression {
                    expr: other_expr, ..
                },
            ) => {
                if &self.expr() == other_expr {
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
        match &self.inner {
            ParameterInner::Expression {
                expr,
                qpy_replay,
                parameter_symbols,
            } => match expr {
                SymbolExpr::Value(_) | SymbolExpr::Symbol(_) => self.clone(),
                _ => ParameterExpression {
                    inner: ParameterInner::Expression {
                        expr: expr.expand(),
                        qpy_replay: qpy_replay.clone(),
                        parameter_symbols: parameter_symbols.clone(),
                    },
                },
            },
            _ => self.clone(),
        }
    }

    /// calculate gradient
    pub fn gradient(&self, param: &Self) -> Result<Self, String> {
        let replay = OPReplay::_INSTRUCTION {
            op: _OPCode::GRAD,
            lhs: Some(ParameterValueType::clone_expr_for_replay(self)),
            rhs: Some(ParameterValueType::clone_expr_for_replay(param)),
        };

        if let ParameterInner::Expression {
            expr,
            qpy_replay: _,
            parameter_symbols,
        } = &self.inner
        {
            if let Some(parameter_symbols) = parameter_symbols {
                if parameter_symbols.is_empty() {
                    return Ok(ParameterExpression {
                        inner: ParameterInner::Expression {
                            expr: SymbolExpr::Value(Value::Int(0)),
                            qpy_replay: Vec::from([replay]),
                            parameter_symbols: None,
                        },
                    });
                }

                let p = match &param.inner {
                    ParameterInner::Expression { expr, .. } => expr,
                    _ => &SymbolExpr::Symbol(Box::new(param.to_string())),
                };
                let expr_grad = match expr.derivative(p) {
                    Ok(expr) => expr,
                    Err(e) => return Err(e),
                };
                match expr_grad.eval(true) {
                    Some(v) => Ok(ParameterExpression {
                        inner: ParameterInner::Expression {
                            expr: SymbolExpr::Value(v),
                            qpy_replay: Vec::from([replay]),
                            parameter_symbols: None,
                        },
                    }),
                    None => {
                        // update parameter symbols
                        let symbols = expr_grad.symbols();
                        let mut new_map = HashSet::<Arc<ParameterExpression>>::new();
                        for symbol in parameter_symbols {
                            if symbols.contains(&symbol.to_string()) {
                                new_map.insert(symbol.clone());
                            }
                        }
                        Ok(ParameterExpression {
                            inner: ParameterInner::Expression {
                                expr: expr_grad,
                                qpy_replay: Vec::from([replay]),
                                parameter_symbols: Some(new_map),
                            },
                        })
                    }
                }
            } else {
                Ok(ParameterExpression {
                    inner: ParameterInner::Expression {
                        expr: SymbolExpr::Value(Value::Int(0)),
                        qpy_replay: Vec::from([replay]),
                        parameter_symbols: None,
                    },
                })
            }
        } else if self.compare_eq(param, true) {
            Ok(ParameterExpression {
                inner: ParameterInner::Expression {
                    expr: SymbolExpr::Value(Value::Int(1)),
                    qpy_replay: Vec::from([replay]),
                    parameter_symbols: None,
                },
            })
        } else {
            Ok(ParameterExpression {
                inner: ParameterInner::Expression {
                    expr: SymbolExpr::Value(Value::Int(0)),
                    qpy_replay: Vec::from([replay]),
                    parameter_symbols: None,
                },
            })
        }
    }

    fn set_vector(&mut self, in_vector: Arc<ParameterVector>) {
        if let ParameterInner::VectorElement {
            name: _,
            index: _,
            uuid: _,
            vector,
        } = &mut self.inner
        {
            *vector = Some(in_vector);
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
        match &self.inner {
            ParameterInner::Expression { expr, .. } => expr == r,
            _ => false,
        }
    }
}

impl PartialEq<Complex64> for ParameterExpression {
    fn eq(&self, c: &Complex64) -> bool {
        match &self.inner {
            ParameterInner::Expression { expr, .. } => expr == c,
            _ => false,
        }
    }
}

// =============================
// Make from Rust native types
// =============================

impl From<i32> for ParameterExpression {
    fn from(v: i32) -> Self {
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: SymbolExpr::Value(symbol_expr::Value::Real(v as f64)),
                qpy_replay: Vec::new(),
                parameter_symbols: None,
            },
        }
    }
}
impl From<i64> for ParameterExpression {
    fn from(v: i64) -> Self {
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: SymbolExpr::Value(symbol_expr::Value::Real(v as f64)),
                qpy_replay: Vec::new(),
                parameter_symbols: None,
            },
        }
    }
}

impl From<u32> for ParameterExpression {
    fn from(v: u32) -> Self {
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: SymbolExpr::Value(symbol_expr::Value::Real(v as f64)),
                qpy_replay: Vec::new(),
                parameter_symbols: None,
            },
        }
    }
}

impl From<f64> for ParameterExpression {
    fn from(v: f64) -> Self {
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: SymbolExpr::Value(symbol_expr::Value::Real(v)),
                qpy_replay: Vec::new(),
                parameter_symbols: None,
            },
        }
    }
}

impl From<Complex64> for ParameterExpression {
    fn from(v: Complex64) -> Self {
        ParameterExpression {
            inner: ParameterInner::Expression {
                expr: SymbolExpr::Value(symbol_expr::Value::Complex(v)),
                qpy_replay: Vec::new(),
                parameter_symbols: None,
            },
        }
    }
}

impl From<&str> for ParameterExpression {
    fn from(s: &str) -> Self {
        ParameterExpression {
            inner: ParameterInner::Symbol {
                name: Box::new(s.to_string()),
                uuid: Uuid::new_v4().as_u128(),
            },
        }
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
    #[pyo3(signature = (symbol_map = None, expr = None, *, _qpy_replay = None))]
    pub fn __new__(
        symbol_map: Option<HashMap<ParameterExpression, PyObject>>,
        expr: Option<String>,
        _qpy_replay: Option<Vec<OPReplay>>,
    ) -> PyResult<Self> {
        let (Some(symbol_map), Some(expr)) = (symbol_map, expr) else {
            return Ok(ParameterExpression::default());
        };

        // check if expr contains replacements for sympy
        let expr = expr
            .replace("__begin_sympy_replace__", "$\\")
            .replace("__end_sympy_replace__", "$");
        match parse_expression(&expr) {
            Ok(expr) => {
                let mut parameter_symbols = HashSet::<Arc<ParameterExpression>>::new();
                for (param, _) in symbol_map {
                    parameter_symbols.insert(Arc::new(param.to_owned()));
                }
                // substitute 'I' to imaginary number i before returning expression
                Ok(ParameterExpression {
                    inner: ParameterInner::Expression {
                        expr: expr.bind(&HashMap::from([(
                            "I".to_string(),
                            symbol_expr::Value::from(Complex64::i()),
                        )])),
                        qpy_replay: _qpy_replay.unwrap_or_default(),
                        parameter_symbols: Some(parameter_symbols),
                    },
                })
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
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
        let ret = match &self.inner {
            ParameterInner::Expression { expr, .. } => expr.optimize().sympify().to_string(),
            _ => self.to_string(),
        };
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
        if let ParameterInner::Expression {
            expr,
            qpy_replay: _,
            parameter_symbols,
        } = &self.inner
        {
            match expr.eval(true) {
                Some(v) => match v {
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
                None => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "Expression with unbound parameters '{:?}' is not numeric",
                    parameter_symbols
                ))),
            }
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Expression is a symbol '{}', not numeric",
                self
            )))
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

    /// check if ParameterVectorElement
    #[getter("is_vector_element")]
    pub fn py_is_vector_element(&self) -> bool {
        self.is_vector_element()
    }

    /// get uuid for this symbol
    #[getter("uuid")]
    pub fn py_get_uuid(&self) -> u128 {
        self.uuid()
    }

    /// get ParameterVector if this is ParameterVectorElement
    #[getter("vector")]
    pub fn py_get_vector(&self) -> PyResult<ParameterVector> {
        if let ParameterInner::VectorElement {
            name: _,
            index: _,
            uuid: _,
            vector: Some(v),
        } = &self.inner
        {
            Ok(Arc::unwrap_or_clone(v.clone()))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Not a vector element",
            ))
        }
    }
    // backward compatibility, some code accesses _vector member directly
    #[getter("_vector")]
    pub fn py_vector(&self) -> PyResult<ParameterVector> {
        self.py_get_vector()
    }

    /// get index in ParameterVector if this is ParameterVectorElement
    #[getter("index")]
    pub fn py_get_index(&self) -> usize {
        if let ParameterInner::VectorElement { name: _, index, .. } = &self.inner {
            *index
        } else {
            0
        }
    }
    // backward compatibility, some code accesses _index member directly
    #[getter("_index")]
    pub fn py_index(&self) -> usize {
        self.py_get_index()
    }

    /// return this as complex if this is numeric
    pub fn __complex__(&self) -> PyResult<Complex64> {
        if let ParameterInner::Expression {
            expr,
            qpy_replay: _,
            parameter_symbols,
        } = &self.inner
        {
            match expr.eval(true) {
                Some(v) => match v {
                    symbol_expr::Value::Real(r) => Ok(Complex64::from(r)),
                    symbol_expr::Value::Int(i) => Ok(Complex64::from(i as f64)),
                    symbol_expr::Value::Complex(c) => Ok(c),
                },
                None => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "ParameterExpression with unbound parameters ({:?}) cannot be cast to a complex.",
                    parameter_symbols
                ))),
            }
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "ParameterExpression is a symbol '{}' cannot be cast to a complex.",
                self
            )))
        }
    }

    /// return this as real if this is numeric
    pub fn __float__(&self) -> PyResult<f64> {
        if let ParameterInner::Expression {
            expr,
            qpy_replay: _,
            parameter_symbols,
        } = &self.inner
        {
            match expr.eval(true) {
                Some(v) => match v {
                    symbol_expr::Value::Real(r) => Ok(r),
                    symbol_expr::Value::Int(i) => Ok(i as f64),
                    symbol_expr::Value::Complex(c) => {
                        if (-symbol_expr::SYMEXPR_EPSILON..symbol_expr::SYMEXPR_EPSILON)
                            .contains(&c.im)
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
                    parameter_symbols
                ))),
            }
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "ParameterExpression is a symbol '{}' cannot be cast to a float.",
                self
            )))
        }
    }

    /// return this as int if this is numeric
    pub fn __int__(&self) -> PyResult<i64> {
        if let ParameterInner::Expression {
            expr,
            qpy_replay: _,
            parameter_symbols,
        } = &self.inner
        {
            match expr.eval(true) {
                Some(v) => match v {
                    symbol_expr::Value::Real(r) => Ok(r as i64),
                    symbol_expr::Value::Int(i) => Ok(i),
                    symbol_expr::Value::Complex(c) => {
                        if (-symbol_expr::SYMEXPR_EPSILON..symbol_expr::SYMEXPR_EPSILON)
                            .contains(&c.im)
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
                    parameter_symbols
                ))),
            }
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "ParameterExpression is a symbol '{}' cannot be cast to int.",
                self
            )))
        }
    }

    /// clone expression
    pub fn __copy__(&self) -> Self {
        self.clone()
    }
    pub fn __deepcopy__(&self, _memo: Option<PyObject>) -> Self {
        self.clone()
    }

    /// return derivative of this expression for param
    #[pyo3(name = "derivative")]
    pub fn py_derivative(&self, param: &ParameterExpression) -> PyResult<ParameterExpression> {
        match self.gradient(param) {
            Ok(expr) => Ok(expr),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }

    /// return conjugate of expression
    #[pyo3(name = "conjugate")]
    pub fn py_conjugate(&self) -> ParameterExpression {
        self.conjugate()
    }

    /// Get the derivative of a parameter expression w.r.t. a specified parameter expression.
    /// Args:
    ///     param (Parameter): Parameter w.r.t. which we want to take the derivative
    /// Returns:
    ///     ParameterExpression representing the gradient of param_expr w.r.t. param
    ///     or complex or float number
    #[pyo3(name = "gradient")]
    pub fn py_gradient(&self, param: &Self, py: Python) -> PyResult<PyObject> {
        match self.gradient(param) {
            Ok(grad) => match &grad.inner {
                ParameterInner::Expression {
                    expr: SymbolExpr::Value(v),
                    ..
                } => match v {
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
    pub fn py_get_parameters(&self, py: Python) -> PyResult<Py<PySet>> {
        let out = PySet::empty(py)?;
        match &self.inner {
            ParameterInner::Expression {
                expr: _,
                qpy_replay: _,
                parameter_symbols,
            } => match &parameter_symbols {
                Some(symbols) => {
                    for s in symbols {
                        // initialize new Parameter/ParameterVectorElement object from ParameterExpression in symbols set
                        match &s.inner {
                            ParameterInner::VectorElement { .. } => {
                                let pve = Py::new(
                                    py,
                                    PyClassInitializer::from(
                                        Arc::<ParameterExpression>::unwrap_or_clone(s.clone()),
                                    )
                                    .add_subclass(PyParameter {})
                                    .add_subclass(PyParameterVectorElement {}),
                                )?;
                                out.add(pve)?;
                            }
                            _ => {
                                let pp = Py::new(
                                    py,
                                    PyClassInitializer::from(
                                        Arc::<ParameterExpression>::unwrap_or_clone(s.clone()),
                                    )
                                    .add_subclass(PyParameter {}),
                                )?;
                                out.add(pp)?;
                            }
                        }
                    }
                    Ok(out.unbind())
                }
                None => Ok(out.unbind()),
            },
            ParameterInner::VectorElement { .. } => {
                let pve = Py::new(
                    py,
                    PyClassInitializer::from(self.clone())
                        .add_subclass(PyParameter {})
                        .add_subclass(PyParameterVectorElement {}),
                )?;
                out.add(pve)?;
                Ok(out.unbind())
            }
            ParameterInner::Symbol { .. } => {
                let pp = Py::new(
                    py,
                    PyClassInitializer::from(self.clone()).add_subclass(PyParameter {}),
                )?;
                out.add(pp)?;
                Ok(out.unbind())
            }
        }
    }
    #[getter("_parameter_symbols")]
    pub fn py_parameter_symbols(&self, py: Python) -> PyResult<Py<PySet>> {
        self.py_get_parameters(py)
    }

    /// return all values in this equation
    #[pyo3(name = "values")]
    pub fn py_values(&self, py: Python) -> PyResult<Vec<PyObject>> {
        match &self.inner {
            ParameterInner::Expression { expr, .. } => expr
                .values()
                .iter()
                .map(|val| match val {
                    symbol_expr::Value::Real(r) => r.into_py_any(py),
                    symbol_expr::Value::Int(i) => i.into_py_any(py),
                    symbol_expr::Value::Complex(c) => c.into_py_any(py),
                })
                .collect(),
            _ => Ok(Vec::new()),
        }
    }

    /// return expression as a string
    #[getter("name")]
    pub fn py_name(&self) -> String {
        self.__str__()
    }

    #[pyo3(name = "assign")]
    pub fn py_assign(&self, param: &ParameterExpression, value: &Bound<PyAny>) -> PyResult<Self> {
        if let Some(e) = _extract_value(value) {
            let eval = matches!(
                &e.inner,
                ParameterInner::Expression {
                    expr: SymbolExpr::Value(_),
                    ..
                }
            );
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
    #[pyo3(name="bind", signature = (in_map, allow_unknown_parameters = None))]
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
    #[pyo3(name="subs", signature = (map, allow_unknown_parameters = None))]
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
            Some(rhs) => {
                if let (
                    ParameterInner::Expression { expr: l, .. },
                    ParameterInner::Expression { expr: r, .. },
                ) = (&self.inner, &rhs.inner)
                {
                    l < r
                } else {
                    false
                }
            }
            None => false,
        }
    }
    pub fn __gt__(&self, rhs: &Bound<PyAny>) -> bool {
        match _extract_value(rhs) {
            Some(rhs) => {
                if let (
                    ParameterInner::Expression { expr: l, .. },
                    ParameterInner::Expression { expr: r, .. },
                ) = (&self.inner, &rhs.inner)
                {
                    l > r
                } else {
                    false
                }
            }
            None => false,
        }
    }

    // unary operators
    pub fn __neg__(&self) -> ParameterExpression {
        self.neg()
    }
    pub fn __pos__(&self) -> ParameterExpression {
        self.pos()
    }
    #[pyo3(name = "sin")]
    pub fn py_sin(&self) -> ParameterExpression {
        self.sin()
    }
    #[pyo3(name = "cos")]
    pub fn py_cos(&self) -> ParameterExpression {
        self.cos()
    }
    #[pyo3(name = "tan")]
    pub fn py_tan(&self) -> ParameterExpression {
        self.tan()
    }
    pub fn arcsin(&self) -> ParameterExpression {
        self.asin()
    }
    pub fn arccos(&self) -> ParameterExpression {
        self.acos()
    }
    pub fn arctan(&self) -> ParameterExpression {
        self.atan()
    }
    #[pyo3(name = "exp")]
    pub fn py_exp(&self) -> ParameterExpression {
        self.exp()
    }
    #[pyo3(name = "log")]
    pub fn py_log(&self) -> ParameterExpression {
        self.log()
    }
    pub fn __abs__(&self) -> ParameterExpression {
        self.abs()
    }
    #[pyo3(name = "abs")]
    pub fn py_abs(&self) -> ParameterExpression {
        self.abs()
    }
    #[pyo3(name = "sign")]
    pub fn py_sign(&self) -> ParameterExpression {
        self.sign()
    }

    // binary operators
    pub fn __add__(&self, rhs: &Bound<PyAny>) -> PyResult<ParameterExpression> {
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
    pub fn __radd__(&self, lhs: &Bound<PyAny>) -> PyResult<ParameterExpression> {
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
    pub fn __sub__(&self, rhs: &Bound<PyAny>) -> PyResult<ParameterExpression> {
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
    pub fn __rsub__(&self, lhs: &Bound<PyAny>) -> PyResult<ParameterExpression> {
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
    pub fn __mul__(&self, rhs: &Bound<PyAny>) -> PyResult<ParameterExpression> {
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
    pub fn __rmul__(&self, lhs: &Bound<PyAny>) -> PyResult<ParameterExpression> {
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

    pub fn __truediv__(&self, rhs: &Bound<PyAny>) -> PyResult<ParameterExpression> {
        match _extract_value(rhs) {
            Some(rhs) => {
                if let ParameterInner::Expression {
                    expr: SymbolExpr::Value(v),
                    ..
                } = &rhs.inner
                {
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

    pub fn __rtruediv__(&self, lhs: &Bound<PyAny>) -> PyResult<ParameterExpression> {
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
    pub fn __pow__(
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
    pub fn __rpow__(
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
        match &self.inner {
            ParameterInner::Expression { expr, .. } => {
                if let SymbolExpr::Symbol(_) = expr {
                    return expr.to_string();
                }
                match expr.eval(true) {
                    Some(e) => e.to_string(),
                    None => expr.optimize().to_string(),
                }
            }
            _ => self.to_string(),
        }
    }

    pub fn __hash__(&self, py: Python) -> PyResult<isize> {
        match &self.inner {
            ParameterInner::Expression { expr, .. } => match expr.eval(true) {
                Some(v) => match v {
                    symbol_expr::Value::Int(i) => i.into_pyobject(py)?.hash(),
                    symbol_expr::Value::Real(r) => r.into_pyobject(py)?.hash(),
                    symbol_expr::Value::Complex(c) => c.into_pyobject(py)?.hash(),
                },
                None => expr.to_string().into_pyobject(py)?.hash(),
            },
            _ => self.to_string().into_pyobject(py)?.hash(),
        }
    }

    // for pickle, we can reproduce equation from expression string
    #[allow(clippy::type_complexity)]
    fn __getstate__(
        &self,
    ) -> PyResult<(
        String,
        Option<u128>,
        Option<usize>,
        Option<ParameterVector>,
        Option<HashMap<String, u128>>,
    )> {
        match &self.inner {
            ParameterInner::Symbol { name, uuid } => {
                Ok((name.as_ref().clone(), Some(*uuid), None, None, None))
            }
            ParameterInner::VectorElement {
                name,
                index,
                uuid,
                vector,
            } => Ok((
                name.as_ref().clone(),
                Some(*uuid),
                Some(*index),
                vector.as_ref().map(|v| Arc::unwrap_or_clone(v.clone())),
                None,
            )),
            ParameterInner::Expression {
                expr: _,
                qpy_replay: _,
                parameter_symbols,
            } => {
                if let Some(symbols) = &parameter_symbols {
                    let mut ret = HashMap::<String, u128>::new();
                    for s in symbols {
                        ret.insert(s.to_string(), s.uuid());
                    }
                    return Ok((self.to_string(), None, None, None, Some(ret)));
                }
                Ok((self.to_string(), None, None, None, None))
            }
        }
    }

    #[allow(clippy::type_complexity)]
    fn __setstate__(
        &mut self,
        state: (
            String,
            Option<u128>,
            Option<usize>,
            Option<ParameterVector>,
            Option<HashMap<String, u128>>,
        ),
    ) -> PyResult<()> {
        match parse_expression(&state.0) {
            Ok(expr) => {
                if let Some(uuid) = state.1 {
                    if let Some(index) = state.2 {
                        if let Some(vector) = state.3 {
                            self.inner = ParameterInner::VectorElement {
                                name: Box::new(expr.to_string()),
                                index,
                                uuid,
                                vector: Some(Arc::new(vector)),
                            };
                        } else {
                            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                                "index of vector element is specified but vector is not specified",
                            ));
                        }
                    } else {
                        self.inner = ParameterInner::Symbol {
                            name: Box::new(expr.to_string()),
                            uuid,
                        };
                    }
                } else if let Some(symbols) = state.4 {
                    let mut parameter_symbols = HashSet::<Arc<ParameterExpression>>::new();
                    for (name, uuid) in symbols {
                        parameter_symbols.insert(Arc::<ParameterExpression>::new(
                            ParameterExpression::new(name, Some(uuid)),
                        ));
                    }
                    self.inner = ParameterInner::Expression {
                        expr,
                        qpy_replay: Vec::new(),
                        parameter_symbols: Some(parameter_symbols),
                    };
                } else {
                    self.inner = ParameterInner::Expression {
                        expr,
                        qpy_replay: Vec::new(),
                        parameter_symbols: None,
                    };
                }
                Ok(())
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }

    pub fn __repr__(&self) -> String {
        match &self.inner {
            ParameterInner::Symbol { .. } => format!("Parameter({})", self),
            ParameterInner::VectorElement { .. } => {
                format!("ParameterVectorElement({})", self)
            }
            ParameterInner::Expression { expr, .. } => {
                format!("ParameterExpression({})", expr.optimize())
            }
        }
    }

    /// return QPY replay
    #[pyo3(name = "replay")]
    pub fn py_get_replay(&self) -> Option<Vec<OPReplay>> {
        match &self.inner {
            ParameterInner::Expression {
                expr: _,
                qpy_replay,
                ..
            } => Some(qpy_replay.clone()),
            _ => None,
        }
    }
    #[getter("_qpy_replay")]
    pub fn py_qpy_replay(&self) -> Option<Vec<OPReplay>> {
        match &self.inner {
            ParameterInner::Expression {
                expr: _,
                qpy_replay,
                ..
            } => Some(qpy_replay.clone()),
            _ => None,
        }
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

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PyObject> {
        // make new vector element object from ParameterExpression for next object
        match slf.inner.next() {
            Some(e) => {
                match Py::new(
                    slf.py(),
                    PyClassInitializer::from(e.clone())
                        .add_subclass(PyParameter {})
                        .add_subclass(PyParameterVectorElement {}),
                ) {
                    Ok(p) => match p.into_py_any(slf.py()) {
                        Ok(p) => Some(p),
                        _ => None,
                    },
                    _ => None,
                }
            }
            None => None,
        }
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
            root_uuid,
            params: Vec::with_capacity(length),
        };

        for i in 0..length {
            let pe = ParameterExpression {
                inner: ParameterInner::VectorElement {
                    name: Box::new(name.clone()),
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
            // initialize new ParameterVectorElement object from ParameterExpression
            let pve = Py::new(
                py,
                PyClassInitializer::from(s.clone())
                    .add_subclass(PyParameter {})
                    .add_subclass(PyParameterVectorElement {}),
            )?;
            out.append(pve)?;
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
            SequenceIndex::Int(index) => {
                // return new ParameterVectorElement object made from ParameterExpression
                let pve = Py::new(
                    py,
                    PyClassInitializer::from(self.params[index].clone())
                        .add_subclass(PyParameter {})
                        .add_subclass(PyParameterVectorElement {}),
                )?;
                pve.into_py_any(py)
            }
            indices => {
                let out = PyList::empty(py);
                for i in indices {
                    let pve = Py::new(
                        py,
                        PyClassInitializer::from(self.params[i].clone())
                            .add_subclass(PyParameter {})
                            .add_subclass(PyParameterVectorElement {}),
                    )?;
                    out.append(pve)?;
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
                        name: Box::new(self.name.clone()),
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
                    name: Box::new(self.name.clone()),
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
