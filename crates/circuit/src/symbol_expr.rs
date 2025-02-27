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

/// symbol_expr.rs
/// symbolic expression engine for parameter expression
use core::f64;
use hashbrown::{HashMap, HashSet};
use std::cmp::Ordering;
use std::cmp::PartialOrd;
use std::convert::From;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

use num_complex::Complex64;

/// node types of expression tree
#[derive(Debug, Clone)]
pub enum SymbolExpr {
    Symbol(Symbol),
    Value(Value),
    Unary(Arc<Unary>),
    Binary(Arc<Binary>),
}

/// symbol with its name
#[derive(Debug, Clone)]
pub struct Symbol {
    name: String,
}

/// Value type, can be integer, real or complex number
#[derive(Debug, Clone)]
pub enum Value {
    Real(f64),
    Int(i64),
    Complex(Complex64),
}

/// unary operation node has 1 branch node
#[derive(Debug, Clone)]
pub struct Unary {
    op: UnaryOps,
    expr: SymbolExpr,
}

/// binary operation node has 2 branch nodes
#[derive(Debug, Clone)]
pub struct Binary {
    op: BinaryOps,
    lhs: SymbolExpr,
    rhs: SymbolExpr,
}

/// definition of unary operations
#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOps {
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
}

/// definition of binary operations
#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOps {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

// functions to make new expr for add
#[inline(always)]
fn _add(lhs: SymbolExpr, rhs: SymbolExpr) -> SymbolExpr {
    if rhs.is_negative() {
        match rhs.neg_opt() {
            Some(e) => SymbolExpr::Binary(Arc::new(Binary {
                op: BinaryOps::Sub,
                lhs,
                rhs: e,
            })),
            None => SymbolExpr::Binary(Arc::new(Binary {
                op: BinaryOps::Sub,
                lhs,
                rhs: _neg(rhs),
            })),
        }
    } else {
        SymbolExpr::Binary(Arc::new(Binary {
            op: BinaryOps::Add,
            lhs,
            rhs,
        }))
    }
}

// functions to make new expr for sub
#[inline(always)]
fn _sub(lhs: SymbolExpr, rhs: SymbolExpr) -> SymbolExpr {
    if rhs.is_negative() {
        match rhs.neg_opt() {
            Some(e) => SymbolExpr::Binary(Arc::new(Binary {
                op: BinaryOps::Add,
                lhs,
                rhs: e,
            })),
            None => SymbolExpr::Binary(Arc::new(Binary {
                op: BinaryOps::Add,
                lhs,
                rhs: _neg(rhs),
            })),
        }
    } else {
        SymbolExpr::Binary(Arc::new(Binary {
            op: BinaryOps::Sub,
            lhs,
            rhs,
        }))
    }
}

// functions to make new expr for mul
#[inline(always)]
fn _mul(lhs: SymbolExpr, rhs: SymbolExpr) -> SymbolExpr {
    SymbolExpr::Binary(Arc::new(Binary {
        op: BinaryOps::Mul,
        lhs,
        rhs,
    }))
}

// functions to make new expr for div
#[inline(always)]
fn _div(lhs: SymbolExpr, rhs: SymbolExpr) -> SymbolExpr {
    SymbolExpr::Binary(Arc::new(Binary {
        op: BinaryOps::Div,
        lhs,
        rhs,
    }))
}

// functions to make new expr for pow
#[inline(always)]
fn _pow(lhs: SymbolExpr, rhs: SymbolExpr) -> SymbolExpr {
    SymbolExpr::Binary(Arc::new(Binary {
        op: BinaryOps::Pow,
        lhs,
        rhs,
    }))
}

// functions to make new expr for neg
#[inline(always)]
fn _neg(expr: SymbolExpr) -> SymbolExpr {
    SymbolExpr::Unary(Arc::new(Unary {
        op: UnaryOps::Neg,
        expr,
    }))
}

impl fmt::Display for SymbolExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                SymbolExpr::Symbol(e) => e.to_string(),
                SymbolExpr::Value(e) => e.to_string(),
                SymbolExpr::Unary(e) => e.to_string(),
                SymbolExpr::Binary(e) => e.to_string(),
            }
        )
    }
}

/// ==================================
/// SymbolExpr implementation
/// ==================================
impl SymbolExpr {
    /// bind value to symbol node
    pub fn bind(&self, maps: &HashMap<String, Value>) -> SymbolExpr {
        match self {
            SymbolExpr::Symbol(e) => e.bind(maps),
            SymbolExpr::Value(e) => SymbolExpr::Value(e.clone()),
            SymbolExpr::Unary(e) => e.bind(maps),
            SymbolExpr::Binary(e) => e.bind(maps),
        }
    }

    /// substitute symbol node to other expression
    pub fn subs(&self, maps: &HashMap<String, SymbolExpr>) -> SymbolExpr {
        match self {
            SymbolExpr::Symbol(e) => e.subs(maps),
            SymbolExpr::Value(e) => SymbolExpr::Value(e.clone()),
            SymbolExpr::Unary(e) => e.subs(maps),
            SymbolExpr::Binary(e) => e.subs(maps),
        }
    }

    /// evaluate the equation
    /// if recursive is false, only this node will be evaluated
    pub fn eval(&self, recurse: bool) -> Option<Value> {
        match self {
            SymbolExpr::Symbol(_) => None,
            SymbolExpr::Value(e) => Some(e.clone()),
            SymbolExpr::Unary(e) => e.eval(recurse),
            SymbolExpr::Binary(e) => e.eval(recurse),
        }
    }

    /// calculate derivative of the equantion for a symbol passed by param
    pub fn derivative(&self, param: &SymbolExpr) -> SymbolExpr {
        if self == param {
            SymbolExpr::Value(Value::Real(1.0))
        } else {
            match self {
                SymbolExpr::Value(_) | SymbolExpr::Symbol(_) => SymbolExpr::Value(Value::Real(0.0)),
                SymbolExpr::Unary(e) => e.derivative(param),
                SymbolExpr::Binary(e) => e.derivative(param),
            }
        }
    }

    /// expand the equation
    pub fn expand(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Symbol(_) => self.clone(),
            SymbolExpr::Value(_) => self.clone(),
            SymbolExpr::Unary(e) => e.expand(),
            SymbolExpr::Binary(e) => e.expand(),
        }
    }

    /// sign operator
    pub fn sign(&self) -> SymbolExpr {
        SymbolExpr::Unary(Arc::new(Unary {
            op: UnaryOps::Sign,
            expr: self.clone(),
        }))
    }

    /// return real number if equation can be evaluated
    pub fn real(&self) -> Option<f64> {
        match self.eval(true) {
            Some(v) => match v {
                Value::Real(r) => Some(r),
                Value::Int(r) => Some(r as f64),
                Value::Complex(c) => Some(c.re),
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
            },
            None => None,
        }
    }

    /// return hashset of all symbols this equation contains
    pub fn symbols(&self) -> HashSet<String> {
        match self {
            SymbolExpr::Symbol(e) => HashSet::<String>::from([e.name.clone()]),
            SymbolExpr::Value(_) => HashSet::<String>::new(),
            SymbolExpr::Unary(e) => e.symbols(),
            SymbolExpr::Binary(e) => e.symbols(),
        }
    }

    /// concatenate all symbols under this node (internal use for sorting nodes)
    pub fn get_symbols_string(&self) -> String {
        match self {
            SymbolExpr::Symbol(e) => e.name.clone(),
            SymbolExpr::Value(_) => String::new(),
            SymbolExpr::Unary(e) => e.expr.get_symbols_string(),
            SymbolExpr::Binary(e) => e.get_symbols_string(),
        }
    }

    /// check if a symbol is in this equation
    pub fn has_symbol(&self, param: &String) -> bool {
        match self {
            SymbolExpr::Symbol(e) => e.name == *param,
            SymbolExpr::Value(_) => false,
            SymbolExpr::Unary(e) => e.has_symbol(param),
            SymbolExpr::Binary(e) => e.has_symbol(param),
        }
    }

    /// return reciprocal of the equation
    pub fn rcp(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Symbol(e) => _div(
                SymbolExpr::Value(Value::Real(1.0)),
                SymbolExpr::Symbol(e.clone()),
            ),
            SymbolExpr::Value(e) => SymbolExpr::Value(e.rcp()),
            SymbolExpr::Unary(e) => _div(
                SymbolExpr::Value(Value::Real(1.0)),
                SymbolExpr::Unary(e.clone()),
            ),
            SymbolExpr::Binary(ref e) => match e.op {
                BinaryOps::Div => SymbolExpr::Binary(Arc::new(Binary {
                    op: e.op.clone(),
                    lhs: e.rhs.clone(),
                    rhs: e.lhs.clone(),
                })),
                _ => _div(SymbolExpr::Value(Value::Real(1.0)), self.clone()),
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
            SymbolExpr::Symbol(e) => SymbolExpr::Symbol(e.clone()),
            SymbolExpr::Value(e) => match e {
                Value::Complex(c) => SymbolExpr::Value(Value::Complex(c.conj())),
                _ => SymbolExpr::Value(e.clone()),
            },
            SymbolExpr::Unary(e) => SymbolExpr::Unary(Arc::new(Unary {
                op: e.op.clone(),
                expr: e.expr.conjugate(),
            })),
            SymbolExpr::Binary(e) => SymbolExpr::Binary(Arc::new(Binary {
                op: e.op.clone(),
                lhs: e.lhs.conjugate(),
                rhs: e.rhs.conjugate(),
            })),
        }
    }

    /// check if complex number or not
    pub fn is_complex(&self) -> Option<bool> {
        match self.eval(true) {
            Some(v) => match v {
                Value::Complex(_) => Some(true),
                _ => Some(false),
            },
            None => None,
        }
    }

    /// check if real number or not
    pub fn is_real(&self) -> Option<bool> {
        match self.eval(true) {
            Some(v) => match v {
                Value::Real(_) => Some(true),
                Value::Int(_) => Some(false),
                Value::Complex(c) => Some(c.im < f64::EPSILON && c.im > -f64::EPSILON),
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

    /// check if evaluated result is 0
    pub fn is_zero(&self) -> bool {
        match self.eval(true) {
            Some(v) => v.is_zero(),
            None => false,
        }
    }

    /// check if evaluated result is 1
    pub fn is_one(&self) -> bool {
        match self.eval(true) {
            Some(v) => v.is_one(),
            None => false,
        }
    }

    /// check if evaluated result is -1
    pub fn is_minus_one(&self) -> bool {
        match self.eval(true) {
            Some(v) => v.is_minus_one(),
            None => false,
        }
    }

    /// check if evaluated result is negative
    fn is_negative(&self) -> bool {
        match self {
            SymbolExpr::Value(v) => v.is_negative(),
            SymbolExpr::Symbol(_) => false,
            SymbolExpr::Unary(u) => match u.op {
                UnaryOps::Abs => false,
                UnaryOps::Neg => !u.expr.is_negative(),
                _ => false, // TO DO add heuristic determination
            },
            SymbolExpr::Binary(b) => match b.op {
                BinaryOps::Mul | BinaryOps::Div => b.lhs.is_negative() ^ b.rhs.is_negative(),
                BinaryOps::Add | BinaryOps::Sub => b.lhs.is_negative(),
                _ => false, // TO DO add heuristic determination for pow
            },
        }
    }

    /// unary operations
    pub fn abs(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.abs()),
            SymbolExpr::Unary(e) => match e.op {
                UnaryOps::Abs | UnaryOps::Neg => e.expr.abs(),
                _ => SymbolExpr::Unary(Arc::new(Unary {
                    op: UnaryOps::Abs,
                    expr: self.clone(),
                })),
            },
            _ => SymbolExpr::Unary(Arc::new(Unary {
                op: UnaryOps::Abs,
                expr: self.clone(),
            })),
        }
    }
    pub fn sin(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.sin()),
            _ => SymbolExpr::Unary(Arc::new(Unary {
                op: UnaryOps::Sin,
                expr: self.clone(),
            })),
        }
    }
    pub fn asin(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.asin()),
            _ => SymbolExpr::Unary(Arc::new(Unary {
                op: UnaryOps::Asin,
                expr: self.clone(),
            })),
        }
    }
    pub fn cos(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.cos()),
            _ => SymbolExpr::Unary(Arc::new(Unary {
                op: UnaryOps::Cos,
                expr: self.clone(),
            })),
        }
    }
    pub fn acos(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.acos()),
            _ => SymbolExpr::Unary(Arc::new(Unary {
                op: UnaryOps::Acos,
                expr: self.clone(),
            })),
        }
    }
    pub fn tan(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.tan()),
            _ => SymbolExpr::Unary(Arc::new(Unary {
                op: UnaryOps::Tan,
                expr: self.clone(),
            })),
        }
    }
    pub fn atan(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.atan()),
            _ => SymbolExpr::Unary(Arc::new(Unary {
                op: UnaryOps::Atan,
                expr: self.clone(),
            })),
        }
    }
    pub fn exp(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.exp()),
            _ => SymbolExpr::Unary(Arc::new(Unary {
                op: UnaryOps::Exp,
                expr: self.clone(),
            })),
        }
    }
    pub fn log(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.log()),
            _ => SymbolExpr::Unary(Arc::new(Unary {
                op: UnaryOps::Log,
                expr: self.clone(),
            })),
        }
    }
    pub fn pow(&self, rhs: &SymbolExpr) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => match rhs {
                SymbolExpr::Value(r) => SymbolExpr::Value(l.pow(r)),
                _ => SymbolExpr::Binary(Arc::new(Binary {
                    op: BinaryOps::Pow,
                    lhs: SymbolExpr::Value(l.clone()),
                    rhs: rhs.clone(),
                })),
            },
            _ => SymbolExpr::Binary(Arc::new(Binary {
                op: BinaryOps::Pow,
                lhs: self.clone(),
                rhs: rhs.clone(),
            })),
        }
    }

    /// Add with heuristic optimization
    fn add_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        if self.is_zero() {
            Some(rhs.clone())
        } else if rhs.is_zero() {
            Some(self.clone())
        } else {
            // if neg operation, call sub_opt
            if let SymbolExpr::Unary(r) = rhs {
                if let UnaryOps::Neg = r.op {
                    return self.sub_opt(&r.expr);
                }
            }

            match self {
                SymbolExpr::Value(e) => e.add_opt(rhs),
                SymbolExpr::Symbol(e) => e.add_opt(rhs),
                SymbolExpr::Unary(e) => match e.add_opt(rhs) {
                    Some(opt) => Some(opt),
                    None => match rhs {
                        // swap nodes by sorting rule
                        SymbolExpr::Binary(r) => {
                            if let BinaryOps::Mul | BinaryOps::Div | BinaryOps::Pow = r.op {
                                if rhs < self {
                                    Some(_add(rhs.clone(), self.clone()))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        }
                        _ => {
                            if rhs < self {
                                Some(_add(rhs.clone(), self.clone()))
                            } else {
                                None
                            }
                        }
                    },
                },
                SymbolExpr::Binary(l) => match l.add_opt(rhs) {
                    Some(opt) => Some(opt),
                    None => {
                        if let BinaryOps::Mul | BinaryOps::Div | BinaryOps::Pow = l.op {
                            // swap nodes by sorting rule
                            match rhs {
                                SymbolExpr::Binary(r) => {
                                    if let BinaryOps::Mul | BinaryOps::Div | BinaryOps::Pow = r.op {
                                        if rhs < self {
                                            Some(_add(rhs.clone(), self.clone()))
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    }
                                }
                                _ => {
                                    if rhs < self {
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
                },
            }
        }
    }

    /// Sub with heuristic optimization
    fn sub_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        if self.is_zero() {
            match rhs.neg_opt() {
                Some(e) => Some(e),
                None => Some(_neg(rhs.clone())),
            }
        } else if rhs.is_zero() {
            Some(self.clone())
        } else {
            // if neg, call add_opt
            if let SymbolExpr::Unary(r) = rhs {
                if let UnaryOps::Neg = r.op {
                    return self.add_opt(&r.expr);
                }
            }

            match self {
                SymbolExpr::Value(e) => e.sub_opt(rhs),
                SymbolExpr::Symbol(e) => e.sub_opt(rhs),
                SymbolExpr::Unary(e) => match e.sub_opt(rhs) {
                    Some(opt) => Some(opt),
                    None => match rhs {
                        // swap nodes by sorting rule
                        SymbolExpr::Binary(r) => {
                            if let BinaryOps::Mul | BinaryOps::Div | BinaryOps::Pow = r.op {
                                if rhs < self {
                                    match rhs.neg_opt() {
                                        Some(e) => Some(_add(e, self.clone())),
                                        None => Some(_add(_neg(rhs.clone()), self.clone())),
                                    }
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        }
                        _ => {
                            if rhs < self {
                                match rhs.neg_opt() {
                                    Some(e) => Some(_add(e, self.clone())),
                                    None => Some(_add(_neg(rhs.clone()), self.clone())),
                                }
                            } else {
                                None
                            }
                        }
                    },
                },
                SymbolExpr::Binary(l) => match l.sub_opt(rhs) {
                    Some(opt) => Some(opt),
                    None => {
                        if let BinaryOps::Mul | BinaryOps::Div | BinaryOps::Pow = l.op {
                            // swap nodes by sorting rule
                            match rhs {
                                SymbolExpr::Binary(r) => {
                                    if let BinaryOps::Mul | BinaryOps::Div | BinaryOps::Pow = r.op {
                                        if rhs < self {
                                            match rhs.neg_opt() {
                                                Some(e) => Some(_add(e, self.clone())),
                                                None => Some(_add(_neg(rhs.clone()), self.clone())),
                                            }
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    }
                                }
                                _ => {
                                    if rhs < self {
                                        match rhs.neg_opt() {
                                            Some(e) => Some(_add(e, self.clone())),
                                            None => Some(_add(_neg(rhs.clone()), self.clone())),
                                        }
                                    } else {
                                        None
                                    }
                                }
                            }
                        } else {
                            None
                        }
                    }
                },
            }
        }
    }

    /// Mul with heuristic optimization
    fn mul_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        if self.is_zero() {
            Some(self.clone())
        } else if rhs.is_zero() || self.is_one() {
            Some(rhs.clone())
        } else if rhs.is_one() {
            Some(self.clone())
        } else if self.is_minus_one() {
            match rhs.neg_opt() {
                Some(e) => Some(e),
                None => Some(_neg(rhs.clone())),
            }
        } else if rhs.is_minus_one() {
            match self.neg_opt() {
                Some(e) => Some(e),
                None => Some(_neg(self.clone())),
            }
        } else {
            if let SymbolExpr::Value(_) | SymbolExpr::Symbol(_) = rhs {
                if let SymbolExpr::Unary(_) = self {
                    return match rhs.mul_opt(self) {
                        Some(e) => Some(e),
                        None => Some(_mul(rhs.clone(), self.clone())),
                    };
                }
            }

            match self {
                SymbolExpr::Unary(e) => e.mul_opt(rhs),
                SymbolExpr::Binary(e) => e.mul_opt(rhs),
                SymbolExpr::Value(e) => e.mul_opt(rhs),
                SymbolExpr::Symbol(e) => e.mul_opt(rhs),
            }
        }
    }
    /// expand with optimization for mul operation
    fn mul_expand(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        if let SymbolExpr::Binary(r) = rhs {
            if let BinaryOps::Add | BinaryOps::Sub = &r.op {
                let el = match self.mul_expand(&r.lhs) {
                    Some(e) => e,
                    None => match self.mul_opt(&r.lhs) {
                        Some(e) => e,
                        None => _mul(self.clone(), r.lhs.clone()),
                    },
                };
                let er = match self.mul_expand(&r.rhs) {
                    Some(e) => e,
                    None => match self.mul_opt(&r.rhs) {
                        Some(e) => e,
                        None => _mul(self.clone(), r.rhs.clone()),
                    },
                };
                return match &r.op {
                    BinaryOps::Sub => match el.sub_opt(&er) {
                        Some(e) => Some(e),
                        None => Some(_sub(el, er)),
                    },
                    _ => match el.add_opt(&er) {
                        Some(e) => Some(e),
                        None => Some(_add(el, er)),
                    },
                };
            }
            if let BinaryOps::Mul = &r.op {
                return match self.mul_expand(&r.lhs) {
                    Some(e) => match e.mul_expand(&r.rhs) {
                        Some(ee) => Some(ee),
                        None => Some(_mul(e, r.rhs.clone())),
                    },
                    None => self.mul_expand(&r.rhs).map(|e| _mul(e, r.lhs.clone())),
                };
            }
            if let BinaryOps::Div = &r.op {
                return match self.mul_expand(&r.lhs) {
                    Some(e) => match e.mul_expand(&r.rhs) {
                        Some(ee) => Some(ee),
                        None => Some(_mul(e, r.rhs.clone())),
                    },
                    None => self.div_expand(&r.rhs).map(|e| _div(e, r.lhs.clone())),
                };
            }
        }
        if let SymbolExpr::Unary(r) = rhs {
            if let UnaryOps::Neg = &r.op {
                return match self.mul_expand(&r.expr) {
                    Some(e) => match e.neg_opt() {
                        Some(ee) => Some(ee),
                        None => Some(_neg(e)),
                    },
                    None => match self.mul_opt(&r.expr) {
                        Some(e) => match e.neg_opt() {
                            Some(ee) => Some(ee),
                            None => Some(_neg(e)),
                        },
                        None => Some(_neg(_mul(self.clone(), r.expr.clone()))),
                    },
                };
            }
        }

        match self {
            SymbolExpr::Unary(l) => l.mul_expand(rhs),
            SymbolExpr::Binary(l) => l.mul_expand(rhs),
            _ => None,
        }
    }

    /// Div with heuristic optimization
    fn div_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        if self.is_zero() {
            Some(self.clone())
        } else if rhs.is_zero() {
            // return inf to detect divide by zero without panic
            Some(SymbolExpr::Value(Value::Real(f64::INFINITY)))
        } else if rhs.is_zero() {
            Some(self.clone())
        } else if rhs.is_minus_one() {
            match self.neg_opt() {
                Some(e) => Some(e),
                None => Some(_neg(self.clone())),
            }
        } else if *self == *rhs {
            let l_is_int = self.is_int().unwrap_or_default();
            let r_is_int = rhs.is_int().unwrap_or_default();
            if l_is_int || r_is_int {
                Some(SymbolExpr::Value(Value::Int(1)))
            } else {
                Some(SymbolExpr::Value(Value::Real(1.0)))
            }
        } else {
            match self {
                SymbolExpr::Unary(e) => e.div_opt(rhs),
                SymbolExpr::Binary(e) => e.div_opt(rhs),
                SymbolExpr::Value(e) => e.div_opt(rhs),
                SymbolExpr::Symbol(e) => e.div_opt(rhs),
            }
        }
    }

    /// expand with optimization for div operation
    fn div_expand(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match self {
            SymbolExpr::Unary(l) => l.div_expand(rhs),
            SymbolExpr::Binary(e) => e.div_expand(rhs),
            _ => self.div_opt(rhs),
        }
    }

    /// optimization for neg
    fn neg_opt(&self) -> Option<SymbolExpr> {
        match self {
            SymbolExpr::Value(v) => Some(SymbolExpr::Value(-v)),
            SymbolExpr::Unary(u) => match &u.op {
                UnaryOps::Neg => Some(u.expr.clone()),
                _ => None,
            },
            SymbolExpr::Binary(b) => match &b.op {
                BinaryOps::Add => match b.lhs.neg_opt() {
                    Some(ln) => match b.rhs.neg_opt() {
                        Some(rn) => Some(_add(ln, rn)),
                        None => Some(_sub(ln, b.rhs.clone())),
                    },
                    None => match b.rhs.neg_opt() {
                        Some(rn) => Some(_add(_neg(b.lhs.clone()), rn)),
                        None => Some(_sub(_neg(b.lhs.clone()), b.rhs.clone())),
                    },
                },
                BinaryOps::Sub => match b.lhs.neg_opt() {
                    Some(ln) => Some(_add(ln, b.rhs.clone())),
                    None => Some(_add(_neg(b.lhs.clone()), b.rhs.clone())),
                },
                BinaryOps::Mul => match b.lhs.neg_opt() {
                    Some(ln) => Some(_mul(ln, b.rhs.clone())),
                    None => b.rhs.neg_opt().map(|rn| _mul(b.lhs.clone(), rn)),
                },
                BinaryOps::Div => match b.lhs.neg_opt() {
                    Some(ln) => Some(_div(ln, b.rhs.clone())),
                    None => b.rhs.neg_opt().map(|rn| _div(b.lhs.clone(), rn)),
                },
                _ => None,
            },
            _ => None,
        }
    }
}

impl Add for SymbolExpr {
    type Output = SymbolExpr;
    fn add(self, rhs: Self) -> SymbolExpr {
        &self + &rhs
    }
}

impl Add for &SymbolExpr {
    type Output = SymbolExpr;
    fn add(self, rhs: Self) -> SymbolExpr {
        match self.add_opt(rhs) {
            Some(e) => e,
            None => _add(self.clone(), rhs.clone()),
        }
    }
}

impl Sub for SymbolExpr {
    type Output = SymbolExpr;
    fn sub(self, rhs: Self) -> SymbolExpr {
        &self - &rhs
    }
}

impl Sub for &SymbolExpr {
    type Output = SymbolExpr;
    fn sub(self, rhs: Self) -> SymbolExpr {
        match self.sub_opt(rhs) {
            Some(e) => e,
            None => _sub(self.clone(), rhs.clone()),
        }
    }
}

impl Mul for SymbolExpr {
    type Output = SymbolExpr;
    fn mul(self, rhs: Self) -> SymbolExpr {
        &self * &rhs
    }
}

impl Mul for &SymbolExpr {
    type Output = SymbolExpr;
    fn mul(self, rhs: Self) -> SymbolExpr {
        match self.mul_opt(rhs) {
            Some(e) => e,
            None => _mul(self.clone(), rhs.clone()),
        }
    }
}

impl Div for SymbolExpr {
    type Output = SymbolExpr;
    fn div(self, rhs: Self) -> SymbolExpr {
        &self / &rhs
    }
}

impl Div for &SymbolExpr {
    type Output = SymbolExpr;
    fn div(self, rhs: Self) -> SymbolExpr {
        match self.div_opt(rhs) {
            Some(e) => e,
            None => _div(self.clone(), rhs.clone()),
        }
    }
}

impl Neg for SymbolExpr {
    type Output = SymbolExpr;
    fn neg(self) -> SymbolExpr {
        -&self
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
        match (self, rexpr) {
            (SymbolExpr::Symbol(l), SymbolExpr::Symbol(r)) => l.name == r.name,
            (SymbolExpr::Value(l), SymbolExpr::Value(r)) => l == r,
            (
                SymbolExpr::Binary(_) | SymbolExpr::Unary(_),
                SymbolExpr::Binary(_) | SymbolExpr::Unary(_),
            ) => {
                let ex_lhs = self.expand();
                let ex_rhs = rexpr.expand();
                let t = &ex_lhs - &ex_rhs;
                match t {
                    SymbolExpr::Value(v) => v.is_zero(),
                    _ => false,
                }
            }
            (SymbolExpr::Binary(_), _) => {
                let ex_lhs = self.expand();

                let t = &ex_lhs - rexpr;
                match t {
                    SymbolExpr::Value(v) => v.is_zero(),
                    _ => false,
                }
            }
            (_, SymbolExpr::Binary(_)) => {
                let ex_rhs = rexpr.expand();
                let t = self - &ex_rhs;
                match t {
                    SymbolExpr::Value(v) => v.is_zero(),
                    _ => false,
                }
            }
            (_, _) => false,
        }
    }
}

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
        match self {
            SymbolExpr::Value(l) => match rhs {
                SymbolExpr::Value(r) => l.partial_cmp(r),
                _ => Some(Ordering::Less),
            },
            SymbolExpr::Symbol(l) => match rhs {
                SymbolExpr::Value(_) => Some(Ordering::Greater),
                SymbolExpr::Symbol(r) => l.name.partial_cmp(&r.name),
                SymbolExpr::Unary(r) => self.partial_cmp(&r.expr),
                _ => Some(Ordering::Less),
            },
            SymbolExpr::Unary(l) => match rhs {
                SymbolExpr::Value(_) => Some(Ordering::Greater),
                SymbolExpr::Unary(r) => l.expr.partial_cmp(&r.expr),
                _ => l.expr.partial_cmp(rhs),
            },
            SymbolExpr::Binary(l) => match rhs {
                SymbolExpr::Value(_) | SymbolExpr::Symbol(_) => match l.op {
                    BinaryOps::Mul | BinaryOps::Div | BinaryOps::Pow => Some(Ordering::Greater),
                    _ => Some(Ordering::Equal),
                },
                SymbolExpr::Unary(r) => self.partial_cmp(&r.expr),
                SymbolExpr::Binary(r) => {
                    let ls = match l.lhs {
                        SymbolExpr::Value(_) => l.rhs.to_string(),
                        _ => l.to_string(),
                    };
                    let rs = match r.lhs {
                        SymbolExpr::Value(_) => r.rhs.to_string(),
                        _ => r.to_string(),
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

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

// ===============================================================
//  implementations for Symbol
// ===============================================================
impl Symbol {
    pub fn new(expr: &str) -> Self {
        Self {
            name: expr.to_string(),
        }
    }

    pub fn bind(&self, maps: &HashMap<String, Value>) -> SymbolExpr {
        match maps.get(&self.name) {
            Some(v) => SymbolExpr::Value(v.clone()),
            None => SymbolExpr::Symbol(self.clone()),
        }
    }

    pub fn subs(&self, maps: &HashMap<String, SymbolExpr>) -> SymbolExpr {
        match maps.get(&self.name) {
            Some(v) => v.clone(),
            None => SymbolExpr::Symbol(self.clone()),
        }
    }

    fn add_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match rhs {
            SymbolExpr::Value(_) => Some(_add(rhs.clone(), SymbolExpr::Symbol(self.clone()))),
            SymbolExpr::Symbol(r) => {
                if r.name == self.name {
                    Some(_mul(
                        SymbolExpr::Value(Value::Int(2)),
                        SymbolExpr::Symbol(self.clone()),
                    ))
                } else if r.name < self.name {
                    Some(_add(rhs.clone(), SymbolExpr::Symbol(self.clone())))
                } else {
                    None
                }
            }
            SymbolExpr::Unary(_) => None,
            SymbolExpr::Binary(r) => match &r.op {
                BinaryOps::Add => match self.add_opt(&r.lhs) {
                    // self + r.lhs + r.rhs
                    Some(rl) => match rl.add_opt(&r.rhs) {
                        Some(rr) => Some(rr),
                        None => Some(_add(rl, r.rhs.clone())),
                    },
                    None => match self.add_opt(&r.rhs) {
                        Some(rr) => match rr.add_opt(&r.lhs) {
                            Some(rl) => Some(rl),
                            None => Some(_add(rr, r.lhs.clone())),
                        },
                        None => None,
                    },
                },
                BinaryOps::Sub => match self.add_opt(&r.lhs) {
                    // self + r.lhs - r.rhs
                    Some(rl) => match rl.sub_opt(&r.rhs) {
                        Some(rr) => Some(rr),
                        None => Some(_sub(rl, r.rhs.clone())),
                    },
                    None => match self.sub_opt(&r.rhs) {
                        Some(rr) => match rr.add_opt(&r.lhs) {
                            Some(rl) => Some(rl),
                            None => Some(_add(rr, r.lhs.clone())),
                        },
                        None => None,
                    },
                },
                _ => None,
            },
        }
    }

    fn sub_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match rhs {
            SymbolExpr::Value(r) => Some(_add(
                SymbolExpr::Value(-r),
                SymbolExpr::Symbol(self.clone()),
            )),
            SymbolExpr::Symbol(r) => {
                if r.name == self.name {
                    Some(SymbolExpr::Value(Value::Int(0)))
                } else if r.name < self.name {
                    Some(_add(_neg(rhs.clone()), SymbolExpr::Symbol(self.clone())))
                } else {
                    None
                }
            }
            SymbolExpr::Unary(_) => None,
            SymbolExpr::Binary(r) => match &r.op {
                BinaryOps::Add => match self.sub_opt(&r.lhs) {
                    // self - r.lhs - r.rhs
                    Some(rl) => match rl.sub_opt(&r.rhs) {
                        Some(rr) => Some(rr),
                        None => Some(_sub(rl, r.rhs.clone())),
                    },
                    None => match self.sub_opt(&r.rhs) {
                        Some(rr) => match rr.sub_opt(&r.lhs) {
                            Some(rl) => Some(rl),
                            None => Some(_sub(rr, r.lhs.clone())),
                        },
                        None => None,
                    },
                },
                BinaryOps::Sub => match self.sub_opt(&r.lhs) {
                    // self - r.lhs + r.rhs
                    Some(rl) => match rl.add_opt(&r.rhs) {
                        Some(rr) => Some(rr),
                        None => Some(_add(rl, r.rhs.clone())),
                    },
                    None => match self.add_opt(&r.rhs) {
                        Some(rr) => match rr.sub_opt(&r.lhs) {
                            Some(rl) => Some(rl),
                            None => Some(_sub(rr, r.lhs.clone())),
                        },
                        None => None,
                    },
                },
                _ => None,
            },
        }
    }

    fn mul_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match rhs {
            SymbolExpr::Value(_) => Some(_mul(rhs.clone(), SymbolExpr::Symbol(self.clone()))),
            SymbolExpr::Symbol(r) => {
                if r.name < self.name {
                    Some(_mul(rhs.clone(), SymbolExpr::Symbol(self.clone())))
                } else {
                    None
                }
            }
            SymbolExpr::Unary(r) => match &r.op {
                UnaryOps::Neg => match &r.expr {
                    SymbolExpr::Value(v) => Some(_mul(
                        SymbolExpr::Value(-v),
                        SymbolExpr::Symbol(self.clone()),
                    )),
                    SymbolExpr::Symbol(s) => {
                        if s.name < self.name {
                            Some(_neg(_mul(r.expr.clone(), SymbolExpr::Symbol(self.clone()))))
                        } else {
                            Some(_neg(_mul(SymbolExpr::Symbol(self.clone()), r.expr.clone())))
                        }
                    }
                    SymbolExpr::Binary(_) => match self.mul_opt(&r.expr) {
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
            _ => None,
        }
    }

    fn div_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match rhs {
            SymbolExpr::Value(r) => Some(_mul(
                SymbolExpr::Value(r.rcp()),
                SymbolExpr::Symbol(self.clone()),
            )),
            _ => None,
        }
    }
}

impl From<&str> for Symbol {
    fn from(v: &str) -> Self {
        Self::new(v)
    }
}

impl PartialEq for Symbol {
    fn eq(&self, r: &Self) -> bool {
        self.name == r.name
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
                    if e.re < f64::EPSILON && e.re > -f64::EPSILON {
                        if e.im < f64::EPSILON && e.im > -f64::EPSILON {
                            0.to_string()
                        } else {
                            format!("{}i", e.im)
                        }
                    } else if e.im < f64::EPSILON && e.im > -f64::EPSILON {
                        e.re.to_string()
                    } else {
                        e.to_string()
                    }
                }
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
        }
    }

    pub fn abs(&self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.abs()),
            Value::Int(e) => Value::Int(e.abs()),
            Value::Complex(e) => Value::Real((e.re * e.re + e.im * e.im).sqrt()),
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
        }
    }
    pub fn log(&self) -> Value {
        match self {
            Value::Real(e) => Value::Real(e.ln()),
            Value::Int(e) => Value::Real((*e as f64).ln()),
            Value::Complex(e) => {
                let t = Value::Complex(e.ln());
                match t.opt_complex() {
                    Some(v) => v,
                    None => t,
                }
            }
        }
    }
    pub fn sqrt(&self) -> Value {
        match self {
            Value::Real(e) => {
                if *e < 0.0 {
                    Value::Complex(Complex64::from(e).powf(0.5))
                } else {
                    Value::Real(e.sqrt())
                }
            }
            Value::Int(e) => {
                if *e < 0 {
                    Value::Complex(Complex64::from(*e as f64).powf(0.5))
                } else {
                    let t = (*e as f64).sqrt();
                    let d = t.floor() - t;
                    if (-f64::EPSILON..f64::EPSILON).contains(&d) {
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
        }
    }
    pub fn pow(&self, p: &Value) -> Value {
        match self {
            Value::Real(e) => match p {
                Value::Real(r) => {
                    if *e < 0.0 {
                        Value::Complex(Complex64::from(e).powf(*r))
                    } else {
                        Value::Real(e.powf(*r))
                    }
                }
                Value::Int(i) => {
                    if *e < 0.0 {
                        Value::Complex(Complex64::from(e).powf(*i as f64))
                    } else {
                        Value::Real(e.powf(*i as f64))
                    }
                }
                Value::Complex(r) => Value::Complex(Complex64::from(e).powc(*r)),
            },
            Value::Int(e) => {
                if *e < 0 {
                    match p {
                        Value::Real(r) => Value::Complex(Complex64::from(*e as f64).powf(*r)),
                        Value::Int(i) => Value::Complex(Complex64::from(*e as f64).powf(*i as f64)),
                        Value::Complex(c) => Value::Complex(Complex64::from(*e as f64).powc(*c)),
                    }
                } else {
                    match p {
                        Value::Real(r) => {
                            let t = (*e as f64).powf(*r);
                            let d = t.floor() - t;
                            if (-f64::EPSILON..f64::EPSILON).contains(&d) {
                                Value::Int(t as i64)
                            } else {
                                Value::Real(t)
                            }
                        }
                        Value::Int(r) => {
                            if *r < 0 {
                                Value::Real((*e as f64).powf(*r as f64))
                            } else {
                                Value::Int(e.pow(*r as u32))
                            }
                        }
                        Value::Complex(r) => Value::Complex(Complex64::from(*e as f64).powc(*r)),
                    }
                }
            }
            Value::Complex(e) => {
                let t = match p {
                    Value::Real(r) => Value::Complex(e.powf(*r)),
                    Value::Int(r) => Value::Complex(e.powf(*r as f64)),
                    Value::Complex(r) => Value::Complex(e.powc(*r)),
                };
                match t.opt_complex() {
                    Some(v) => v,
                    None => t,
                }
            }
        }
    }
    pub fn rcp(&self) -> Value {
        match self {
            Value::Real(e) => Value::Real(1.0 / e),
            Value::Int(e) => {
                let t = 1.0 / (*e as f64);
                let d = t.floor() - t;
                if (-f64::EPSILON..f64::EPSILON).contains(&d) {
                    Value::Int(t as i64)
                } else {
                    Value::Real(t)
                }
            }
            Value::Complex(e) => Value::Complex(1.0 / e),
        }
    }
    pub fn sign(&self) -> Value {
        match self {
            Value::Real(e) => {
                if *e > f64::EPSILON {
                    Value::Real(1.0)
                } else if *e < -f64::EPSILON {
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
            Value::Complex(_) => self.clone(),
        }
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Value::Real(r) => *r < f64::EPSILON && *r > -f64::EPSILON,
            Value::Int(i) => *i == 0,
            Value::Complex(c) => {
                c.re < f64::EPSILON
                    && c.re > -f64::EPSILON
                    && c.im < f64::EPSILON
                    && c.im > -f64::EPSILON
            }
        }
    }
    pub fn is_one(&self) -> bool {
        match self {
            Value::Real(r) => *r - 1.0 < f64::EPSILON && *r - 1.0 > -f64::EPSILON,
            Value::Int(i) => *i == 1,
            Value::Complex(c) => {
                c.re - 1.0 < f64::EPSILON
                    && c.re - 1.0 > -f64::EPSILON
                    && c.im < f64::EPSILON
                    && c.im > -f64::EPSILON
            }
        }
    }
    pub fn is_minus_one(&self) -> bool {
        match self {
            Value::Real(r) => *r + 1.0 < f64::EPSILON && *r + 1.0 > -f64::EPSILON,
            Value::Int(i) => *i == -1,
            Value::Complex(c) => {
                c.re + 1.0 < f64::EPSILON
                    && c.re + 1.0 > -f64::EPSILON
                    && c.im < f64::EPSILON
                    && c.im > -f64::EPSILON
            }
        }
    }

    pub fn is_negative(&self) -> bool {
        match self {
            Value::Real(r) => *r < 0.0,
            Value::Int(i) => *i < 0,
            Value::Complex(c) => {
                (c.re < 0.0 && c.im < f64::EPSILON && c.im > -f64::EPSILON)
                    || (c.im < 0.0 && c.re < f64::EPSILON && c.re > -f64::EPSILON)
            }
        }
    }

    fn add_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match rhs {
            SymbolExpr::Value(r) => Some(SymbolExpr::Value(self + r)),
            SymbolExpr::Unary(_) => None,
            SymbolExpr::Binary(r) => match &r.op {
                BinaryOps::Add => match self.add_opt(&r.lhs) {
                    // self + r.lhs + r.rhs
                    Some(rl) => match rl.add_opt(&r.rhs) {
                        Some(rr) => Some(rr),
                        None => Some(_add(rl, r.rhs.clone())),
                    },
                    None => match self.add_opt(&r.rhs) {
                        Some(rr) => match rr.add_opt(&r.lhs) {
                            Some(rl) => Some(rl),
                            None => Some(_add(rr, r.lhs.clone())),
                        },
                        None => None,
                    },
                },
                BinaryOps::Sub => match self.add_opt(&r.lhs) {
                    // self + r.lhs - r.rhs
                    Some(rl) => match rl.sub_opt(&r.rhs) {
                        Some(rr) => Some(rr),
                        None => Some(_sub(rl, r.rhs.clone())),
                    },
                    None => match self.sub_opt(&r.rhs) {
                        Some(rr) => match rr.add_opt(&r.lhs) {
                            Some(rl) => Some(rl),
                            None => Some(_add(rr, r.lhs.clone())),
                        },
                        None => None,
                    },
                },
                _ => None,
            },
            _ => None,
        }
    }

    fn sub_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match rhs {
            SymbolExpr::Value(r) => Some(SymbolExpr::Value(self - r)),
            SymbolExpr::Unary(_) => None,
            SymbolExpr::Binary(r) => match &r.op {
                BinaryOps::Add => match self.sub_opt(&r.lhs) {
                    // self - r.lhs - r.rhs
                    Some(rl) => match rl.sub_opt(&r.rhs) {
                        Some(rr) => Some(rr),
                        None => Some(_sub(rl, r.rhs.clone())),
                    },
                    None => match self.sub_opt(&r.rhs) {
                        Some(rr) => match rr.sub_opt(&r.lhs) {
                            Some(rl) => Some(rl),
                            None => Some(_sub(rr, r.lhs.clone())),
                        },
                        None => None,
                    },
                },
                BinaryOps::Sub => match self.sub_opt(&r.lhs) {
                    // self - r.lhs + r.rhs
                    Some(rl) => match rl.add_opt(&r.rhs) {
                        Some(rr) => Some(rr),
                        None => Some(_add(rl, r.rhs.clone())),
                    },
                    None => match self.add_opt(&r.rhs) {
                        Some(rr) => match rr.sub_opt(&r.lhs) {
                            Some(rl) => Some(rl),
                            None => Some(_sub(rr, r.lhs.clone())),
                        },
                        None => None,
                    },
                },
                _ => None,
            },
            _ => None,
        }
    }

    fn mul_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match rhs {
            SymbolExpr::Value(r) => Some(SymbolExpr::Value(self * r)),
            SymbolExpr::Unary(r) => match &r.op {
                UnaryOps::Neg => {
                    let l = SymbolExpr::Value(-self);
                    match l.mul_opt(&r.expr) {
                        Some(e) => Some(e),
                        None => Some(_mul(l, r.expr.clone())),
                    }
                }
                _ => None,
            },
            SymbolExpr::Binary(r) => match &r.op {
                BinaryOps::Mul => match self.mul_opt(&r.lhs) {
                    Some(e) => match e.mul_opt(&r.rhs) {
                        Some(ee) => Some(ee),
                        None => Some(_mul(e, r.rhs.clone())),
                    },
                    None => self.mul_opt(&r.rhs).map(|e| _mul(e, r.lhs.clone())),
                },
                BinaryOps::Div => match self.mul_opt(&r.lhs) {
                    Some(e) => Some(_div(e, r.rhs.clone())),
                    None => self.div_opt(&r.rhs).map(|e| _mul(e, r.lhs.clone())),
                },
                _ => None,
            },
            _ => None,
        }
    }

    fn div_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match rhs {
            SymbolExpr::Value(r) => Some(SymbolExpr::Value(self / r)),
            SymbolExpr::Unary(r) => match &r.op {
                UnaryOps::Neg => self.div_opt(&r.expr).map(_neg),
                _ => None,
            },
            SymbolExpr::Binary(r) => match &r.lhs {
                SymbolExpr::Value(v) => match &r.op {
                    BinaryOps::Mul => Some(_div(SymbolExpr::Value(self / v), r.rhs.clone())),
                    BinaryOps::Div => Some(_mul(SymbolExpr::Value(self / v), r.rhs.clone())),
                    _ => None,
                },
                _ => match &r.rhs {
                    SymbolExpr::Value(v) => match &r.op {
                        BinaryOps::Mul => Some(_div(SymbolExpr::Value(self / v), r.lhs.clone())),
                        BinaryOps::Div => Some(_div(SymbolExpr::Value(self * v), r.lhs.clone())),
                        _ => None,
                    },
                    _ => None,
                },
            },
            _ => None,
        }
    }

    fn opt_complex(&self) -> Option<Value> {
        match self {
            Value::Complex(c) => {
                if c.im < f64::EPSILON && c.im > -f64::EPSILON {
                    Some(Value::Real(c.re))
                } else {
                    None
                }
            }
            _ => None,
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
        if v.im < f64::EPSILON && v.im > -f64::EPSILON {
            Value::Real(v.re)
        } else {
            Value::Complex(v)
        }
    }
}

impl Add for Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Value {
        &self + &rhs
    }
}

impl Add for &Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Value {
        let t = match self {
            Value::Real(l) => match rhs {
                Value::Real(r) => Value::Real(l + r),
                Value::Int(r) => Value::Real(l + *r as f64),
                Value::Complex(r) => Value::Complex(l + r),
            },
            Value::Int(l) => match rhs {
                Value::Real(r) => Value::Real(*l as f64 + r),
                Value::Int(r) => Value::Int(l + r),
                Value::Complex(r) => Value::Complex(*l as f64 + r),
            },
            Value::Complex(l) => match rhs {
                Value::Real(r) => Value::Complex(l + r),
                Value::Int(r) => Value::Complex(l + *r as f64),
                Value::Complex(r) => Value::Complex(l + r),
            },
        };
        match t.opt_complex() {
            Some(v) => v,
            None => t,
        }
    }
}

impl Sub for Value {
    type Output = Value;
    fn sub(self, rhs: Self) -> Value {
        &self - &rhs
    }
}

impl Sub for &Value {
    type Output = Value;
    fn sub(self, rhs: Self) -> Value {
        let t = match self {
            Value::Real(l) => match rhs {
                Value::Real(r) => Value::Real(l - r),
                Value::Int(r) => Value::Real(l - *r as f64),
                Value::Complex(r) => Value::Complex(l - r),
            },
            Value::Int(l) => match rhs {
                Value::Real(r) => Value::Real(*l as f64 - r),
                Value::Int(r) => Value::Int(l - r),
                Value::Complex(r) => Value::Complex(*l as f64 - r),
            },
            Value::Complex(l) => match rhs {
                Value::Real(r) => Value::Complex(l - r),
                Value::Int(r) => Value::Complex(l - *r as f64),
                Value::Complex(r) => Value::Complex(l - r),
            },
        };
        match t.opt_complex() {
            Some(v) => v,
            None => t,
        }
    }
}

impl Mul for Value {
    type Output = Value;
    fn mul(self, rhs: Self) -> Value {
        &self * &rhs
    }
}

impl Mul for &Value {
    type Output = Value;
    fn mul(self, rhs: Self) -> Value {
        let t = match self {
            Value::Real(l) => match rhs {
                Value::Real(r) => Value::Real(l * r),
                Value::Int(r) => Value::Real(l * *r as f64),
                Value::Complex(r) => Value::Complex(l * r),
            },
            Value::Int(l) => match rhs {
                Value::Real(r) => Value::Real(*l as f64 * r),
                Value::Int(r) => Value::Int(l * r),
                Value::Complex(r) => Value::Complex(*l as f64 * r),
            },
            Value::Complex(l) => match rhs {
                Value::Real(r) => Value::Complex(l * r),
                Value::Int(r) => Value::Complex(l * *r as f64),
                Value::Complex(r) => Value::Complex(l * r),
            },
        };
        match t.opt_complex() {
            Some(v) => v,
            None => t,
        }
    }
}

impl Div for Value {
    type Output = Value;
    fn div(self, rhs: Self) -> Value {
        &self / &rhs
    }
}

impl Div for &Value {
    type Output = Value;
    fn div(self, rhs: Self) -> Value {
        let t = match self {
            Value::Real(l) => match rhs {
                Value::Real(r) => Value::Real(l / r),
                Value::Int(r) => Value::Real(l / *r as f64),
                Value::Complex(r) => Value::Complex(l / r),
            },
            Value::Int(l) => {
                if *rhs == 0.0 {
                    return Value::Real(f64::INFINITY);
                }
                match rhs {
                    Value::Real(r) => Value::Real(*l as f64 / r),
                    Value::Int(r) => {
                        let t = *l as f64 / *r as f64;
                        let d = t.floor() - t;
                        if (-f64::EPSILON..f64::EPSILON).contains(&d) {
                            Value::Int(t as i64)
                        } else {
                            Value::Real(t)
                        }
                    }
                    Value::Complex(r) => Value::Complex(*l as f64 / r),
                }
            }
            Value::Complex(l) => match rhs {
                Value::Real(r) => Value::Complex(l / r),
                Value::Int(r) => Value::Complex(l / *r as f64),
                Value::Complex(r) => Value::Complex(l / r),
            },
        };
        match t.opt_complex() {
            Some(v) => v,
            None => t,
        }
    }
}

impl Neg for Value {
    type Output = Value;
    fn neg(self) -> Value {
        -&self
    }
}

impl Neg for &Value {
    type Output = Value;
    fn neg(self) -> Value {
        match self {
            Value::Real(v) => Value::Real(-v),
            Value::Int(v) => Value::Int(-v),
            Value::Complex(v) => Value::Complex(-v),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, r: &Self) -> bool {
        match self {
            Value::Real(e) => match r {
                Value::Real(rv) => (e - rv).abs() < f64::EPSILON,
                Value::Int(rv) => (e - *rv as f64).abs() < f64::EPSILON,
                Value::Complex(rv) => {
                    let t = Complex64::from(*e) - rv;
                    t.re < f64::EPSILON
                        && t.re > -f64::EPSILON
                        && t.im < f64::EPSILON
                        && t.im > -f64::EPSILON
                }
            },
            Value::Int(e) => match r {
                Value::Int(rv) => e == rv,
                Value::Real(rv) => (*e as f64 - rv).abs() < f64::EPSILON,
                Value::Complex(rv) => {
                    let t = Complex64::from(*e as f64) - rv;
                    t.re < f64::EPSILON
                        && t.re > -f64::EPSILON
                        && t.im < f64::EPSILON
                        && t.im > -f64::EPSILON
                }
            },
            Value::Complex(e) => match r {
                Value::Real(rv) => {
                    let t = *e - Complex64::from(rv);
                    t.re < f64::EPSILON
                        && t.re > -f64::EPSILON
                        && t.im < f64::EPSILON
                        && t.im > -f64::EPSILON
                }
                Value::Int(rv) => {
                    let t = *e - Complex64::from(*rv as f64);
                    t.re < f64::EPSILON
                        && t.re > -f64::EPSILON
                        && t.im < f64::EPSILON
                        && t.im > -f64::EPSILON
                }
                Value::Complex(rv) => {
                    let t = *e - rv;
                    t.re < f64::EPSILON
                        && t.re > -f64::EPSILON
                        && t.im < f64::EPSILON
                        && t.im > -f64::EPSILON
                }
            },
        }
    }
}

impl PartialEq<f64> for Value {
    fn eq(&self, r: &f64) -> bool {
        match self {
            Value::Real(e) => (e - r).abs() < f64::EPSILON,
            Value::Int(e) => (*e as f64 - r).abs() < f64::EPSILON,
            Value::Complex(e) => {
                let t = *e - Complex64::from(r);
                t.re < f64::EPSILON
                    && t.re > -f64::EPSILON
                    && t.im < f64::EPSILON
                    && t.im > -f64::EPSILON
            }
        }
    }
}

impl PartialEq<Complex64> for Value {
    fn eq(&self, r: &Complex64) -> bool {
        match self {
            Value::Real(e) => {
                let t = Complex64::from(*e) - r;
                t.re < f64::EPSILON
                    && t.re > -f64::EPSILON
                    && t.im < f64::EPSILON
                    && t.im > -f64::EPSILON
            }
            Value::Int(e) => {
                let t = Complex64::from(*e as f64) - r;
                t.re < f64::EPSILON
                    && t.re > -f64::EPSILON
                    && t.im < f64::EPSILON
                    && t.im > -f64::EPSILON
            }
            Value::Complex(e) => {
                let t = *e - r;
                t.re < f64::EPSILON
                    && t.re > -f64::EPSILON
                    && t.im < f64::EPSILON
                    && t.im > -f64::EPSILON
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
            },
            Value::Int(l) => match rhs {
                Value::Real(r) => (*l as f64).partial_cmp(r),
                Value::Int(r) => l.partial_cmp(r),
                Value::Complex(_) => None,
            },
            Value::Complex(_) => None,
        }
    }
}

impl fmt::Display for Unary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.expr.to_string();
        write!(
            f,
            "{}",
            match self.op {
                UnaryOps::Abs => format!("abs({})", s),
                UnaryOps::Neg => match &self.expr {
                    SymbolExpr::Value(e) => (-e).to_string(),
                    SymbolExpr::Binary(e) => match e.op {
                        BinaryOps::Add | BinaryOps::Sub => format!("-({})", s),
                        _ => format!("-{}", s),
                    },
                    _ => format!("-{}", s),
                },
                UnaryOps::Sin => format!("sin({})", s),
                UnaryOps::Asin => format!("asin({})", s),
                UnaryOps::Cos => format!("cos({})", s),
                UnaryOps::Acos => format!("acos({})", s),
                UnaryOps::Tan => format!("tan({})", s),
                UnaryOps::Atan => format!("atan({})", s),
                UnaryOps::Exp => format!("exp({})", s),
                UnaryOps::Log => format!("log({})", s),
                UnaryOps::Sign => format!("sign({})", s),
            }
        )
    }
}

// ===============================================================
//  implementations for Unary operators
// ===============================================================
impl Unary {
    pub fn new(op: UnaryOps, expr: SymbolExpr) -> Self {
        Self { op, expr }
    }

    pub fn bind(&self, maps: &HashMap<String, Value>) -> SymbolExpr {
        let new_expr = Unary {
            op: self.op.clone(),
            expr: self.expr.bind(maps),
        };
        match new_expr.clone().eval(false) {
            Some(v) => SymbolExpr::Value(v.clone()),
            None => SymbolExpr::Unary(Arc::new(new_expr)),
        }
    }

    pub fn subs(&self, maps: &HashMap<String, SymbolExpr>) -> SymbolExpr {
        let new_expr = Unary {
            op: self.op.clone(),
            expr: self.expr.subs(maps),
        };
        match new_expr.clone().eval(false) {
            Some(v) => SymbolExpr::Value(v.clone()),
            None => SymbolExpr::Unary(Arc::new(new_expr)),
        }
    }

    pub fn derivative(&self, param: &SymbolExpr) -> SymbolExpr {
        let expr_d = self.expr.derivative(param);
        match self.op {
            UnaryOps::Abs => {
                self.expr.clone() * expr_d
                    / SymbolExpr::Unary(Arc::new(Unary {
                        op: self.op.clone(),
                        expr: self.expr.clone(),
                    }))
            }
            UnaryOps::Neg => SymbolExpr::Unary(Arc::new(Unary {
                op: UnaryOps::Neg,
                expr: expr_d,
            })),
            UnaryOps::Sin => {
                let lhs = SymbolExpr::Unary(Arc::new(Unary {
                    op: UnaryOps::Cos,
                    expr: self.expr.clone(),
                }));
                lhs * expr_d
            }
            UnaryOps::Asin => {
                let d = SymbolExpr::Value(Value::Real(1.0)) - self.expr.clone() * self.expr.clone();
                let lhs = match d {
                    SymbolExpr::Value(v) => SymbolExpr::Value(v.sqrt()),
                    _ => SymbolExpr::Binary(Arc::new(Binary {
                        op: BinaryOps::Pow,
                        lhs: d,
                        rhs: SymbolExpr::Value(Value::Real(0.5)),
                    })),
                };
                lhs * expr_d
            }
            UnaryOps::Cos => {
                let lhs = SymbolExpr::Unary(Arc::new(Unary {
                    op: UnaryOps::Sin,
                    expr: self.expr.clone(),
                }));
                -lhs * expr_d
            }
            UnaryOps::Acos => {
                let d = SymbolExpr::Value(Value::Real(1.0)) - self.expr.clone() * self.expr.clone();
                let lhs = match d {
                    SymbolExpr::Value(v) => SymbolExpr::Value(v.sqrt()),
                    _ => SymbolExpr::Binary(Arc::new(Binary {
                        op: BinaryOps::Pow,
                        lhs: d,
                        rhs: SymbolExpr::Value(Value::Real(0.5)),
                    })),
                };
                -lhs * expr_d
            }
            UnaryOps::Tan => {
                let d = SymbolExpr::Unary(Arc::new(Unary {
                    op: UnaryOps::Cos,
                    expr: self.expr.clone(),
                }));
                expr_d / d.clone() / d
            }
            UnaryOps::Atan => {
                let d = SymbolExpr::Value(Value::Real(1.0)) + self.expr.clone() * self.expr.clone();
                expr_d / d
            }
            UnaryOps::Exp => {
                SymbolExpr::Unary(Arc::new(Unary {
                    op: UnaryOps::Exp,
                    expr: self.expr.clone(),
                })) * expr_d
            }
            UnaryOps::Log => expr_d / self.expr.clone(),
            UnaryOps::Sign => SymbolExpr::Unary(Arc::new(Unary {
                op: UnaryOps::Sign,
                expr: expr_d,
            })),
        }
    }

    pub fn eval(&self, recurse: bool) -> Option<Value> {
        let val: Value;
        if recurse {
            match self.expr.eval(recurse) {
                Some(v) => val = v,
                None => return None,
            }
        } else {
            match &self.expr {
                SymbolExpr::Value(e) => val = e.clone(),
                _ => return None,
            }
        }
        let ret = match self.op {
            UnaryOps::Abs => val.abs(),
            UnaryOps::Neg => -val,
            UnaryOps::Sin => val.sin(),
            UnaryOps::Asin => val.asin(),
            UnaryOps::Cos => val.cos(),
            UnaryOps::Acos => val.acos(),
            UnaryOps::Tan => val.tan(),
            UnaryOps::Atan => val.atan(),
            UnaryOps::Exp => val.exp(),
            UnaryOps::Log => val.log(),
            UnaryOps::Sign => val.sign(),
        };
        match ret {
            Value::Real(_) => Some(ret),
            Value::Int(_) => Some(ret),
            Value::Complex(c) => {
                if c.im < f64::EPSILON && c.im > -f64::EPSILON {
                    Some(Value::Real(c.re))
                } else {
                    Some(ret)
                }
            }
        }
    }

    pub fn expand(&self) -> SymbolExpr {
        let ex = self.expr.expand();
        match self.op {
            UnaryOps::Neg => match ex.neg_opt() {
                Some(ne) => ne,
                None => _neg(ex),
            },
            _ => SymbolExpr::Unary(Arc::new(Unary {
                op: self.op.clone(),
                expr: ex,
            })),
        }
    }

    pub fn symbols(&self) -> HashSet<String> {
        self.expr.symbols()
    }

    pub fn has_symbol(&self, param: &String) -> bool {
        self.expr.has_symbol(param)
    }

    // Add with heuristic optimization
    fn add_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        if let SymbolExpr::Binary(r) = rhs {
            if let BinaryOps::Add = r.op {
                return match self.add_opt(&r.lhs) {
                    // self + r.lhs + r.rhs
                    Some(rl) => match rl.add_opt(&r.rhs) {
                        Some(rr) => Some(rr),
                        None => Some(_add(rl, r.rhs.clone())),
                    },
                    None => match self.add_opt(&r.rhs) {
                        Some(rr) => match rr.add_opt(&r.lhs) {
                            Some(rl) => Some(rl),
                            None => Some(_add(rr, r.lhs.clone())),
                        },
                        None => None,
                    },
                };
            } else if let BinaryOps::Sub = r.op {
                return match self.add_opt(&r.lhs) {
                    // self + r.lhs - r.rhs
                    Some(rl) => match rl.sub_opt(&r.rhs) {
                        Some(rr) => Some(rr),
                        None => Some(_sub(rl, r.rhs.clone())),
                    },
                    None => match self.sub_opt(&r.rhs) {
                        Some(rr) => match rr.add_opt(&r.lhs) {
                            Some(rl) => Some(rl),
                            None => Some(_add(rr, r.lhs.clone())),
                        },
                        None => None,
                    },
                };
            }
        }

        if let UnaryOps::Neg = self.op {
            if let Some(e) = self.expr.sub_opt(rhs) {
                return match e.neg_opt() {
                    Some(ee) => Some(ee),
                    None => Some(_neg(e)),
                };
            }
        }
        match rhs {
            SymbolExpr::Unary(r) => {
                if self.op == r.op {
                    let t = self.expr.expand() + r.expr.expand();
                    if t.is_zero() {
                        Some(SymbolExpr::Value(Value::Int(0)))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }
    // Sub with heuristic optimization
    fn sub_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        if let SymbolExpr::Binary(r) = rhs {
            if let BinaryOps::Add = r.op {
                return match self.sub_opt(&r.lhs) {
                    // self - r.lhs - r.rhs
                    Some(rl) => match rl.sub_opt(&r.rhs) {
                        Some(rr) => Some(rr),
                        None => Some(_sub(rl, r.rhs.clone())),
                    },
                    None => match self.sub_opt(&r.rhs) {
                        Some(rr) => match rr.sub_opt(&r.lhs) {
                            Some(rl) => Some(rl),
                            None => Some(_sub(rr, r.lhs.clone())),
                        },
                        None => None,
                    },
                };
            } else if let BinaryOps::Sub = r.op {
                return match self.sub_opt(&r.lhs) {
                    // self - r.lhs + r.rhs
                    Some(rl) => match rl.add_opt(&r.rhs) {
                        Some(rr) => Some(rr),
                        None => Some(_add(rl, r.rhs.clone())),
                    },
                    None => match self.add_opt(&r.rhs) {
                        Some(rr) => match rr.sub_opt(&r.lhs) {
                            Some(rl) => Some(rl),
                            None => Some(_sub(rr, r.lhs.clone())),
                        },
                        None => None,
                    },
                };
            }
        }

        if let UnaryOps::Neg = self.op {
            if let Some(e) = self.expr.add_opt(rhs) {
                return match e.neg_opt() {
                    Some(ee) => Some(ee),
                    None => Some(_neg(e)),
                };
            }
        }
        match rhs {
            SymbolExpr::Unary(r) => {
                if self.op == r.op {
                    let t = self.expr.expand() - r.expr.expand();
                    if t.is_zero() {
                        Some(SymbolExpr::Value(Value::Int(0)))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    // mul with heuristic optimization
    fn mul_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match self.op {
            UnaryOps::Neg => match self.expr.mul_opt(rhs) {
                Some(e) => match e.neg_opt() {
                    Some(ee) => Some(ee),
                    None => Some(_neg(e)),
                },
                None => None,
            },
            UnaryOps::Abs => match rhs {
                SymbolExpr::Unary(r) => match &r.op {
                    UnaryOps::Abs => match self.expr.mul_opt(&r.expr) {
                        Some(e) => Some(SymbolExpr::Unary(Arc::new(Unary {
                            op: UnaryOps::Abs,
                            expr: e,
                        }))),
                        None => Some(SymbolExpr::Unary(Arc::new(Unary {
                            op: UnaryOps::Abs,
                            expr: _mul(self.expr.clone(), r.expr.clone()),
                        }))),
                    },
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        }
    }
    fn mul_expand(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match self.op {
            UnaryOps::Neg => match self.expr.mul_expand(rhs) {
                Some(e) => match e.neg_opt() {
                    Some(ee) => Some(ee),
                    None => Some(_neg(e)),
                },
                None => match self.expr.mul_opt(rhs) {
                    Some(e) => match e.neg_opt() {
                        Some(ee) => Some(ee),
                        None => Some(_neg(e)),
                    },
                    None => None,
                },
            },
            _ => None,
        }
    }

    // div with heuristic optimization
    fn div_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match self.op {
            UnaryOps::Neg => match self.expr.div_opt(rhs) {
                Some(e) => match e.neg_opt() {
                    Some(ee) => Some(ee),
                    None => Some(_neg(e)),
                },
                None => None,
            },
            UnaryOps::Abs => match rhs {
                SymbolExpr::Unary(r) => match &r.op {
                    UnaryOps::Abs => match self.expr.div_opt(&r.expr) {
                        Some(e) => Some(SymbolExpr::Unary(Arc::new(Unary {
                            op: UnaryOps::Abs,
                            expr: e,
                        }))),
                        None => Some(SymbolExpr::Unary(Arc::new(Unary {
                            op: UnaryOps::Abs,
                            expr: _div(self.expr.clone(), r.expr.clone()),
                        }))),
                    },
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        }
    }
    fn div_expand(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match self.op {
            UnaryOps::Neg => match self.expr.div_expand(rhs) {
                Some(e) => match e.neg_opt() {
                    Some(ee) => Some(ee),
                    None => Some(_neg(e)),
                },
                None => match self.expr.div_opt(rhs) {
                    Some(e) => match e.neg_opt() {
                        Some(ee) => Some(ee),
                        None => Some(_neg(e)),
                    },
                    None => None,
                },
            },
            _ => None,
        }
    }
}

impl PartialEq for Unary {
    fn eq(&self, r: &Self) -> bool {
        self.op == r.op && self.expr == r.expr
    }
}

impl fmt::Display for Binary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s_lhs = self.lhs.to_string();
        let s_rhs = self.rhs.to_string();
        let op_lhs = match &self.lhs {
            SymbolExpr::Binary(e) => matches!(e.op, BinaryOps::Add | BinaryOps::Sub),
            SymbolExpr::Value(e) => match e {
                Value::Real(v) => *v < 0.0,
                Value::Int(v) => *v < 0,
                Value::Complex(_) => true,
            },
            _ => false,
        };
        let op_rhs = match &self.rhs {
            SymbolExpr::Binary(e) => match e.op {
                BinaryOps::Add | BinaryOps::Sub => true,
                _ => matches!(self.op, BinaryOps::Div),
            },
            SymbolExpr::Value(e) => match e {
                Value::Real(v) => *v < 0.0,
                Value::Int(v) => *v < 0,
                Value::Complex(_) => true,
            },
            _ => false,
        };

        write!(
            f,
            "{}",
            match self.op {
                BinaryOps::Add => match &self.rhs {
                    SymbolExpr::Unary(r) => match r.op {
                        UnaryOps::Neg => {
                            if s_rhs.as_str().char_indices().nth(0).unwrap().1 == '-' {
                                format!("{} {}", s_lhs, s_rhs)
                            } else {
                                format!("{} + {}", s_lhs, s_rhs)
                            }
                        }
                        _ => format!("{} + {}", s_lhs, s_rhs),
                    },
                    _ => format!("{} + {}", s_lhs, s_rhs),
                },
                BinaryOps::Sub => match &self.rhs {
                    SymbolExpr::Unary(r) => match r.op {
                        UnaryOps::Neg => {
                            if s_rhs.as_str().char_indices().nth(0).unwrap().1 == '-' {
                                let st = s_rhs.char_indices().nth(0).unwrap().0;
                                let ed = s_rhs.char_indices().nth(1).unwrap().0;
                                let s_rhs_new: &str = &s_rhs.as_str()[st..ed];
                                format!("{} + {}", s_lhs, s_rhs_new)
                            } else if op_rhs {
                                format!("{} -({})", s_lhs, s_rhs)
                            } else {
                                format!("{} - {}", s_lhs, s_rhs)
                            }
                        }
                        _ => {
                            if op_rhs {
                                format!("{} -({})", s_lhs, s_rhs)
                            } else {
                                format!("{} - {}", s_lhs, s_rhs)
                            }
                        }
                    },
                    _ => {
                        if op_rhs {
                            format!("{} -({})", s_lhs, s_rhs)
                        } else {
                            format!("{} - {}", s_lhs, s_rhs)
                        }
                    }
                },
                BinaryOps::Mul => {
                    if op_lhs {
                        if op_rhs {
                            format!("({})*({})", s_lhs, s_rhs)
                        } else {
                            format!("({})*{}", s_lhs, s_rhs)
                        }
                    } else if op_rhs {
                        format!("{}*({})", s_lhs, s_rhs)
                    } else {
                        format!("{}*{}", s_lhs, s_rhs)
                    }
                }
                BinaryOps::Div => {
                    if op_lhs {
                        if op_rhs {
                            format!("({})/({})", s_lhs, s_rhs)
                        } else {
                            format!("({})/{}", s_lhs, s_rhs)
                        }
                    } else if op_rhs {
                        format!("{}/({})", s_lhs, s_rhs)
                    } else {
                        format!("{}/{}", s_lhs, s_rhs)
                    }
                }
                BinaryOps::Pow => match &self.lhs {
                    SymbolExpr::Binary(_) | SymbolExpr::Unary(_) => match &self.rhs {
                        SymbolExpr::Binary(_) | SymbolExpr::Unary(_) => {
                            format!("({})**({})", s_lhs, s_rhs)
                        }
                        SymbolExpr::Value(r) => {
                            if r.as_real() < 0.0 {
                                format!("({})**({})", s_lhs, s_rhs)
                            } else {
                                format!("({})**{}", s_lhs, s_rhs)
                            }
                        }
                        _ => format!("({})**{}", s_lhs, s_rhs),
                    },
                    SymbolExpr::Value(l) => {
                        if l.as_real() < 0.0 {
                            match &self.rhs {
                                SymbolExpr::Binary(_) | SymbolExpr::Unary(_) => {
                                    format!("({})**({})", s_lhs, s_rhs)
                                }
                                _ => format!("({})**{}", s_lhs, s_rhs),
                            }
                        } else {
                            match &self.rhs {
                                SymbolExpr::Binary(_) | SymbolExpr::Unary(_) => {
                                    format!("{}**({})", s_lhs, s_rhs)
                                }
                                _ => format!("{}**{}", s_lhs, s_rhs),
                            }
                        }
                    }
                    _ => match &self.rhs {
                        SymbolExpr::Binary(_) | SymbolExpr::Unary(_) => {
                            format!("{}**({})", s_lhs, s_rhs)
                        }
                        SymbolExpr::Value(r) => {
                            if r.as_real() < 0.0 {
                                format!("{}**({})", s_lhs, s_rhs)
                            } else {
                                format!("{}**{}", s_lhs, s_rhs)
                            }
                        }
                        _ => format!("{}**{}", s_lhs, s_rhs),
                    },
                },
            }
        )
    }
}

// ===============================================================
//  implementations for Binary operators
// ===============================================================
impl Binary {
    pub fn new(op: BinaryOps, lhs: SymbolExpr, rhs: SymbolExpr) -> Self {
        Self { op, lhs, rhs }
    }

    pub fn get_symbols_string(&self) -> String {
        self.lhs.get_symbols_string() + &self.rhs.get_symbols_string()
    }

    pub fn bind(&self, maps: &HashMap<String, Value>) -> SymbolExpr {
        let new_lhs = self.lhs.bind(maps);
        let new_rhs = self.rhs.bind(maps);
        match self.op {
            BinaryOps::Add => match new_lhs.add_opt(&new_rhs) {
                Some(e) => e,
                None => _add(new_lhs, new_rhs),
            },
            BinaryOps::Sub => match new_lhs.sub_opt(&new_rhs) {
                Some(e) => e,
                None => _sub(new_lhs, new_rhs),
            },
            BinaryOps::Mul => match new_lhs.mul_opt(&new_rhs) {
                Some(e) => e,
                None => _mul(new_lhs, new_rhs),
            },
            BinaryOps::Div => match new_lhs.div_opt(&new_rhs) {
                Some(e) => e,
                None => _div(new_lhs, new_rhs),
            },
            BinaryOps::Pow => new_lhs.pow(&new_rhs),
        }
    }

    pub fn subs(&self, maps: &HashMap<String, SymbolExpr>) -> SymbolExpr {
        let new_lhs = self.lhs.subs(maps);
        let new_rhs = self.rhs.subs(maps);
        match self.op {
            BinaryOps::Add => match new_lhs.add_opt(&new_rhs) {
                Some(e) => e,
                None => _add(new_lhs, new_rhs),
            },
            BinaryOps::Sub => match new_lhs.sub_opt(&new_rhs) {
                Some(e) => e,
                None => _sub(new_lhs, new_rhs),
            },
            BinaryOps::Mul => match new_lhs.mul_opt(&new_rhs) {
                Some(e) => e,
                None => _mul(new_lhs, new_rhs),
            },
            BinaryOps::Div => match new_lhs.div_opt(&new_rhs) {
                Some(e) => e,
                None => _div(new_lhs, new_rhs),
            },
            BinaryOps::Pow => new_lhs.pow(&new_rhs),
        }
    }

    pub fn derivative(&self, param: &SymbolExpr) -> SymbolExpr {
        match self.op {
            BinaryOps::Add => self.lhs.derivative(param) + self.rhs.derivative(param),
            BinaryOps::Sub => self.lhs.derivative(param) - self.rhs.derivative(param),
            BinaryOps::Mul => {
                self.lhs.derivative(param) * self.rhs.clone()
                    + self.lhs.clone() * self.rhs.derivative(param)
            }
            BinaryOps::Div => {
                (self.lhs.derivative(param) * self.rhs.clone()
                    - self.lhs.clone() * self.rhs.derivative(param))
                    / self.rhs.clone()
                    / self.rhs.clone()
            }
            BinaryOps::Pow => {
                if !self.lhs.has_symbol(&param.to_string()) {
                    if !self.rhs.has_symbol(&param.to_string()) {
                        SymbolExpr::Value(Value::Real(0.0))
                    } else {
                        let rhs = SymbolExpr::Unary(Arc::new(Unary {
                            op: UnaryOps::Log,
                            expr: self.lhs.clone(),
                        }));
                        SymbolExpr::Binary(Arc::new(Binary {
                            op: BinaryOps::Mul,
                            lhs: SymbolExpr::Binary(Arc::new(Binary {
                                op: BinaryOps::Pow,
                                lhs: self.lhs.clone(),
                                rhs: self.rhs.clone(),
                            })),
                            rhs,
                        }))
                    }
                } else if !self.rhs.has_symbol(&param.to_string()) {
                    let rhs = self.rhs.clone() - SymbolExpr::Value(Value::Real(1.0));
                    self.rhs.clone()
                        * SymbolExpr::Binary(Arc::new(Binary {
                            op: BinaryOps::Pow,
                            lhs: self.lhs.clone(),
                            rhs,
                        }))
                } else {
                    let new_expr = SymbolExpr::Unary(Arc::new(Unary {
                        op: UnaryOps::Exp,
                        expr: SymbolExpr::Binary(Arc::new(Binary {
                            op: BinaryOps::Mul,
                            lhs: SymbolExpr::Unary(Arc::new(Unary {
                                op: UnaryOps::Log,
                                expr: self.lhs.clone(),
                            })),
                            rhs: self.rhs.clone(),
                        })),
                    }));
                    new_expr.derivative(param)
                }
            }
        }
    }

    pub fn eval(&self, recurse: bool) -> Option<Value> {
        let lval: Value;
        let rval: Value;
        if recurse {
            match (self.lhs.eval(true), self.rhs.eval(true)) {
                (Some(left), Some(right)) => {
                    lval = left;
                    rval = right;
                }
                _ => return None,
            }
        } else {
            match (&self.lhs, &self.rhs) {
                (SymbolExpr::Value(l), SymbolExpr::Value(r)) => {
                    lval = l.clone();
                    rval = r.clone();
                }
                _ => return None,
            }
        }
        let ret = match self.op {
            BinaryOps::Add => lval + rval,
            BinaryOps::Sub => lval - rval,
            BinaryOps::Mul => lval * rval,
            BinaryOps::Div => lval / rval,
            BinaryOps::Pow => lval.pow(&rval),
        };
        match ret {
            Value::Real(_) => Some(ret),
            Value::Int(_) => Some(ret),
            Value::Complex(c) => {
                if c.im < f64::EPSILON && c.im > -f64::EPSILON {
                    Some(Value::Real(c.re))
                } else {
                    Some(ret)
                }
            }
        }
    }

    pub fn expand(&self) -> SymbolExpr {
        match self.op {
            BinaryOps::Mul => match self.lhs.mul_expand(&self.rhs) {
                Some(e) => e,
                None => _mul(self.lhs.clone(), self.rhs.clone()),
            },
            BinaryOps::Div => match self.lhs.div_expand(&self.rhs) {
                Some(e) => e,
                None => _div(self.lhs.clone(), self.rhs.clone()),
            },
            BinaryOps::Add => match self.lhs.add_opt(&self.rhs) {
                Some(e) => e,
                None => _add(self.lhs.clone(), self.rhs.clone()),
            },
            BinaryOps::Sub => match self.lhs.sub_opt(&self.rhs) {
                Some(e) => e,
                None => _sub(self.lhs.clone(), self.rhs.clone()),
            },
            _ => _pow(self.lhs.expand(), self.rhs.expand()), // TO DO : add expand for pow
        }
    }

    pub fn symbols(&self) -> HashSet<String> {
        let mut symbols = HashSet::<String>::new();
        for s in self.lhs.symbols().union(&self.rhs.symbols()) {
            symbols.insert(s.to_string());
        }
        symbols
    }

    pub fn has_symbol(&self, param: &String) -> bool {
        self.lhs.has_symbol(param) | self.rhs.has_symbol(param)
    }

    // Add with heuristic optimization
    fn add_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        if let SymbolExpr::Binary(r) = rhs {
            if self.op == r.op {
                if let BinaryOps::Mul | BinaryOps::Div | BinaryOps::Pow = self.op {
                    if let (SymbolExpr::Value(rv), SymbolExpr::Value(lv)) = (&self.lhs, &r.lhs) {
                        if self.rhs.expand().to_string() == r.rhs.expand().to_string() {
                            return match SymbolExpr::Value(rv + lv).mul_opt(&self.rhs) {
                                Some(e) => Some(e),
                                None => Some(_mul(SymbolExpr::Value(rv + lv), self.rhs.clone())),
                            };
                        }
                    }

                    if let Some(e) = rhs.neg_opt() {
                        if self.expand().to_string() == e.expand().to_string() {
                            return Some(SymbolExpr::Value(Value::Int(0)));
                        }
                    }
                }
            }
        }

        if let SymbolExpr::Binary(r) = rhs {
            if let BinaryOps::Add = &r.op {
                if let Some(e) = self.add_opt(&r.lhs) {
                    return match e.add_opt(&r.rhs) {
                        Some(ee) => Some(ee),
                        None => Some(_add(e, r.rhs.clone())),
                    };
                }
                if let Some(e) = self.add_opt(&r.rhs) {
                    return match e.add_opt(&r.lhs) {
                        Some(ee) => Some(ee),
                        None => Some(_add(e, r.lhs.clone())),
                    };
                }
            }
            if let BinaryOps::Sub = &r.op {
                if let Some(e) = self.add_opt(&r.lhs) {
                    return match e.sub_opt(&r.rhs) {
                        Some(ee) => Some(ee),
                        None => Some(_sub(e, r.rhs.clone())),
                    };
                }
                if let Some(e) = self.sub_opt(&r.rhs) {
                    return match e.add_opt(&r.lhs) {
                        Some(ee) => Some(ee),
                        None => Some(_add(e, r.lhs.clone())),
                    };
                }
            }
        }

        match &self.op {
            BinaryOps::Add => {
                if let Some(e) = self.lhs.add_opt(rhs) {
                    return match e.add_opt(&self.rhs) {
                        Some(ee) => Some(ee),
                        None => Some(_add(e, self.rhs.clone())),
                    };
                }
                if let Some(e) = self.rhs.add_opt(rhs) {
                    return match self.lhs.add_opt(&e) {
                        Some(ee) => Some(ee),
                        None => Some(_add(self.lhs.clone(), e)),
                    };
                }
                None
            }
            BinaryOps::Sub => {
                if let Some(e) = self.lhs.add_opt(rhs) {
                    return match e.sub_opt(&self.rhs) {
                        Some(ee) => Some(ee),
                        None => Some(_sub(e, self.rhs.clone())),
                    };
                }
                if let Some(e) = rhs.sub_opt(&self.rhs) {
                    return match self.lhs.add_opt(&e) {
                        Some(ee) => Some(ee),
                        None => Some(_add(self.lhs.clone(), e)),
                    };
                }
                None
            }
            _ => None,
        }
    }

    // Sub with heuristic optimization
    fn sub_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        if let SymbolExpr::Binary(r) = rhs {
            if self.op == r.op {
                if let BinaryOps::Mul | BinaryOps::Div | BinaryOps::Pow = self.op {
                    if let (SymbolExpr::Value(rv), SymbolExpr::Value(lv)) = (&self.lhs, &r.lhs) {
                        if self.rhs.expand().to_string() == r.rhs.expand().to_string() {
                            return match SymbolExpr::Value(rv - lv).mul_opt(&self.rhs) {
                                Some(e) => Some(e),
                                None => Some(_mul(SymbolExpr::Value(rv - lv), self.rhs.clone())),
                            };
                        }
                    }
                    if self.expand().to_string() == rhs.expand().to_string() {
                        return Some(SymbolExpr::Value(Value::Int(0)));
                    }
                }
            }
        }

        if let SymbolExpr::Binary(r) = rhs {
            if let BinaryOps::Add = &r.op {
                if let Some(e) = self.sub_opt(&r.lhs) {
                    return match e.sub_opt(&r.rhs) {
                        Some(ee) => Some(ee),
                        None => Some(_sub(e, r.rhs.clone())),
                    };
                }
                if let Some(e) = self.sub_opt(&r.rhs) {
                    return match e.sub_opt(&r.lhs) {
                        Some(ee) => Some(ee),
                        None => Some(_sub(e, r.lhs.clone())),
                    };
                }
            }
            if let BinaryOps::Sub = &r.op {
                if let Some(e) = self.sub_opt(&r.lhs) {
                    return match e.add_opt(&r.rhs) {
                        Some(ee) => Some(ee),
                        None => Some(_add(e, r.rhs.clone())),
                    };
                }
                if let Some(e) = self.add_opt(&r.rhs) {
                    return match e.sub_opt(&r.lhs) {
                        Some(ee) => Some(ee),
                        None => Some(_sub(e, r.lhs.clone())),
                    };
                }
            }
        }

        match &self.op {
            BinaryOps::Add => {
                if let Some(e) = self.lhs.sub_opt(rhs) {
                    return match e.add_opt(&self.rhs) {
                        Some(ee) => Some(ee),
                        None => Some(_add(e, self.rhs.clone())),
                    };
                }
                if let Some(e) = self.rhs.sub_opt(rhs) {
                    return match self.lhs.add_opt(&e) {
                        Some(ee) => Some(ee),
                        None => Some(_add(self.lhs.clone(), e)),
                    };
                }
                None
            }
            BinaryOps::Sub => {
                if let Some(e) = self.lhs.sub_opt(rhs) {
                    return match e.sub_opt(&self.rhs) {
                        Some(ee) => Some(ee),
                        None => Some(_sub(e, self.rhs.clone())),
                    };
                }
                if let Some(e) = self.rhs.add_opt(rhs) {
                    return match self.lhs.sub_opt(&e) {
                        Some(ee) => Some(ee),
                        None => Some(_sub(self.lhs.clone(), e)),
                    };
                }
                None
            }
            _ => None,
        }
    }

    // Mul with heuristic optimization
    fn mul_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        if let SymbolExpr::Binary(r) = rhs {
            if let BinaryOps::Mul = &r.op {
                if let Some(e) = self.mul_opt(&r.lhs) {
                    return match e.mul_opt(&r.rhs) {
                        Some(ee) => Some(ee),
                        None => Some(_mul(e, r.rhs.clone())),
                    };
                }
                if let Some(e) = self.mul_opt(&r.rhs) {
                    return match e.mul_opt(&r.lhs) {
                        Some(ee) => Some(ee),
                        None => Some(_mul(e, r.lhs.clone())),
                    };
                }
            }
            if let BinaryOps::Div = &r.op {
                if let Some(e) = self.mul_opt(&r.lhs) {
                    return match e.div_opt(&r.rhs) {
                        Some(ee) => Some(ee),
                        None => Some(_div(e, r.rhs.clone())),
                    };
                }
                if let Some(e) = self.div_opt(&r.rhs) {
                    return match e.mul_opt(&r.lhs) {
                        Some(ee) => Some(ee),
                        None => Some(_mul(e, r.lhs.clone())),
                    };
                }
            }
        }

        if let BinaryOps::Mul = &self.op {
            if let Some(e) = self.lhs.mul_opt(rhs) {
                return match e.mul_opt(&self.rhs) {
                    Some(ee) => Some(ee),
                    None => Some(_mul(e, self.rhs.clone())),
                };
            }
            if let Some(e) = self.rhs.mul_opt(rhs) {
                return match self.lhs.mul_opt(&e) {
                    Some(ee) => Some(ee),
                    None => Some(_mul(self.lhs.clone(), e)),
                };
            }
        } else if let BinaryOps::Div = &self.op {
            if let Some(e) = self.lhs.mul_opt(rhs) {
                return match e.div_opt(&self.rhs) {
                    Some(ee) => Some(ee),
                    None => Some(_div(e, self.rhs.clone())),
                };
            }
            if let Some(e) = rhs.div_opt(&self.rhs) {
                return match self.lhs.mul_opt(&e) {
                    Some(ee) => Some(ee),
                    None => Some(_mul(self.lhs.clone(), e)),
                };
            }
        }
        None
    }

    fn mul_expand(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match &self.op {
            BinaryOps::Add | BinaryOps::Sub => {
                let l = match self.lhs.mul_expand(rhs) {
                    Some(e) => e,
                    None => match self.lhs.mul_opt(rhs) {
                        Some(e) => e,
                        None => _mul(self.lhs.clone(), rhs.clone()),
                    },
                };
                let r = match self.rhs.mul_expand(rhs) {
                    Some(e) => e,
                    None => match self.rhs.mul_opt(rhs) {
                        Some(e) => e,
                        None => _mul(self.rhs.clone(), rhs.clone()),
                    },
                };
                match &self.op {
                    BinaryOps::Sub => match l.sub_opt(&r) {
                        Some(e) => Some(e),
                        None => Some(_sub(l, r)),
                    },
                    _ => match l.add_opt(&r) {
                        Some(e) => Some(e),
                        None => Some(_add(l, r)),
                    },
                }
            }
            BinaryOps::Mul => match self.lhs.mul_expand(rhs) {
                Some(e) => match e.mul_expand(&self.rhs) {
                    Some(ee) => Some(ee),
                    None => match e.mul_opt(&self.rhs) {
                        Some(ee) => Some(ee),
                        None => Some(_mul(e, self.rhs.clone())),
                    },
                },
                None => match self.rhs.mul_expand(rhs) {
                    Some(e) => match self.lhs.mul_expand(&e) {
                        Some(ee) => Some(ee),
                        None => match self.lhs.mul_opt(&e) {
                            Some(ee) => Some(ee),
                            None => Some(_mul(self.lhs.clone(), e)),
                        },
                    },
                    None => None,
                },
            },
            BinaryOps::Div => match self.lhs.div_expand(rhs) {
                Some(e) => Some(_div(e, self.rhs.clone())),
                None => self.rhs.div_expand(rhs).map(|e| _div(self.lhs.clone(), e)),
            },
            _ => None,
        }
    }

    // Div with heuristic optimization
    fn div_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        if let SymbolExpr::Binary(r) = rhs {
            if let BinaryOps::Mul = &r.op {
                if let Some(e) = self.div_opt(&r.lhs) {
                    return match e.div_opt(&r.rhs) {
                        Some(ee) => Some(ee),
                        None => Some(_div(e, r.rhs.clone())),
                    };
                }
                if let Some(e) = self.div_opt(&r.rhs) {
                    return match e.div_opt(&r.lhs) {
                        Some(ee) => Some(ee),
                        None => Some(_div(e, r.lhs.clone())),
                    };
                }
            }
            if let BinaryOps::Div = &r.op {
                if let Some(e) = self.mul_opt(&r.rhs) {
                    return match e.div_opt(&r.lhs) {
                        Some(ee) => Some(ee),
                        None => Some(_div(e, r.lhs.clone())),
                    };
                }
                if let Some(e) = self.div_opt(&r.lhs) {
                    return match e.mul_opt(&r.rhs) {
                        Some(ee) => Some(ee),
                        None => Some(_mul(e, r.rhs.clone())),
                    };
                }
            }
        }

        if let BinaryOps::Mul = &self.op {
            if let Some(e) = self.lhs.div_opt(rhs) {
                return match e.mul_opt(&self.rhs) {
                    Some(ee) => Some(ee),
                    None => Some(_mul(e, self.rhs.clone())),
                };
            }
            if let Some(e) = self.rhs.div_opt(rhs) {
                return match self.lhs.mul_opt(&e) {
                    Some(ee) => Some(ee),
                    None => Some(_mul(self.lhs.clone(), e)),
                };
            }
        } else if let BinaryOps::Div = &self.op {
            if let Some(e) = self.rhs.mul_opt(rhs) {
                return match self.lhs.div_opt(&e) {
                    Some(ee) => Some(ee),
                    None => Some(_div(self.lhs.clone(), e)),
                };
            }
            if let Some(e) = self.lhs.div_opt(rhs) {
                return match e.div_opt(&self.rhs) {
                    Some(ee) => Some(ee),
                    None => Some(_div(e, self.rhs.clone())),
                };
            }
        }
        None
    }

    fn div_expand(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match &self.op {
            BinaryOps::Add | BinaryOps::Sub => {
                let l = match self.lhs.div_expand(rhs) {
                    Some(e) => e,
                    None => match self.lhs.div_opt(rhs) {
                        Some(e) => e,
                        None => _div(self.lhs.clone(), rhs.clone()),
                    },
                };
                let r = match self.rhs.div_expand(rhs) {
                    Some(e) => e,
                    None => match self.rhs.div_opt(rhs) {
                        Some(e) => e,
                        None => _div(self.rhs.clone(), rhs.clone()),
                    },
                };
                match &self.op {
                    BinaryOps::Sub => match l.sub_opt(&r) {
                        Some(e) => Some(e),
                        None => Some(_sub(l, r)),
                    },
                    _ => match l.add_opt(&r) {
                        Some(e) => Some(e),
                        None => Some(_add(l, r)),
                    },
                }
            }
            _ => None,
        }
    }
}

impl PartialEq for Binary {
    fn eq(&self, rhs: &Self) -> bool {
        if self.op != rhs.op {
            return false;
        }
        match self.op {
            BinaryOps::Mul | BinaryOps::Div => {
                self.expand().to_string() == rhs.expand().to_string()
            }
            _ => false, // we can not evaluate equality at this point
        }
    }
}
