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

use num_complex::Complex64;

/// node types of expression tree
#[derive(Debug, Clone)]
pub enum SymbolExpr {
    Symbol(Symbol),
    Value(Value),
    Unary {
        op: UnaryOp,
        expr: Box<SymbolExpr>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<SymbolExpr>,
        rhs: Box<SymbolExpr>,
    },
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
    if rhs.is_negative() {
        match rhs.neg_opt() {
            Some(e) => SymbolExpr::Binary {
                op: BinaryOp::Sub,
                lhs: Box::new(lhs),
                rhs: Box::new(e),
            },
            None => SymbolExpr::Binary {
                op: BinaryOp::Sub,
                lhs: Box::new(lhs),
                rhs: Box::new(_neg(rhs)),
            },
        }
    } else {
        SymbolExpr::Binary {
            op: BinaryOp::Add,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }
}

// functions to make new expr for sub
#[inline(always)]
fn _sub(lhs: SymbolExpr, rhs: SymbolExpr) -> SymbolExpr {
    if rhs.is_negative() {
        match rhs.neg_opt() {
            Some(e) => SymbolExpr::Binary {
                op: BinaryOp::Add,
                lhs: Box::new(lhs),
                rhs: Box::new(e),
            },
            None => SymbolExpr::Binary {
                op: BinaryOp::Add,
                lhs: Box::new(lhs),
                rhs: Box::new(_neg(rhs)),
            },
        }
    } else {
        SymbolExpr::Binary {
            op: BinaryOp::Sub,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }
}

// functions to make new expr for mul
#[inline(always)]
fn _mul(lhs: SymbolExpr, rhs: SymbolExpr) -> SymbolExpr {
    SymbolExpr::Binary {
        op: BinaryOp::Mul,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }
}

// functions to make new expr for div
#[inline(always)]
fn _div(lhs: SymbolExpr, rhs: SymbolExpr) -> SymbolExpr {
    SymbolExpr::Binary {
        op: BinaryOp::Div,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }
}

// functions to make new expr for pow
#[inline(always)]
fn _pow(lhs: SymbolExpr, rhs: SymbolExpr) -> SymbolExpr {
    SymbolExpr::Binary {
        op: BinaryOp::Pow,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }
}

// functions to make new expr for neg
#[inline(always)]
fn _neg(expr: SymbolExpr) -> SymbolExpr {
    SymbolExpr::Unary {
        op: UnaryOp::Neg,
        expr: Box::new(expr),
    }
}

impl fmt::Display for SymbolExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                SymbolExpr::Symbol(e) => e.to_string(),
                SymbolExpr::Value(e) => e.to_string(),
                SymbolExpr::Unary{op, expr} => {
                    let s = expr.to_string();
                    match op {
                        UnaryOp::Abs => format!("abs({})", s),
                        UnaryOp::Neg => match &**expr {
                            SymbolExpr::Value(e) => (-e).to_string(),
                            SymbolExpr::Binary{op: eop, lhs: _, rhs: _} => match eop {
                                BinaryOp::Add | BinaryOp::Sub => format!("-({})", s),
                                _ => format!("-{}", s),
                            },
                            _ => format!("-{}", s),
                        },
                        UnaryOp::Sin => format!("sin({})", s),
                        UnaryOp::Asin => format!("asin({})", s),
                        UnaryOp::Cos => format!("cos({})", s),
                        UnaryOp::Acos => format!("acos({})", s),
                        UnaryOp::Tan => format!("tan({})", s),
                        UnaryOp::Atan => format!("atan({})", s),
                        UnaryOp::Exp => format!("exp({})", s),
                        UnaryOp::Log => format!("log({})", s),
                        UnaryOp::Sign => format!("sign({})", s),
                        UnaryOp::Conj => format!("conj({})", s),
                    }
                },
                SymbolExpr::Binary{op, lhs, rhs} => {
                    let s_lhs = lhs.to_string();
                    let s_rhs = rhs.to_string();
                    let op_lhs = match &**lhs {
                        SymbolExpr::Binary{op: lop, lhs: _, rhs: _} => matches!(lop, BinaryOp::Add | BinaryOp::Sub),
                        SymbolExpr::Value(e) => match e {
                            Value::Real(v) => *v < 0.0,
                            Value::Int(v) => *v < 0,
                            Value::Complex(_) => true,
                        },
                        _ => false,
                    };
                    let op_rhs = match &**rhs {
                        SymbolExpr::Binary{op: rop, lhs: _, rhs: _} => match rop {
                            BinaryOp::Add | BinaryOp::Sub => true,
                            _ => matches!(op, BinaryOp::Div),
                        },
                        SymbolExpr::Value(e) => match e {
                            Value::Real(v) => *v < 0.0,
                            Value::Int(v) => *v < 0,
                            Value::Complex(_) => true,
                        },
                        _ => false,
                    };

                    match op {
                        BinaryOp::Add => match &**rhs {
                            SymbolExpr::Unary{op: rop, expr: _} => match rop {
                                UnaryOp::Neg => {
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
                        BinaryOp::Sub => match &**rhs {
                            SymbolExpr::Unary{op: rop, expr: _} => match rop {
                                UnaryOp::Neg => {
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
                        BinaryOp::Mul => {
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
                        BinaryOp::Div => {
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
                        BinaryOp::Pow => match &**lhs {
                            SymbolExpr::Binary{op: _, lhs: _, rhs: _} | SymbolExpr::Unary{op: _, expr: _} => match &**rhs {
                                SymbolExpr::Binary{op: _, lhs: _, rhs: _} | SymbolExpr::Unary{op: _, expr: _} => {
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
                                    match &**rhs {
                                        SymbolExpr::Binary{op: _, lhs: _, rhs: _} | SymbolExpr::Unary{op: _, expr: _} => {
                                            format!("({})**({})", s_lhs, s_rhs)
                                        }
                                        _ => format!("({})**{}", s_lhs, s_rhs),
                                    }
                                } else {
                                    match &**rhs {
                                        SymbolExpr::Binary{op: _, lhs: _, rhs: _} | SymbolExpr::Unary{op: _, expr: _} => {
                                            format!("{}**({})", s_lhs, s_rhs)
                                        }
                                        _ => format!("{}**{}", s_lhs, s_rhs),
                                    }
                                }
                            }
                            _ => match &**rhs {
                                SymbolExpr::Binary{op: _, lhs: _, rhs: _} | SymbolExpr::Unary{op: _, expr: _} => {
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
                }
            },
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
            SymbolExpr::Unary{op, expr} => {
                let new_expr = SymbolExpr::Unary {
                    op: op.clone(),
                    expr: Box::new(expr.bind(maps)),
                };
                match new_expr.eval(false) {
                    Some(v) => SymbolExpr::Value(v.clone()),
                    None => new_expr,
                }
            },
            SymbolExpr::Binary{op, lhs, rhs} => {
                let new_lhs = lhs.bind(maps);
                let new_rhs = rhs.bind(maps);
                /*
                match op {
                    BinaryOp::Add => _add(new_lhs, new_rhs),
                    BinaryOp::Sub => _sub(new_lhs, new_rhs),
                    BinaryOp::Mul => _mul(new_lhs, new_rhs),
                    BinaryOp::Div => _div(new_lhs, new_rhs),
                    BinaryOp::Pow => _pow(new_lhs, new_rhs),
                }
                */
                match op {
                    BinaryOp::Add => match new_lhs.add_opt(&new_rhs) {
                        Some(e) => e,
                        None => _add(new_lhs, new_rhs),
                    },
                    BinaryOp::Sub => match new_lhs.sub_opt(&new_rhs) {
                        Some(e) => e,
                        None => _sub(new_lhs, new_rhs),
                    },
                    BinaryOp::Mul => match new_lhs.mul_opt(&new_rhs) {
                        Some(e) => e,
                        None => _mul(new_lhs, new_rhs),
                    },
                    BinaryOp::Div => match new_lhs.div_opt(&new_rhs) {
                        Some(e) => e,
                        None => _div(new_lhs, new_rhs),
                    },
                    BinaryOp::Pow => new_lhs.pow(&new_rhs),
                }
            }        
        }
    }

    /// substitute symbol node to other expression
    pub fn subs(&self, maps: &HashMap<String, SymbolExpr>) -> SymbolExpr {
        match self {
            SymbolExpr::Symbol(e) => e.subs(maps),
            SymbolExpr::Value(e) => SymbolExpr::Value(e.clone()),
            SymbolExpr::Unary{op, expr} => {
                let new_expr = SymbolExpr::Unary {
                    op: op.clone(),
                    expr: Box::new(expr.subs(maps)),
                };
                match new_expr.eval(false) {
                    Some(v) => SymbolExpr::Value(v.clone()),
                    None => new_expr,
                }
            },
            SymbolExpr::Binary{op, lhs, rhs} => {
                let new_lhs = lhs.subs(maps);
                let new_rhs = rhs.subs(maps);
                /*
                match op {
                    BinaryOp::Add => _add(new_lhs, new_rhs),
                    BinaryOp::Sub => _sub(new_lhs, new_rhs),
                    BinaryOp::Mul => _mul(new_lhs, new_rhs),
                    BinaryOp::Div => _div(new_lhs, new_rhs),
                    BinaryOp::Pow => _pow(new_lhs, new_rhs),
                }
                */
                match op {
                    BinaryOp::Add => match new_lhs.add_opt(&new_rhs) {
                        Some(e) => e,
                        None => _add(new_lhs, new_rhs),
                    },
                    BinaryOp::Sub => match new_lhs.sub_opt(&new_rhs) {
                        Some(e) => e,
                        None => _sub(new_lhs, new_rhs),
                    },
                    BinaryOp::Mul => match new_lhs.mul_opt(&new_rhs) {
                        Some(e) => e,
                        None => _mul(new_lhs, new_rhs),
                    },
                    BinaryOp::Div => match new_lhs.div_opt(&new_rhs) {
                        Some(e) => e,
                        None => _div(new_lhs, new_rhs),
                    },
                    BinaryOp::Pow => new_lhs.pow(&new_rhs),
                }
            }
        }
    }

    /// evaluate the equation
    /// if recursive is false, only this node will be evaluated
    pub fn eval(&self, recurse: bool) -> Option<Value> {
        match self {
            SymbolExpr::Symbol(_) => None,
            SymbolExpr::Value(e) => Some(e.clone()),
            SymbolExpr::Unary{op, expr} => {
                let val: Value;
                if recurse {
                    match expr.eval(recurse) {
                        Some(v) => val = v,
                        None => return None,
                    }
                } else {
                    match &**expr {
                        SymbolExpr::Value(e) => val = e.clone(),
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
                        if c.im < f64::EPSILON && c.im > -f64::EPSILON {
                            Some(Value::Real(c.re))
                        } else {
                            Some(ret)
                        }
                    }
                }
            },        
            SymbolExpr::Binary{op, lhs, rhs} => {
                let lval: Value;
                let rval: Value;
                if recurse {
                    match (lhs.eval(true), rhs.eval(true)) {
                        (Some(left), Some(right)) => {
                            lval = left;
                            rval = right;
                        }
                        _ => return None,
                    }
                } else {
                    match (&**lhs, &**rhs) {
                        (SymbolExpr::Value(l), SymbolExpr::Value(r)) => {
                            lval = l.clone();
                            rval = r.clone();
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
                        if c.im < f64::EPSILON && c.im > -f64::EPSILON {
                            Some(Value::Real(c.re))
                        } else {
                            Some(ret)
                        }
                    }
                }
            }
        }
    }

    /// calculate derivative of the equantion for a symbol passed by param
    pub fn derivative(&self, param: &SymbolExpr) -> SymbolExpr {
        if self == param {
            SymbolExpr::Value(Value::Real(1.0))
        } else {
            match self {
                SymbolExpr::Value(_) | SymbolExpr::Symbol(_) => SymbolExpr::Value(Value::Real(0.0)),
                SymbolExpr::Unary{op, expr} => {
                    let expr_d = expr.derivative(param);
                    match op {
                        UnaryOp::Abs => {
                            &(&**expr * &expr_d)
                                / &SymbolExpr::Unary {
                                    op: op.clone(),
                                    expr: Box::new(*expr.clone()),
                                }
                        }
                        UnaryOp::Neg => SymbolExpr::Unary {
                            op: UnaryOp::Neg,
                            expr: Box::new(expr_d),
                        },
                        UnaryOp::Sin => {
                            let lhs = SymbolExpr::Unary {
                                op: UnaryOp::Cos,
                                expr: Box::new(*expr.clone()),
                            };
                            lhs * expr_d
                        }
                        UnaryOp::Asin => {
                            let d = &SymbolExpr::Value(Value::Real(1.0)) - &(&**expr * &**expr);
                            let lhs = match d {
                                SymbolExpr::Value(v) => SymbolExpr::Value(v.sqrt()),
                                _ => _pow(d, SymbolExpr::Value(Value::Real(0.5))),
                            };
                            &lhs * &expr_d
                        }
                        UnaryOp::Cos => {
                            let lhs = SymbolExpr::Unary {
                                op: UnaryOp::Sin,
                                expr: Box::new(*expr.clone()),
                            };
                            &-&lhs * &expr_d
                        }
                        UnaryOp::Acos => {
                            let d = &SymbolExpr::Value(Value::Real(1.0)) - &(&**expr * &**expr);
                            let lhs = match d {
                                SymbolExpr::Value(v) => SymbolExpr::Value(v.sqrt()),
                                _ => _pow(d, SymbolExpr::Value(Value::Real(0.5))),
                            };
                            &-&lhs * &expr_d
                        }
                        UnaryOp::Tan => {
                            let d = SymbolExpr::Unary {
                                op: UnaryOp::Cos,
                                expr: Box::new(*expr.clone()),
                            };
                            &(&expr_d / &d) / &d
                        }
                        UnaryOp::Atan => {
                            let d = &SymbolExpr::Value(Value::Real(1.0)) + &(&**expr * &**expr);
                            &expr_d / &d
                        }
                        UnaryOp::Exp => {
                            &SymbolExpr::Unary {
                                op: UnaryOp::Exp,
                                expr: Box::new(*expr.clone()),
                            } * &expr_d
                        }
                        UnaryOp::Log => &expr_d / &**expr,
                        UnaryOp::Sign => SymbolExpr::Unary {
                            op: UnaryOp::Sign,
                            expr: Box::new(expr_d),
                        },
                        UnaryOp::Conj => SymbolExpr::Unary {
                            op: UnaryOp::Conj,
                            expr: Box::new(expr_d),
                        },
                    }
                },
                SymbolExpr::Binary{op, lhs, rhs} => {
                    match op {
                        BinaryOp::Add => &lhs.derivative(param) + &rhs.derivative(param),
                        BinaryOp::Sub => &lhs.derivative(param) - &rhs.derivative(param),
                        BinaryOp::Mul => {
                            &(&lhs.derivative(param) * &**rhs)
                                + &(&**lhs * &rhs.derivative(param))
                        }
                        BinaryOp::Div => {
                            &(&(&(&lhs.derivative(param) * &**rhs)
                                - &(&**lhs * &rhs.derivative(param)))
                                / &**rhs)
                                / &**rhs
                        }
                        BinaryOp::Pow => {
                            if !lhs.has_symbol(&param.to_string()) {
                                if !rhs.has_symbol(&param.to_string()) {
                                    SymbolExpr::Value(Value::Real(0.0))
                                } else {
                                    _mul(
                                        SymbolExpr::Binary {
                                            op: BinaryOp::Pow,
                                            lhs: Box::new(*lhs.clone()),
                                            rhs: Box::new(*rhs.clone()),
                                        },
                                         SymbolExpr::Unary {
                                            op: UnaryOp::Log,
                                            expr: Box::new(*lhs.clone()),
                                        },
                                    )
                                }
                            } else if !rhs.has_symbol(&param.to_string()) {
                                &**rhs * &SymbolExpr::Binary {
                                        op: BinaryOp::Pow,
                                        lhs: Box::new(*lhs.clone()),
                                        rhs: Box::new(&**rhs - &SymbolExpr::Value(Value::Real(1.0))),
                                    }
                            } else {
                                let new_expr = SymbolExpr::Unary {
                                    op: UnaryOp::Exp,
                                    expr: Box::new(_mul(
                                        SymbolExpr::Unary {
                                            op: UnaryOp::Log,
                                            expr: Box::new(*lhs.clone()),
                                        },
                                        *rhs.clone()
                                    )),
                                };
                                new_expr.derivative(param)
                            }
                        }
                    }
                }
            }
        }
    }

    /// expand the equation
    pub fn expand(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Symbol(_) => self.clone(),
            SymbolExpr::Value(_) => self.clone(),
            SymbolExpr::Unary{op, expr} => {
                let ex = expr.expand();
                match op {
                    UnaryOp::Neg => match ex.neg_opt() {
                        Some(ne) => ne,
                        None => _neg(ex),
                    },
                    _ => SymbolExpr::Unary {
                        op: op.clone(),
                        expr: Box::new(ex),
                    },
                }
            },
            SymbolExpr::Binary{op, lhs, rhs} => {
                /*
                match op {
                    BinaryOp::Mul => _mul(*lhs.clone(), *rhs.clone()),
                    BinaryOp::Div => _div(*lhs.clone(), *rhs.clone()),
                    BinaryOp::Add => _add(*lhs.clone(), *rhs.clone()),
                    BinaryOp::Sub => _sub(*lhs.clone(), *rhs.clone()),
                    _ => _pow(lhs.expand(), rhs.expand()), // TO DO : add expand for pow
                }
                */
                match op {
                    BinaryOp::Mul => match lhs.mul_expand(rhs) {
                        Some(e) => e,
                        None => _mul(*lhs.clone(), *rhs.clone()),
                    },
                    BinaryOp::Div => match lhs.div_expand(rhs) {
                        Some(e) => e,
                        None => _div(*lhs.clone(), *rhs.clone()),
                    },
                    BinaryOp::Add => match lhs.add_opt(rhs) {
                        Some(e) => e,
                        None => _add(*lhs.clone(), *rhs.clone()),
                    },
                    BinaryOp::Sub => match lhs.sub_opt(rhs) {
                        Some(e) => e,
                        None => _sub(*lhs.clone(), *rhs.clone()),
                    },
                    _ => _pow(lhs.expand(), rhs.expand()), // TO DO : add expand for pow
                }
            }
        }
    }

    /// sign operator
    pub fn sign(&self) -> SymbolExpr {
        SymbolExpr::Unary {
            op: UnaryOp::Sign,
            expr: Box::new(self.clone()),
        }
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
            SymbolExpr::Unary{op: _, expr} => {
                expr.symbols()
            },
            SymbolExpr::Binary{op: _, lhs, rhs} => {
                let mut symbols = HashSet::<String>::new();
                for s in lhs.symbols().union(&rhs.symbols()) {
                    symbols.insert(s.to_string());
                }
                symbols
            }
        }
    }

    /// return all numbers in the equation
    pub fn values(&self) -> Vec<Value> {
        match self {
            SymbolExpr::Symbol(_) => Vec::<Value>::new(),
            SymbolExpr::Value(v) => Vec::<Value>::from([v.clone()]),
            SymbolExpr::Unary{op: _, expr} => expr.values(),
            SymbolExpr::Binary{op: _, lhs, rhs} => {
                let mut l = lhs.values();
                let r = rhs.values();
                l.extend(r);
                l
            }
        }
    }

    /// concatenate all symbols under this node (internal use for sorting nodes)
    fn get_symbols_string(&self) -> String {
        match self {
            SymbolExpr::Symbol(e) => e.name.clone(),
            SymbolExpr::Value(_) => String::new(),
            SymbolExpr::Unary{op: _, expr} => expr.get_symbols_string(),
            SymbolExpr::Binary{op: _, lhs, rhs} => lhs.get_symbols_string() + &rhs.get_symbols_string(),
        }
    }

    /// check if a symbol is in this equation
    pub fn has_symbol(&self, param: &String) -> bool {
        match self {
            SymbolExpr::Symbol(e) => e.name == *param,
            SymbolExpr::Value(_) => false,
            SymbolExpr::Unary{op: _, expr} => expr.has_symbol(param),
            SymbolExpr::Binary{op: _, lhs, rhs} => lhs.has_symbol(param) | rhs.has_symbol(param),
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
            SymbolExpr::Unary{op: _, expr: _} => _div(
                SymbolExpr::Value(Value::Real(1.0)),
                self.clone()
            ),
            SymbolExpr::Binary{op, lhs, rhs} => match op {
                BinaryOp::Div => SymbolExpr::Binary {
                    op: op.clone(),
                    lhs: Box::new(*rhs.clone()),
                    rhs: Box::new(*lhs.clone()),
                },
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
            SymbolExpr::Symbol(_) => SymbolExpr::Unary {
                op: UnaryOp::Conj,
                expr: Box::new(self.clone()),
            },
            SymbolExpr::Value(e) => match e {
                Value::Complex(c) => SymbolExpr::Value(Value::Complex(c.conj())),
                _ => SymbolExpr::Value(e.clone()),
            },
            SymbolExpr::Unary{op, expr} => SymbolExpr::Unary {
                op: op.clone(),
                expr: Box::new(expr.conjugate()),
            },
            SymbolExpr::Binary{op, lhs, rhs} => SymbolExpr::Binary {
                op: op.clone(),
                lhs: Box::new(lhs.conjugate()),
                rhs: Box::new(rhs.conjugate()),
            },
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
            SymbolExpr::Unary{op, expr} => match op {
                UnaryOp::Abs => false,
                UnaryOp::Neg => !expr.is_negative(),
                _ => false, // TO DO add heuristic determination
            },
            SymbolExpr::Binary{op, lhs, rhs} => match op {
                BinaryOp::Mul | BinaryOp::Div => lhs.is_negative() ^ rhs.is_negative(),
                BinaryOp::Add | BinaryOp::Sub => lhs.is_negative(),
                _ => false, // TO DO add heuristic determination for pow
            },
        }
    }

    /// unary operations
    pub fn abs(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.abs()),
            SymbolExpr::Unary{op, expr} => match op {
                UnaryOp::Abs | UnaryOp::Neg => expr.abs(),
                _ => SymbolExpr::Unary{
                    op: UnaryOp::Abs,
                    expr: Box::new(self.clone()),
                },
            },
            _ => SymbolExpr::Unary {
                op: UnaryOp::Abs,
                expr: Box::new(self.clone()),
            },
        }
    }
    pub fn sin(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.sin()),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Sin,
                expr: Box::new(self.clone()),
            },
        }
    }
    pub fn asin(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.asin()),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Asin,
                expr: Box::new(self.clone()),
            },
        }
    }
    pub fn cos(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.cos()),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Cos,
                expr: Box::new(self.clone()),
            },
        }
    }
    pub fn acos(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.acos()),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Acos,
                expr: Box::new(self.clone()),
            },
        }
    }
    pub fn tan(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.tan()),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Tan,
                expr: Box::new(self.clone()),
            },
        }
    }
    pub fn atan(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.atan()),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Atan,
                expr: Box::new(self.clone()),
            },
        }
    }
    pub fn exp(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.exp()),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Exp,
                expr: Box::new(self.clone()),
            },
        }
    }
    pub fn log(&self) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => SymbolExpr::Value(l.log()),
            _ => SymbolExpr::Unary {
                op: UnaryOp::Log,
                expr: Box::new(self.clone()),
            },
        }
    }
    pub fn pow(&self, rhs: &SymbolExpr) -> SymbolExpr {
        match self {
            SymbolExpr::Value(l) => match rhs {
                SymbolExpr::Value(r) => SymbolExpr::Value(l.pow(r)),
                _ => SymbolExpr::Binary {
                    op: BinaryOp::Pow,
                    lhs: Box::new(SymbolExpr::Value(l.clone())),
                    rhs: Box::new(rhs.clone()),
                },
            },
            _ => SymbolExpr::Binary {
                op: BinaryOp::Pow,
                lhs: Box::new(self.clone()),
                rhs: Box::new(rhs.clone()),
            },
        }
    }

    // Add with heuristic optimization
    fn add_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        if self.is_zero() {
            Some(rhs.clone())
        } else if rhs.is_zero() {
            Some(self.clone())
        } else {
            // if neg operation, call sub_opt
            if let SymbolExpr::Unary{op, expr} = rhs {
                if let UnaryOp::Neg = op {
                    return self.sub_opt(&expr);
                }
            } else if let SymbolExpr::Binary{op, lhs: r_lhs, rhs: r_rhs} = rhs {
                // recursive optimization for add and sub
                match op {
                    BinaryOp::Add => match self.add_opt(r_lhs) {
                        // self + r.lhs + r.rhs
                        Some(rl) => match rl.add_opt(r_rhs) {
                            Some(rr) => Some(rr),
                            None => Some(_add(rl, *r_rhs.clone())),
                        },
                        None => match self.add_opt(r_rhs) {
                            Some(rr) => match rr.add_opt(r_lhs) {
                                Some(rl) => Some(rl),
                                None => Some(_add(rr, *r_lhs.clone())),
                            },
                            None => None,
                        },
                    },
                    BinaryOp::Sub => match self.add_opt(r_lhs) {
                        // self + r.lhs - r.rhs
                        Some(rl) => match rl.sub_opt(r_rhs) {
                            Some(rr) => Some(rr),
                            None => Some(_sub(rl, *r_rhs.clone())),
                        },
                        None => match self.sub_opt(r_rhs) {
                            Some(rr) => match rr.add_opt(r_lhs) {
                                Some(rl) => Some(rl),
                                None => Some(_add(rr, *r_lhs.clone())),
                            },
                            None => None,
                        },
                    },
                    _ => None,
                };
            }

            // optimization for each node type
            match self {
                SymbolExpr::Value(l) => match rhs {
                    SymbolExpr::Value(r) => Some(SymbolExpr::Value(l + r)),
                    _ => None,
                },
                SymbolExpr::Symbol(l) => match rhs {
                    SymbolExpr::Value(_) => Some(_add(rhs.clone(), self.clone())),
                    SymbolExpr::Symbol(r) => {
                        if r.name == l.name {
                            Some(_mul(
                                SymbolExpr::Value(Value::Int(2)),
                                self.clone(),
                            ))
                        } else if r.name < l.name {
                            Some(_add(rhs.clone(), self.clone()))
                        } else {
                            None
                        }
                    }
                    _ => None,  // TO DO add optimization for adding binary mul/div case
                },
                SymbolExpr::Unary{op, expr} => {
                    if let UnaryOp::Neg = op {
                        if let Some(e) = expr.sub_opt(rhs) {
                            return match e.neg_opt() {
                                Some(ee) => Some(ee),
                                None => Some(_neg(e)),
                            };
                        }
                    } else if let SymbolExpr::Unary{op: rop, expr: rexpr} = rhs {
                        if op == rop {
                            let t = expr.expand() + rexpr.expand();
                            if t.is_zero() {
                                return Some(SymbolExpr::Value(Value::Int(0)));
                            }
                        }
                    }

                    // swap nodes by sorting rule
                    match rhs {
                        SymbolExpr::Binary{op: rop, lhs: _, rhs: _} => {
                            if let BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow = rop {
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
                },
                SymbolExpr::Binary{op, lhs: l_lhs, rhs: l_rhs} => {
                    if let SymbolExpr::Binary{op: rop, lhs: r_lhs, rhs: r_rhs} = rhs {
                        if op == rop {
                            if let BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow = op {
                                if let (SymbolExpr::Value(rv), SymbolExpr::Value(lv)) = (&**l_lhs, &**r_lhs) {
                                    if l_rhs.expand().to_string() == r_rhs.expand().to_string() {
                                        return match SymbolExpr::Value(rv + lv).mul_opt(l_rhs) {
                                            Some(e) => Some(e),
                                            None => Some(_mul(SymbolExpr::Value(rv + lv), *l_rhs.clone())),
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
                    if let BinaryOp::Add = op {
                        if let Some(e) = l_lhs.add_opt(rhs) {
                            return match e.add_opt(l_rhs) {
                                Some(ee) => Some(ee),
                                None => Some(_add(e, *l_rhs.clone())),
                            };
                        }
                        if let Some(e) = l_rhs.add_opt(rhs) {
                            return match l_lhs.add_opt(&e) {
                                Some(ee) => Some(ee),
                                None => Some(_add(*l_lhs.clone(), e)),
                            };
                        }
                    } else if let BinaryOp::Sub = op {
                        if let Some(e) = l_lhs.add_opt(rhs) {
                            return match e.sub_opt(l_rhs) {
                                Some(ee) => Some(ee),
                                None => Some(_sub(e, *l_rhs.clone())),
                            };
                        }
                        if let Some(e) = rhs.sub_opt(l_rhs) {
                            return match l_lhs.add_opt(&e) {
                                Some(ee) => Some(ee),
                                None => Some(_add(*l_lhs.clone(), e)),
                            };
                        }
                    }
            
                    // swap nodes by sorting rule
                    if let BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow = op {
                        match rhs {
                            SymbolExpr::Binary{op: rop, lhs: _, rhs: _} => {
                                if let BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow = rop {
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
            if let SymbolExpr::Unary{op, expr} = rhs {
                if let UnaryOp::Neg = op {
                    return self.add_opt(expr);
                }
            } else if let SymbolExpr::Binary{op, lhs: r_lhs, rhs: r_rhs} = rhs {
                // recursive optimization for add and sub
                match op {
                    BinaryOp::Add => match self.sub_opt(r_lhs) {
                        // self - r.lhs - r.rhs
                        Some(rl) => match rl.sub_opt(r_rhs) {
                            Some(rr) => Some(rr),
                            None => Some(_sub(rl, *r_rhs.clone())),
                        },
                        None => match self.sub_opt(r_rhs) {
                            Some(rr) => match rr.sub_opt(r_lhs) {
                                Some(rl) => Some(rl),
                                None => Some(_sub(rr, *r_lhs.clone())),
                            },
                            None => None,
                        },
                    },
                    BinaryOp::Sub => match self.sub_opt(r_lhs) {
                        // self - r.lhs + r.rhs
                        Some(rl) => match rl.add_opt(r_rhs) {
                            Some(rr) => Some(rr),
                            None => Some(_add(rl, *r_rhs.clone())),
                        },
                        None => match self.add_opt(r_rhs) {
                            Some(rr) => match rr.sub_opt(r_lhs) {
                                Some(rl) => Some(rl),
                                None => Some(_sub(rr, *r_lhs.clone())),
                            },
                            None => None,
                        },
                    },
                    _ => None,
                };
            }

            // optimization for each type
            match self {
                SymbolExpr::Value(l) => match &rhs {
                    SymbolExpr::Value(r) => Some(SymbolExpr::Value(l - r)),
                    _ => None,
                },
                SymbolExpr::Symbol(l) => match &rhs {
                    SymbolExpr::Value(r) => Some(_add(
                        SymbolExpr::Value(-r),
                        self.clone(),
                    )),
                    SymbolExpr::Symbol(r) => {
                        if r.name == l.name {
                            Some(SymbolExpr::Value(Value::Int(0)))
                        } else if r.name < l.name {
                            Some(_add(_neg(rhs.clone()), self.clone()))
                        } else {
                            None
                        }
                    },
                    _ => None,  // TO DO add optimization for adding binary mul/div case
                },
                SymbolExpr::Unary{op, expr} => {
                    if let UnaryOp::Neg = op {
                        if let Some(e) = expr.add_opt(rhs) {
                            return match e.neg_opt() {
                                Some(ee) => Some(ee),
                                None => Some(_neg(e)),
                            };
                        }
                    }
                    if let SymbolExpr::Unary{op: rop, expr: rexpr} = rhs {
                        if op == rop {
                            let t = expr.expand() - rexpr.expand();
                            if t.is_zero() {
                                return Some(SymbolExpr::Value(Value::Int(0)));
                            }
                        }
                    }

                    // swap nodes by sorting rule
                    match rhs {
                        SymbolExpr::Binary{op: rop, lhs: _, rhs: _} => {
                            if let BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow = rop {
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
                },
                SymbolExpr::Binary{op, lhs: l_lhs, rhs: l_rhs} => {
                    if let SymbolExpr::Binary{op: rop, lhs: r_lhs, rhs: r_rhs} = rhs {
                        if op == rop {
                            if let BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow = op {
                                if let (SymbolExpr::Value(rv), SymbolExpr::Value(lv)) = (&**l_lhs, &**r_lhs) {
                                    if l_rhs.expand().to_string() == r_rhs.expand().to_string() {
                                        return match SymbolExpr::Value(rv - lv).mul_opt(l_rhs) {
                                            Some(e) => Some(e),
                                            None => Some(_mul(SymbolExpr::Value(rv - lv), *l_rhs.clone())),
                                        };
                                    }
                                }
                                if self.expand().to_string() == rhs.expand().to_string() {
                                    return Some(SymbolExpr::Value(Value::Int(0)));
                                }
                            }
                        }
                    }
                    if let BinaryOp::Add = op {
                        if let Some(e) = l_lhs.sub_opt(rhs) {
                            return match e.add_opt(l_rhs) {
                                Some(ee) => Some(ee),
                                None => Some(_add(e, *l_rhs.clone())),
                            };
                        }
                        if let Some(e) = l_rhs.sub_opt(rhs) {
                            return match l_lhs.add_opt(&e) {
                                Some(ee) => Some(ee),
                                None => Some(_add(*l_lhs.clone(), e)),
                            };
                        }
                    }
                    if let BinaryOp::Sub = op {
                        if let Some(e) = l_lhs.sub_opt(rhs) {
                            return match e.sub_opt(l_rhs) {
                                Some(ee) => Some(ee),
                                None => Some(_sub(e, *l_rhs.clone())),
                            };
                        }
                        if let Some(e) = l_rhs.add_opt(rhs) {
                            return match l_lhs.sub_opt(&e) {
                                Some(ee) => Some(ee),
                                None => Some(_sub(*l_lhs.clone(), e)),
                            };
                        }
                    }
            
                    // swap nodes by sorting rule
                    if let BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow = op {
                        match rhs {
                            SymbolExpr::Binary{op: rop, lhs: _, rhs: _} => {
                                if let BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow = rop {
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
                if let SymbolExpr::Unary{op: _, expr: _} = self {
                    return match rhs.mul_opt(self) {
                        Some(e) => Some(e),
                        None => Some(_mul(rhs.clone(), self.clone())),
                    };
                }
            }

            match self {
                SymbolExpr::Value(e) => e.mul_opt(rhs),
                SymbolExpr::Symbol(e) => e.mul_opt(rhs),
                SymbolExpr::Unary{op, expr} => match op {
                    UnaryOp::Neg => match expr.mul_opt(rhs) {
                        Some(e) => match e.neg_opt() {
                            Some(ee) => Some(ee),
                            None => Some(_neg(e)),
                        },
                        None => None,
                    },
                    UnaryOp::Abs => match rhs {
                        SymbolExpr::Unary{op: rop, expr: rexpr} => match &rop {
                            UnaryOp::Abs => match expr.mul_opt(rexpr) {
                                Some(e) => Some(SymbolExpr::Unary {
                                    op: UnaryOp::Abs,
                                    expr: Box::new(e),
                                }),
                                None => Some(SymbolExpr::Unary {
                                    op: UnaryOp::Abs,
                                    expr: Box::new(_mul(*expr.clone(), *rexpr.clone())),
                                }),
                            },
                            _ => None,
                        },
                        _ => None,
                    },
                    _ => None,
                },
                SymbolExpr::Binary{op,lhs: l_lhs, rhs: l_rhs} => {
                    if let SymbolExpr::Binary{op: rop, lhs: r_lhs, rhs: r_rhs} = rhs {
                        if let BinaryOp::Mul = &rop {
                            if let Some(e) = self.mul_opt(r_lhs) {
                                return match e.mul_opt(r_rhs) {
                                    Some(ee) => Some(ee),
                                    None => Some(_mul(e, *r_rhs.clone())),
                                };
                            }
                            if let Some(e) = self.mul_opt(r_rhs) {
                                return match e.mul_opt(r_lhs) {
                                    Some(ee) => Some(ee),
                                    None => Some(_mul(e, *r_lhs.clone())),
                                };
                            }
                        }
                        if let BinaryOp::Div = &rop {
                            if let Some(e) = self.mul_opt(r_lhs) {
                                return match e.div_opt(r_rhs) {
                                    Some(ee) => Some(ee),
                                    None => Some(_div(e, *r_rhs.clone())),
                                };
                            }
                            if let Some(e) = self.div_opt(r_rhs) {
                                return match e.mul_opt(r_lhs) {
                                    Some(ee) => Some(ee),
                                    None => Some(_mul(e, *r_lhs.clone())),
                                };
                            }
                        }
                    }
            
                    if let BinaryOp::Mul = &op {
                        if let Some(e) = l_lhs.mul_opt(rhs) {
                            return match e.mul_opt(l_rhs) {
                                Some(ee) => Some(ee),
                                None => Some(_mul(e, *l_rhs.clone())),
                            };
                        }
                        if let Some(e) = l_rhs.mul_opt(rhs) {
                            return match l_lhs.mul_opt(&e) {
                                Some(ee) => Some(ee),
                                None => Some(_mul(*l_lhs.clone(), e)),
                            };
                        }
                    } else if let BinaryOp::Div = &op {
                        if let Some(e) = l_lhs.mul_opt(rhs) {
                            return match e.div_opt(l_rhs) {
                                Some(ee) => Some(ee),
                                None => Some(_div(e, *l_rhs.clone())),
                            };
                        }
                        if let Some(e) = rhs.div_opt(l_rhs) {
                            return match l_lhs.mul_opt(&e) {
                                Some(ee) => Some(ee),
                                None => Some(_mul(*l_lhs.clone(), e)),
                            };
                        }
                    }
                    None            
                },
            }
        }
    }
    /// expand with optimization for mul operation
    fn mul_expand(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        if let SymbolExpr::Binary{op: rop, lhs: r_lhs, rhs: r_rhs} = rhs {
            if let BinaryOp::Add | BinaryOp::Sub = &rop {
                let el = match self.mul_expand(r_lhs) {
                    Some(e) => e,
                    None => match self.mul_opt(r_lhs) {
                        Some(e) => e,
                        None => _mul(self.clone(), *r_lhs.clone()),
                    },
                };
                let er = match self.mul_expand(r_rhs) {
                    Some(e) => e,
                    None => match self.mul_opt(r_rhs) {
                        Some(e) => e,
                        None => _mul(self.clone(), *r_rhs.clone()),
                    },
                };
                return match &rop {
                    BinaryOp::Sub => match el.sub_opt(&er) {
                        Some(e) => Some(e),
                        None => Some(_sub(el, er)),
                    },
                    _ => match el.add_opt(&er) {
                        Some(e) => Some(e),
                        None => Some(_add(el, er)),
                    },
                };
            }
            if let BinaryOp::Mul = &rop {
                return match self.mul_expand(r_lhs) {
                    Some(e) => match e.mul_expand(r_rhs) {
                        Some(ee) => Some(ee),
                        None => Some(_mul(e, *r_rhs.clone())),
                    },
                    None => self.mul_expand(r_rhs).map(|e| _mul(e, *r_lhs.clone())),
                };
            }
            if let BinaryOp::Div = &rop {
                return match self.mul_expand(r_lhs) {
                    Some(e) => match e.mul_expand(r_rhs) {
                        Some(ee) => Some(ee),
                        None => Some(_mul(e, *r_rhs.clone())),
                    },
                    None => self.div_expand(r_rhs).map(|e| _div(e, *r_lhs.clone())),
                };
            }
        }
        if let SymbolExpr::Unary{op: rop, expr: rexpr} = rhs {
            if let UnaryOp::Neg = &rop {
                return match self.mul_expand(rexpr) {
                    Some(e) => match e.neg_opt() {
                        Some(ee) => Some(ee),
                        None => Some(_neg(e)),
                    },
                    None => match self.mul_opt(rexpr) {
                        Some(e) => match e.neg_opt() {
                            Some(ee) => Some(ee),
                            None => Some(_neg(e)),
                        },
                        None => Some(_neg(_mul(self.clone(), *rexpr.clone()))),
                    },
                };
            }
        }

        match self {
            SymbolExpr::Unary{op, expr} => match op {
                UnaryOp::Neg => match expr.mul_expand(rhs) {
                    Some(e) => match e.neg_opt() {
                        Some(ee) => Some(ee),
                        None => Some(_neg(e)),
                    },
                    None => match expr.mul_opt(rhs) {
                        Some(e) => match e.neg_opt() {
                            Some(ee) => Some(ee),
                            None => Some(_neg(e)),
                        },
                        None => None,
                    },
                },
                _ => None,
            },    
            SymbolExpr::Binary{op, lhs: l_lhs, rhs: l_rhs} => match &op {
                BinaryOp::Add | BinaryOp::Sub => {
                    let l = match l_lhs.mul_expand(rhs) {
                        Some(e) => e,
                        None => match l_lhs.mul_opt(rhs) {
                            Some(e) => e,
                            None => _mul(*l_lhs.clone(), rhs.clone()),
                        },
                    };
                    let r = match l_rhs.mul_expand(rhs) {
                        Some(e) => e,
                        None => match l_rhs.mul_opt(rhs) {
                            Some(e) => e,
                            None => _mul(*l_rhs.clone(), rhs.clone()),
                        },
                    };
                    match &op {
                        BinaryOp::Sub => match l.sub_opt(&r) {
                            Some(e) => Some(e),
                            None => Some(_sub(l, r)),
                        },
                        _ => match l.add_opt(&r) {
                            Some(e) => Some(e),
                            None => Some(_add(l, r)),
                        },
                    }
                }
                BinaryOp::Mul => match l_lhs.mul_expand(rhs) {
                    Some(e) => match e.mul_expand(l_rhs) {
                        Some(ee) => Some(ee),
                        None => match e.mul_opt(l_rhs) {
                            Some(ee) => Some(ee),
                            None => Some(_mul(e, *l_rhs.clone())),
                        },
                    },
                    None => match l_rhs.mul_expand(rhs) {
                        Some(e) => match l_lhs.mul_expand(&e) {
                            Some(ee) => Some(ee),
                            None => match l_lhs.mul_opt(&e) {
                                Some(ee) => Some(ee),
                                None => Some(_mul(*l_lhs.clone(), e)),
                            },
                        },
                        None => None,
                    },
                },
                BinaryOp::Div => match l_lhs.div_expand(rhs) {
                    Some(e) => Some(_div(e, *l_rhs.clone())),
                    None => l_rhs.div_expand(rhs).map(|e| _div(*l_lhs.clone(), e)),
                },
                _ => None,
            },
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
                SymbolExpr::Value(e) => e.div_opt(rhs),
                SymbolExpr::Symbol(e) => e.div_opt(rhs),
                SymbolExpr::Unary{op, expr} => match op {
                    UnaryOp::Neg => match expr.div_opt(rhs) {
                        Some(e) => match e.neg_opt() {
                            Some(ee) => Some(ee),
                            None => Some(_neg(e)),
                        },
                        None => None,
                    },
                    UnaryOp::Abs => match rhs {
                        SymbolExpr::Unary{op: rop, expr: rexpr} => match &rop {
                            UnaryOp::Abs => match expr.div_opt(rexpr) {
                                Some(e) => Some(SymbolExpr::Unary {
                                    op: UnaryOp::Abs,
                                    expr: Box::new(e),
                                }),
                                None => Some(SymbolExpr::Unary {
                                    op: UnaryOp::Abs,
                                    expr: Box::new(_div(*expr.clone(), *rexpr.clone())),
                                }),
                            },
                            _ => None,
                        },
                        _ => None,
                    },
                    _ => None,
                },
                SymbolExpr::Binary{op, lhs: l_lhs, rhs: l_rhs} => {
                    if let SymbolExpr::Binary{op: rop, lhs: r_lhs, rhs: r_rhs} = rhs {
                        if let BinaryOp::Mul = &rop {
                            if let Some(e) = self.div_opt(r_lhs) {
                                return match e.div_opt(r_rhs) {
                                    Some(ee) => Some(ee),
                                    None => Some(_div(e, *r_rhs.clone())),
                                };
                            }
                            if let Some(e) = self.div_opt(r_rhs) {
                                return match e.div_opt(r_lhs) {
                                    Some(ee) => Some(ee),
                                    None => Some(_div(e, *r_lhs.clone())),
                                };
                            }
                        }
                        if let BinaryOp::Div = &rop {
                            if let Some(e) = self.mul_opt(r_rhs) {
                                return match e.div_opt(r_lhs) {
                                    Some(ee) => Some(ee),
                                    None => Some(_div(e, *r_lhs.clone())),
                                };
                            }
                            if let Some(e) = self.div_opt(r_lhs) {
                                return match e.mul_opt(r_rhs) {
                                    Some(ee) => Some(ee),
                                    None => Some(_mul(e, *r_rhs.clone())),
                                };
                            }
                        }
                    }
            
                    if let BinaryOp::Mul = &op {
                        if let Some(e) = l_lhs.div_opt(rhs) {
                            return match e.mul_opt(l_rhs) {
                                Some(ee) => Some(ee),
                                None => Some(_mul(e, *l_rhs.clone())),
                            };
                        }
                        if let Some(e) = l_rhs.div_opt(rhs) {
                            return match l_lhs.mul_opt(&e) {
                                Some(ee) => Some(ee),
                                None => Some(_mul(*l_lhs.clone(), e)),
                            };
                        }
                    } else if let BinaryOp::Div = &op {
                        if let Some(e) = l_rhs.mul_opt(rhs) {
                            return match l_lhs.div_opt(&e) {
                                Some(ee) => Some(ee),
                                None => Some(_div(*l_lhs.clone(), e)),
                            };
                        }
                        if let Some(e) = l_lhs.div_opt(rhs) {
                            return match e.div_opt(l_rhs) {
                                Some(ee) => Some(ee),
                                None => Some(_div(e, *l_rhs.clone())),
                            };
                        }
                    }
                    None
                }
            }
        }
    }

    /// expand with optimization for div operation
    fn div_expand(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match self {
            SymbolExpr::Unary{op, expr} => match op {
                UnaryOp::Neg => match expr.div_expand(rhs) {
                    Some(e) => match e.neg_opt() {
                        Some(ee) => Some(ee),
                        None => Some(_neg(e)),
                    },
                    None => match expr.div_opt(rhs) {
                        Some(e) => match e.neg_opt() {
                            Some(ee) => Some(ee),
                            None => Some(_neg(e)),
                        },
                        None => None,
                    },
                },
                _ => None,
            },
            SymbolExpr::Binary{op, lhs: l_lhs, rhs: l_rhs} => match &op {
                BinaryOp::Add | BinaryOp::Sub => {
                    let l = match l_lhs.div_expand(rhs) {
                        Some(e) => e,
                        None => match l_lhs.div_opt(rhs) {
                            Some(e) => e,
                            None => _div(*l_lhs.clone(), rhs.clone()),
                        },
                    };
                    let r = match l_rhs.div_expand(rhs) {
                        Some(e) => e,
                        None => match l_rhs.div_opt(rhs) {
                            Some(e) => e,
                            None => _div(*l_rhs.clone(), rhs.clone()),
                        },
                    };
                    match &op {
                        BinaryOp::Sub => match l.sub_opt(&r) {
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
            },
            _ => self.div_opt(rhs),
        }
    }

    /// optimization for neg
    fn neg_opt(&self) -> Option<SymbolExpr> {
        match self {
            SymbolExpr::Value(v) => Some(SymbolExpr::Value(-v)),
            SymbolExpr::Unary{op, expr} => match &op {
                UnaryOp::Neg => Some(*expr.clone()),
                _ => None,
            },
            SymbolExpr::Binary{op, lhs, rhs} => match &op {
                BinaryOp::Add => match lhs.neg_opt() {
                    Some(ln) => match rhs.neg_opt() {
                        Some(rn) => Some(_add(ln, rn)),
                        None => Some(_sub(ln, *rhs.clone())),
                    },
                    None => match rhs.neg_opt() {
                        Some(rn) => Some(_add(_neg(*lhs.clone()), rn)),
                        None => Some(_sub(_neg(*lhs.clone()), *rhs.clone())),
                    },
                },
                BinaryOp::Sub => match lhs.neg_opt() {
                    Some(ln) => Some(_add(ln, *rhs.clone())),
                    None => Some(_add(_neg(*lhs.clone()), *rhs.clone())),
                },
                BinaryOp::Mul => match lhs.neg_opt() {
                    Some(ln) => Some(_mul(ln, *rhs.clone())),
                    None => rhs.neg_opt().map(|rn| _mul(*lhs.clone(), rn)),
                },
                BinaryOp::Div => match lhs.neg_opt() {
                    Some(ln) => Some(_div(ln, *rhs.clone())),
                    None => rhs.neg_opt().map(|rn| _div(*lhs.clone(), rn)),
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
        //_add(self.clone(), rhs.clone())
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
        //_sub(self.clone(), rhs.clone())
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
        //_mul(self.clone(), rhs.clone())
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
        //_div(self.clone(), rhs.clone())
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
        //_neg(self.clone())
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
                SymbolExpr::Binary{op: _, lhs: _, rhs: _} | SymbolExpr::Unary{op: _, expr: _},
                SymbolExpr::Binary{op: _, lhs: _, rhs: _} | SymbolExpr::Unary{op: _, expr: _},
            ) => {
                let ex_lhs = self.expand();
                let ex_rhs = rexpr.expand();
                let t = &ex_lhs - &ex_rhs;
                match t {
                    SymbolExpr::Value(v) => v.is_zero(),
                    _ => false,
                }
            }
            (SymbolExpr::Binary{op: _, lhs: _, rhs: _}, _) => {
                let ex_lhs = self.expand();

                let t = &ex_lhs - rexpr;
                match t {
                    SymbolExpr::Value(v) => v.is_zero(),
                    _ => false,
                }
            }
            (_, SymbolExpr::Binary{op: _, lhs: _, rhs: _}) => {
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
                SymbolExpr::Unary{op: _, expr} => self.partial_cmp(expr),
                _ => Some(Ordering::Less),
            },
            SymbolExpr::Unary{op: _, expr} => match rhs {
                SymbolExpr::Value(_) => Some(Ordering::Greater),
                SymbolExpr::Unary{op: _, expr: rexpr} => expr.partial_cmp(rexpr),
                _ => (&**expr).partial_cmp(rhs),
            },
            SymbolExpr::Binary{op, lhs: ll, rhs: lr} => match rhs {
                SymbolExpr::Value(_) | SymbolExpr::Symbol(_) => match op {
                    BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow => Some(Ordering::Greater),
                    _ => Some(Ordering::Equal),
                },
                SymbolExpr::Unary{op: _, expr} => self.partial_cmp(expr),
                SymbolExpr::Binary{op: _, lhs: rl, rhs: rr} => {
                    let ls = match **ll {
                        SymbolExpr::Value(_) => lr.to_string(),
                        _ => ll.to_string(),
                    };
                    let rs = match **rl {
                        SymbolExpr::Value(_) => rr.to_string(),
                        _ => lr.to_string(),
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
            SymbolExpr::Unary{op, expr} => match &op {
                UnaryOp::Neg => match &**expr {
                    SymbolExpr::Value(v) => Some(_mul(
                        SymbolExpr::Value(-v),
                        SymbolExpr::Symbol(self.clone()),
                    )),
                    SymbolExpr::Symbol(s) => {
                        if s.name < self.name {
                            Some(_neg(_mul(*expr.clone(), SymbolExpr::Symbol(self.clone()))))
                        } else {
                            Some(_neg(_mul(SymbolExpr::Symbol(self.clone()), *expr.clone())))
                        }
                    }
                    SymbolExpr::Binary{op: _, lhs: _, rhs: _} => match self.mul_opt(expr) {
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
                    if *e < 0.0 && r.fract() != 0. {
                        Value::Complex(Complex64::from(e).powf(*r))
                    } else {
                        Value::Real(e.powf(*r))
                    }
                }
                Value::Int(i) => Value::Real(e.powf(*i as f64)),
                Value::Complex(r) => Value::Complex(Complex64::from(e).powc(*r)),
            },
            Value::Int(e) => match p {
                Value::Real(r) => {
                    if *e < 0 && r.fract() != 0. {
                        Value::Complex(Complex64::from(*e as f64).powf(*r))
                    } else {
                        let t = (*e as f64).powf(*r);
                        let d = t.floor() - t;
                        if (-f64::EPSILON..f64::EPSILON).contains(&d) {
                            Value::Int(t as i64)
                        } else {
                            Value::Real(t)
                        }
                    }
                }
                Value::Int(r) => {
                    if *r < 0 {
                        Value::Real((*e as f64).powf(*r as f64))
                    } else {
                        Value::Int(e.pow(*r as u32))
                    }
                }
                Value::Complex(c) => Value::Complex(Complex64::from(*e as f64).powc(*c)),
            },
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

    fn mul_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match rhs {
            SymbolExpr::Value(r) => Some(SymbolExpr::Value(self * r)),
            SymbolExpr::Unary{op, expr} => match op {
                UnaryOp::Neg => {
                    let l = SymbolExpr::Value(-self);
                    match l.mul_opt(expr) {
                        Some(e) => Some(e),
                        None => Some(_mul(l, *expr.clone())),
                    }
                }
                _ => None,
            },
            SymbolExpr::Binary{op, lhs: l, rhs: r} => match op {
                BinaryOp::Mul => match self.mul_opt(l) {
                    Some(e) => match e.mul_opt(r) {
                        Some(ee) => Some(ee),
                        None => Some(_mul(e, *r.clone())),
                    },
                    None => self.mul_opt(r).map(|e| _mul(e, *l.clone())),
                },
                BinaryOp::Div => match self.mul_opt(l) {
                    Some(e) => Some(_div(e, *r.clone())),
                    None => self.div_opt(r).map(|e| _mul(e, *l.clone())),
                },
                _ => None,
            },
            _ => None,
        }
    }

    fn div_opt(&self, rhs: &SymbolExpr) -> Option<SymbolExpr> {
        match rhs {
            SymbolExpr::Value(r) => Some(SymbolExpr::Value(self / r)),
            SymbolExpr::Unary{op, expr} => match op {
                UnaryOp::Neg => self.div_opt(expr).map(_neg),
                _ => None,
            },
            SymbolExpr::Binary{op, lhs: l, rhs: r} => match &**l {
                SymbolExpr::Value(v) => match op {
                    BinaryOp::Mul => Some(_div(SymbolExpr::Value(self / v), *r.clone())),
                    BinaryOp::Div => Some(_mul(SymbolExpr::Value(self / v), *r.clone())),
                    _ => None,
                },
                _ => match &**r {
                    SymbolExpr::Value(v) => match op {
                        BinaryOp::Mul => Some(_div(SymbolExpr::Value(self / v), *l.clone())),
                        BinaryOp::Div => Some(_div(SymbolExpr::Value(self * v), *l.clone())),
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
