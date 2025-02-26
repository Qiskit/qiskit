// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::fmt;

use hashbrown::{HashMap, HashSet};
use std::convert::From;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::symbol_expr::{SymbolExpr, Value};
use crate::symbol_parser::parse_expression;

use num_complex::Complex64;

/// Parameter Expression
#[derive(Debug)]
pub struct ParameterExpression {
    expr_: SymbolExpr,
}

impl Default for ParameterExpression {
    /// default constructor returns zero
    fn default() -> Self {
        Self {
            expr_: SymbolExpr::Value(Value::Real(0.0)),
        }
    }
}

impl fmt::Display for ParameterExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expr_)
    }
}

impl ParameterExpression {
    /// create new ParameterExpression from string expression
    pub fn new(expr: &str) -> Self {
        Self {
            expr_: parse_expression(expr),
        }
    }

    /// return number of symbols in this expression
    pub fn num_symbols(&self) -> usize {
        self.expr_.symbols().len()
    }

    /// get hashset of all the symbols used in this expression
    pub fn symbols(&self) -> HashSet<String> {
        self.expr_.symbols()
    }

    /// check if the symbol is used in this expression
    pub fn has_symbol(&self, symbol: String) -> bool {
        self.expr_.symbols().contains(&symbol)
    }

    /// bind values to symbols given by input hashmap
    pub fn bind(&self, maps: HashMap<String, Value>) -> Result<Self, &str> {
        let bound = self.expr_.bind(&maps);
        match bound {
            SymbolExpr::Value(ref v) => match v {
                Value::Real(r) => {
                    if *r == f64::INFINITY {
                        Err("zero division occurs while binding parameter")
                    } else {
                        Ok(Self { expr_: bound })
                    }
                }
                Value::Int(_) => Ok(Self { expr_: bound }),
                Value::Complex(c) => {
                    if c.re == f64::INFINITY || c.im == f64::INFINITY {
                        Err("zero division occurs while binding parameter")
                    } else if c.im < f64::EPSILON && c.im > -f64::EPSILON {
                        Ok(Self {
                            expr_: SymbolExpr::Value(Value::Real(c.re)),
                        })
                    } else {
                        Ok(Self { expr_: bound })
                    }
                }
            },
            _ => Ok(Self { expr_: bound }),
        }
    }

    /// substitute symbols to expressions (or values) given by hash map
    pub fn subs(&self, map: &HashMap<String, Self>) -> Self {
        let subs_map: HashMap<String, SymbolExpr> = map
            .iter()
            .map(|(key, val)| (key.clone(), val.expr_.clone()))
            .collect();
        ParameterExpression {
            expr_: self.expr_.subs(&subs_map),
        }
    }

    /// return floating point value if expression does not include symbols
    pub fn float(&self) -> Option<f64> {
        match self.expr_.eval(true) {
            Some(v) => match v {
                Value::Real(r) => Some(r),
                Value::Int(r) => Some(r as f64),
                Value::Complex(_) => None,
            },
            None => None,
        }
    }

    /// return integer value if expression does not include symbols
    pub fn int(&self) -> Option<i64> {
        match self.expr_.eval(true) {
            Some(v) => match v {
                Value::Real(r) => Some(r as i64),
                Value::Int(r) => Some(r),
                Value::Complex(_) => None,
            },
            None => None,
        }
    }

    /// return complex value if expression does not include symbols
    pub fn complex(&self) -> Option<Complex64> {
        match self.expr_.eval(true) {
            Some(v) => match v {
                Value::Real(r) => Some(Complex64::from(r)),
                Value::Int(r) => Some(Complex64::from(r as f64)),
                Value::Complex(c) => Some(c),
            },
            None => None,
        }
    }

    /// return conjugate of value
    pub fn conjugate(&self) -> Self {
        Self {
            expr_: self.expr_.conjugate(),
        }
    }
    /// return expression of derivative for param
    pub fn derivative(&self, param: &Self) -> Self {
        Self {
            expr_: self.expr_.derivative(&param.expr_),
        }
    }

    /// expand espression
    pub fn expand(&self) -> Self {
        Self {
            expr_: self.expr_.expand(),
        }
    }

    /// check if complex or not
    pub fn is_complex(&self) -> Option<bool> {
        self.expr_.is_complex()
    }
    /// check if floating point or not
    pub fn is_real(&self) -> Option<bool> {
        self.expr_.is_real()
    }
    /// check if integer or not
    pub fn is_int(&self) -> Option<bool> {
        self.expr_.is_int()
    }

    /// add 2 expressions
    fn add_expr(&self, rhs: &Self) -> Self {
        Self {
            expr_: &self.expr_ + &rhs.expr_,
        }
    }

    /// add other expression
    fn add_assign_expr(&mut self, rhs: &Self) {
        self.expr_ = &self.expr_ + &rhs.expr_;
    }

    /// subtract 2 expressions
    fn sub_expr(&self, rhs: &Self) -> Self {
        Self {
            expr_: &self.expr_ - &rhs.expr_,
        }
    }

    /// subtract other expression
    fn sub_assign_expr(&mut self, rhs: &Self) {
        self.expr_ = &self.expr_ - &rhs.expr_;
    }

    /// multiply 2 expressions
    fn mul_expr(&self, rhs: &Self) -> Self {
        Self {
            expr_: &self.expr_ * &rhs.expr_,
        }
    }

    /// multiply other expression
    fn mul_assign_expr(&mut self, rhs: &Self) {
        self.expr_ = &self.expr_ * &rhs.expr_;
    }

    /// divide expression
    fn div_expr(&self, rhs: &Self) -> Self {
        Self {
            expr_: &self.expr_ / &rhs.expr_,
        }
    }

    /// divide by other expression
    fn div_assign_expr(&mut self, rhs: &Self) {
        self.expr_ = &self.expr_ / &rhs.expr_;
    }

    /// sin of expression
    pub fn sin(&self) -> Self {
        Self {
            expr_: self.expr_.sin(),
        }
    }

    /// cos of expression
    pub fn cos(&self) -> Self {
        Self {
            expr_: self.expr_.cos(),
        }
    }

    /// tan of expression
    pub fn tan(&self) -> Self {
        Self {
            expr_: self.expr_.tan(),
        }
    }

    /// arcsin of expression
    pub fn arcsin(&self) -> Self {
        Self {
            expr_: self.expr_.asin(),
        }
    }

    /// arccos of expression
    pub fn arccos(&self) -> Self {
        Self {
            expr_: self.expr_.acos(),
        }
    }

    /// arctan of expression
    pub fn arctan(&self) -> Self {
        Self {
            expr_: self.expr_.atan(),
        }
    }

    /// exp of expression
    pub fn exp(&self) -> Self {
        Self {
            expr_: self.expr_.exp(),
        }
    }

    /// log of expression
    pub fn log(&self) -> Self {
        Self {
            expr_: self.expr_.log(),
        }
    }

    /// abs of expression
    pub fn abs(&self) -> Self {
        Self {
            expr_: self.expr_.abs(),
        }
    }

    /// pow of expression
    pub fn pow<T: Into<Self>>(&self, prm: T) -> Self {
        let t: ParameterExpression = prm.into();
        Self {
            expr_: self.expr_.pow(&t.expr_),
        }
    }
}

impl Clone for ParameterExpression {
    fn clone(&self) -> Self {
        Self {
            expr_: self.expr_.clone(),
        }
    }
}

impl PartialEq for ParameterExpression {
    fn eq(&self, rprm: &Self) -> bool {
        self.expr_ == rprm.expr_
    }
}

// =============================
// Make from Rust native types
// =============================

impl From<i32> for ParameterExpression {
    fn from(v: i32) -> Self {
        Self {
            expr_: SymbolExpr::Value(Value::Real(v as f64)),
        }
    }
}
impl From<i64> for ParameterExpression {
    fn from(v: i64) -> Self {
        Self {
            expr_: SymbolExpr::Value(Value::Real(v as f64)),
        }
    }
}

impl From<u32> for ParameterExpression {
    fn from(v: u32) -> Self {
        Self {
            expr_: SymbolExpr::Value(Value::Real(v as f64)),
        }
    }
}

impl From<f64> for ParameterExpression {
    fn from(v: f64) -> Self {
        Self {
            expr_: SymbolExpr::Value(Value::Real(v)),
        }
    }
}

impl From<Complex64> for ParameterExpression {
    fn from(v: Complex64) -> Self {
        Self {
            expr_: SymbolExpr::Value(Value::Complex(v)),
        }
    }
}

impl From<&str> for ParameterExpression {
    fn from(v: &str) -> Self {
        Self::new(v)
    }
}

impl From<&SymbolExpr> for ParameterExpression {
    fn from(expr: &SymbolExpr) -> Self {
        Self {
            expr_: expr.clone(),
        }
    }
}

// =============================
// Unary operations
// =============================
impl Neg for ParameterExpression {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            expr_: -&self.expr_,
        }
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
                self.add_expr(&other.into())
            }
        }

        impl AddAssign<$t> for ParameterExpression {
            #[inline]
            #[track_caller]
            fn add_assign(&mut self, other: $t) {
                self.add_assign_expr(&other.into())
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
                self.sub_expr(&other.into())
            }
        }

        impl SubAssign<$t> for ParameterExpression {
            #[inline]
            #[track_caller]
            fn sub_assign(&mut self, other: $t) {
                self.sub_assign_expr(&other.into())
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
                self.mul_expr(&other.into())
            }
        }

        impl MulAssign<$t> for ParameterExpression {
            #[inline]
            #[track_caller]
            fn mul_assign(&mut self, other: $t) {
                self.mul_assign_expr(&other.into())
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
                self.div_expr(&other.into())
            }
        }

        impl DivAssign<$t> for ParameterExpression {
            #[inline]
            #[track_caller]
            fn div_assign(&mut self, other: $t) {
                self.div_assign_expr(&other.into())
            }
        }
    )*)
}

div_impl_expr! {f64 i32 u32 ParameterExpression}
