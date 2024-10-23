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

// ParameterExpression class using symengine C wrapper interface

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)]
#![allow(dead_code)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::convert::From;
use std::ffi::{CStr, CString};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign, Neg};

#[derive(Debug, PartialEq)]
pub struct ParameterExpression {
    pub expr_: *mut basic_struct,
}

impl Drop for ParameterExpression {
    fn drop(&mut self) {
        unsafe {
            basic_free_heap(self.expr_);
        }
    }
}

impl Clone for ParameterExpression {
    fn clone(&self) -> Self {
        let _ret = Self::default();
        unsafe {
            basic_assign(_ret.expr_, self.expr_);
        }
        _ret
    }
}

impl ParameterExpression {
    pub fn default() -> Self {
        unsafe {
            Self {
                expr_: basic_new_heap(),
            }
        }
    }

    pub fn new(expr: &str) -> Self {
        let cexpr = CString::new(expr).unwrap();
        let _ret = Self::default();
        unsafe {
            basic_parse(_ret.expr_, cexpr.as_ptr());
        }
        _ret
    }

    pub fn from_ptr(expr: *mut basic_struct) -> Self {
        let _ret = Self::default();
        unsafe {
            basic_assign(_ret.expr_, expr);
        }
        _ret
    }

    pub fn as_str(&self) -> &str {
        let _ret = unsafe { CStr::from_ptr(basic_str(self.expr_)) };
        _ret.to_str().unwrap()
    }

    pub fn bind(self, param: &str, value: f64) {
        let cexpr = CString::new(param).unwrap();
        unsafe {
            let mut key = basic_new_heap();
            let mut val = basic_new_heap();
            let mut map = mapbasicbasic_new();
            basic_parse(key, cexpr.as_ptr());
            real_double_set_d(val, value);
            mapbasicbasic_insert(map, key, val);
            basic_subs(self.expr_, self.expr_, map);
            mapbasicbasic_free(map);
            basic_free_heap(key);
            basic_free_heap(val);
        }
    }

    fn add_expr(self, rhs: Self) -> Self {
        let _ret = Self::default();
        unsafe {
            basic_add(_ret.expr_, self.expr_, rhs.expr_);
        }
        _ret
    }

    fn add_assign_expr(&mut self, rhs: Self) {
        unsafe {
            basic_add(self.expr_, self.expr_, rhs.expr_);
        }
    }

    fn sub_expr(self, rhs: Self) -> Self {
        let _ret = Self::default();
        unsafe {
            basic_sub(_ret.expr_, self.expr_, rhs.expr_);
        }
        _ret
    }

    fn sub_assign_expr(&mut self, rhs: Self) {
        unsafe {
            basic_sub(self.expr_, self.expr_, rhs.expr_);
        }
    }

    fn mul_expr(self, rhs: Self) -> Self {
        let _ret = Self::default();
        unsafe {
            basic_mul(_ret.expr_, self.expr_, rhs.expr_);
        }
        _ret
    }

    fn mul_assign_expr(&mut self, rhs: Self) {
        unsafe {
            basic_mul(self.expr_, self.expr_, rhs.expr_);
        }
    }

    fn div_expr(self, rhs: Self) -> Self {
        let _ret = Self::default();
        unsafe {
            basic_div(_ret.expr_, self.expr_, rhs.expr_);
        }
        _ret
    }

    fn div_assign_expr(&mut self, rhs: Self) {
        unsafe {
            basic_div(self.expr_, self.expr_, rhs.expr_);
        }
    }

    fn sin(self) -> Self {
        let _ret = Self::default();
        unsafe {
            basic_sin(_ret.expr_, self.expr_);
        }
        _ret
    }

    fn cos(self) -> Self {
        let _ret = Self::default();
        unsafe {
            basic_cos(_ret.expr_, self.expr_);
        }
        _ret
    }

    fn tan(self) -> Self {
        let _ret = Self::default();
        unsafe {
            basic_tan(_ret.expr_, self.expr_);
        }
        _ret
    }

    fn arcsin(self) -> Self {
        let _ret = Self::default();
        unsafe {
            basic_asin(_ret.expr_, self.expr_);
        }
        _ret
    }

    fn arccos(self) -> Self {
        let _ret = Self::default();
        unsafe {
            basic_acos(_ret.expr_, self.expr_);
        }
        _ret
    }

    fn arctan(self) -> Self {
        let _ret = Self::default();
        unsafe {
            basic_atan(_ret.expr_, self.expr_);
        }
        _ret
    }

    fn exp(self) -> Self {
        let _ret = Self::default();
        unsafe {
            basic_exp(_ret.expr_, self.expr_);
        }
        _ret
    }

    fn log(self) -> Self {
        let _ret = Self::default();
        unsafe {
            basic_log(_ret.expr_, self.expr_);
        }
        _ret
    }

    fn abs(self) -> Self {
        let _ret = Self::default();
        unsafe {
            basic_abs(_ret.expr_, self.expr_);
        }
        _ret
    }

    fn pow<T: Into<Self>>(self, prm: T) -> Self {
        let _ret = Self::default();
        unsafe {
            basic_pow(_ret.expr_, self.expr_, prm.into().expr_);
        }
        _ret
    }  
}

// =============================
// Make from Rust native types
// =============================
impl From<i64> for ParameterExpression {
    fn from(v: i64) -> Self {
        let _ret = Self::default();
        unsafe {
            integer_set_si(_ret.expr_, v);
        }
        _ret
    }
}
impl From<u64> for ParameterExpression {
    fn from(v: u64) -> Self {
        let _ret = Self::default();
        unsafe {
            integer_set_ui(_ret.expr_, v);
        }
        _ret
    }
}

impl From<i32> for ParameterExpression {
    fn from(v: i32) -> Self {
        let _ret = Self::default();
        unsafe {
            integer_set_si(_ret.expr_, v as i64);
        }
        _ret
    }
}
impl From<u32> for ParameterExpression {
    fn from(v: u32) -> Self {
        let _ret = Self::default();
        unsafe {
            integer_set_ui(_ret.expr_, v as u64);
        }
        _ret
    }
}

impl From<f64> for ParameterExpression {
    fn from(v: f64) -> Self {
        let _ret = Self::default();
        unsafe {
            real_double_set_d(_ret.expr_, v);
        }
        _ret
    }
}

// =============================
// Unary operations
// =============================
impl Neg for ParameterExpression {
    type Output = Self;

    fn neg(self) -> Self::Output {
        unsafe {
            basic_neg(self.expr_, self.expr_);
        }
        self
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
                self.add_expr(other.into())
            }
        }

        impl AddAssign<$t> for ParameterExpression {
            #[inline]
            #[track_caller]
            fn add_assign(&mut self, other: $t) {
                self.add_assign_expr(other.into())
            }
        }
    )*)
}

add_impl_expr! {f64 i32 u32 i64 u64 ParameterExpression}

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
                self.sub_expr(other.into())
            }
        }

        impl SubAssign<$t> for ParameterExpression {
            #[inline]
            #[track_caller]
            fn sub_assign(&mut self, other: $t) {
                self.sub_assign_expr(other.into())
            }
        }
    )*)
}

sub_impl_expr! {f64 i32 u32 i64 u64 ParameterExpression}

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
                self.mul_expr(other.into())
            }
        }

        impl MulAssign<$t> for ParameterExpression {
            #[inline]
            #[track_caller]
            fn mul_assign(&mut self, other: $t) {
                self.mul_assign_expr(other.into())
            }
        }
    )*)
}

mul_impl_expr! {f64 i32 u32 i64 u64 ParameterExpression}

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
                self.div_expr(other.into())
            }
        }

        impl DivAssign<$t> for ParameterExpression {
            #[inline]
            #[track_caller]
            fn div_assign(&mut self, other: $t) {
                self.div_assign_expr(other.into())
            }
        }
    )*)
}

div_impl_expr! {f64 i32 u32 i64 u64 ParameterExpression}




