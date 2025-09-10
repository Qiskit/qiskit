// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use hashbrown::HashMap;
use num_complex::Complex64;
use std::ffi::{c_char, CStr, CString};
use std::ptr;
use std::sync::Arc;

use crate::pointers::{const_ptr_as_ref, mut_ptr_as_ref};

use qiskit_circuit::operations::Param;
use qiskit_circuit::parameter::parameter_expression::{ParameterError, ParameterExpression};
use qiskit_circuit::parameter::symbol_expr::{Symbol, SymbolExpr, Value};

/// @ingroup QkParam
/// Construct a new Parameter with a name of symbol
///
/// @param name The name of symbol.
///
/// @return A pointer to the created QkParam.
///
/// # Example
///
///     QkParam *a = qk_param_new("a");
///
/// # Safety
///
/// name must be a pointer to valid C string to name of symbol
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_param_new(name: *const c_char) -> *mut Param {
    let name = unsafe { CStr::from_ptr(name).to_str().unwrap() };
    let symbol = Symbol::new(name, None, None);
    let expr = ParameterExpression::from_symbol(symbol);
    let param = Param::ParameterExpression(Arc::new(expr));
    Box::into_raw(Box::new(param))
}

/// @ingroup QkParam
/// Free the QkParam.
///
/// @param expr A pointer to the QkParam to free.
///
/// # Example
///
///     QkParam *a = qk_param_new("a");
///     qk_param_free(a);
///
/// # Safety
///
/// Behavior is undefined if ``expr`` is not either null or a valid pointer to a
/// [ParameterExpression].
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_param_free(expr: *mut Param) {
    if !expr.is_null() {
        if !expr.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(expr);
        }
    }
}

/// @ingroup QkParam
/// Construct a new Param from value
///
/// @param value real number for a new QkParam
///
/// @return A pointer to the created QkParam.
///
/// # Example
///
///     QkParam *r = qk_param_from_value(2.5);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_param_from_double(value: f64) -> *mut Param {
    let value = Param::Float(value);
    Box::into_raw(Box::new(value))
}

/// @ingroup QkParam
/// Construct a new QkParam from complex number
///
/// @param value complex number for a new QkParam
///
/// @return A pointer to the created QkParam.
///
/// # Example
///
///     QkParam *c = qk_param_from_complex((1.0,1.0));
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_param_from_complex(value: Complex64) -> *mut Param {
    let value = SymbolExpr::Value(Value::Complex(value));
    let expr = ParameterExpression::from_symbol_expr(value);
    let param = Param::ParameterExpression(Arc::new(expr));
    Box::into_raw(Box::new(param))
}

/// @ingroup QkParam
/// Construct a new QkParam from rational number
///   TODO : this function needs PR #14686
/// @param numerator numerator of rational number
/// @param denominator denominator of rational number
///
/// @return A pointer to the created QkParam.
///
/// # Example
///
///     QkParam *c = qk_param_from_rational(1, 2);
///
//#[no_mangle]
//#[cfg(feature = "cbinding")]
//pub extern "C" fn qk_param_from_rational(numerator: i64, denominator: i64) -> *mut Param {
//    let value = SymbolExpr::Value(Value::Rational{numerator, denominator});
//    let expr = ParameterExpression::from_symbol_expr(value);
//    let param = Param::ParameterExpression(Arc::new(expr));
//    Box::into_raw(Box::new(param))
//}

/// @ingroup QkParam
/// copy QkParam
///
/// @param name The name of symbol.
///
/// @return A pointer to the created QkParam.
///
/// # Example
///
///     QkParam *b = qk_param_copy(a);
///
/// # Safety
///
/// expr should be valid pointer to parameter to be copied
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_param_copy(expr: *const Param) -> *mut Param {
    let expr = unsafe { const_ptr_as_ref(expr) };
    Box::into_raw(Box::new(expr.clone()))
}

/// @ingroup QkParam
/// get string expression.
///
/// @param expr A pointer to the Param
///
/// # Example
///
///     QkParam *a = qk_param_new("a");
///     char* str = qk_param_to_string(a);
///     printf(str);
///
/// # Safety
///
/// expr should be valid pointer to parameter
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_param_to_string(expr: *const Param) -> *mut c_char {
    let expr = unsafe { const_ptr_as_ref(expr) };
    let str = match expr {
        Param::ParameterExpression(expr) => expr.to_string(),
        Param::Float(f) => f.to_string(),
        Param::Obj(o) => o.to_string(),
    };
    let out = CString::new(str.to_string()).unwrap();
    out.into_raw()
}

/// @ingroup QkParam
/// add 2 expressions.
///
/// @param lhs A pointer to the left hand side QkParam.
/// @param rhs A pointer to the right hand side QkParam.
///
/// @return A pointer to the new QkParam.
///
/// # Example
///
///     QkParam *a = qk_param_new("a");
///     QkParam *b = qk_param_new("b");
///     QkParam *c = qk_param_add(a, b);
///
/// # Safety
///
/// lhs and rhs should be valid pointer to parameter
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_param_add(lhs: *const Param, rhs: *const Param) -> *mut Param {
    let lhs = unsafe { const_ptr_as_ref(lhs) };
    let rhs = unsafe { const_ptr_as_ref(rhs) };
    let out = match (lhs, rhs) {
        (Param::ParameterExpression(lhs), Param::ParameterExpression(rhs)) => lhs.add(rhs),
        (Param::ParameterExpression(lhs), Param::Float(rhs)) => {
            lhs.add(&ParameterExpression::from_f64(*rhs))
        }
        (Param::Float(lhs), Param::ParameterExpression(rhs)) => {
            ParameterExpression::from_f64(*lhs).add(rhs)
        }
        // PyObject should be supported in C-API ?
        (_, _) => Err(ParameterError::OperatorNotSupported(
            "add for PyObject".to_string(),
        )),
    };
    match out {
        Ok(out) => Box::into_raw(Box::new(Param::ParameterExpression(Arc::new(out)))),
        Err(_) => ptr::null_mut(),
    }
}

/// @ingroup QkParam
/// sub 2 expressions.
///
/// @param lhs A pointer to the left hand side QkParam.
/// @param rhs A pointer to the right hand side QkParam.
///
/// @return A pointer to the new QkParam.
///
/// # Example
///
///     QkParam *a = qk_param_new("a");
///     QkParam *b = qk_param_new("b");
///     QkParam *c = qk_param_sub(a, b);
///
/// # Safety
///
/// lhs and rhs should be valid pointer to parameter
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_param_sub(lhs: *const Param, rhs: *const Param) -> *mut Param {
    let lhs = unsafe { const_ptr_as_ref(lhs) };
    let rhs = unsafe { const_ptr_as_ref(rhs) };
    let out = match (lhs, rhs) {
        (Param::ParameterExpression(lhs), Param::ParameterExpression(rhs)) => lhs.sub(rhs),
        (Param::ParameterExpression(lhs), Param::Float(rhs)) => {
            lhs.sub(&ParameterExpression::from_f64(*rhs))
        }
        (Param::Float(lhs), Param::ParameterExpression(rhs)) => {
            ParameterExpression::from_f64(*lhs).sub(rhs)
        }
        // PyObject should be supported in C-API ?
        (_, _) => Err(ParameterError::OperatorNotSupported(
            "sub for PyObject".to_string(),
        )),
    };
    match out {
        Ok(out) => Box::into_raw(Box::new(Param::ParameterExpression(Arc::new(out)))),
        Err(_) => ptr::null_mut(),
    }
}

/// @ingroup QkParam
/// mul 2 expressions.
///
/// @param lhs A pointer to the left hand side QkParam.
/// @param rhs A pointer to the right hand side QkParam.
///
/// @return A pointer to the new QkParam.
///
/// # Example
///
///     QkParam *a = qk_param_new("a");
///     QkParam *b = qk_param_new("b");
///     QkParam *c = qk_param_mul(a, b);
///
/// # Safety
///
/// lhs and rhs should be valid pointer to parameter
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_param_mul(lhs: *const Param, rhs: *const Param) -> *mut Param {
    let lhs = unsafe { const_ptr_as_ref(lhs) };
    let rhs = unsafe { const_ptr_as_ref(rhs) };
    let out = match (lhs, rhs) {
        (Param::ParameterExpression(lhs), Param::ParameterExpression(rhs)) => lhs.mul(rhs),
        (Param::ParameterExpression(lhs), Param::Float(rhs)) => {
            lhs.mul(&ParameterExpression::from_f64(*rhs))
        }
        (Param::Float(lhs), Param::ParameterExpression(rhs)) => {
            ParameterExpression::from_f64(*lhs).mul(rhs)
        }
        // PyObject should be supported in C-API ?
        (_, _) => Err(ParameterError::OperatorNotSupported(
            "mul for PyObject".to_string(),
        )),
    };
    match out {
        Ok(out) => Box::into_raw(Box::new(Param::ParameterExpression(Arc::new(out)))),
        Err(_) => ptr::null_mut(),
    }
}

/// @ingroup QkParam
/// div 2 expressions.
///
/// @param lhs A pointer to the left hand side QkParam.
/// @param rhs A pointer to the right hand side QkParam.
///
/// @return A pointer to the new QkParam.
///
/// # Example
///
///     QkParam *a = qk_param_new("a");
///     QkParam *b = qk_param_new("b");
///     QkParam *c = qk_param_div(a, b);
///
/// # Safety
///
/// lhs and rhs should be valid pointer to parameter
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_param_div(lhs: *const Param, rhs: *const Param) -> *mut Param {
    let lhs = unsafe { const_ptr_as_ref(lhs) };
    let rhs = unsafe { const_ptr_as_ref(rhs) };
    let out = match (lhs, rhs) {
        (Param::ParameterExpression(lhs), Param::ParameterExpression(rhs)) => lhs.div(rhs),
        (Param::ParameterExpression(lhs), Param::Float(rhs)) => {
            lhs.div(&ParameterExpression::from_f64(*rhs))
        }
        (Param::Float(lhs), Param::ParameterExpression(rhs)) => {
            ParameterExpression::from_f64(*lhs).div(rhs)
        }
        // PyObject should be supported in C-API ?
        (_, _) => Err(ParameterError::OperatorNotSupported(
            "sub for PyObject".to_string(),
        )),
    };
    match out {
        Ok(out) => Box::into_raw(Box::new(Param::ParameterExpression(Arc::new(out)))),
        Err(_) => ptr::null_mut(),
    }
}

/// @ingroup QkParam
/// calculate pow of expressions.
///
/// @param base A pointer to the base QkParam.
/// @param exp A pointer to the exponent QkParam.
///
/// @return A pointer to the new QkParam.
///
/// # Example
///
///     QkParam *a = qk_param_new("a");
///     QkParam *b = qk_param_new("b");
///     QkParam *c = qk_param_pow(a, b);
///
/// # Safety
///
/// lhs and rhs should be valid pointer to parameter
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_param_pow(lhs: *const Param, rhs: *const Param) -> *mut Param {
    let lhs = unsafe { const_ptr_as_ref(lhs) };
    let rhs = unsafe { const_ptr_as_ref(rhs) };
    let out = match (lhs, rhs) {
        (Param::ParameterExpression(lhs), Param::ParameterExpression(rhs)) => lhs.pow(rhs),
        (Param::ParameterExpression(lhs), Param::Float(rhs)) => {
            lhs.pow(&ParameterExpression::from_f64(*rhs))
        }
        (Param::Float(lhs), Param::ParameterExpression(rhs)) => {
            ParameterExpression::from_f64(*lhs).pow(rhs)
        }
        // PyObject should be supported in C-API ?
        (_, _) => Err(ParameterError::OperatorNotSupported(
            "pow for PyObject".to_string(),
        )),
    };
    match out {
        Ok(out) => Box::into_raw(Box::new(Param::ParameterExpression(Arc::new(out)))),
        Err(_) => ptr::null_mut(),
    }
}

/// @ingroup QkParam
/// sin of expressions.
///
/// @param expr A pointer to the QkParam.
///
/// @return A pointer to the new QkParam.
///
/// # Example
///
///     QkParam *a = qk_param_new("a");
///     QkParam *out = qk_param_sin(a);
///
/// # Safety
///
/// expr should be valid pointer to parameter
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_param_sin(expr: *const Param) -> *mut Param {
    let expr = unsafe { const_ptr_as_ref(expr) };
    match expr {
        Param::ParameterExpression(expr) => {
            Box::into_raw(Box::new(Param::ParameterExpression(Arc::new(expr.sin()))))
        }
        Param::Float(f) => Box::into_raw(Box::new(Param::Float(f.sin()))),
        // PyObject should be supported in C-API ?
        _ => ptr::null_mut(),
    }
}

/// @ingroup QkParam
/// cos of expressions.
///
/// @param expr A pointer to the QkParam.
///
/// @return A pointer to the new QkParam.
///
/// # Example
///
///     QkParam *a = qk_param_new("a");
///     QkParam *out = qk_param_cos(a);
///
/// # Safety
///
/// expr should be valid pointer to parameter
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_param_cos(expr: *const Param) -> *mut Param {
    let expr = unsafe { const_ptr_as_ref(expr) };
    match expr {
        Param::ParameterExpression(expr) => {
            Box::into_raw(Box::new(Param::ParameterExpression(Arc::new(expr.cos()))))
        }
        Param::Float(f) => Box::into_raw(Box::new(Param::Float(f.cos()))),
        // PyObject should be supported in C-API ?
        _ => ptr::null_mut(),
    }
}

/// @ingroup QkParam
/// tan of expressions.
///
/// @param expr A pointer to the QkParam.
///
/// @return A pointer to the new QkParam.
///
/// # Example
///
///     QkParam *a = qk_param_new("a");
///     QkParam *out = qk_param_tan(a);
///
/// # Safety
///
/// expr should be valid pointer to parameter
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_param_tan(expr: *const Param) -> *mut Param {
    let expr = unsafe { const_ptr_as_ref(expr) };
    match expr {
        Param::ParameterExpression(expr) => {
            Box::into_raw(Box::new(Param::ParameterExpression(Arc::new(expr.tan()))))
        }
        Param::Float(f) => Box::into_raw(Box::new(Param::Float(f.tan()))),
        // PyObject should be supported in C-API ?
        _ => ptr::null_mut(),
    }
}

/// @ingroup QkParam
/// arcsin of expressions.
///
/// @param expr A pointer to the QkParam.
///
/// @return A pointer to the new QkParam.
///
/// # Example
///
///     QkParam *a = qk_param_new("a");
///     QkParam *out = qk_param_asin(a);
///
/// # Safety
///
/// expr should be valid pointer to parameter
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_param_asin(expr: *const Param) -> *mut Param {
    let expr = unsafe { const_ptr_as_ref(expr) };
    match expr {
        Param::ParameterExpression(expr) => {
            Box::into_raw(Box::new(Param::ParameterExpression(Arc::new(expr.asin()))))
        }
        Param::Float(f) => Box::into_raw(Box::new(Param::Float(f.asin()))),
        // PyObject should be supported in C-API ?
        _ => ptr::null_mut(),
    }
}

/// @ingroup QkParam
/// arccos of expressions.
///
/// @param expr A pointer to the QkParam.
///
/// @return A pointer to the new QkParam.
///
/// # Example
///
///     QkParam *a = qk_param_new("a");
///     QkParam *out = qk_param_acos(a);
///
/// # Safety
///
/// expr should be valid pointer to parameter
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_param_acos(expr: *const Param) -> *mut Param {
    let expr = unsafe { const_ptr_as_ref(expr) };
    match expr {
        Param::ParameterExpression(expr) => {
            Box::into_raw(Box::new(Param::ParameterExpression(Arc::new(expr.acos()))))
        }
        Param::Float(f) => Box::into_raw(Box::new(Param::Float(f.acos()))),
        // PyObject should be supported in C-API ?
        _ => ptr::null_mut(),
    }
}

/// @ingroup QkParam
/// arctan of expressions.
///
/// @param expr A pointer to the QkParam.
///
/// @return A pointer to the new QkParam.
///
/// # Example
///
///     QkParam *a = qk_param_new("a");
///     QkParam *out = qk_param_atan(a);
///
/// # Safety
///
/// expr should be valid pointer to parameter
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_param_atan(expr: *const Param) -> *mut Param {
    let expr = unsafe { const_ptr_as_ref(expr) };
    match expr {
        Param::ParameterExpression(expr) => {
            Box::into_raw(Box::new(Param::ParameterExpression(Arc::new(expr.atan()))))
        }
        Param::Float(f) => Box::into_raw(Box::new(Param::Float(f.atan()))),
        // PyObject should be supported in C-API ?
        _ => ptr::null_mut(),
    }
}

/// @ingroup QkParam
/// log of expressions.
///
/// @param expr A pointer to the QkParam.
///
/// @return A pointer to the new QkParam.
///
/// # Example
///
///     QkParam *a = qk_param_new("a");
///     QkParam *out = qk_param_log(a);
///
/// # Safety
///
/// expr should be valid pointer to parameter
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_param_log(expr: *const Param) -> *mut Param {
    let expr = unsafe { const_ptr_as_ref(expr) };
    match expr {
        Param::ParameterExpression(expr) => {
            Box::into_raw(Box::new(Param::ParameterExpression(Arc::new(expr.log()))))
        }
        Param::Float(f) => Box::into_raw(Box::new(Param::Float(f.ln()))),
        // PyObject should be supported in C-API ?
        _ => ptr::null_mut(),
    }
}

/// @ingroup QkParam
/// exp of expressions.
///
/// @param expr A pointer to the QkParam.
///
/// @return A pointer to the new QkParam.
///
/// # Example
///
///     QkParam *a = qk_param_new("a");
///     QkParam *out = qk_param_exp(a);
///
/// # Safety
///
/// expr should be valid pointer to parameter
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_param_exp(expr: *const Param) -> *mut Param {
    let expr = unsafe { const_ptr_as_ref(expr) };
    match expr {
        Param::ParameterExpression(expr) => {
            Box::into_raw(Box::new(Param::ParameterExpression(Arc::new(expr.exp()))))
        }
        Param::Float(f) => Box::into_raw(Box::new(Param::Float(f.exp()))),
        // PyObject should be supported in C-API ?
        _ => ptr::null_mut(),
    }
}

/// @ingroup QkParam
/// abs of expressions.
///
/// @param expr A pointer to the QkParam.
///
/// @return A pointer to the new QkParam.
///
/// # Example
///
///     QkParam *a = qk_param_new("a");
///     QkParam *out = qk_param_abs(a);
///
/// # Safety
///
/// expr should be valid pointer to parameter
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_param_abs(expr: *const Param) -> *mut Param {
    let expr = unsafe { const_ptr_as_ref(expr) };
    match expr {
        Param::ParameterExpression(expr) => {
            Box::into_raw(Box::new(Param::ParameterExpression(Arc::new(expr.abs()))))
        }
        Param::Float(f) => Box::into_raw(Box::new(Param::Float(f.abs()))),
        // PyObject should be supported in C-API ?
        _ => ptr::null_mut(),
    }
}

/// @ingroup QkParam
/// sign of expressions.
///
/// @param expr A pointer to the QkParam.
///
/// @return A pointer to the new QkParam.
///
/// # Example
///
///     QkParam *a = qk_param_new("a");
///     QkParam *out = qk_param_sign(a);
///
/// # Safety
///
/// expr should be valid pointer to parameter
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_param_sign(expr: *const Param) -> *mut Param {
    let expr = unsafe { const_ptr_as_ref(expr) };
    match expr {
        Param::ParameterExpression(expr) => {
            Box::into_raw(Box::new(Param::ParameterExpression(Arc::new(expr.sign()))))
        }
        Param::Float(f) => {
            if *f < 0. {
                Box::into_raw(Box::new(Param::Float(-1.)))
            } else if *f > 0. {
                Box::into_raw(Box::new(Param::Float(1.)))
            } else {
                Box::into_raw(Box::new(Param::Float(0.)))
            }
        }
        // PyObject should be supported in C-API ?
        _ => ptr::null_mut(),
    }
}

/// @ingroup QkParam
/// neg of expressions.
///
/// @param expr A pointer to the QkParam.
///
/// @return A pointer to the new QkParam.
///
/// # Example
///
///     QkParam *a = qk_param_new("a");
///     QkParam *out = qk_param_neg(a);
///
/// # Safety
///
/// expr should be valid pointer to parameter
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_param_neg(expr: *const Param) -> *mut Param {
    let expr = unsafe { const_ptr_as_ref(expr) };
    match expr {
        Param::ParameterExpression(expr) => {
            Box::into_raw(Box::new(Param::ParameterExpression(Arc::new(expr.neg()))))
        }
        Param::Float(f) => Box::into_raw(Box::new(Param::Float(-f))),
        // PyObject should be supported in C-API ?
        _ => ptr::null_mut(),
    }
}

/// @ingroup QkParam
/// conjugate of expressions.
///
/// @param expr A pointer to the QkParam.
///
/// @return A pointer to the new QkParam.
///
/// # Example
///
///     QkParam *a = qk_param_new("a");
///     QkParam *c = qk_param_from_complex((0.0, 1.0));
///     QkParam *b = qk_param_add(a, c);
///     QkParam *out = qk_param_conj(b);
///
/// # Safety
///
/// expr should be valid pointer to parameter
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_param_conj(expr: *const Param) -> *mut Param {
    let expr = unsafe { const_ptr_as_ref(expr) };
    match expr {
        Param::ParameterExpression(expr) => Box::into_raw(Box::new(Param::ParameterExpression(
            Arc::new(expr.conjugate()),
        ))),
        Param::Float(_) => Box::into_raw(Box::new(Param::Float(0.))),
        // PyObject should be supported in C-API ?
        _ => ptr::null_mut(),
    }
}

/// @ingroup QkParam
/// compare 2 expressions.
///
/// @param lhs A pointer to the left hand side QkParam.
/// @param rhs A pointer to the right hand side QkParam.
///
/// @return true if 2 expressions are equal.
///
/// # Example
///
///     QkParam *a = qk_param_new("a");
///     QkParam *one = qk_param_from_value(1.0);
///     QkParam *mone = qk_param_fromn_value(-1.0);
///     QkParam *x = qk_param_add(a, one);
///     QkParam *y = qk_param_sub(a, mone);
///     QkParam *c = qk_param_compare_eq(x, y);
///
/// # Safety
///
/// lhs and rhs should be valid pointer to parameter
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_param_compare_eq(lhs: *const Param, rhs: *const Param) -> bool {
    let lhs = unsafe { const_ptr_as_ref(lhs) };
    let rhs = unsafe { const_ptr_as_ref(rhs) };
    lhs.eq(rhs).unwrap()
}

/// @ingroup QkParam
/// bind parameters.
///
/// @param expr A pointer to the QkParam
/// @param keys An array of pointer to the QkParam.
/// @param values An array of values to be bound.
/// @param num number of pairs of (parameter, value)
///
/// @return A pointer to the new QkParam.
///
/// # Example
///
///     QkParam *a = qk_param_new("a");
///     QkParam *b = qk_param_new("b");
///     QkParam *c = qk_param_add(a, b);
///     QkParam *t = qk_param_bind(c, {b}, {1.5}, 1);
///
/// # Safety
///
/// expr should be valid pointer to parameter
/// keys and values should be valid pointer to array in C
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_param_bind(
    expr: *const Param,
    keys: *const *const Param,
    values: *const f64,
    num: usize,
) -> *mut Param {
    let expr = unsafe { const_ptr_as_ref(expr) };
    if let Param::ParameterExpression(expr) = expr {
        let keys = unsafe { std::slice::from_raw_parts(keys, num).iter().map(|k| const_ptr_as_ref(*k)) };
        let values = unsafe { std::slice::from_raw_parts(values, num) };

        let mut map: HashMap<Symbol, Value> = HashMap::new();
        keys.zip(values.iter()).for_each(|m| {
            if let Param::ParameterExpression(e) = m.0 {
                if let Ok(symbol) = e.try_to_symbol() {
                    map.insert(symbol, Value::Real(*m.1));
                }
            }
        });
        let map_ref: HashMap<&Symbol, Value> = map.iter().map(|t| (t.0, *t.1)).collect();
        let out = expr.bind(&map_ref, false);
        match out {
            Ok(out) => Box::into_raw(Box::new(Param::ParameterExpression(Arc::new(out)))),
            Err(_) => ptr::null_mut(),
        }
    } else {
        // return copy of input parameter
        Box::into_raw(Box::new(expr.clone()))
    }
}

/// @ingroup QkParam
/// substitute parameters.
///
/// @param expr A pointer to the QkParam
/// @param keys An array of pointer to the QkParam.
/// @param subs An array of pointer to the new QkParam.
/// @param num number of pairs of (parameter, subs)
///
/// @return A pointer to the new QkParam.
///
/// # Example
///
///     QkParam *a = qk_param_new("a");
///     QkParam *b = qk_param_new("b");
///     QkParam *c = qk_param_new("c");
///     QkParam *d = qk_param_add(a, b);
///     QkParam *t = qk_param_subs(d, {b}, {c}, 1);
///
/// # Safety
///
/// expr should be valid pointer to parameter
/// keys and subs should be valid pointer to array in C
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_param_subs(
    expr: *const Param,
    keys: *const *const Param,
    subs: *const *const Param,
    num: usize,
) -> *mut Param {
    let expr = unsafe { const_ptr_as_ref(expr) };
    if let Param::ParameterExpression(expr) = expr {
        let keys = unsafe { std::slice::from_raw_parts(keys, num).iter().map(|k| const_ptr_as_ref(*k)) };
        let subs = unsafe { std::slice::from_raw_parts(subs, num).iter().map(|s| const_ptr_as_ref(*s)) };

        let mut map: HashMap<Symbol, ParameterExpression> = HashMap::new();
        keys.zip(subs).for_each(|m| {
            if let Param::ParameterExpression(e) = m.0 {
                if let Ok(symbol) = e.try_to_symbol() {
                    if let Param::ParameterExpression(s) = m.1 {
                        map.insert(symbol, s.as_ref().clone());
                    } else if let Param::Float(f) = m.1 {
                        map.insert(symbol, ParameterExpression::from_f64(*f));
                    }
                }
            }
        });
        let out = expr.subs(&map, false);
        match out {
            Ok(out) => Box::into_raw(Box::new(Param::ParameterExpression(Arc::new(out)))),
            Err(_) => ptr::null_mut(),
        }
    } else {
        // return copy of input parameter
        Box::into_raw(Box::new(expr.clone()))
    }
}

/// @ingroup QkParam
/// evaluate the expression and return as real number
///
/// @param expr A pointer to the QkParam.
/// @param out A pointer to store result of evaluation
///
/// @return true if expression can be evaluated as real
///
/// # Example
///
///     QkParam *a = qk_param_new("a");
///     QkParam *b = qk_param_new("ab);
///     QkParam *x = qk_param_add(a, b);
///     QkParam *y = qk_param_bind(x, {a, b}, {1.0, 2.0}, 2);
///     double out;
///     qk_param_as_real(y)
///
/// # Safety
///
/// expr should be valid pointer to parameter
/// out should be valid pointer to double in C
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_param_as_real(expr: *const Param, out: *mut f64) -> bool {
    let expr = unsafe { const_ptr_as_ref(expr) };
    let out = unsafe { mut_ptr_as_ref(out) };
    match expr {
        Param::ParameterExpression(expr) => match expr.try_to_value(true) {
            Ok(v) => {
                *out = v.as_real();
                true
            }
            Err(_) => false,
        },
        Param::Float(f) => {
            *out = *f;
            true
        }
        // PyObject should be supported in C-API ?
        _ => false,
    }
}
