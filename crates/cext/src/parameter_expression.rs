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
use std::ptr::null;

use crate::pointers::const_ptr_as_ref;

use qiskit_circuit::parameter_expression::ParameterExpression;

/// @ingroup QkParameterExpression
/// Construct a new Parameter with a name of symbol
///
/// @param name The name of symbol.
///
/// @return A pointer to the created ParameterExpression.
///
/// # Example
///
///     QkParameterExpression *a = qk_parameter_symbol("a");
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_parameter_symbol(name: *mut c_char) -> *mut ParameterExpression {
    let name = unsafe { CStr::from_ptr(name).to_str().unwrap() };
    let symbol = ParameterExpression::new(name.to_string(), None);
    Box::into_raw(Box::new(symbol))
}

/// @ingroup QkParameterExpression
/// Free the ParameterExpression.
///
/// @param expr A pointer to the ParameterExpression to free.
///
/// # Example
///
///     QkParameterExpression *a = qk_parameter_symbol("a");
///     qk_parameter_free(a);
///
/// # Safety
///
/// Behavior is undefined if ``expr`` is not either null or a valid pointer to a
/// [ParameterExpression].
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_parameter_free(expr: *mut ParameterExpression) {
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

/// @ingroup QkParameterExpression
/// copy ParameterExpression
///
/// @param name The name of symbol.
///
/// @return A pointer to the created ParameterExpression.
///
/// # Example
///
///     QkParameterExpression *b = qk_parameter_copy(a);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_parameter_copy(expr: *const ParameterExpression) -> *mut ParameterExpression {
    let expr = unsafe { const_ptr_as_ref(expr) };
    Box::into_raw(Box::new(expr.clone()))
}

/// @ingroup QkParameterExpression
/// get string expression.
///
/// @param expr A pointer to the ParameterExpression.
///
/// # Example
///
///     QkParameterExpression *a = qk_parameter_symbol("a");
///     char* str = qk_parameter_to_string(a);
///     printf(str);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_parameter_to_string(expr: *const ParameterExpression) -> *const c_char {
    let expr = unsafe { const_ptr_as_ref(expr) };
    let out = CString::new(expr.to_string()).unwrap();
    out.into_raw()
}

/// @ingroup QkParameterExpression
/// Construct a new ParameterExpression from real number
///
/// @param value real number for a new ParameterExpression
///
/// @return A pointer to the created ParameterExpression.
///
/// # Example
///
///     QkParameterExpression *r = qk_parameter_from_real(2.5);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_parameter_from_real(value: f64) -> *mut ParameterExpression {
    let value = ParameterExpression::from(value);
    Box::into_raw(Box::new(value))
}

/// @ingroup QkParameterExpression
/// Construct a new ParameterExpression from interger number
///
/// @param value integer number for a new ParameterExpression
///
/// @return A pointer to the created ParameterExpression.
///
/// # Example
///
///     QkParameterExpression *i = qk_parameter_from_int(1);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_parameter_from_int(value: i64) -> *mut ParameterExpression {
    let value = ParameterExpression::from(value);
    Box::into_raw(Box::new(value))
}

/// @ingroup QkParameterExpression
/// Construct a new ParameterExpression from complex number
///
/// @param value complex number for a new ParameterExpression
///
/// @return A pointer to the created ParameterExpression.
///
/// # Example
///
///     QkParameterExpression *c = qk_parameter_from_complex((1.0,1.0));
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_parameter_from_complex(value: Complex64) -> *mut ParameterExpression {
    let value = ParameterExpression::from(value);
    Box::into_raw(Box::new(value))
}

/// @ingroup QkParameterExpression
/// add 2 expressions.
///
/// @param lhs A pointer to the left hand side ParameterExpression.
/// @param rhs A pointer to the right hand side ParameterExpression.
///
/// @return A pointer to the new ParameterExpression.
///
/// # Example
///
///     QkParameterExpression *a = qk_parameter_symbol("a");
///     QkParameterExpression *b = qk_parameter_symbol("b");
///     QkParameterExpression *c = qk_parameter_add(a, b);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_parameter_add(
    lhs: *const ParameterExpression,
    rhs: *const ParameterExpression,
) -> *mut ParameterExpression {
    let lhs = unsafe { const_ptr_as_ref(lhs) };
    let rhs = unsafe { const_ptr_as_ref(rhs) };
    let out = lhs + rhs;
    Box::into_raw(Box::new(out))
}

/// @ingroup QkParameterExpression
/// sub 2 expressions.
///
/// @param lhs A pointer to the left hand side ParameterExpression.
/// @param rhs A pointer to the right hand side ParameterExpression.
///
/// @return A pointer to the new ParameterExpression.
///
/// # Example
///
///     QkParameterExpression *a = qk_parameter_symbol("a");
///     QkParameterExpression *b = qk_parameter_symbol("b");
///     QkParameterExpression *c = qk_parameter_sub(a, b);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_parameter_sub(
    lhs: *const ParameterExpression,
    rhs: *const ParameterExpression,
) -> *mut ParameterExpression {
    let lhs = unsafe { const_ptr_as_ref(lhs) };
    let rhs = unsafe { const_ptr_as_ref(rhs) };
    let out = lhs - rhs;
    Box::into_raw(Box::new(out))
}

/// @ingroup QkParameterExpression
/// mul 2 expressions.
///
/// @param lhs A pointer to the left hand side ParameterExpression.
/// @param rhs A pointer to the right hand side ParameterExpression.
///
/// @return A pointer to the new ParameterExpression.
///
/// # Example
///
///     QkParameterExpression *a = qk_parameter_symbol("a");
///     QkParameterExpression *b = qk_parameter_symbol("b");
///     QkParameterExpression *c = qk_parameter_mul(a, b);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_parameter_mul(
    lhs: *const ParameterExpression,
    rhs: *const ParameterExpression,
) -> *mut ParameterExpression {
    let lhs = unsafe { const_ptr_as_ref(lhs) };
    let rhs = unsafe { const_ptr_as_ref(rhs) };
    let out = lhs * rhs;
    Box::into_raw(Box::new(out))
}

/// @ingroup QkParameterExpression
/// div 2 expressions.
///
/// @param lhs A pointer to the left hand side ParameterExpression.
/// @param rhs A pointer to the right hand side ParameterExpression.
///
/// @return A pointer to the new ParameterExpression.
///
/// # Example
///
///     QkParameterExpression *a = qk_parameter_symbol("a");
///     QkParameterExpression *b = qk_parameter_symbol("b");
///     QkParameterExpression *c = qk_parameter_div(a, b);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_parameter_div(
    lhs: *const ParameterExpression,
    rhs: *const ParameterExpression,
) -> *mut ParameterExpression {
    let lhs = unsafe { const_ptr_as_ref(lhs) };
    let rhs = unsafe { const_ptr_as_ref(rhs) };
    let out = lhs / rhs;
    Box::into_raw(Box::new(out))
}

/// @ingroup QkParameterExpression
/// calculate pow of expressions.
///
/// @param base A pointer to the base ParameterExpression.
/// @param exp A pointer to the exponent ParameterExpression.
///
/// @return A pointer to the new ParameterExpression.
///
/// # Example
///
///     QkParameterExpression *a = qk_parameter_symbol("a");
///     QkParameterExpression *b = qk_parameter_symbol("b");
///     QkParameterExpression *c = qk_parameter_pow(a, b);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_parameter_pow(
    base: *const ParameterExpression,
    exp: *const ParameterExpression,
) -> *mut ParameterExpression {
    let base = unsafe { const_ptr_as_ref(base) };
    let exp = unsafe { const_ptr_as_ref(exp) };
    let out = base.pow(exp);
    Box::into_raw(Box::new(out))
}

/// @ingroup QkParameterExpression
/// sin of expressions.
///
/// @param expr A pointer to the ParameterExpression.
///
/// @return A pointer to the new ParameterExpression.
///
/// # Example
///
///     QkParameterExpression *a = qk_parameter_symbol("a");
///     QkParameterExpression *out = qk_parameter_sin(a);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_parameter_sin(
    expr: *const ParameterExpression,
) -> *mut ParameterExpression {
    let expr = unsafe { const_ptr_as_ref(expr) };
    let out = expr.sin();
    Box::into_raw(Box::new(out))
}

/// @ingroup QkParameterExpression
/// cos of expressions.
///
/// @param expr A pointer to the ParameterExpression.
///
/// @return A pointer to the new ParameterExpression.
///
/// # Example
///
///     QkParameterExpression *a = qk_parameter_symbol("a");
///     QkParameterExpression *out = qk_parameter_cos(a);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_parameter_cos(
    expr: *const ParameterExpression,
) -> *mut ParameterExpression {
    let expr = unsafe { const_ptr_as_ref(expr) };
    let out = expr.cos();
    Box::into_raw(Box::new(out))
}

/// @ingroup QkParameterExpression
/// tan of expressions.
///
/// @param expr A pointer to the ParameterExpression.
///
/// @return A pointer to the new ParameterExpression.
///
/// # Example
///
///     QkParameterExpression *a = qk_parameter_symbol("a");
///     QkParameterExpression *out = qk_parameter_tan(a);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_parameter_tan(
    expr: *const ParameterExpression,
) -> *mut ParameterExpression {
    let expr = unsafe { const_ptr_as_ref(expr) };
    let out = expr.tan();
    Box::into_raw(Box::new(out))
}

/// @ingroup QkParameterExpression
/// arcsin of expressions.
///
/// @param expr A pointer to the ParameterExpression.
///
/// @return A pointer to the new ParameterExpression.
///
/// # Example
///
///     QkParameterExpression *a = qk_parameter_symbol("a");
///     QkParameterExpression *out = qk_parameter_asin(a);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_parameter_asin(
    expr: *const ParameterExpression,
) -> *mut ParameterExpression {
    let expr = unsafe { const_ptr_as_ref(expr) };
    let out = expr.arcsin();
    Box::into_raw(Box::new(out))
}

/// @ingroup QkParameterExpression
/// arccos of expressions.
///
/// @param expr A pointer to the ParameterExpression.
///
/// @return A pointer to the new ParameterExpression.
///
/// # Example
///
///     QkParameterExpression *a = qk_parameter_symbol("a");
///     QkParameterExpression *out = qk_parameter_acos(a);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_parameter_acos(
    expr: *const ParameterExpression,
) -> *mut ParameterExpression {
    let expr = unsafe { const_ptr_as_ref(expr) };
    let out = expr.arccos();
    Box::into_raw(Box::new(out))
}

/// @ingroup QkParameterExpression
/// arctan of expressions.
///
/// @param expr A pointer to the ParameterExpression.
///
/// @return A pointer to the new ParameterExpression.
///
/// # Example
///
///     QkParameterExpression *a = qk_parameter_symbol("a");
///     QkParameterExpression *out = qk_parameter_atan(a);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_parameter_atan(
    expr: *const ParameterExpression,
) -> *mut ParameterExpression {
    let expr = unsafe { const_ptr_as_ref(expr) };
    let out = expr.arctan();
    Box::into_raw(Box::new(out))
}

/// @ingroup QkParameterExpression
/// log of expressions.
///
/// @param expr A pointer to the ParameterExpression.
///
/// @return A pointer to the new ParameterExpression.
///
/// # Example
///
///     QkParameterExpression *a = qk_parameter_symbol("a");
///     QkParameterExpression *out = qk_parameter_log(a);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_parameter_log(
    expr: *const ParameterExpression,
) -> *mut ParameterExpression {
    let expr = unsafe { const_ptr_as_ref(expr) };
    let out = expr.log();
    Box::into_raw(Box::new(out))
}

/// @ingroup QkParameterExpression
/// exp of expressions.
///
/// @param expr A pointer to the ParameterExpression.
///
/// @return A pointer to the new ParameterExpression.
///
/// # Example
///
///     QkParameterExpression *a = qk_parameter_symbol("a");
///     QkParameterExpression *out = qk_parameter_exp(a);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_parameter_exp(
    expr: *const ParameterExpression,
) -> *mut ParameterExpression {
    let expr = unsafe { const_ptr_as_ref(expr) };
    let out = expr.exp();
    Box::into_raw(Box::new(out))
}

/// @ingroup QkParameterExpression
/// abs of expressions.
///
/// @param expr A pointer to the ParameterExpression.
///
/// @return A pointer to the new ParameterExpression.
///
/// # Example
///
///     QkParameterExpression *a = qk_parameter_symbol("a");
///     QkParameterExpression *out = qk_parameter_abs(a);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_parameter_abs(
    expr: *const ParameterExpression,
) -> *mut ParameterExpression {
    let expr = unsafe { const_ptr_as_ref(expr) };
    let out = expr.abs();
    Box::into_raw(Box::new(out))
}

/// @ingroup QkParameterExpression
/// sign of expressions.
///
/// @param expr A pointer to the ParameterExpression.
///
/// @return A pointer to the new ParameterExpression.
///
/// # Example
///
///     QkParameterExpression *a = qk_parameter_symbol("a");
///     QkParameterExpression *out = qk_parameter_sign(a);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_parameter_sign(
    expr: *const ParameterExpression,
) -> *mut ParameterExpression {
    let expr = unsafe { const_ptr_as_ref(expr) };
    let out = expr.sign();
    Box::into_raw(Box::new(out))
}

/// @ingroup QkParameterExpression
/// neg of expressions.
///
/// @param expr A pointer to the ParameterExpression.
///
/// @return A pointer to the new ParameterExpression.
///
/// # Example
///
///     QkParameterExpression *a = qk_parameter_symbol("a");
///     QkParameterExpression *out = qk_parameter_neg(a);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_parameter_neg(
    expr: *const ParameterExpression,
) -> *mut ParameterExpression {
    let expr = unsafe { const_ptr_as_ref(expr) };
    let out = -expr;
    Box::into_raw(Box::new(out))
}

/// @ingroup QkParameterExpression
/// conjugate of expressions.
///
/// @param expr A pointer to the ParameterExpression.
///
/// @return A pointer to the new ParameterExpression.
///
/// # Example
///
///     QkParameterExpression *a = qk_parameter_symbol("a");
///     QkParameterExpression *c = qk_parameter_complex((0.0, 1.0));
///     QkParameterExpression *b = qk_parameter_add(a, c);
///     QkParameterExpression *out = qk_parameter_conj(b);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_parameter_conj(
    expr: *const ParameterExpression,
) -> *mut ParameterExpression {
    let expr = unsafe { const_ptr_as_ref(expr) };
    let out = expr.conjugate();
    Box::into_raw(Box::new(out))
}

/// @ingroup QkParameterExpression
/// compare 2 expressions.
///
/// @param lhs A pointer to the left hand side ParameterExpression.
/// @param rhs A pointer to the right hand side ParameterExpression.
///
/// @return true if 2 expressions are equal.
///
/// # Example
///
///     QkParameterExpression *a = qk_parameter_symbol("a");
///     QkParameterExpression *one = qk_parameter_real(1.0);
///     QkParameterExpression *mone = qk_parameter_real(-1.0);
///     QkParameterExpression *x = qk_parameter_add(a, one);
///     QkParameterExpression *y = qk_parameter_sub(a, mone);
///     QkParameterExpression *c = qk_parameter_compare_eq(x, y);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_parameter_compare_eq(
    lhs: *const ParameterExpression,
    rhs: *const ParameterExpression,
) -> bool {
    let lhs = unsafe { const_ptr_as_ref(lhs) };
    let rhs = unsafe { const_ptr_as_ref(rhs) };
    lhs == rhs
}

/// @ingroup QkParameterExpression
/// substitute expressions.
///
/// @param expr A pointer to the ParameterExpression
/// @param keys An array of pointer to the ParameterExpression.
/// @param values An array of pointer to the ParameterExpression.
/// @param num number of pairs ob substitutions
///
/// @return A pointer to the new ParameterExpression.
///
/// # Example
///
///     QkParameterExpression *a = qk_parameter_symbol("a");
///     QkParameterExpression *b = qk_parameter_symbol("b");
///     QkParameterExpression *c = qk_parameter_add(a, b);
///     QkParameterExpression *v = qk_parameter_from_real(1.5);
///     QkParameterExpression *t = qk_parameter_subs({b}, {v}, 1);
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_parameter_subs(
    expr: *const ParameterExpression,
    keys: *const ParameterExpression,
    values: *const ParameterExpression,
    num: usize,
) -> *mut ParameterExpression {
    let expr = unsafe { const_ptr_as_ref(expr) };
    let keys = unsafe { std::slice::from_raw_parts(keys, num) };
    let values = unsafe { std::slice::from_raw_parts(values, num) };
    let map: HashMap<_, _> = keys.iter().zip(values.iter()).collect();
    let out = expr.substitute(map, true);
    match out {
        Ok(out) => Box::into_raw(Box::new(out)),
        Err(err) => {
            println!("ERROR : {}", err);
            null::<ParameterExpression>() as *mut ParameterExpression
        }
    }
}

/// @ingroup QkParameterExpression
/// check if the expressions is numeric or not.
///
/// @param expr A pointer to the ParameterExpression.
///
/// @return true if numeric.
///
/// # Example
///
///     QkParameterExpression *a = qk_parameter_symbol("a");
///     QkParameterExpression *one = qk_parameter_real(1.0);
///     QkParameterExpression *x = qk_parameter_add(a, one);
///     QkParameterExpression *y = qk_parameter_substitute({a}, {one}, 1);
///     if (qk_parameter_is_numeric(y))
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_parameter_is_numeric(expr: *const ParameterExpression) -> bool {
    let expr = unsafe { const_ptr_as_ref(expr) };
    expr.is_numeric()
}

/// @ingroup QkParameterExpression
/// check if the expressions is symbol or not.
///
/// @param expr A pointer to the ParameterExpression.
///
/// @return true if symbol.
///
/// # Example
///
///     QkParameterExpression *a = qk_parameter_symbol("a");
///     if (qk_parameter_is_symbol(a))
///
#[no_mangle]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_parameter_is_symbol(expr: *const ParameterExpression) -> bool {
    let expr = unsafe { const_ptr_as_ref(expr) };
    expr.is_symbol()
}
