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

use std::ffi::{c_char, CStr};

use qiskit_circuit::operations::Param;
use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
use qiskit_circuit::parameter::symbol_expr::Symbol;

/// @ingroup QkParam
/// Create a new `QkParam` as symbol.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_param_new(name: *const c_char) -> *mut Param {
    let name = unsafe {
        CStr::from_ptr(name)
            .to_str()
            .expect("Invalid UTF-8 character")
            .to_string()
    };
    let symbol = Symbol::new(&name, None, None);
    let expr = ParameterExpression::from_symbol(symbol);
    let param = Param::ParameterExpression(expr);
    Box::into_raw(Box::new(param))
}

/// @ingroup QkParam
/// Create a new `QkParam` from a double.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_param_from_double(value: f64) -> *mut Param {
    let expr = ParameterExpression::from_f64(value);
    let param = Param::ParameterExpression(expr);
    Box::into_raw(Box::new(param))
}

/// @ingroup QkParam
/// Free a `QkParam`.
#[no_mangle]
#[cfg(feature = "cbinding")]
pub extern "C" fn qk_param_free(param: *mut Param) {
    if !param.is_null() {
        if !param.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(param);
        }
    }
}
