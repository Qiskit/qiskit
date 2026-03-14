// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// The `boolean_struct` module here is `doc(hidden)` but `pub`.  We need it to specify the trait
// bound correctly to make `borrow_mut` available.
use pyo3::PyClass;
use pyo3::prelude::*;
use pyo3::pyclass::boolean_struct::False;

/// Borrow a pointer to a Rust-native object from a Python object.
///
/// The returned pointer derives its lifetime from the lifetime of `ob`.  The reference `ob` is only
/// borrowed by the function.
///
/// If the `PyObject` is not of the correct type, the null pointer is returned and the Python
/// exception state is set.
///
/// # Safety
///
/// `ob` must point to a valid PyObject.
pub unsafe fn borrow<T>(py: Python, ob: *mut ::pyo3::ffi::PyObject) -> *mut T
where
    T: PyClass<Frozen = False>,
{
    // SAFETY: per documentation, `ob` points to a valid PyObject.  The lifetime of the
    // `Borrowed` is valid because we immediately consume it back into a pointer, whose
    // lifetime is thus tied to the incoming `ob`.
    match unsafe { Borrowed::from_ptr(py, ob) }.cast::<T>() {
        Ok(ob) => &mut *ob.borrow_mut(),
        Err(e) => {
            PyErr::from(e).restore(py);
            ::std::ptr::null_mut()
        }
    }
}

/// Extract a pointer to a Rust-native object from a PyObject representing a PyClass, storing the
/// result in `address`.
///
/// This is used to define Python-space "converter" functions for use with the `PyArg_Parse*` family
/// of functions.
///
/// On success, returns 1 and writes out the pointer in `address`.  On failure, returns 0, sets the
/// Python exception state and leaves `address` untouched.
///
/// # Safety
///
/// `object` must point to a valid PyObject.  `address` must point to enough space to write a
/// pointer to.
pub unsafe fn convert<T>(
    py: Python,
    object: *mut ::pyo3::ffi::PyObject,
    address: *mut ::std::ffi::c_void,
) -> ::std::ffi::c_int
where
    T: PyClass<Frozen = False>,
{
    // SAFETY: per documentation, `object` points to a valid PyObject.
    let native = unsafe { borrow::<T>(py, object) };
    if native.is_null() {
        0
    } else {
        // SAFETY: per documentation, `address` is a pointer to a valid storage location of
        // the correct type.
        unsafe { address.cast::<*mut T>().write(native) };
        1
    }
}
