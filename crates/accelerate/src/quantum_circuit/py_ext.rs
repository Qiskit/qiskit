// This code is part of Qiskit.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//! Contains helper functions for creating [Py<T>] (GIL-independent)
//! objects without creating an intermediate owned reference. These functions
//! are faster than PyO3's list and tuple factory methods when the caller
//! doesn't need to dereference the newly constructed object (i.e. if the
//! resulting [Py<T>] will simply be stored in a Rust struct).
//!
//! The reason this is faster is because PyO3 tracks owned references and
//! will perform deallocation when the active [GILPool] goes out of scope.
//! If we don't need to dereference the [Py<T>], then we can skip the
//! tracking and deallocation.

use pyo3::ffi::Py_ssize_t;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use pyo3::{ffi, AsPyPointer, PyNativeType};

pub fn tuple_new(py: Python<'_>, items: Vec<PyObject>) -> Py<PyTuple> {
    unsafe {
        let ptr = ffi::PyTuple_New(items.len() as Py_ssize_t);
        let tup: Py<PyTuple> = Py::from_owned_ptr(py, ptr);
        for (i, obj) in items.into_iter().enumerate() {
            ffi::PyTuple_SetItem(ptr, i as Py_ssize_t, obj.into_ptr());
        }
        tup
    }
}

pub fn tuple_new_empty(py: Python<'_>) -> Py<PyTuple> {
    unsafe { Py::from_owned_ptr(py, ffi::PyTuple_New(0)) }
}

pub fn tuple_from_list(list: &PyList) -> Py<PyTuple> {
    unsafe { Py::from_owned_ptr(list.py(), ffi::PyList_AsTuple(list.as_ptr())) }
}
