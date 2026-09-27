// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Custom operations implemented natively in Rust.
//!
//! Each operation here implements [`qiskit_circuit::operations::CustomOperation`], which is what
//! makes it usable in a circuit and says nothing about Python. Operations that also have a
//! Python-space class implement [`PyConvertible`] separately; this crate names the concrete types,
//! so it is the one place that can build the lookup tables `qiskit-circuit` needs to convert between
//! a `&dyn CustomOperation` and its Python-space object. See [`qiskit_circuit::py_convertible`]
//! for why the two directions (Rust-to-Python, Python-to-Rust) are wired up differently.
//!
//! To add an operation: implement it in a submodule, implement [`PyConvertible`] for it if it has
//! a Python class, then add one line each to [`CONVERSIONS_TO_PYTHON_TABLE`] and
//! [`CONVERSIONS_FROM_PYTHON_TABLE`] -- the compile-time Rust-to-Python and Python-to-Rust tables.

use std::any::TypeId;

use pyo3::prelude::*;

use qiskit_circuit::py_convertible::{
    ConversionFromPythonEntry, ConversionToPythonEntry, create_py_op_for, extract_from_py_for,
    register_conversions_from_python, register_conversions_to_python,
};

pub mod qft;

/// The compile-time Rust-to-Python conversion table for every custom operation in this crate.
static CONVERSIONS_TO_PYTHON_TABLE: &[ConversionToPythonEntry] = &[ConversionToPythonEntry {
    type_id: TypeId::of::<qft::QftGate>,
    create: create_py_op_for::<qft::QftGate>,
}];

/// The compile-time Python-to-Rust conversion table for every custom operation in this crate.
static CONVERSIONS_FROM_PYTHON_TABLE: &[ConversionFromPythonEntry] = &[ConversionFromPythonEntry {
    name: "qft",
    extract: extract_from_py_for::<qft::QftGate>,
}];

/// Register the Python conversions for every custom operation in this crate.
///
/// This must run before any circuit containing these operations crosses the Python boundary; it is
/// called from this crate's module initialisation.
pub fn register_custom_operations() -> PyResult<()> {
    register_conversions_to_python(CONVERSIONS_TO_PYTHON_TABLE);
    register_conversions_from_python(CONVERSIONS_FROM_PYTHON_TABLE);
    Ok(())
}
