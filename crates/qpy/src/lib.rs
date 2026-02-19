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

// QPY involves deserializing untrusted user input, which means it's almost _never_ safe to make
// assertions about it.  Better just to completely deny use of these panicking methods.  This is
// done here rather than in `Cargo.toml` so we don't override the workspace inheritance of lints.
#![deny(clippy::unwrap_used, clippy::expect_used)]

use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::{Bound, PyResult, wrap_pyfunction};

mod annotations;
mod bytes;
mod circuit_reader;
mod circuit_writer;
mod consts;
mod error;
mod expr;
mod formats;
mod params;
mod py_methods;
mod value;

// Re-export the error types for use throughout the module
pub use error::{QpyError, from_binrw_error, to_binrw_error};

// Import the Python exception for UnsupportedFeatureForVersion
// We don't import QpyError from Python since we have our own Rust type
use pyo3::import_exception;
import_exception!(qiskit.qpy.exceptions, UnsupportedFeatureForVersion);

/// Internal module supplying the QPY capabilities.  The entries in it should largely
/// be re-exposed directly to public Python space.
pub fn qpy(module: &Bound<PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(circuit_writer::py_write_circuit, module)?)?;
    module.add_function(wrap_pyfunction!(circuit_reader::py_read_circuit, module)?)?;
    Ok(())
}
