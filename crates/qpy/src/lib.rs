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

use pyo3::{pyfunction, wrap_pyfunction, Bound, PyResult, Python};
use pyo3::types::PyModule;
use pyo3::prelude::*;

mod circuits;
mod params;

#[pyfunction]
pub fn hello_world(
    _py: Python,
) -> PyResult<usize> {
    println!("Hello world from qpy module");
    return Ok(42);
}

/// Internal module supplying the QPY capabilities.  The entries in it should largely
/// be re-exposed directly to public Python space.
pub fn qpy(module: &Bound<PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(hello_world, module)?)?;
    module.add_function(wrap_pyfunction!(circuits::py_write_instructions, module)?)?;
    Ok(())
}
