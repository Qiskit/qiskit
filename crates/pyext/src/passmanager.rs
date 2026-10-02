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

use pyo3::prelude::*;

/// The pass type that we define the `dyn Pass` stuff on.
#[pyclass]
pub struct PyPass;
/// Rust-native wrapper that mediates access to the lifetime-bound `&mut PassContext` (somehow -
/// possibly unsafe).
#[pyclass]
pub struct PassContextHandle;
/// Wrapper around the Rust-native pass manager; we can take this as owned when we create it from
/// Python or give a Rust-created one to Python.
#[pyclass(name = "PassManager")]
pub struct PyPassManager;

#[pymodule(name = "passmanager")]
pub fn passmanager_mod(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPass>()?;
    m.add_class::<PyPassManager>()?;
    m.add_class::<PassContextHandle>()?;
    Ok(())
}
