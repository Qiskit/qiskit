// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

mod constructors;
mod expr;
mod ordering;
mod types;
mod visitors;

pub use self::{expr::Expr, types::Type};
use pyo3::prelude::*;

pub(crate) fn register_python(m: &Bound<PyModule>) -> PyResult<()> {
    expr::register_python(m)?;
    types::register_python(m)?;
    Ok(())
}
