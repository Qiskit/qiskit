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

mod ordering;
mod types;

pub use ordering::{cast_kind, is_subtype, is_supertype, order, CastKind, Ordering};
pub use types::Type;

use pyo3::prelude::*;

pub(crate) fn register_python(m: &Bound<PyModule>) -> PyResult<()> {
    types::register_python(&m)?;

    let ordering_mod = PyModule::new(m.py(), "ordering")?;
    ordering::register_python(&ordering_mod)?;
    m.add_submodule(&ordering_mod)?;

    Ok(())
}
