// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

pub mod marginalization;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pymodule]
pub fn results(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(marginalization::marginal_counts))?;
    m.add_wrapped(wrap_pyfunction!(marginalization::marginal_distribution))?;
    Ok(())
}
