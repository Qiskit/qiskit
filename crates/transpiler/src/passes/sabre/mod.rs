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

mod dag;
mod distance;
mod heuristic;
mod layer;
mod layout;
mod neighbors;
mod route;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub fn sabre(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(route::sabre_routing))?;
    m.add_wrapped(wrap_pyfunction!(layout::sabre_layout_and_routing))?;
    m.add_class::<route::PyRoutingTarget>()?;
    m.add_class::<heuristic::SetScaling>()?;
    m.add_class::<heuristic::Heuristic>()?;
    m.add_class::<heuristic::BasicHeuristic>()?;
    m.add_class::<heuristic::LookaheadHeuristic>()?;
    m.add_class::<heuristic::DecayHeuristic>()?;
    Ok(())
}
