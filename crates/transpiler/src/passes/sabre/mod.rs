// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

mod dag;
pub mod heuristic;
mod layer;
mod layout;
pub(crate) mod route;
mod vec_map;

use pyo3::prelude::*;

pub(crate) use heuristic::Heuristic;
pub(crate) use heuristic::SetScaling;
pub use layout::sabre_layout_and_routing;
pub(crate) use route::sabre_routing;

#[pymodule(submodule, module = "qiskit._accelerate")]
pub mod sabre {
    #[pymodule_export]
    use super::{
        heuristic::{BasicHeuristic, DecayHeuristic, Heuristic, LookaheadHeuristic, SetScaling},
        layout::py_sabre_layout_and_routing,
        route::{PyRoutingTarget, py_sabre_routing},
    };
}
