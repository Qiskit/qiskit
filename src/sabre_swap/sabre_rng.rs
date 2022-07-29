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

use pyo3::prelude::*;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;

/// A rng container that shares an rng state between python and sabre's rust
/// code. It should be initialized once and passed to
/// ``sabre_score_heuristic`` to avoid recreating a rng on the inner loop
#[pyclass(module = "qiskit._accelerate.sabre_swap")]
#[pyo3(text_signature = "(/)")]
#[derive(Clone, Debug)]
pub struct SabreRng {
    pub rng: Pcg64Mcg,
}

#[pymethods]
impl SabreRng {
    #[new]
    pub fn new(seed: u64) -> Self {
        SabreRng {
            rng: Pcg64Mcg::seed_from_u64(seed),
        }
    }
}
