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

pub mod converters;
pub mod marginalization;

use pyo3::prelude::*;

#[pymodule(submodule)]
pub mod results {
    #[pymodule_export]
    use super::marginalization::{
        marginal_counts, marginal_distribution, marginal_measure_level_0,
        marginal_measure_level_0_avg, marginal_measure_level_1, marginal_measure_level_1_avg,
        marginal_memory,
    };
}
