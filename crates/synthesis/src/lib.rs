// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

pub mod clifford;
mod diagonal;
pub mod discrete_basis;
pub mod euler_one_qubit_decomposer;
pub mod evolution;
pub mod linalg;
pub mod linear;
pub mod linear_phase;
pub mod matrix;
mod multi_controlled;
pub mod pauli_evolution;
pub mod pauli_products;
mod permutation;
mod qft;
pub mod qsd;
pub mod ross_selinger;
pub mod two_qubit_decompose;
mod uc_gate;
mod ucrz;
mod utils;

use pyo3::import_exception;
use pyo3::prelude::*;

import_exception!(qiskit.exceptions, QiskitError);

#[pymodule(submodule)]
pub mod synthesis {
    #[pymodule_export]
    use super::{
        clifford::clifford, diagonal::diagonal, discrete_basis::discrete_basis,
        evolution::evolution, linear::linear, linear_phase::linear_phase,
        multi_controlled::multi_controlled, pauli_products::pauli_products_mod,
        permutation::permutation, qft::qft, uc_gate::uc_gate,
    };
}
