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

use std::env;

use pyo3::import_exception;

pub mod circuit_library;
pub mod convert_2q_block_matrix;
pub mod cos_sin_decomp;
pub mod euler_one_qubit_decomposer;
pub mod gate_metrics;
pub mod isometry;
pub mod optimize_1q_gates;
pub mod pauli_exp_val;
pub mod quantum_info;
pub mod results;
pub mod sampled_exp_val;
pub mod sparse_observable;
pub mod sparse_pauli_op;
pub mod synthesis;
pub mod two_qubit_decompose;
pub mod uc_gate;
pub mod unitary_compose;

mod rayon_ext;
#[cfg(test)]
mod test;

#[inline]
pub fn getenv_use_multiple_threads() -> bool {
    let parallel_context = env::var("QISKIT_IN_PARALLEL")
        .unwrap_or_else(|_| "FALSE".to_string())
        .to_uppercase()
        == "TRUE";
    let force_threads = env::var("QISKIT_FORCE_THREADS")
        .unwrap_or_else(|_| "FALSE".to_string())
        .to_uppercase()
        == "TRUE";
    !parallel_context || force_threads
}

import_exception!(qiskit.exceptions, QiskitError);
import_exception!(qiskit.circuit.exceptions, CircuitError);
