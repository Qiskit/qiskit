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

use pyo3::import_exception;

pub mod circuit_duration;
pub mod circuit_library;
pub mod isometry;
pub mod optimize_1q_gates;
pub mod pauli_exp_val;
pub mod results;
pub mod sampled_exp_val;
pub mod twirling;
pub mod uc_gate;
pub mod unitary_synthesis;
pub mod utils;
pub mod vf2_layout;

mod rayon_ext;
#[cfg(test)]
mod test;
mod unitary_compose;

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
    
    let result = !parallel_context || force_threads;
    
    // Log threading decision if debug logging is enabled
    if env::var("QISKIT_DEBUG_THREADING").is_ok() {
        eprintln!(
            "Rust threading decision: {} (parallel_context={}, force_threads={})",
            if result { "MULTI_THREADED" } else { "SINGLE_THREADED" },
            parallel_context,
            force_threads
        );
    }
    
    result
}
import_exception!(qiskit.exceptions, QiskitError);
import_exception!(qiskit.circuit.exceptions, CircuitError);
