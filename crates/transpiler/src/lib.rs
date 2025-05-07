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

pub mod commutation_checker;
pub mod equivalence;
pub mod passes;
pub mod target;

// TODO: Move these to qiskit-accelerate (or another crate) after the
// qiskit-transpiler dependencies in qiskit-accelerate are separated into
// crates for quantum_info and/or synthesis and we can remove qiskit-accelerate
// from the dependencies list.
pub mod circuit_duration;
pub mod twirling;

use pyo3::import_exception_bound;

import_exception_bound! {qiskit.exceptions, QiskitError}
import_exception_bound! {qiskit.transpiler.exceptions, TranspilerError}
