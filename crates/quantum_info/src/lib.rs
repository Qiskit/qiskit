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

pub mod convert_2q_block_matrix;
pub mod pauli_lindblad_map;
pub mod sparse_observable;
pub mod sparse_pauli_op;
pub mod unitary_compose;
pub mod versor_u2;

mod rayon_ext;
#[cfg(test)]
mod test;

use pyo3::import_exception;

import_exception!(qiskit.exceptions, QiskitError);
