// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;

mod basis_decomposer;
mod common;
mod controlled_u_decomposer;
mod gate_sequence;
mod weyl_decomposition;

pub use basis_decomposer::{TwoQubitBasisDecomposer, two_qubit_decompose_up_to_diagonal};
pub use controlled_u_decomposer::{RXXEquivalent, TwoQubitControlledUDecomposer};
pub use gate_sequence::TwoQubitGateSequence;
pub use weyl_decomposition::{Specialization, TwoQubitWeylDecomposition};

#[pymodule(submodule)]
pub mod two_qubit_decompose {
    #[pymodule_export]
    use super::{
        basis_decomposer::{TwoQubitBasisDecomposer, py_two_qubit_decompose_up_to_diagonal},
        common::{local_equivalence, py_trace_to_fid, py_ud, two_qubit_local_invariants},
        controlled_u_decomposer::TwoQubitControlledUDecomposer,
        weyl_decomposition::{
            _num_basis_gates, Specialization, TwoQubitWeylDecomposition,
            py_decompose_two_qubit_product_gate, weyl_coordinates,
        },
    };
}
