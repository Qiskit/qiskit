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

pub mod binary;
pub mod bitwise;
pub mod reduction;

use crate::program_node::CallInputError;
use crate::tensor::TensorError;
use thiserror::Error;

/// Errors returned by [`crate::program_node::ProgramNode`] implementations in this module.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MathNodeError {
    /// The input tree did not match the contract declared by `input_types`.
    #[error(transparent)]
    Input(#[from] CallInputError),
    /// A tensor operation failed (dtype or shape mismatch).
    #[error(transparent)]
    Tensor(#[from] TensorError),
}
