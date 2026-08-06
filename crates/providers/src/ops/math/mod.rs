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

mod binary;
mod bitwise;
mod reduction;

pub use binary::*;
pub use bitwise::*;
pub use reduction::*;

use crate::ops::CallInputError;
use crate::tensor::TensorError;
use thiserror::Error;

/// Errors returned by [`crate::ops::ProgramOp`] implementations in this module.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MathOpError {
    /// The input tree did not match the contract declared by `input_types`.
    #[error(transparent)]
    Input(#[from] CallInputError),
    /// A tensor operation failed (dtype or shape mismatch).
    #[error(transparent)]
    Tensor(#[from] TensorError),
    /// The requested axis was out of bounds for the tensor's number of dimensions.
    #[error("axis {axis} is out of bounds for tensor with {ndim} dimension(s)")]
    InvalidAxis { axis: usize, ndim: usize },
}

/// Validate that `axis` is a valid axis index for a tensor with `ndim` dimensions.
pub(crate) fn check_axis(axis: usize, ndim: usize) -> Result<(), MathOpError> {
    if axis >= ndim {
        return Err(MathOpError::InvalidAxis { axis, ndim });
    }
    Ok(())
}
