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

//! What the arithmetic, bitwise and reduction nodes reject.

use crate::tensor::{DType, Dim, TensorError};
use thiserror::Error;

/// Errors returned by the arithmetic, bitwise and reduction [`OpNodeType`](super::OpNodeType)
/// implementations.
///
/// Operands are positional, so an offending one is named by index. The node itself is named by
/// whatever wraps the error: a [`ProgramFunction`](crate::ProgramFunction) attaches its type name.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MathNodeError {
    /// An operand's dtype is not the single one this node accepts.
    #[error("operand {operand}: expected dtype {expected}, got {actual}")]
    WrongDType {
        operand: usize,
        expected: DType,
        actual: DType,
    },
    /// An operand's dtype is not among those this node accepts.
    #[error("operand {operand}: dtype {dtype} is not supported")]
    UnsupportedDType { operand: usize, dtype: DType },
    /// Two operands promote to a dtype this node does not compute.
    ///
    /// An elementwise operation works in the promoted dtype, so that is what has to be admitted.
    /// Both operand dtypes are named, because neither is at fault on its own: `add` takes a `Bit`
    /// operand happily alongside an `F64` one, and refuses only two of them together.
    #[error("operands of dtype {lhs} and {rhs} promote to {dtype}, which is not supported")]
    UnsupportedPromotion {
        lhs: DType,
        rhs: DType,
        dtype: DType,
    },
    /// A tensor operation failed (dtype or shape mismatch).
    #[error(transparent)]
    Tensor(#[from] TensorError),
    /// The requested axis was out of bounds for the tensor's number of dimensions.
    #[error("axis {axis} is out of bounds for tensor with {ndim} dimension(s)")]
    InvalidAxis { axis: usize, ndim: usize },
}

/// Validate that an operand's dtype is `expected`, naming the offending operand by its position.
pub(super) fn check_dtype(
    operand: usize,
    actual: DType,
    expected: DType,
) -> Result<(), MathNodeError> {
    if actual != expected {
        return Err(MathNodeError::WrongDType {
            operand,
            expected,
            actual,
        });
    }
    Ok(())
}

/// Validate that `axis` is a valid axis index for a tensor with `ndim` dimensions.
pub(super) fn check_axis(axis: usize, ndim: usize) -> Result<(), MathNodeError> {
    if axis >= ndim {
        return Err(MathNodeError::InvalidAxis { axis, ndim });
    }
    Ok(())
}

/// Remove `axis` from a shape, bounds-checked against its rank.
///
/// A reduction divides by the reduced axis's size, but only at run time, which is when that size is
/// known, and the axis is gone from the result type either way. Surviving axes are copied out
/// untouched. So a [`Dim::Bounded`] axis passes straight through whether or not it is the one being
/// folded.
pub(super) fn reduced_shape(axis: usize, shape: &[Dim]) -> Result<Vec<Dim>, MathNodeError> {
    check_axis(axis, shape.len())?;
    let mut shape = shape.to_vec();
    shape.remove(axis);
    Ok(shape)
}
