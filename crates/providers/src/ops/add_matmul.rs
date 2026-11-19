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

//! A helper that adds a matrix product with broadcasting to a [`ProgramFunction`].

use thiserror::Error;

use super::MathOpError;
use crate::program::{FunctionError, ProgramFunction, Value};

/// Errors from [`add_matmul`].
#[derive(Debug, Error)]
pub enum AddMatMulError {
    /// An operand was not produced by this function.
    #[error("operand {operand} was not produced by this function")]
    UnknownOperand { operand: usize },
    /// The operand types cannot be multiplied.
    #[error(transparent)]
    Math(#[from] MathOpError),
    /// The function rejected an instruction.
    #[error(transparent)]
    Function(#[from] FunctionError),
}

/// Add the matrix product `x @ y` to `f`, broadcasting the leading axes, and return the product.
///
/// The last two axes are multiplied and the leading axes are broadcast:
/// `(*a, m, k) @ (*b, k, n) -> (*bcast(a, b), m, n)`. Both operands need at least two axes.
pub fn add_matmul(f: &mut ProgramFunction, x: Value, y: Value) -> Result<Value, AddMatMulError> {
    let _ = (f, x, y);
    todo!()
}
