// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use qiskit_accelerate::sparse_observable::ArithmeticError;
use thiserror::Error;

/// Errors related to C input.
#[derive(Error, Debug)]
pub enum CInputError {
    #[error("Unexpected null pointer.")]
    NullPointerError,
    #[error("Non-aligned memory.")]
    AlignmentError,
    #[error("Index out of bounds.")]
    IndexError,
}

/// Integer exit codes returned to C.
#[repr(u32)]
pub enum ExitCode {
    Success = 0, // these need to be fixed for backward compat
    CInputError = 100,
    NullPointerError = 101,
    AlignmentError = 102,
    IndexError = 103,
    ArithmeticError = 200,
    MismatchedQubits = 201,
}

impl From<ArithmeticError> for ExitCode {
    fn from(value: ArithmeticError) -> Self {
        match value {
            ArithmeticError::MismatchedQubits { left: _, right: _ } => ExitCode::MismatchedQubits,
        }
    }
}

impl From<CInputError> for ExitCode {
    fn from(value: CInputError) -> Self {
        match value {
            CInputError::AlignmentError => ExitCode::AlignmentError,
            CInputError::NullPointerError => ExitCode::NullPointerError,
            CInputError::IndexError => ExitCode::IndexError,
        }
    }
}
