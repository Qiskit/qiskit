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

use qiskit_quantum_info::sparse_observable::ArithmeticError;
use qiskit_transpiler::target::TargetError;
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
    /// Success.
    Success = 0,
    /// Error related to data input.
    CInputError = 100,
    /// Unexpected null pointer.
    NullPointerError = 101,
    /// Pointer is not aligned to expected data.
    AlignmentError = 102,
    /// Index out of bounds.
    IndexError = 103,
    /// Error related to arithmetic operations or similar.
    ArithmeticError = 200,
    /// Mismatching number of qubits.
    MismatchedQubits = 201,
    /// Matrix is not unitary.
    ExpectedUnitary = 202,
    /// Target related error
    TargetError = 300,
    /// Instruction already exists in the Target
    TargetInstAlreadyExists = 301,
    /// Properties with incorrect qargs was added
    TargetQargMismatch = 302,
    /// Trying to query into the target with non-existent qargs.
    TargetInvalidQargsKey = 303,
    /// Querying an operation that doesn't exist in the Target.
    TargetInvalidInstKey = 304,
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

impl From<TargetError> for ExitCode {
    fn from(value: TargetError) -> Self {
        match value {
            TargetError::InvalidKey(_) => ExitCode::TargetInvalidInstKey,
            TargetError::AlreadyExists(_) => ExitCode::TargetInstAlreadyExists,
            TargetError::QargsMismatch {
                instruction: _,
                arguments: _,
            } => ExitCode::TargetQargMismatch,
            TargetError::InvalidQargsKey {
                instruction: _,
                arguments: _,
            } => ExitCode::TargetInvalidQargsKey,
            _ => ExitCode::TargetError,
        }
    }
}
