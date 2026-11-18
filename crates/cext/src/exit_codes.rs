// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use qiskit_circuit::{circuit_data::CircuitDataError, parameter_table::ParameterTableError};
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
#[derive(PartialEq, Eq, Debug)]
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
    /// Duplicate index.
    DuplicateIndexError = 104,
    /// Invalid ``QkOperationKind``.
    InvalidOperationKind = 105,
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
    /// Transpilation failed
    TranspilerError = 400,
    /// Incompatible types.
    IncompatibleTypes = 401,
    /// Type casting error.
    CastingError = 402,
    /// QkDag operation error
    DagError = 500,
    /// The DAGs have mismatching qubit/clbit amounts during compose.
    DagComposeMismatch = 501,
    /// One or more bit indices were not found during compose.
    DagComposeMissingBit = 502,
    /// Errors concerning parameter handling.
    ParameterError = 600,
    /// Parameter name conflict.
    ParameterNameConflict = 601,
    /// QPY serialization, deserialization, or file I/O failed.
    QpyError = 700,
    /// QkCircuit operation error
    CircuitError = 800,
    /// Incorrect delay unit added to the circuit. Use specialized function.
    IncorrectDelayUnit = 801,
    /// CustomOperation error
    CustomOperation = 900,
    /// Repeated vtable slot error
    CustomOperationRepeatedSlot = 901,
}

impl ExitCode {
    pub fn from_error<E: Exit>(error: &E) -> ExitCode {
        error.exit_code()
    }

    pub fn from_result<T, E: Exit>(result: &Result<T, E>) -> ExitCode {
        if let Err(error) = result {
            error.exit_code()
        } else {
            ExitCode::Success
        }
    }
}

pub trait Exit {
    fn exit_code(&self) -> ExitCode;
}

impl Exit for ArithmeticError {
    fn exit_code(&self) -> ExitCode {
        match self {
            Self::MismatchedQubits { .. } => ExitCode::MismatchedQubits,
            Self::InvalidOperation(_) => ExitCode::ArithmeticError,
            Self::DuplicatedIndex => ExitCode::DuplicateIndexError,
            Self::OutOfBounds(_) => ExitCode::IndexError,
        }
    }
}

impl Exit for CircuitDataError {
    fn exit_code(&self) -> ExitCode {
        match self {
            Self::ParameterTableError(error) => error.exit_code(),
            _ => ExitCode::ParameterError,
        }
    }
}

impl Exit for ParameterTableError {
    fn exit_code(&self) -> ExitCode {
        match self {
            Self::NameConflict(_) => ExitCode::ParameterNameConflict,
            _ => ExitCode::ParameterError,
        }
    }
}

impl Exit for CInputError {
    fn exit_code(&self) -> ExitCode {
        match self {
            Self::NullPointerError => ExitCode::NullPointerError,
            Self::AlignmentError => ExitCode::AlignmentError,
            Self::IndexError => ExitCode::IndexError,
        }
    }
}

impl Exit for TargetError {
    fn exit_code(&self) -> ExitCode {
        match self {
            TargetError::InvalidKey(_) => ExitCode::TargetInvalidInstKey,
            TargetError::AlreadyExists(_) => ExitCode::TargetInstAlreadyExists,
            TargetError::QargsMismatch { .. } => ExitCode::TargetQargMismatch,
            TargetError::InvalidQargsKey { .. } => ExitCode::TargetInvalidQargsKey,
            _ => ExitCode::TargetError,
        }
    }
}
