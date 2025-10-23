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

use crate::{
    object_registry::ObjectRegistryError, parameter::parameter_expression::ParameterError,
    parameter_table::ParameterTableError, py_error::CircuitError as PyCircuitError,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    PyErr,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CircuitError {
    #[error("cannot have fewer physical qubits ({0}) than virtual ({1})")]
    PhysicalQubitMismatch(u32, usize), // ValueError
    #[error("Qubit at index {0} exceeds circuit capacity.")]
    QubitExceedsCapacity(usize),
    #[error("Clbit at index {0} exceeds circuit capacity.")]
    ClbitExceedsCapacity(usize),
    #[error("Replacement 'qubits' of size {0} must contain at least {1} bits.")]
    ReplaceBitsQubitMismatch(usize, usize), // ValueError
    #[error("Replacement 'clbits' of size {0} must contain at least {1} bits.")]
    ReplaceBitsClbitMismatch(usize, usize), // ValueError
    #[error("register name \"{0}\" already exists")]
    RegisterNameExists(String),
    #[error("Invalid type for global phase: {0}")]
    InvalidGlobalPhase(String), // TypeError
    #[error("Mismatching number of values and parameters. For partial binding please pass a mapping of {{parameter: value}} pairs.")]
    ParamAssignmentMismatch, // ValueError
    #[error("An invalid parameter was provided.")]
    ParamAssignmentInvalidParam, // ValueError
    #[error("Internal error: circuit parameter table is inconsistent.")]
    InvalidParameterTable, // Runtime Error
    #[error("Cannot assign object ({0}) object to parameter.")]
    ParamAssignmentObjectInvalid(String), // TypeError
    #[error("Incorrect type after binding parameter for gate '{0}', '{1}'")]
    ParamAssignmentInvalidBinding(String, String),
    #[error("Name conflict adding parameter '{0}'")]
    ParamAssignmentRepeatedName(String),
    #[error("Error adding Var to circuit: {0}")]
    AddVar(String),
    #[error("Error adding Stretch to circuit: {0}")]
    AddStretch(String),
    #[error("Error finding bit argument: {0}")]
    BitArgument(String),
    #[error(transparent)]
    ParameterError(#[from] ParameterError),
    #[error(transparent)]
    ParameterTableError(#[from] ParameterTableError),
    #[error(transparent)]
    RegistryError(#[from] ObjectRegistryError),
}

impl From<CircuitError> for PyErr {
    fn from(value: CircuitError) -> Self {
        match value {
            CircuitError::PhysicalQubitMismatch(_, _)
            | CircuitError::ReplaceBitsQubitMismatch(_, _)
            | CircuitError::ReplaceBitsClbitMismatch(_, _)
            | CircuitError::ParamAssignmentMismatch
            | CircuitError::ParamAssignmentInvalidParam => PyValueError::new_err(value.to_string()),
            CircuitError::InvalidGlobalPhase(_) | CircuitError::ParamAssignmentObjectInvalid(_) => {
                PyTypeError::new_err(value.to_string())
            }
            CircuitError::InvalidParameterTable => PyRuntimeError::new_err(value.to_string()),
            CircuitError::ParameterTableError(error) => error.into(),
            CircuitError::RegistryError(error) => error.into(),
            _ => PyCircuitError::new_err(value.to_string()),
        }
    }
}
