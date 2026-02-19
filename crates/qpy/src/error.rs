// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::PyAnyMethods;
use pyo3::{PyErr, Python};
use thiserror::Error;

/// Errors that can occur during QPY serialization and deserialization operations.
///
/// This error type is used internally within the QPY module. It is converted to
/// Python exceptions only at the boundary when returning to Python space.
#[derive(Error, Debug)]
pub enum QpyError {
    /// An unsupported feature was encountered for the target QPY version
    #[error(
        "'{feature}' is not supported in QPY version {version}. Minimum required version is {min_version}"
    )]
    UnsupportedFeatureForVersion {
        feature: String,
        version: u32,
        min_version: u32,
    },

    /// Invalid value type encountered during serialization/deserialization
    #[error("invalid value type: expected {expected}, got {actual}")]
    InvalidValueType { expected: String, actual: String },

    /// Failed to serialize data
    #[error("serialization failed: {0}")]
    SerializationError(String),

    /// Failed to deserialize data
    #[error("deserialization failed: {0}")]
    DeserializationError(String),

    /// Invalid QPY format or corrupted data
    #[error("invalid QPY format: {0}")]
    InvalidFormat(String),

    /// Missing required data
    #[error("missing required data: {0}")]
    MissingData(String),

    /// Invalid instruction or operation
    #[error("invalid instruction: {0}")]
    InvalidInstruction(String),

    /// Invalid parameter
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// Invalid register reference
    #[error("invalid register: {0}")]
    InvalidRegister(String),

    /// Invalid bit reference
    #[error("invalid bit reference: {0}")]
    InvalidBit(String),

    /// Type conversion error
    #[error("type conversion failed: {0}")]
    ConversionError(String),

    /// Circuit data error
    #[error("circuit data error: {0}")]
    CircuitError(String),

    /// Custom instruction not found
    #[error("custom instruction '{0}' not found")]
    CustomInstructionNotFound(String),

    /// Invalid annotation
    #[error("invalid annotation: {0}")]
    InvalidAnnotation(String),

    /// Invalid expression
    #[error("invalid expression: {0}")]
    InvalidExpression(String),

    /// Invalid layout
    #[error("invalid layout: {0}")]
    InvalidLayout(String),

    /// Metadata serialization/deserialization error
    #[error("metadata error: {0}")]
    MetadataError(String),

    /// I/O error during file operations
    #[error(transparent)]
    IoError(#[from] std::io::Error),

    /// UTF-8 conversion error
    #[error(transparent)]
    Utf8Error(#[from] std::str::Utf8Error),

    /// Integer conversion error
    #[error(transparent)]
    IntConversionError(#[from] std::num::TryFromIntError),

    /// Circuit data error from the circuit module
    #[error(transparent)]
    CircuitDataError(#[from] qiskit_circuit::circuit_data::CircuitDataError),

    /// Parameter error from the circuit module
    #[error(transparent)]
    ParameterError(#[from] qiskit_circuit::parameter::parameter_expression::ParameterError),

    /// BinRW parsing error
    #[error("binary parsing error: {0}")]
    BinRwError(String),

    /// Python error that occurred during a Python call
    /// This wraps PyErr but stores it as a string to avoid lifetime issues
    #[error("Python error: {0}")]
    PythonError(String),
}

impl From<QpyError> for PyErr {
    fn from(error: QpyError) -> Self {
        match error {
            QpyError::UnsupportedFeatureForVersion {
                feature,
                version,
                min_version,
            } => {
                // Use the imported exception type from lib.rs
                Python::attach(|py| {
                    let qpy_exceptions = py
                        .import("qiskit.qpy.exceptions")
                        .expect("Failed to import qiskit.qpy.exceptions");
                    let exc_type = qpy_exceptions
                        .getattr("UnsupportedFeatureForVersion")
                        .expect("Failed to get UnsupportedFeatureForVersion");
                    let exc_instance = exc_type
                        .call1((feature, min_version, version))
                        .expect("Failed to create UnsupportedFeatureForVersion");
                    PyErr::from_value(exc_instance)
                })
            }
            QpyError::InvalidValueType { expected, actual } => PyValueError::new_err(format!(
                "invalid value type: expected {expected}, got {actual}"
            )),
            QpyError::InvalidFormat(msg)
            | QpyError::MissingData(msg)
            | QpyError::InvalidInstruction(msg)
            | QpyError::InvalidParameter(msg)
            | QpyError::InvalidRegister(msg)
            | QpyError::InvalidBit(msg)
            | QpyError::ConversionError(msg)
            | QpyError::CircuitError(msg)
            | QpyError::InvalidAnnotation(msg)
            | QpyError::InvalidExpression(msg)
            | QpyError::InvalidLayout(msg)
            | QpyError::MetadataError(msg)
            | QpyError::SerializationError(msg)
            | QpyError::DeserializationError(msg) => PyValueError::new_err(msg),
            QpyError::CustomInstructionNotFound(name) => {
                PyValueError::new_err(format!("custom instruction '{name}' not found"))
            }
            QpyError::IoError(e) => PyRuntimeError::new_err(format!("I/O error: {e}")),
            QpyError::Utf8Error(e) => PyValueError::new_err(format!("UTF-8 error: {e}")),
            QpyError::IntConversionError(e) => {
                PyValueError::new_err(format!("integer conversion error: {e}"))
            }
            QpyError::CircuitDataError(e) => e.into(),
            QpyError::ParameterError(e) => e.into(),
            QpyError::BinRwError(msg) => {
                PyRuntimeError::new_err(format!("binary parsing error: {msg}"))
            }
            QpyError::PythonError(msg) => PyRuntimeError::new_err(msg),
        }
    }
}

impl From<PyErr> for QpyError {
    fn from(err: PyErr) -> Self {
        QpyError::PythonError(err.to_string())
    }
}

impl From<binrw::Error> for QpyError {
    fn from(err: binrw::Error) -> Self {
        QpyError::BinRwError(err.to_string())
    }
}

/// Helper function to convert a binrw error to QpyError for use in binrw parsing
pub fn to_binrw_error<W: std::io::Seek, E: std::error::Error + Send + Sync + 'static>(
    writer: &mut W,
    err: E,
) -> binrw::Error {
    binrw::Error::Custom {
        pos: writer.stream_position().unwrap_or(0),
        err: Box::new(err),
    }
}

/// Helper function to convert binrw errors back to QpyError
pub fn from_binrw_error(err: binrw::Error) -> QpyError {
    QpyError::BinRwError(err.to_string())
}

// Made with Bob
