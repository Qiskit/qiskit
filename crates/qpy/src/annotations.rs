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

use crate::bytes::Bytes;
use crate::error::QpyError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

/// Handles QPY annotations at the boundary appropriate to the caller.
///
/// Python QPY entry points delegate annotation handling to the existing Python state objects.
/// Native callers do not initialize any Python annotation machinery and fail only if the payload
/// actually requires annotation handling.
#[derive(Debug)]
pub enum AnnotationHandler {
    Python {
        factories: Py<PyDict>,
        serialization_state: Py<PyAny>,
        deserialization_state: Py<PyAny>,
    },
    Native,
}

impl AnnotationHandler {
    pub fn python(factories: &Py<PyDict>) -> Result<Self, QpyError> {
        Python::attach(|py| {
            let module = py.import("qiskit.qpy.binary_io.circuits")?;
            let serialization_state = module
                .getattr("_AnnotationSerializationState")?
                .call1((factories.bind(py),))?
                .unbind();
            let deserialization_state = module
                .getattr("_AnnotationDeserializationState")?
                .call1((factories.bind(py),))?
                .unbind();
            Ok(Self::Python {
                factories: factories.clone_ref(py),
                serialization_state,
                deserialization_state,
            })
        })
    }

    // we will use this as part of the python-independance path
    #[allow(dead_code)]
    pub fn native() -> Self {
        Self::Native
    }

    /// Create independent annotation state for a nested circuit while preserving the caller mode.
    pub fn child(&self) -> Result<Self, QpyError> {
        match self {
            Self::Python { factories, .. } => Self::python(factories),
            Self::Native => Ok(Self::Native),
        }
    }

    pub fn serialize(&self, annotation: &Py<PyAny>) -> Result<(u32, Bytes), QpyError> {
        match self {
            Self::Python {
                serialization_state,
                ..
            } => Python::attach(|py| {
                Ok(serialization_state
                    .call_method1(py, "serialize", (annotation,))?
                    .extract(py)?)
            }),
            Self::Native => Err(Self::native_error("serialize")),
        }
    }

    pub fn load(&self, index: u32, payload: Bytes) -> Result<Py<PyAny>, QpyError> {
        match self {
            Self::Python {
                deserialization_state,
                ..
            } => Python::attach(|py| {
                Ok(deserialization_state.call_method1(py, "load", (index, payload))?)
            }),
            Self::Native => Err(Self::native_error("deserialize")),
        }
    }

    pub fn dump_serializers(&self) -> Result<Vec<(String, Bytes)>, QpyError> {
        match self {
            Self::Python {
                serialization_state,
                ..
            } => Python::attach(|py| {
                Ok(serialization_state
                    .call_method0(py, "dump_states")?
                    .extract(py)?)
            }),
            Self::Native => Ok(Vec::new()),
        }
    }

    pub fn load_deserializers(&self, data: Vec<(String, Bytes)>) -> Result<(), QpyError> {
        if data.is_empty() {
            return Ok(());
        }
        match self {
            Self::Python {
                deserialization_state,
                ..
            } => Python::attach(|py| {
                for (namespace, state) in data {
                    deserialization_state.call_method1(py, "initialize", (namespace, state))?;
                }
                Ok(())
            }),
            Self::Native => Err(Self::native_error("deserialize")),
        }
    }

    fn native_error(action: &str) -> QpyError {
        QpyError::AnnotationError(format!(
            "native QPY cannot {action} circuits containing annotations"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_handler_is_inert_without_annotations() {
        assert!(
            AnnotationHandler::native()
                .dump_serializers()
                .is_ok_and(|states| states.is_empty())
        );
        assert!(
            AnnotationHandler::native()
                .load_deserializers(Vec::new())
                .is_ok()
        );
    }

    #[test]
    fn native_handler_rejects_annotations() {
        let handler = AnnotationHandler::native();
        assert!(matches!(
            handler.load(0, Bytes::new()),
            Err(QpyError::AnnotationError(_))
        ));
        assert!(matches!(
            handler.load_deserializers(vec![("namespace".to_owned(), Bytes::new())]),
            Err(QpyError::AnnotationError(_))
        ));
    }
}
