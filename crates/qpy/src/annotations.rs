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

use std::sync::Arc;

use qiskit_circuit::annotation::{
    Annotation, AnnotationFromPython, NativeLoaders, PythonAnnotation,
};

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
    Native {
        namespaces: Vec<String>,
        loaders: NativeLoaders,
    },
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

    pub fn native(namespaces: Vec<String>, loaders: NativeLoaders) -> Self {
        Self::Native {
            namespaces,
            loaders,
        }
    }

    /// Create independent annotation state for a nested circuit while preserving the caller mode.
    pub fn child(&self) -> Result<Self, QpyError> {
        match self {
            Self::Python { factories, .. } => Self::python(factories),
            Self::Native { loaders, .. } => Ok(Self::native(Vec::new(), loaders.clone())),
        }
    }

    pub fn serialize(
        &mut self,
        annotation: &Arc<dyn Annotation>,
    ) -> Result<(u32, Bytes), QpyError> {
        match self {
            Self::Python {
                serialization_state,
                ..
            } => Python::attach(|py| {
                let Some(ob) = annotation.downcast_ref::<PythonAnnotation>() else {
                    return Err(QpyError::AnnotationError(
                        "Rust native annotations cannot be serialized by the Python QPY path"
                            .to_owned(),
                    ));
                };
                Ok(serialization_state
                    .call_method1(py, "serialize", (ob.annotation(py),))?
                    .extract(py)?)
            }),
            Self::Native { namespaces, .. } => {
                let ns = annotation.namespace();
                let index = if let Some(i) = namespaces.iter().position(|a| a == ns) {
                    i
                } else {
                    namespaces.push(annotation.namespace().to_string());
                    namespaces.len() - 1
                };
                let Some(payload) = annotation.payload() else {
                    return Err(QpyError::AnnotationError(format!(
                        "Could not find an appropriate serializer for namespace {ns}."
                    )));
                };
                Ok((index as u32, format!("{ns}\x00{payload}").into()))
            }
        }
    }

    pub fn load_py(&self, py: Python, index: u32, payload: Bytes) -> Result<Py<PyAny>, QpyError> {
        match self {
            Self::Python {
                deserialization_state,
                ..
            } => Ok(deserialization_state.call_method1(py, "load", (index, payload))?),
            Self::Native { .. } => Err(Self::native_error("deserialize")),
        }
    }

    pub fn load(&self, index: u32, payload: Bytes) -> Result<Arc<dyn Annotation>, QpyError> {
        match self {
            Self::Python { .. } => Python::attach(|py| {
                Ok(self
                    .load_py(py, index, payload)?
                    .bind(py)
                    .extract::<AnnotationFromPython>()?
                    .0)
            }),
            Self::Native { loaders, .. } => {
                let text: &str = (&payload).try_into()?;
                let (ns, payload) = text.split_once("\x00").ok_or_else(|| {
                    QpyError::AnnotationError("Incorrectly formatted payload.".to_owned())
                })?;
                loaders.load(ns, payload).ok_or_else(|| {
                    QpyError::AnnotationError(format!("Could not find a deserializer for {ns}."))
                })
            }
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
            Self::Native { namespaces, .. } => Ok(namespaces
                .iter()
                .map(|ns| (ns.clone(), Bytes::new()))
                .collect()),
        }
    }

    pub fn load_deserializers(&mut self, data: Vec<(String, Bytes)>) -> Result<(), QpyError> {
        if data.is_empty() {
            return Ok(());
        }
        match self {
            Self::Python {
                deserialization_state,
                ..
            } => Python::attach(|py| {
                for (ns, state) in data {
                    deserialization_state.call_method1(py, "initialize", (ns, state))?;
                }
                Ok(())
            }),
            Self::Native { namespaces, .. } => {
                *namespaces = data.into_iter().map(|(ns, _)| ns).collect();
                Ok(())
            }
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
    use std::{assert_eq, assert_ne};

    #[derive(Debug, Clone, PartialEq)]
    struct Tag(&'static str);

    impl Annotation for Tag {
        fn namespace(&self) -> &str {
            "tag"
        }

        fn payload(&self) -> Option<std::borrow::Cow<'_, str>> {
            Some(std::borrow::Cow::Borrowed(self.0))
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Mark;

    impl Annotation for Mark {
        fn namespace(&self) -> &str {
            "mark"
        }

        fn payload(&self) -> Option<std::borrow::Cow<'_, str>> {
            Some(std::borrow::Cow::Borrowed("mark"))
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct NonSerializable;

    impl Annotation for NonSerializable {
        fn namespace(&self) -> &str {
            "no_payload"
        }
    }

    #[test]
    fn native_handler_is_inert_without_annotations() {
        assert!(
            AnnotationHandler::native(Vec::new(), NativeLoaders::default())
                .dump_serializers()
                .is_ok_and(|states| states.is_empty())
        );
        assert!(
            AnnotationHandler::native(Vec::new(), NativeLoaders::default())
                .load_deserializers(Vec::new())
                .is_ok()
        );
    }

    #[test]
    fn test_native_serialize() -> Result<(), Box<dyn std::error::Error>> {
        let annotation: Arc<dyn Annotation> = Arc::new(Tag("my_tag"));
        let other_annotation: Arc<dyn Annotation> = Arc::new(Tag("my_other_tag"));
        let mark: Arc<dyn Annotation> = Arc::new(Mark);
        let mut handler = AnnotationHandler::native(Vec::new(), NativeLoaders::default());

        let (idx, payload) = handler.serialize(&annotation)?;
        let (other_idx, other_payload) = handler.serialize(&other_annotation)?;
        let (mark_idx, mark_payload) = handler.serialize(&mark)?;

        assert_eq!(TryInto::<&str>::try_into(&payload)?, "tag\x00my_tag");
        assert_eq!(
            TryInto::<&str>::try_into(&other_payload)?,
            "tag\x00my_other_tag"
        );
        assert_eq!(TryInto::<&str>::try_into(&mark_payload)?, "mark\x00mark");

        assert_eq!(idx, other_idx);
        assert_ne!(idx, mark_idx);

        Ok(())
    }

    #[test]
    fn test_native_serialize_error() {
        let annotation: Arc<dyn Annotation> = Arc::new(NonSerializable);
        let mut handler = AnnotationHandler::native(Vec::new(), NativeLoaders::default());

        assert!(matches!(
            handler.serialize(&annotation),
            Err(QpyError::AnnotationError(_))
        ));
    }

    #[test]
    fn test_native_dump_serializers() -> Result<(), Box<dyn std::error::Error>> {
        let handler = AnnotationHandler::native(
            vec![
                "randomization".to_string(),
                "randomization.twirl".to_string(),
            ],
            NativeLoaders::default(),
        );
        let serializers = handler
            .dump_serializers()?
            .into_iter()
            .map(|(s, _)| s)
            .collect::<Vec<_>>();
        assert_eq!(
            serializers,
            vec![
                "randomization".to_string(),
                "randomization.twirl".to_string()
            ]
        );
        Ok(())
    }

    #[test]
    fn test_native_load_deserializers() {
        let mut handler = AnnotationHandler::native(Vec::new(), NativeLoaders::default());
        assert!(
            handler
                .load_deserializers(vec![("a.namespace".to_string(), Bytes::new())])
                .is_ok()
        );
    }

    #[test]
    fn test_native_load_error() {
        let handler = AnnotationHandler::native(Vec::new(), NativeLoaders::default());
        assert!(matches!(
            handler.load(0, "bad_payload".into()),
            Err(QpyError::AnnotationError(_))
        ));
    }

    #[test]
    fn test_native_child() -> Result<(), Box<dyn std::error::Error>> {
        let handler =
            AnnotationHandler::native(vec!["some.namespace".to_string()], NativeLoaders::default())
                .child()?;
        assert!(
            handler
                .dump_serializers()
                .is_ok_and(|states| states.is_empty())
        );
        Ok(())
    }
}
