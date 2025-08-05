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
use crate::bytes::Bytes;
use hashbrown::HashMap;
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator, PyNotImplemented};
/// The purpose of AnnotationHandler is to keep track of all the annotation serializers/deserializers at once.
/// Each namespace has one potential serializer/deserializer corresponding to it, given in the factories list.
/// When serializing the first time a namespace is encountered, the serializer is generated from the factory,
/// but it remains inside `potential_serializers` as long as it did not manage to serialize any annotation.
/// The way to determine whether a serialization attempt succeeded is to check whether the result of `dump_annotation`
/// was `NotImplemented`.
pub struct AnnotationHandler<'a> {
    pub annotation_factories: &'a Bound<'a, PyDict>,
    factories: HashMap<String, Py<PyAny>>,
    pub serializers: HashMap<String, (usize, Py<PyAny>)>,
    potential_serializers: HashMap<String, Py<PyAny>>,
    pub deserializers: Vec<Py<PyAny>>,
}

impl<'a> AnnotationHandler<'a> {
    pub fn new(annotation_factories: &'a Bound<'a, PyDict>) -> Self {
        let mut factories = HashMap::with_capacity(annotation_factories.len());
        for (key, value) in annotation_factories.iter() {
            if let Ok(key_string) = key.extract() {
                // we ignore non-string keys since they will not be invoked during serialization
                // where we choose serializer according to the namespace string
                factories.insert(key_string, value.clone().unbind());
            }
        }

        let serializers = HashMap::new();
        let potential_serializers = HashMap::new();
        // deserializers are created on demand based on the QPY annotation headers
        let deserializers = Vec::new();
        AnnotationHandler {
            annotation_factories,
            factories,
            serializers,
            potential_serializers,
            deserializers,
        }
    }
    pub fn serialize(&mut self, annotation: &Bound<PyAny>) -> PyResult<(u32, Bytes)> {
        let py = annotation.py();
        let annotation_namespace: String = annotation.getattr("namespace")?.extract()?;
        let annotation_module = py.import("qiskit.circuit.annotation")?;
        let namespace_iter_func = annotation_module.getattr("iter_namespaces")?;
        let namespace_iter =
            PyIterator::from_object(&namespace_iter_func.call1((&annotation_namespace,))?)?;
        for namespace_res in namespace_iter {
            let namespace = namespace_res?;
            let namespace_string: String = namespace.clone().extract()?;
            // first check for an active serializer
            if let Some((index, serializer)) = self.serializers.get(&namespace_string) {
                let result = serializer.call_method1(
                    py,
                    "dump_annotation",
                    (namespace.clone(), annotation),
                )?;
                if !result.is(PyNotImplemented::get(py)) {
                    let result_bytes: Bytes = result.extract(py)?;
                    return Ok((*index as u32, result_bytes));
                }
            } else if let Some(serializer) = self.potential_serializers.get(&namespace_string) {
                let result = serializer.call_method1(
                    py,
                    "dump_annotation",
                    (namespace.clone(), annotation),
                )?;
                if !result.is(PyNotImplemented::get(py)) {
                    let index = self.serializers.len();
                    let result_bytes: Bytes = result.extract(py)?;
                    self.serializers
                        .insert(namespace_string.clone(), (index, serializer.clone()));
                    return Ok((index as u32, result_bytes));
                }
            }
            // no serializer, let's try to create one from the corresponding factory
            else if let Some(factory) = self.factories.get(&namespace_string) {
                let serializer = factory.call0(py)?;
                let result = serializer.call_method1(
                    py,
                    "dump_annotation",
                    (namespace.clone(), annotation),
                )?;
                if !result.is(PyNotImplemented::get(py)) {
                    let index = self.serializers.len();
                    let result_bytes: Bytes = result.extract(py)?;
                    self.serializers
                        .insert(namespace_string.clone(), (index, serializer));
                    return Ok((index as u32, result_bytes));
                } else {
                    // This time the serializer failed, but we might want to try it again without
                    // having to reconstruct it from the factory
                    self.potential_serializers
                        .insert(namespace_string.clone(), serializer);
                }
            }
        }
        Ok((0, Bytes(Vec::new())))
    }

    pub fn load(&self, py: Python, index: u32, payload: Bytes) -> PyResult<Py<PyAny>> {
        if let Some(deserializer) = self.deserializers.get(index as usize) {
            deserializer.call_method1(py, "load_annotation", (payload,))
        } else {
            Err(PyIndexError::new_err(format!(
                "Annotation deserializer index {:?} out of range (0...{:?}), is too large and would overflow", index, self.deserializers.len())))
        }
    }

    pub fn dump_serializers(&self) -> PyResult<Vec<(String, Bytes)>> {
        // we need to be a little careful to keep the result sorted by index order
        let mut result: Vec<Option<(String, Bytes)>> = vec![None; self.serializers.len()];
        Python::with_gil(|py| -> PyResult<()> {
            for (namespace, (index, serializer)) in &self.serializers {
                let state: Bytes = serializer.call_method0(py, "dump_state")?.extract(py)?;
                result[*index] = Some((namespace.clone(), state));
            }
            Ok(())
        })?;
        Ok(result.into_iter().flatten().collect())
    }

    pub fn load_deserializers(&mut self, data: Vec<(String, Bytes)>) -> PyResult<()> {
        Python::with_gil(|py| -> PyResult<()> {
            for (namespace, state) in data {
                if let Some(factory) = self.factories.get(&namespace) {
                    let serializer = factory.call0(py)?;
                    serializer.call_method1(
                        py,
                        "load_state",
                        (namespace.clone(), state.clone()),
                    )?;
                    self.deserializers.push(serializer);
                };
            }
            Ok(())
        })?;
        Ok(())
    }
}
