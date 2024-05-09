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

use super::property_map::PropsMap;
use hashbrown::{hash_set::IntoIter, HashSet};
use indexmap::IndexMap;
use itertools::Itertools;
use pyo3::{exceptions::PyKeyError, prelude::*, pyclass, types::PyDict};

type GateMapType = IndexMap<String, PropsMap>;
type GateMapIterType = IntoIter<String>;

#[pyclass]
pub struct GateMapIter {
    iter: GateMapIterType,
}

#[pymethods]
impl GateMapIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<String> {
        slf.iter.next()
    }
}

#[pyclass(mapping)]
#[derive(Debug, Clone)]
pub struct GateMap {
    pub map: GateMapType,
}

#[pymethods]
impl GateMap {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn __contains__(&self, key: &Bound<PyAny>) -> bool {
        if let Ok(key) = key.extract::<String>() {
            self.map.contains_key(&key)
        } else {
            false
        }
    }

    fn __eq__(slf: PyRef<Self>, other: &Bound<PyAny>) -> PyResult<bool> {
        if let Ok(dict) = other.downcast::<PyDict>() {
            for key in dict.keys() {
                if let Ok(key) = key.extract::<String>() {
                    if !slf.map.contains_key(&key) {
                        return Ok(false);
                    }
                } else {
                    return Ok(false);
                }
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn __getitem__(&self, key: String) -> PyResult<PropsMap> {
        if let Some(item) = self.map.get(&key) {
            Ok(item.to_owned())
        } else {
            Err(PyKeyError::new_err(format!(
                "Key {:#?} not in target.",
                key
            )))
        }
    }

    #[pyo3(signature = (key, default=None))]
    fn get(slf: PyRef<Self>, key: String, default: Option<Bound<PyAny>>) -> PyObject {
        match slf.__getitem__(key) {
            Ok(value) => value.into_py(slf.py()),
            Err(_) => match default {
                Some(value) => value.into(),
                None => slf.py().None(),
            },
        }
    }

    fn __len__(slf: PyRef<Self>) -> usize {
        slf.map.len()
    }

    pub fn __iter__(&self, py: Python<'_>) -> PyResult<Py<GateMapIter>> {
        let iter = GateMapIter {
            iter: self
                .map
                .keys()
                .cloned()
                .collect::<HashSet<String>>()
                .into_iter(),
        };
        Py::new(py, iter)
    }

    pub fn keys(&self) -> HashSet<String> {
        self.map.keys().cloned().collect()
    }

    pub fn values(&self) -> Vec<PropsMap> {
        self.map.clone().into_values().collect_vec()
    }

    pub fn items(&self) -> Vec<(String, PropsMap)> {
        self.map.clone().into_iter().collect_vec()
    }

    fn __setstate__(&mut self, state: (GateMapType,)) -> PyResult<()> {
        self.map = state.0;
        Ok(())
    }

    fn __getstate__(&self) -> (GateMapType,) {
        (self.map.clone(),)
    }
}

impl Default for GateMap {
    fn default() -> Self {
        Self {
            map: IndexMap::new(),
        }
    }
}
