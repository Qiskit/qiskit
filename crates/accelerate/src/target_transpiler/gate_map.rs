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
use hashbrown::HashSet;
use indexmap::{set::IntoIter, IndexMap, IndexSet};
use itertools::Itertools;
use pyo3::{
    exceptions::PyKeyError,
    prelude::*,
    pyclass,
    types::{PyDict, PySet},
};

type GateMapType = IndexMap<String, Py<PropsMap>>;
type GateMapIterType = IntoIter<String>;

#[pyclass(sequence)]
pub struct GateMapKeys {
    keys: IndexSet<String>,
}

#[pymethods]
impl GateMapKeys {
    fn __iter__(slf: PyRef<Self>) -> PyResult<Py<GateMapIter>> {
        let iter = GateMapIter {
            iter: slf.keys.clone().into_iter(),
        };
        Py::new(slf.py(), iter)
    }

    fn __eq__(slf: PyRef<Self>, other: Bound<PySet>) -> PyResult<bool> {
        for item in other.iter() {
            let key = item.extract::<String>()?;
            if !(slf.keys.contains(&key)) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn __len__(slf: PyRef<Self>) -> usize {
        slf.keys.len()
    }

    fn __sub__(&self, other: HashSet<String>) -> GateMapKeys {
        GateMapKeys {
            keys: self
                .keys
                .iter()
                .cloned()
                .collect::<HashSet<String>>()
                .difference(&other)
                .cloned()
                .collect::<IndexSet<String>>(),
        }
    }

    fn __contains__(slf: PyRef<Self>, obj: String) -> PyResult<bool> {
        Ok(slf.keys.contains(&obj))
    }

    fn __repr__(slf: PyRef<Self>) -> String {
        let mut output = "gate_map_keys[".to_owned();
        output.push_str(slf.keys.iter().join(", ").as_str());
        output.push(']');
        output
    }
}

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

#[pyclass(mapping, module = "qiskit._accelerate.target")]
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

    pub fn __getitem__(&self, py: Python<'_>, key: String) -> PyResult<Py<PropsMap>> {
        if let Some(item) = self.map.get(&key) {
            Ok(item.clone_ref(py))
        } else {
            Err(PyKeyError::new_err(format!(
                "Key {:#?} not in target.",
                key
            )))
        }
    }

    #[pyo3(signature = (key, default=None))]
    fn get(slf: PyRef<Self>, key: String, default: Option<Bound<PyAny>>) -> PyObject {
        match slf.__getitem__(slf.py(), key) {
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
            iter: self.keys().keys.into_iter(),
        };
        Py::new(py, iter)
    }

    pub fn keys(&self) -> GateMapKeys {
        GateMapKeys {
            keys: self.map.keys().cloned().collect::<IndexSet<String>>(),
        }
    }

    pub fn values(&self) -> Vec<Py<PropsMap>> {
        self.map.values().cloned().collect_vec()
    }

    pub fn items(&self) -> Vec<(String, Py<PropsMap>)> {
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
