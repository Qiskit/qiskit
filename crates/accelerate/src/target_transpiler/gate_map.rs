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

use super::{macro_rules::key_like_set_iterator, property_map::PropsMap};
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

// Creates a Key-Like object for the Gate Map keys.
// This is done in an effort to keep the insertion order of the keys.
key_like_set_iterator!(
    GateMapKeys,
    GateMapIter,
    keys,
    String,
    IntoIter<String>,
    "",
    "gate_map_keys"
);

/**
Mapping of Instruction Names and ``PropsMaps`` (``Qargs``: ``InstructionProperties``) present
on the ``Target``.

This structure keeps track of which qubits an instruction is affecting and the properties of
said instruction on those qubits.
 */
#[pyclass(mapping, module = "qiskit._accelerate.target")]
#[derive(Debug, Clone)]
pub struct GateMap {
    pub map: GateMapType,
}

#[pymethods]
impl GateMap {
    /// Create empty instance of a GateMap.
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    /// Checks whether an instruction is present on the ``Target``'s gate map.
    pub fn __contains__(&self, key: &Bound<PyAny>) -> bool {
        if let Ok(key) = key.extract::<String>() {
            self.map.contains_key(&key)
        } else {
            false
        }
    }

    /// Check the equality of two gate_maps in the Python space.
    fn __eq__(slf: PyRef<Self>, other: &Bound<PyAny>) -> PyResult<bool> {
        if let Ok(dict) = other.downcast::<PyDict>() {
            for key in dict.keys() {
                if let Ok(key) = key.extract::<String>() {
                    if !slf.map.contains_key(&key) {
                        return Ok(false);
                    } else if let (Some(value), Ok(Some(other_value))) =
                        (slf.map.get(&key), dict.get_item(key))
                    {
                        let comparison = other_value.eq(value)?;
                        if !comparison {
                            return Ok(false);
                        }
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

    /// Access a value in the GateMap using a Key.
    ///
    /// Args:
    ///     key (str): The instruction name key.
    ///
    /// Return:
    ///     ``PropsMap`` object at that slot.
    /// Raises:
    ///     KeyError if the ``key`` is not in the ``GateMap``.
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

    /// Access a value in the GateMap using a Key.
    ///
    /// Args:
    ///     key (str): The instruction name key.
    ///     default (Option[PyAny]): The default value to be returned.
    ///
    /// Returns:
    ///    ``PropsMap`` value if found, otherwise returns ``default``.
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

    /// Returns number of present keys in the GateMap
    fn __len__(slf: PyRef<Self>) -> usize {
        slf.map.len()
    }

    /// Returns the iterator of the Keys in the GateMap.
    pub fn __iter__(&self, py: Python<'_>) -> PyResult<Py<GateMapIter>> {
        let iter = GateMapIter {
            iter: self.keys().keys.into_iter(),
        };
        Py::new(py, iter)
    }

    /// Returns the Keys in the GateMap as an ordered set of Strings.
    pub fn keys(&self) -> GateMapKeys {
        GateMapKeys {
            keys: self.map.keys().cloned().collect::<IndexSet<String>>(),
        }
    }

    /// Returns the values of the GateMap as a list of ``PropsMap`` objects.
    pub fn values(&self) -> Vec<Py<PropsMap>> {
        self.map.values().cloned().collect_vec()
    }

    /// Returns they (keys, values) pairs as a list of (``str``, ``PropsMap``)
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
