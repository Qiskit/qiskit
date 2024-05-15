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

use hashbrown::HashSet;
use indexmap::{set::IntoIter as IndexSetIntoIter, IndexMap, IndexSet};
use itertools::Itertools;
use pyo3::types::{PyMapping, PySet};
use pyo3::{exceptions::PyKeyError, prelude::*, pyclass};

use super::instruction_properties::InstructionProperties;
use super::macro_rules::qargs_key_like_set_iterator;
use super::qargs::{Qargs, QargsOrTuple};

type KeyIterType = IndexSetIntoIter<Option<Qargs>>;
pub type PropsMapItemsType = Vec<(Option<Qargs>, Option<Py<InstructionProperties>>)>;

qargs_key_like_set_iterator!(
    PropsMapKeys,
    PropsMapIter,
    keys,
    KeyIterType,
    "",
    "props_map_keys"
);

type PropsMapKV = IndexMap<Option<Qargs>, Option<Py<InstructionProperties>>>;
/**
   Mapping containing the properties of an instruction. Represents the relation
   ``Qarg : InstructionProperties``.

   Made to directly avoid conversions from an ``IndexMap`` structure in rust to a Python dict.
*/
#[pyclass(mapping, module = "qiskit._accelerate.target")]
#[derive(Debug, Clone)]
pub struct PropsMap {
    pub map: PropsMapKV,
}

#[pymethods]
impl PropsMap {
    /// Create new instance of PropsMap.
    ///
    /// Args:
    ///     map (``dict[Qargs: InstructionProperties ]``):
    ///         mapping of optional ``Qargs`` and optional``InstructionProperties``.
    #[new]
    pub fn new(map: Option<PropsMapKV>) -> Self {
        match map {
            Some(map) => PropsMap { map },
            None => PropsMap::default(),
        }
    }

    /// Check whether some qargs are part of this PropsMap
    fn __contains__(&self, key: &Bound<PyAny>) -> bool {
        if let Ok(key) = key.extract::<Option<QargsOrTuple>>() {
            let qarg = key.map(|qarg| qarg.parse_qargs());
            self.map.contains_key(&qarg)
        } else {
            false
        }
    }

    /// Check whether the partial equality of two PropMaps.
    ///
    /// Partial equality is considered because ``InstructionProperties`` is non comparable.
    fn __eq__(slf: PyRef<Self>, other: &Bound<PyAny>) -> PyResult<bool> {
        if let Ok(dict) = other.downcast::<PyMapping>() {
            for key in dict.keys()?.iter()? {
                if let Ok(qargs) = key?.extract::<Option<QargsOrTuple>>() {
                    let qargs = qargs.map(|qargs| qargs.parse_qargs());
                    if !slf.map.contains_key(&qargs) {
                        return Ok(false);
                    }
                } else {
                    return Ok(false);
                }
            }
            Ok(true)
        } else if let Ok(prop_keys) = other.extract::<PropsMap>() {
            for key in prop_keys.map.keys() {
                if !slf.map.contains_key(key) {
                    return Ok(false);
                }
            }
            return Ok(true);
        } else {
            return Ok(false);
        }
    }

    /// Access a value in the GateMap using a Key.
    ///
    /// Args:
    ///     key (``Qargs``): The instruction name key.
    ///
    /// Return:
    ///     ``InstructionProperties`` object at that slot.
    /// Raises:
    ///     KeyError if the ``key`` is not in the ``PropsMap``.
    fn __getitem__(&self, py: Python<'_>, key: Option<QargsOrTuple>) -> PyResult<PyObject> {
        let key = key.map(|qargs| qargs.parse_qargs());
        if let Some(item) = self.map.get(&key) {
            Ok(item.to_object(py))
        } else {
            Err(PyKeyError::new_err(format!(
                "Key {:#?} not in target.",
                key.unwrap_or_default().vec
            )))
        }
    }

    /// Access a value in the GateMap using a Key.
    ///
    /// Args:
    ///     key (str): The instruction name key.
    ///     default (PyAny): The default value to be returned.
    ///
    /// Returns:
    ///    ``PropsMap`` value if found, otherwise returns ``default``.
    #[pyo3(signature = (key, default=None))]
    fn get(
        &self,
        py: Python<'_>,
        key: Option<QargsOrTuple>,
        default: Option<Bound<PyAny>>,
    ) -> PyObject {
        match self.__getitem__(py, key) {
            Ok(value) => value,
            Err(_) => match default {
                Some(value) => value.into(),
                None => py.None(),
            },
        }
    }
    /// Returns the number of keys present
    fn __len__(slf: PyRef<Self>) -> usize {
        slf.map.len()
    }

    /// Returns an iterator over the keys of the PropsMap.
    fn __iter__(slf: PyRef<Self>) -> PyResult<Py<PropsMapIter>> {
        let iter = PropsMapIter {
            iter: slf
                .map
                .keys()
                .cloned()
                .collect::<IndexSet<Option<Qargs>>>()
                .into_iter(),
        };
        Py::new(slf.py(), iter)
    }

    /// Returns an ordered set with all the Keys in the PropsMap.
    pub fn keys(&self) -> PropsMapKeys {
        PropsMapKeys {
            keys: self.map.keys().cloned().collect(),
        }
    }

    /// Returns a list with all the values in the PropsMap.
    fn values(&self) -> Vec<Option<Py<InstructionProperties>>> {
        self.map.values().cloned().collect_vec()
    }

    /// Returns a list with all they (key, value) pairs (``Qargs``, ``InstructionProperties``)
    fn items(&self) -> PropsMapItemsType {
        self.map.clone().into_iter().collect_vec()
    }

    fn __setstate__(&mut self, state: (PropsMapKV,)) -> PyResult<()> {
        self.map = state.0;
        Ok(())
    }

    fn __getstate__(&self) -> (PropsMapKV,) {
        (self.map.clone(),)
    }
}

impl Default for PropsMap {
    fn default() -> Self {
        Self {
            map: IndexMap::new(),
        }
    }
}
