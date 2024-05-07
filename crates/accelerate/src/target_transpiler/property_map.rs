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

use hashbrown::{hash_set::IntoIter as HashSetIntoIter, HashSet};
use indexmap::{map::IntoKeys, IndexMap};
use itertools::Itertools;
use pyo3::{exceptions::PyKeyError, prelude::*, pyclass, types::PySequence};

use super::instruction_properties::InstructionProperties;
use super::qargs::{Qargs, QargsOrTuple};

enum PropsMapIterTypes {
    Iter(IntoKeys<Option<Qargs>, Option<Py<InstructionProperties>>>),
    Keys(HashSetIntoIter<Option<Qargs>>),
}

#[pyclass]
struct PropsMapIter {
    iter: PropsMapIterTypes,
}

#[pymethods]
impl PropsMapIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyObject {
        match &mut slf.iter {
            PropsMapIterTypes::Iter(iter) => iter.next().into_py(slf.py()),
            PropsMapIterTypes::Keys(iter) => iter.next().into_py(slf.py()),
        }
    }
}

#[pyclass(sequence)]
#[derive(Debug, Clone)]
pub struct PropsMapKeys {
    pub keys: HashSet<Option<Qargs>>,
}

#[pymethods]
impl PropsMapKeys {
    #[new]
    fn new(keys: HashSet<Option<Qargs>>) -> Self {
        Self { keys }
    }

    fn __iter__(slf: PyRef<Self>) -> PyResult<Py<PropsMapIter>> {
        let iter = PropsMapIter {
            iter: PropsMapIterTypes::Keys(slf.keys.clone().into_iter()),
        };
        Py::new(slf.py(), iter)
    }

    fn __eq__(slf: PyRef<Self>, other: Bound<PySequence>) -> PyResult<bool> {
        for item in other.iter()? {
            let qargs = item?
                .extract::<Option<QargsOrTuple>>()?
                .map(|qargs| qargs.parse_qargs());
            if !(slf.keys.contains(&qargs)) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn __len__(slf: PyRef<Self>) -> usize {
        slf.keys.len()
    }

    fn __contains__(slf: PyRef<Self>, obj: Option<QargsOrTuple>) -> PyResult<bool> {
        let obj = obj.map(|obj| obj.parse_qargs());
        Ok(slf.keys.contains(&obj))
    }

    fn __repr__(slf: PyRef<Self>) -> String {
        let mut output = "prop_map_keys[".to_owned();
        output.push_str(
            slf.keys
                .iter()
                .map(|x| {
                    if let Some(x) = x {
                        x.to_string()
                    } else {
                        "None".to_owned()
                    }
                })
                .join(", ")
                .as_str(),
        );
        output.push(']');
        output
    }
}

type PropsMapKV = IndexMap<Option<Qargs>, Option<Py<InstructionProperties>>>;
/**
   Mapping containing the properties of an instruction. Represents the relation
   ``Qarg : InstructionProperties``.

   Made to directly avoid conversions from an ``IndexMap`` structure in rust to a Python dict.
*/
#[pyclass(mapping)]
#[derive(Debug, Clone)]
pub struct PropsMap {
    pub map: PropsMapKV,
}

#[pymethods]
impl PropsMap {
    #[new]
    pub fn new(map: PropsMapKV) -> Self {
        PropsMap { map }
    }

    fn __contains__(&self, key: Bound<PyAny>) -> bool {
        if let Ok(key) = key.extract::<Option<QargsOrTuple>>() {
            let qarg = key.map(|qarg| qarg.parse_qargs());
            self.map.contains_key(&qarg)
        } else {
            false
        }
    }

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

    fn __len__(slf: PyRef<Self>) -> usize {
        slf.map.len()
    }

    fn __iter__(slf: PyRef<Self>) -> PyResult<Py<PropsMapIter>> {
        let iter = PropsMapIter {
            iter: PropsMapIterTypes::Iter(slf.map.clone().into_keys()),
        };
        Py::new(slf.py(), iter)
    }

    pub fn keys(&self) -> PropsMapKeys {
        PropsMapKeys::new(self.map.keys().cloned().collect())
    }

    fn values(&self) -> Vec<Option<Py<InstructionProperties>>> {
        self.map.clone().into_values().collect_vec()
    }

    fn items(&self) -> Vec<(Option<Qargs>, Option<Py<InstructionProperties>>)> {
        self.map.clone().into_iter().collect_vec()
    }
}
