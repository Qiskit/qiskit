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

use pyo3::prelude::*;
use pyo3::{import_exception, intern, PyObject};

import_exception!(qiskit.circuit.exceptions, CircuitError);

use hashbrown::{HashMap, HashSet};

/// The index value in a `ParamEntry` that indicates the global phase.
pub const GLOBAL_PHASE_INDEX: usize = usize::MAX;

#[pyclass(freelist = 20, module = "qiskit._accelerate.circuit")]
pub(crate) struct ParamEntryKeys {
    keys: Vec<(usize, usize)>,
    iter_pos: usize,
}

#[pymethods]
impl ParamEntryKeys {
    fn __iter__(slf: PyRef<Self>) -> Py<ParamEntryKeys> {
        slf.into()
    }

    fn __next__(mut slf: PyRefMut<Self>) -> Option<(usize, usize)> {
        if slf.iter_pos < slf.keys.len() {
            let res = Some(slf.keys[slf.iter_pos]);
            slf.iter_pos += 1;
            res
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
#[pyclass(freelist = 20, module = "qiskit._accelerate.circuit")]
pub(crate) struct ParamEntry {
    /// Mapping of tuple of instruction index (in CircuitData) and parameter index to the actual
    /// parameter object
    pub index_ids: HashSet<(usize, usize)>,
}

impl ParamEntry {
    pub fn add(&mut self, inst_index: usize, param_index: usize) {
        self.index_ids.insert((inst_index, param_index));
    }

    pub fn discard(&mut self, inst_index: usize, param_index: usize) {
        self.index_ids.remove(&(inst_index, param_index));
    }
}

#[pymethods]
impl ParamEntry {
    #[new]
    pub fn new(inst_index: usize, param_index: usize) -> Self {
        ParamEntry {
            index_ids: HashSet::from([(inst_index, param_index)]),
        }
    }

    pub fn __len__(&self) -> usize {
        self.index_ids.len()
    }

    pub fn __contains__(&self, key: (usize, usize)) -> bool {
        self.index_ids.contains(&key)
    }

    pub fn __iter__(&self) -> ParamEntryKeys {
        ParamEntryKeys {
            keys: self.index_ids.iter().copied().collect(),
            iter_pos: 0,
        }
    }
}

#[derive(Clone, Debug)]
#[pyclass(freelist = 20, module = "qiskit._accelerate.circuit")]
pub(crate) struct ParamTable {
    /// Mapping of parameter uuid (as an int) to the Parameter Entry
    pub table: HashMap<u128, ParamEntry>,
    /// Mapping of parameter name to uuid as an int
    pub names: HashMap<String, u128>,
    /// Mapping of uuid to a parameter object
    pub uuid_map: HashMap<u128, PyObject>,
}

impl ParamTable {
    pub fn insert(&mut self, py: Python, parameter: PyObject, entry: ParamEntry) -> PyResult<()> {
        let uuid: u128 = parameter
            .getattr(py, intern!(py, "_uuid"))?
            .getattr(py, intern!(py, "int"))?
            .extract(py)?;
        let name: String = parameter.getattr(py, intern!(py, "name"))?.extract(py)?;

        if self.names.contains_key(&name) && !self.table.contains_key(&uuid) {
            return Err(CircuitError::new_err(format!(
                "Name conflict on adding parameter: {}",
                name
            )));
        }
        self.table.insert(uuid, entry);
        self.names.insert(name, uuid);
        self.uuid_map.insert(uuid, parameter);
        Ok(())
    }

    pub fn discard_references(
        &mut self,
        uuid: u128,
        inst_index: usize,
        param_index: usize,
        name: String,
    ) {
        if let Some(refs) = self.table.get_mut(&uuid) {
            if refs.__len__() == 1 {
                self.table.remove(&uuid);
                self.names.remove(&name);
                self.uuid_map.remove(&uuid);
            } else {
                refs.discard(inst_index, param_index);
            }
        }
    }
}

#[pymethods]
impl ParamTable {
    #[new]
    pub fn new() -> Self {
        ParamTable {
            table: HashMap::new(),
            names: HashMap::new(),
            uuid_map: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.table.clear();
        self.names.clear();
        self.uuid_map.clear();
    }

    pub fn pop(&mut self, key: u128, name: &str) -> Option<ParamEntry> {
        self.names.remove(name);
        self.uuid_map.remove(&key);
        self.table.remove(&key)
    }

    fn set(&mut self, uuid: u128, name: String, param: PyObject, refs: ParamEntry) {
        self.names.insert(name, uuid);
        self.table.insert(uuid, refs);
        self.uuid_map.insert(uuid, param);
    }

    pub fn get_param_from_name(&self, py: Python, name: String) -> Option<PyObject> {
        self.names
            .get(&name)
            .map(|x| self.uuid_map.get(x).map(|y| y.clone_ref(py)))?
    }
}
