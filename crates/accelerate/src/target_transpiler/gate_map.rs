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

use std::{ops::{Deref, DerefMut}, sync::OnceLock};

use indexmap::IndexMap;
use pyo3::{prelude::*, sync::OnceLockExt, types::{PyDict, PyTuple}};
use super::{instruction_properties::InstructionProperties, Qargs};

/// This structure is supposed to mimic the behavior of a [PyDict] without
/// strictly requiring any python interaction whatsoever.
#[derive(Debug, Clone, Default)]
pub(crate) struct GateMap {
    map: IndexMap<String, IndexMap<Qargs, Option<InstructionProperties>>>,
    cached: OnceLock<Py<PyDict>>
}

impl GateMap {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: IndexMap::with_capacity(capacity),
            ..Default::default()
        }
    }

    pub fn cached_py_dict(&self, py: Python) -> &PyDict {

    }

    fn generate_py_dict(&self, py: Python) -> Bound<PyDict> {
        let dict = PyDict::new(py);

    }
}

impl Deref for GateMap {
    type Target = IndexMap<String, IndexMap<Qargs, Option<InstructionProperties>>>;

    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

impl DerefMut for GateMap {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.map
    }
}


#[derive(Debug, Clone, Default)]
pub(crate) struct PropsMap {
    map: IndexMap<Qargs, Option<InstructionProperties>>,
    cached: OnceLock<Py<PyDict>>
}

impl PropsMap {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: IndexMap::with_capacity(capacity),
            ..Default::default()
        }
    }

    pub fn cached_py_dict(&self, py: Python) -> PyResult<&Py<PyDict>> {
        let res = self.cached.get_or_init_py_attached(py, || self.generate_py_dict(py).unwrap());
        let res_bound = res.bind(py);
        if res_bound.len() == self.map.len() {
            Ok(res)
        } else if res_bound.len() < self.map.len() {
            let len_diff = self.map.len() - res_bound.len();
            for idx in (self.map.len()-len_diff)..self.map.len() {
                let (key,val) = self.map
                res_bound.set_item(key, value)
            }
            Ok(res)
        }

    }

    fn generate_py_dict(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for (qargs, inst_prop) in self.map.iter() {
            dict.set_item(qargs.map(|items| PyTuple::new(py, items)).transpose()?, inst_prop.clone())?;
        }
        Ok(dict.unbind())
    }
}

impl Deref for PropsMap {
    type Target = IndexMap<Qargs, Option<InstructionProperties>>;

    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

impl DerefMut for PropsMap {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.map
    }
}