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

use std::{
    ops::{Deref, DerefMut},
    sync::OnceLock,
};

use super::{instruction_properties::InstructionProperties, Qargs};
use indexmap::{map::IntoIter, IndexMap};
use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    sync::OnceLockExt,
    types::{IntoPyDict, PyDict},
};

/// This structure is supposed to mimic the behavior of a [PyDict]
/// for a `Target's` gate map without requiring to
#[derive(Debug, Default)]
pub struct GateMap {
    map: IndexMap<String, PropsMap>,
    cached: OnceLock<Py<PyDict>>,
}

impl GateMap {
    pub fn new() -> Self {
        Default::default()
    }

    // TODO: Remove once `Target` is built from rust.
    #[allow(dead_code)]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: IndexMap::with_capacity(capacity),
            ..Default::default()
        }
    }

    pub fn cached_py_dict(&self, py: Python) -> &Py<PyDict> {
        self.cached
            .get_or_init(|| self.map.clone().into_py_dict(py).unwrap().into())
    }
}

impl Clone for GateMap {
    fn clone(&self) -> Self {
        GateMap {
            map: self.map.clone(),
            cached: Default::default(),
        }
    }
}

impl<'py> IntoPyObject<'py> for GateMap {
    type Target = PyDict;

    type Output = Bound<'py, PyDict>;

    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.cached_py_dict(py).clone_ref(py).into_bound(py))
    }
}

impl<'py, 'a> IntoPyObject<'py> for &'a GateMap {
    type Target = PyDict;

    type Output = Borrowed<'a, 'py, PyDict>;

    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.cached_py_dict(py).bind_borrowed(py))
    }
}

impl<'py> FromPyObject<'py> for GateMap {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(Self {
            map: ob.extract()?,
            cached: ob.downcast::<PyDict>()?.clone().unbind().into(),
        })
    }
}

impl Deref for GateMap {
    type Target = IndexMap<String, PropsMap>;

    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

impl DerefMut for GateMap {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.cached.take();
        &mut self.map
    }
}

#[derive(Debug, Default)]
pub struct PropsMap {
    map: IndexMap<Qargs, Option<InstructionProperties>>,
    cached: OnceLock<Py<PyDict>>,
}

impl PropsMap {
    // TODO: Remove once `Target` can be created python free.
    #[allow(dead_code)]
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
        let res = self
            .cached
            .get_or_init_py_attached(py, || self.generate_py_dict(py).unwrap());
        let res_bound = res.bind(py);
        let res_bound_len = res_bound.len();
        if res_bound.len() == self.map.len() {
            Ok(res)
        } else if res_bound.len() < self.map.len() {
            for idx in res_bound_len..self.map.len() {
                let (key, val) = self.map.get_index(idx).unwrap();
                res_bound.set_item(key.as_ref(), val.clone())?;
            }
            Ok(res)
        } else {
            Err(PyRuntimeError::new_err("The python and rust counterparts of this gatemap are out of sync, perhaps something modified it?"))
        }
    }

    fn generate_py_dict(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for (qargs, inst_prop) in self.map.iter() {
            dict.set_item(qargs.as_ref(), inst_prop.clone())?;
        }
        Ok(dict.unbind())
    }
}

impl IntoIterator for PropsMap {
    type Item = (Qargs, Option<InstructionProperties>);

    type IntoIter = IntoIter<Qargs, Option<InstructionProperties>>;

    fn into_iter(self) -> Self::IntoIter {
        self.map.into_iter()
    }
}

impl From<IndexMap<Qargs, Option<InstructionProperties>>> for PropsMap {
    fn from(value: IndexMap<Qargs, Option<InstructionProperties>>) -> Self {
        Self {
            map: value,
            ..Default::default()
        }
    }
}

impl Clone for PropsMap {
    fn clone(&self) -> Self {
        PropsMap {
            map: self.map.clone(),
            cached: Default::default(),
        }
    }
}

impl<'py> IntoPyObject<'py> for PropsMap {
    type Target = PyDict;

    type Output = Bound<'py, PyDict>;

    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.cached_py_dict(py)?.clone_ref(py).into_bound(py))
    }
}

impl<'py, 'a> IntoPyObject<'py> for &'a PropsMap {
    type Target = PyDict;

    type Output = Borrowed<'a, 'py, PyDict>;

    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.cached_py_dict(py)?.bind_borrowed(py))
    }
}

impl<'py> FromPyObject<'py> for PropsMap {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let ob = ob.downcast::<PyDict>()?;
        let mut map: IndexMap<Qargs, Option<InstructionProperties>> =
            IndexMap::with_capacity(ob.len() - 1);
        for (key, val) in ob.iter() {
            let key: Qargs = key.extract()?;
            let val: Option<InstructionProperties> = val.extract()?;
            map.insert(key, val);
        }
        Ok(Self {
            map,
            cached: ob.clone().unbind().into(),
        })
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
