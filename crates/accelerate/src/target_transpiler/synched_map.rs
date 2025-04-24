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

use std::hash::Hash;
use std::ops::Index;
use std::sync::OnceLock;

use indexmap::map::{Iter, Keys, Values};
use indexmap::Equivalent;
use indexmap::{map::IntoIter, IndexMap, IndexSet};
use pyo3::intern;
use pyo3::types::PyIterator;
use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    types::{IntoPyDict, PyDict},
};

/// This structure is supposed to mimic the behavior of a [PyDict] that is
/// strictly additive.
#[derive(Debug)]
pub struct SynchedMap<K, V> {
    /// The underlying IndexMap.
    map: IndexMap<K, V>,
    /// The cached python dictionary
    cached: OnceLock<Py<PyDict>>,
    /// The queue with the changed items to be updated.
    changed: IndexSet<usize>,
}

impl<K, V> SynchedMap<K, V>
where
    K: Hash + Eq,
{
    pub fn new() -> Self {
        Self::default()
    }

    // TODO: Remove once `Target` is built from rust.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: IndexMap::with_capacity(capacity),
            ..Default::default()
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let index = self.cached.get().map(|_| {
            if let Some(idx) = self.map.get_index_of(&key) {
                idx
            } else {
                self.map.len()
            }
        });
        let result = self.map.insert(key, value);
        if let Some(index) = index {
            self.changed.insert(index);
        }
        result
    }

    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        Q: ?Sized + Equivalent<K> + Hash + Eq,
    {
        self.map.contains_key(key)
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        Q: ?Sized + Equivalent<K> + Hash + Eq,
    {
        self.map.get(key)
    }

    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        Q: ?Sized + Equivalent<K> + Hash + Eq,
    {
        if let Some(Some(idx)) = self.cached.get().map(|_| self.map.get_index_of(key)) {
            self.changed.insert(idx);
        }
        self.map.get_mut(key)
    }

    pub fn get_index(&self, index: usize) -> Option<(&K, &V)> {
        self.map.get_index(index)
    }

    pub fn keys(&self) -> Keys<'_, K, V> {
        self.map.keys()
    }

    pub fn values(&self) -> Values<'_, K, V> {
        self.map.values()
    }

    pub fn iter(&self) -> Iter<'_, K, V> {
        self.map.iter()
    }

    pub fn into_iter(self) -> IntoIter<K, V> {
        self.map.into_iter()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

impl<'py, K, V> SynchedMap<K, V>
where
    K: IntoPyObject<'py> + Hash + Eq + Clone,
    V: IntoPyObject<'py> + Clone,
{
    fn validate(&mut self, py: Python<'py>) {
        if let Some(cell) = self.cached.get() {
            let bound = cell.bind(py);
            for index in std::mem::take(&mut self.changed) {
                let (key, value) = self
                    .map
                    .get_index(index)
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .unwrap();
                bound.set_item(key, value).unwrap();
            }
        } else {
            self.cached.get_or_init(|| {
                self.map
                    .clone()
                    .into_py_dict(py)
                    .map(|dict| dict.unbind())
                    .unwrap()
            });
        }
    }

    pub fn cached_py_dict(&mut self, py: Python<'py>) -> PyResult<&Py<PyDict>> {
        self.validate(py);
        self.cached.get().ok_or(PyRuntimeError::new_err(
            "Error extracting cache, maybe out of sync.",
        ))
    }
}

impl<'py, K, V> SynchedMap<K, V>
where
    K: IntoPyObject<'py> + FromPyObject<'py> + Hash + Eq + Clone,
    V: IntoPyObject<'py> + FromPyObject<'py> + Clone,
{
    pub fn py_insert(
        &mut self,
        py: Python<'py>,
        key: PyObject,
        value: PyObject,
    ) -> PyResult<Option<V>> {
        let key_native = key.extract(py)?;
        let value_native = value.extract(py)?;
        let cached = self.cached_py_dict(py)?;
        cached.bind(py).set_item(key, value)?;
        Ok(self.map.insert(key_native, value_native))
    }

    pub fn py_get(
        &mut self,
        py: Python<'py>,
        key: &PyObject,
        default: Option<PyObject>,
    ) -> PyResult<Option<Py<PyAny>>> {
        if let Some(value) = self.cached_py_dict(py)?.bind(py).get_item(key)? {
            Ok(Some(value.unbind()))
        } else {
            Ok(default)
        }
    }

    pub fn py_get_item(&mut self, py: Python<'py>, key: &PyObject) -> PyResult<Bound<'py, PyAny>> {
        self.cached_py_dict(py)?.bind(py).as_any().get_item(key)
    }

    pub fn py_keys(&mut self, py: Python<'py>) -> PyResult<PyObject> {
        self.cached_py_dict(py)?
            .call_method0(py, intern!(py, "keys"))
    }

    pub fn py_values(&mut self, py: Python<'py>) -> PyResult<PyObject> {
        self.cached_py_dict(py)?
            .call_method0(py, intern!(py, "values"))
    }

    pub fn py_items(&mut self, py: Python<'py>) -> PyResult<PyObject> {
        self.cached_py_dict(py)?
            .call_method0(py, intern!(py, "items"))
    }

    pub fn py_iter(&mut self, py: Python<'py>) -> PyResult<Py<PyIterator>> {
        Ok(self
            .cached_py_dict(py)?
            .bind(py)
            .as_any()
            .try_iter()?
            .unbind())
    }
}

impl<K, V> Default for SynchedMap<K, V> {
    fn default() -> Self {
        Self {
            map: Default::default(),
            cached: Default::default(),
            changed: Default::default(),
        }
    }
}

impl<'py, K, V> IntoPyObject<'py> for SynchedMap<K, V>
where
    K: IntoPyObject<'py> + Hash + Eq + Clone,
    V: IntoPyObject<'py> + Clone,
{
    type Target = PyDict;

    type Output = Bound<'py, PyDict>;

    type Error = PyErr;

    fn into_pyobject(mut self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        self.cached_py_dict(py).map(|ob| ob.bind(py).clone())
    }
}

impl<'py, K, V> FromPyObject<'py> for SynchedMap<K, V>
where
    K: FromPyObject<'py> + Hash + Eq,
    V: FromPyObject<'py>,
{
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let into_dict: Bound<'py, PyDict> = ob.clone().downcast_into()?;
        let extracted_map = into_dict.extract()?;
        Ok(Self {
            map: extracted_map,
            cached: into_dict.unbind().into(),
            changed: Default::default(),
        })
    }
}

impl<K, V> Clone for SynchedMap<K, V>
where
    K: Clone,
    V: Clone,
{
    fn clone(&self) -> Self {
        Self {
            map: self.map.clone(),
            cached: OnceLock::new(),
            changed: Default::default(),
        }
    }
}

impl<K, V> Extend<(K, V)> for SynchedMap<K, V>
where
    K: Hash + Eq,
{
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        self.map.extend(iter)
    }
}

impl<K, V> From<IndexMap<K, V>> for SynchedMap<K, V> {
    fn from(value: IndexMap<K, V>) -> Self {
        Self {
            map: value,
            ..Default::default()
        }
    }
}

impl<K, V> IntoIterator for SynchedMap<K, V> {
    type Item = (K, V);

    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.map.into_iter()
    }
}

impl<K, V, Q> Index<&Q> for SynchedMap<K, V>
where
    Q: ?Sized + Equivalent<K> + Hash,
{
    type Output = V;

    fn index(&self, index: &Q) -> &Self::Output {
        self.map.index(index)
    }
}
