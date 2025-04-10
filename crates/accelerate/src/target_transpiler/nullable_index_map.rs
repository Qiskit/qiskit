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

use ahash::RandomState;
use indexmap::{
    map::{IntoIter as BaseIntoIter, Iter as BaseIter, Keys as BaseKeys, Values as BaseValues},
    Equivalent, IndexMap,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustworkx_core::dictmap::InitWithHasher;
use std::ops::Index;
use std::{hash::Hash, mem::swap};
type BaseMap<K, V> = IndexMap<K, V, RandomState>;

///
/// An `IndexMap`-like structure thet can be used when one of the keys can have a `None` value.
///
/// This structure is essentially a wrapper around the `IndexMap<K, V>` struct that allows the
/// storage of `Option<K>` key values as `K`` and keep an extra slot reserved only for the
/// `None` instance. There are some upsides to this including:
///
/// The ability to index using Option<&K> to index a specific key.
/// Store keys as non option wrapped to obtain references to K instead of reference to Option<K>.
///
/// **Warning:** This is an experimental feature and should be used with care as it does not
/// fully implement all the methods present in `IndexMap<K, V>` due to API limitations.
#[derive(Debug)]
pub struct NullableIndexMap<K, V> {
    map: BaseMap<K, V>,
    null_val: Option<(usize, V)>,
}

// Implement `Clone` manually to make it an implicit trait.
impl<K, V> Clone for NullableIndexMap<K, V>
where
    K: Clone,
    V: Clone,
{
    fn clone(&self) -> Self {
        Self {
            map: self.map.clone(),
            null_val: self.null_val.clone(),
        }
    }
}

impl<K, V> NullableIndexMap<K, V> {
    /// Creates an instance of `NullableIndexMap<K, V>`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an instance of `NullableIndexMap<K, V>` with capacity to hold `n`+1 key-value
    /// pairs.
    ///
    /// Notice that an extra space needs to be alotted to store the instance of `None` a
    /// key.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            map: BaseMap::with_capacity(n),
            null_val: None,
        }
    }

    /// Queries into the index of the element within the `IndexMap` if it exists.
    // TODO: Remove once `NullableIndexMap` is being consumed.
    #[allow(dead_code)]
    pub fn get_index(&self, key: usize) -> Option<(Option<&K>, &V)> {
        if let Some((index, val)) = self.null_val.as_ref() {
            if index == &key {
                Some((None, val))
            } else {
                let key = if &key > index { key - 1 } else { key };
                self.map.get_index(key).map(|(k, v)| (Some(k), v))
            }
        } else {
            self.map.get_index(key).map(|(k, v)| (Some(k), v))
        }
    }

    /// Returns an iterator over references of the key-value pairs of the map.
    // TODO: Remove once `NullableIndexMap` is being consumed.
    #[allow(dead_code)]
    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            map: self.map.iter(),
            null_value: self.null_val.as_ref(),
            current_index: 0,
        }
    }

    /// Returns an iterator over references of the keys present in the map.
    pub fn keys(&self) -> Keys<K, V> {
        Keys {
            map_keys: self.map.keys(),
            null_value: self.null_val.as_ref().map(|(i, _)| *i),
            current_index: 0,
        }
    }

    /// Returns an iterator over references of all the values present in the map.
    pub fn values(&self) -> Values<K, V> {
        Values {
            map_values: self.map.values(),
            null_value: self.null_val.as_ref(),
            current_index: 0,
        }
    }

    /// Returns the number of key-value pairs present in the map.
    pub fn len(&self) -> usize {
        self.map.len() + self.null_val.is_some() as usize
    }
}

impl<K, V> NullableIndexMap<K, V>
where
    K: Hash + Eq,
{
    /// Inserts a `value` in the slot alotted to `key`.
    ///
    /// If a previous value existed there previously it will be returned, otherwise
    /// `None` will be returned.
    pub fn insert(&mut self, key: Option<K>, value: V) -> Option<V> {
        match key {
            Some(key) => self.map.insert(key, value),
            None => {
                let mut old_val = Some((self.len(), value));
                swap(&mut old_val, &mut self.null_val);
                old_val.map(|val| val.1)
            }
        }
    }
}

impl<K, V> NullableIndexMap<K, V> {
    /// Returns a reference to the value stored at `key`, if it does not exist
    /// `None` is returned instead.
    pub fn get<Q>(&self, key: Option<&Q>) -> Option<&V>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        match key {
            Some(key) => self.map.get(key),
            None => self.null_val.as_ref().map(|val| &val.1),
        }
    }

    /// Returns a mutable reference to the value stored at `key`, if it does not
    /// exist `None` is returned instead.
    pub fn get_mut<Q>(&mut self, key: Option<&Q>) -> Option<&mut V>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        match key {
            Some(key) => self.map.get_mut(key),
            None => self.null_val.as_mut().map(|val| &mut val.1),
        }
    }

    /// Returns `true` if the map contains a slot indexed by `key`, otherwise `false`.
    pub fn contains_key<Q>(&self, key: Option<&Q>) -> bool
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        match key {
            Some(key) => self.map.contains_key(key),
            None => self.null_val.is_some(),
        }
    }

    /// Removes the entry allotted to `key` from the map and returns it. The index of
    /// this entry is then replaced by the entry located at the last index.
    ///
    /// In the case of a `None` entry, the index of `None` will be preserved relative
    /// to the length of the map, while the last "real" value will be swapped.
    ///
    /// `None` will be returned if the `key` is not present in the map.
    pub fn swap_remove<Q>(&mut self, key: Option<&Q>) -> Option<V>
    where
        Q: ?Sized + Hash + Eq + Equivalent<K>,
    {
        match key {
            Some(key) => {
                let ret = self.map.swap_remove(key);
                if self
                    .null_val
                    .as_ref()
                    .is_some_and(|(idx, _)| self.len() <= *idx)
                {
                    if let Some((key, _)) = self.null_val.as_mut() {
                        *key -= 1;
                    }
                }
                ret
            }
            None => self.null_val.take().map(|(_, val)| val),
        }
    }

    /// Removes the entry allotted to `key` from the map and returns it.
    /// The indices of the elements following it will then shift.
    ///
    /// `None` will be returned if the `key` is not present in the map.
    pub fn shift_remove<Q>(&mut self, key: Option<&Q>) -> Option<V>
    where
        Q: ?Sized + Hash + Eq + Equivalent<K>,
    {
        match key {
            Some(key) => {
                if let Some(index) = self.map.get_index_of(key) {
                    if self.null_val.as_ref().is_some_and(|(idx, _)| index < *idx) {
                        if let Some((key, _)) = self.null_val.as_mut() {
                            *key -= 1;
                        }
                    }
                    return self.map.shift_remove(key);
                }
                None
            }
            None => self.null_val.take().map(|(_, val)| val),
        }
    }

    /// Queries into the index of the element within the `IndexMap` if it exists.
    // TODO: Remove once `NullableIndexMap` is being consumed.
    #[allow(dead_code)]
    pub fn get_index_of<Q>(&self, key: Option<&Q>) -> Option<usize>
    where
        Q: ?Sized + Hash + Eq + Equivalent<K>,
    {
        match key {
            Some(key) => self.map.get_index_of(key).map(|idx| {
                if self
                    .null_val
                    .as_ref()
                    .is_some_and(|(n_idx, _)| &idx > n_idx)
                {
                    idx + 1
                } else {
                    idx
                }
            }),
            None => self.null_val.as_ref().map(|(idx, _)| *idx),
        }
    }
}

impl<K, V> FromIterator<(Option<K>, V)> for NullableIndexMap<K, V>
where
    K: Hash + Eq,
{
    fn from_iter<T: IntoIterator<Item = (Option<K>, V)>>(iter: T) -> Self {
        let mut null_val = None;
        let filtered = iter
            .into_iter()
            .enumerate()
            .filter_map(|(index, item)| match item {
                (Some(key), value) => Some((key, value)),
                (None, value) => {
                    null_val = Some((index, value));
                    None
                }
            });
        Self {
            map: IndexMap::from_iter(filtered),
            null_val,
        }
    }
}

impl<K, V> Extend<(Option<K>, V)> for NullableIndexMap<K, V>
where
    K: Hash + Eq,
{
    fn extend<T: IntoIterator<Item = (Option<K>, V)>>(&mut self, iter: T) {
        let filtered = iter
            .into_iter()
            .enumerate()
            .filter_map(|(index, item)| match item {
                (Some(key), value) => Some((key, value)),
                (None, value) => {
                    self.null_val = Some((index, value));
                    None
                }
            });
        self.map.extend(filtered)
    }
}

impl<K, V> IntoIterator for NullableIndexMap<K, V> {
    type Item = (Option<K>, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            map: self.map.into_iter(),
            null_value: self.null_val,
            current_index: 0,
        }
    }
}

/// Iterator for the key-value pairs in `NullableIndexMap`.
pub struct Iter<'a, K, V> {
    map: BaseIter<'a, K, V>,
    null_value: Option<&'a (usize, V)>,
    current_index: usize,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (Option<&'a K>, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        let val = if self
            .null_value
            .as_ref()
            .is_some_and(|(index, _)| *index == self.current_index)
        {
            Some((None, &self.null_value.take().unwrap().1))
        } else {
            self.map.next().map(|(k, v)| (Some(k), v))
        };
        val.inspect(|_| {
            self.current_index += 1;
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.map.size_hint().0 + self.null_value.is_some() as usize,
            self.map
                .size_hint()
                .1
                .map(|hint| hint + self.null_value.is_some() as usize),
        )
    }
}

impl<K, V> ExactSizeIterator for Iter<'_, K, V> {
    fn len(&self) -> usize {
        self.map.len() + self.null_value.is_some() as usize
    }
}

/// Owned iterator over the key-value pairs in `NullableIndexMap`.
pub struct IntoIter<K, V> {
    map: BaseIntoIter<K, V>,
    null_value: Option<(usize, V)>,
    current_index: usize,
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (Option<K>, V);

    fn next(&mut self) -> Option<Self::Item> {
        let val = if self
            .null_value
            .as_ref()
            .is_some_and(|(index, _)| *index == self.current_index)
        {
            Some((None, self.null_value.take().unwrap().1))
        } else {
            self.map.next().map(|(k, v)| (Some(k), v))
        };
        val.inspect(|_| {
            self.current_index += 1;
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.map.size_hint().0 + self.null_value.is_some() as usize,
            self.map
                .size_hint()
                .1
                .map(|hint| hint + self.null_value.is_some() as usize),
        )
    }
}

impl<K, V> ExactSizeIterator for IntoIter<K, V>
where
    V: Clone,
{
    fn len(&self) -> usize {
        self.map.len() + self.null_value.is_some() as usize
    }
}

/// Iterator over the keys of a `NullableIndexMap`.
pub struct Keys<'a, K, V> {
    map_keys: BaseKeys<'a, K, V>,
    null_value: Option<usize>,
    current_index: usize,
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = Option<&'a K>;

    fn next(&mut self) -> Option<Self::Item> {
        let val = if self
            .null_value
            .as_ref()
            .is_some_and(|index| *index == self.current_index)
        {
            Some(None)
        } else {
            self.map_keys.next().map(Some)
        };

        val.inspect(|_| {
            self.current_index += 1;
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.map_keys.size_hint().0 + self.null_value.is_some() as usize,
            self.map_keys
                .size_hint()
                .1
                .map(|hint| hint + self.null_value.is_some() as usize),
        )
    }
}

impl<K, V> ExactSizeIterator for Keys<'_, K, V> {
    fn len(&self) -> usize {
        self.map_keys.len() + self.null_value.is_some() as usize
    }
}

/// Iterator over the values of a `NullableIndexMap`.
pub struct Values<'a, K, V> {
    map_values: BaseValues<'a, K, V>,
    null_value: Option<&'a (usize, V)>,
    current_index: usize,
}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        let val = if self
            .null_value
            .as_ref()
            .is_some_and(|(index, _)| *index == self.current_index)
        {
            Some(&self.null_value.take().unwrap().1)
        } else {
            self.map_values.next()
        };
        val.inspect(|_| {
            self.current_index += 1;
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (
            self.map_values.size_hint().0 + self.null_value.is_some() as usize,
            self.map_values
                .size_hint()
                .1
                .map(|hint| hint + self.null_value.is_some() as usize),
        )
    }
}

impl<K, V> ExactSizeIterator for Values<'_, K, V> {
    fn len(&self) -> usize {
        self.map_values.len() + self.null_value.is_some() as usize
    }
}

impl<K, V> Index<Option<&K>> for NullableIndexMap<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    type Output = V;
    fn index(&self, index: Option<&K>) -> &Self::Output {
        match index {
            Some(k) => self.map.index(k),
            None => match &self.null_val {
                Some(val) => &val.1,
                None => panic!("The provided key is not present in map: None"),
            },
        }
    }
}

impl<K, V> Index<usize> for NullableIndexMap<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    type Output = V;
    fn index(&self, index: usize) -> &Self::Output {
        self.get_index(index)
            .unwrap_or_else(|| {
                panic!(
                    "index out of bounds: the len is {len} but the index is {index}",
                    len = self.len()
                );
            })
            .1
    }
}

impl<K, V> Default for NullableIndexMap<K, V> {
    fn default() -> Self {
        Self {
            map: IndexMap::default(),
            null_val: None,
        }
    }
}

impl<'py, K, V> FromPyObject<'py> for NullableIndexMap<K, V>
where
    K: FromPyObject<'py> + Eq + Hash,
    V: FromPyObject<'py>,
{
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let ob_as_dict = ob.downcast::<PyDict>()?;
        let mut null_val: Option<(usize, V)> = None;
        // For this capacity we might be off by 1 item exactly.
        let mut indexmap: IndexMap<K, V, RandomState> =
            IndexMap::with_capacity_and_hasher(ob.len()?, RandomState::new());
        for (idx, (key, val)) in ob_as_dict.iter().enumerate() {
            let key_as_type: Option<K> = key.extract()?;
            let val_as_type: V = val.extract()?;
            match key_as_type {
                Some(key) => {
                    indexmap.insert(key, val_as_type);
                }
                None => {
                    null_val = Some((idx, val_as_type));
                }
            }
        }
        Ok(Self {
            map: indexmap,
            null_val,
        })
    }
}

impl<'py, K, V> IntoPyObject<'py> for NullableIndexMap<K, V>
where
    K: IntoPyObject<'py>,
    V: IntoPyObject<'py>,
{
    type Target = PyDict;
    type Output = Bound<'py, PyDict>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let py_dict = PyDict::new(py);
        for (key, value) in self.into_iter() {
            py_dict.set_item(key, value)?;
        }
        Ok(py_dict)
    }
}

#[cfg(test)]
mod test {
    use super::NullableIndexMap;

    #[test]
    fn test_insert_index() {
        // Test adding items and testing insertion order.
        let mut map: NullableIndexMap<&str, &str> = NullableIndexMap::new();

        // Add a non-null value.
        map.insert(Some("Hello"), "Hi!");
        assert_eq!(map.len(), 1);
        assert_eq!("Hi!", map[Some(&"Hello")]);
        assert_eq!((Some(&"Hello"), &"Hi!"), map.get_index(0).unwrap());

        // Add a null value.
        map.insert(None, "Bye!");
        assert_eq!(map.len(), 2);
        assert_eq!((None, &"Bye!"), map.get_index(1).unwrap());
        assert_eq!("Bye!", map[None]);
        assert_eq!((Some(&"Hello"), &"Hi!"), map.get_index(0).unwrap());

        // Add another non-null value.
        map.insert(Some("Goodbye"), "See ya!");
        assert_eq!("See ya!", map[Some(&"Goodbye")]);
        assert_eq!((Some(&"Hello"), &"Hi!"), map.get_index(0).unwrap());
        assert_eq!((None, &"Bye!"), map.get_index(1).unwrap());
        assert_eq!((Some(&"Goodbye"), &"See ya!"), map.get_index(2).unwrap());
    }

    #[test]
    fn test_from_iter() {
        let k_v_pairs: [(Option<i32>, &str); 4] = [
            (Some(2), "Yay"),
            (Some(1), "Nay"),
            (None, "..."),
            (Some(4), "Schrodinger"),
        ];
        let ind_map: NullableIndexMap<i32, &str> = k_v_pairs.into_iter().collect();

        // Check length
        assert_eq!(ind_map.len(), k_v_pairs.len());
        // Check item order
        assert_eq!(
            &k_v_pairs,
            ind_map
                .iter()
                .map(|(k, v)| (k.cloned(), *v))
                .collect::<Vec<(Option<i32>, &str)>>()
                .as_slice()
        );
        // Check key order
        assert_eq!(
            k_v_pairs
                .iter()
                .map(|(k, _)| k.as_ref())
                .collect::<Vec<_>>(),
            ind_map.keys().collect::<Vec<_>>()
        );
        // Check value order
        assert_eq!(
            k_v_pairs.iter().map(|(_, v)| v).collect::<Vec<_>>(),
            ind_map.values().collect::<Vec<_>>()
        );

        for index in 0..ind_map.len() {
            // Check key indexing access
            assert_eq!(
                k_v_pairs[index].0,
                ind_map.get_index(index).unwrap().0.copied()
            );
            // Check value indexing access
            assert_eq!(k_v_pairs[index].1, ind_map[index])
        }
    }

    #[test]
    fn test_extend() {
        let k_v_pairs: [(Option<i32>, &str); 4] = [
            (Some(2), "Yay"),
            (Some(1), "Nay"),
            (None, "..."),
            (Some(4), "Schrodinger"),
        ];
        let mut ind_map: NullableIndexMap<i32, &str> = k_v_pairs.into_iter().collect();

        // Test extend
        let k_v_pairs_extend: [(Option<i32>, &str); 2] = [(Some(3), "Whaddis"), (None, "Cat")];
        ind_map.extend(k_v_pairs_extend);

        for (key, value) in k_v_pairs_extend.iter() {
            // Check if new values are updated.
            assert_eq!(ind_map[key.as_ref()], *value)
        }

        // Check complete map
        let ideal = [
            (Some(2), "Yay"),
            (Some(1), "Nay"),
            (Some(4), "Schrodinger"),
            (Some(3), "Whaddis"),
            (None, "Cat"),
        ];
        assert_eq!(&ideal, ind_map.into_iter().collect::<Vec<_>>().as_slice());
    }

    #[test]
    fn test_swap_remove() {
        let k_v_pairs = [
            (Some(2), "Yay"),
            (Some(1), "Nay"),
            (None, "..."),
            (Some(4), "Schrodinger"),
            (Some(3), "OIIAI"),
        ];
        let mut ind_map: NullableIndexMap<i32, &str> = k_v_pairs.into_iter().collect();

        // Check length
        assert_eq!(ind_map.len(), k_v_pairs.len());

        // Remove first element
        // Shift should happen between the first and last real elements
        // Null should remain unaffected
        ind_map.swap_remove(Some(&2));

        // Check length
        assert_eq!(ind_map.len(), k_v_pairs.len() - 1);
        // Check order
        let new_order = [
            (Some(3), "OIIAI"),
            (Some(1), "Nay"),
            (None, "..."),
            (Some(4), "Schrodinger"),
        ];
        // Check item order
        assert_eq!(
            &new_order,
            ind_map
                .iter()
                .map(|(k, v)| (k.cloned(), *v))
                .collect::<Vec<(Option<i32>, &str)>>()
                .as_slice()
        );
        // Check key order
        assert_eq!(
            new_order
                .iter()
                .map(|(k, _)| k.as_ref())
                .collect::<Vec<_>>(),
            ind_map.keys().collect::<Vec<_>>()
        );
        // Check value order
        assert_eq!(
            new_order.iter().map(|(_, v)| v).collect::<Vec<_>>(),
            ind_map.values().collect::<Vec<_>>()
        );

        // Remove last element
        // Nothing should change within the map
        ind_map.swap_remove(Some(&4));

        // Check length
        assert_eq!(ind_map.len(), k_v_pairs.len() - 2);
        // Check order
        let new_order = [(Some(3), "OIIAI"), (Some(1), "Nay"), (None, "...")];
        // Check item order
        assert_eq!(
            &new_order,
            ind_map
                .iter()
                .map(|(k, v)| (k.cloned(), *v))
                .collect::<Vec<(Option<i32>, &str)>>()
                .as_slice()
        );
        // Check key order
        assert_eq!(
            new_order
                .iter()
                .map(|(k, _)| k.as_ref())
                .collect::<Vec<_>>(),
            ind_map.keys().collect::<Vec<_>>()
        );
        // Check value order
        assert_eq!(
            new_order.iter().map(|(_, v)| v).collect::<Vec<_>>(),
            ind_map.values().collect::<Vec<_>>()
        );

        // Remove element before a `None` element, but not first
        // Nothing should change within the map
        ind_map.swap_remove(Some(&1));

        // Check length
        assert_eq!(ind_map.len(), k_v_pairs.len() - 3);
        // Check order
        let new_order = [(Some(3), "OIIAI"), (None, "...")];
        // Check item order
        assert_eq!(
            &new_order,
            ind_map
                .iter()
                .map(|(k, v)| (k.cloned(), *v))
                .collect::<Vec<(Option<i32>, &str)>>()
                .as_slice()
        );
        // Check key order
        assert_eq!(
            new_order
                .iter()
                .map(|(k, _)| k.as_ref())
                .collect::<Vec<_>>(),
            ind_map.keys().collect::<Vec<_>>()
        );
        // Check value order
        assert_eq!(
            new_order.iter().map(|(_, v)| v).collect::<Vec<_>>(),
            ind_map.values().collect::<Vec<_>>()
        );

        // Remove `None` element
        // Nothing should change within the map
        ind_map.swap_remove(None::<&i32>);
        // Check length
        assert_eq!(ind_map.len(), k_v_pairs.len() - 4);
        // Check order
        let new_order = [(Some(3), "OIIAI")];
        // Check item order
        assert_eq!(
            &new_order,
            ind_map
                .iter()
                .map(|(k, v)| (k.cloned(), *v))
                .collect::<Vec<(Option<i32>, &str)>>()
                .as_slice()
        );
        // Check key order
        assert_eq!(
            new_order
                .iter()
                .map(|(k, _)| k.as_ref())
                .collect::<Vec<_>>(),
            ind_map.keys().collect::<Vec<_>>()
        );
        // Check value order
        assert_eq!(
            new_order.iter().map(|(_, v)| v).collect::<Vec<_>>(),
            ind_map.values().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_shift_remove() {
        let mut k_v_pairs = vec![
            (Some(2), "Yay"),
            (Some(1), "Nay"),
            (None, "..."),
            (Some(4), "Schrodinger"),
            (Some(3), "OIIAI"),
        ];
        let mut ind_map: NullableIndexMap<i32, &str> = k_v_pairs.clone().into_iter().collect();

        // Check length
        assert_eq!(ind_map.len(), k_v_pairs.len());

        // Remove first element
        // Order should be preserved
        ind_map.shift_remove(Some(&2));
        k_v_pairs.remove(0);

        // Check length
        assert_eq!(ind_map.len(), k_v_pairs.len());
        // Check item order
        assert_eq!(
            &k_v_pairs,
            ind_map
                .iter()
                .map(|(k, v)| (k.cloned(), *v))
                .collect::<Vec<(Option<i32>, &str)>>()
                .as_slice()
        );
        // Check key order
        assert_eq!(
            k_v_pairs
                .iter()
                .map(|(k, _)| k.as_ref())
                .collect::<Vec<_>>(),
            ind_map.keys().collect::<Vec<_>>()
        );
        // Check value order
        assert_eq!(
            k_v_pairs.iter().map(|(_, v)| v).collect::<Vec<_>>(),
            ind_map.values().collect::<Vec<_>>()
        );

        // Remove last element
        // Order should be preserved
        ind_map.shift_remove(Some(&4));
        k_v_pairs.remove(2);

        // Check length
        assert_eq!(ind_map.len(), k_v_pairs.len());
        // Check item order
        assert_eq!(
            &k_v_pairs,
            ind_map
                .iter()
                .map(|(k, v)| (k.cloned(), *v))
                .collect::<Vec<(Option<i32>, &str)>>()
                .as_slice()
        );
        // Check key order
        assert_eq!(
            k_v_pairs
                .iter()
                .map(|(k, _)| k.as_ref())
                .collect::<Vec<_>>(),
            ind_map.keys().collect::<Vec<_>>()
        );
        // Check value order
        assert_eq!(
            k_v_pairs.iter().map(|(_, v)| v).collect::<Vec<_>>(),
            ind_map.values().collect::<Vec<_>>()
        );

        // Remove element before a `None` element, but not first
        // Order should be preserved
        ind_map.shift_remove(Some(&1));
        k_v_pairs.remove(0);

        // Check length
        assert_eq!(ind_map.len(), k_v_pairs.len());
        // Check item order
        assert_eq!(
            &k_v_pairs,
            ind_map
                .iter()
                .map(|(k, v)| (k.cloned(), *v))
                .collect::<Vec<(Option<i32>, &str)>>()
                .as_slice()
        );
        // Check key order
        assert_eq!(
            k_v_pairs
                .iter()
                .map(|(k, _)| k.as_ref())
                .collect::<Vec<_>>(),
            ind_map.keys().collect::<Vec<_>>()
        );
        // Check value order
        assert_eq!(
            k_v_pairs.iter().map(|(_, v)| v).collect::<Vec<_>>(),
            ind_map.values().collect::<Vec<_>>()
        );

        // Remove `None` element
        // Order should be preserved
        ind_map.shift_remove(None::<&i32>);
        k_v_pairs.remove(0);

        // Check length
        assert_eq!(ind_map.len(), k_v_pairs.len());
        // Check item order
        assert_eq!(
            &k_v_pairs,
            ind_map
                .iter()
                .map(|(k, v)| (k.cloned(), *v))
                .collect::<Vec<(Option<i32>, &str)>>()
                .as_slice()
        );
        // Check key order
        assert_eq!(
            k_v_pairs
                .iter()
                .map(|(k, _)| k.as_ref())
                .collect::<Vec<_>>(),
            ind_map.keys().collect::<Vec<_>>()
        );
        // Check value order
        assert_eq!(
            k_v_pairs.iter().map(|(_, v)| v).collect::<Vec<_>>(),
            ind_map.values().collect::<Vec<_>>()
        );
    }
}
