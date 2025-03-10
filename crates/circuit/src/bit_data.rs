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

use crate::bit::ShareableBit;
use crate::BitType;

use hashbrown::HashMap;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;

/// Private wrapper for Python-side objects that implements [Hash] and [Eq], allowing them to be
/// used in Rust hash-based sets and maps.
///
/// Python's `hash()` is called on the wrapped object during construction and returned from Rust's
/// [Hash] trait impl.  The impl of [PartialEq] first compares the native Py pointers to determine
/// equality. If these are not equal, only then does it call `repr()` on both sides, which has a
/// significant performance advantage.
#[derive(Clone, Debug)]
pub struct VarAsKey {
    /// Python's `hash()` of the wrapped instance.
    hash: isize,
    /// The wrapped instance.
    ob: PyObject,
}

impl VarAsKey {
    pub fn new(bit: &Bound<PyAny>) -> Self {
        VarAsKey {
            // This really shouldn't fail, but if it does,
            // we'll just use 0.
            hash: bit.hash().unwrap_or(0),
            ob: bit.clone().unbind(),
        }
    }

    /// Safely clones the underlying python object reference
    pub fn clone_ref(&self, py: Python) -> Self {
        Self {
            hash: self.hash,
            ob: self.ob.clone_ref(py),
        }
    }
}

impl Hash for VarAsKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_isize(self.hash);
    }
}

impl<'a, 'py> From<&'a Bound<'py, PyAny>> for VarAsKey {
    fn from(value: &'a Bound<'py, PyAny>) -> Self {
        VarAsKey::new(value)
    }
}

impl<'py> From<Bound<'py, PyAny>> for VarAsKey {
    fn from(value: Bound<'py, PyAny>) -> Self {
        VarAsKey::new(&value)
    }
}

impl<'py> IntoPyObject<'py> for VarAsKey {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.ob.bind(py).clone())
    }
}

impl<'py> IntoPyObject<'py> for &VarAsKey {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.ob.bind(py).clone())
    }
}

impl PartialEq for VarAsKey {
    fn eq(&self, other: &Self) -> bool {
        self.ob.is(&other.ob)
            || Python::with_gil(|py| {
                self.ob
                    .bind(py)
                    .repr()
                    .unwrap()
                    .as_any()
                    .eq(other.ob.bind(py).repr().unwrap())
                    .unwrap()
            })
    }
}
impl Eq for VarAsKey {}

// VarAsKey isn't really a _bit_, but it's still the shareable object between circuits, and naming
// thing is hard.
impl ShareableBit for VarAsKey {
    type Subclass = ();
    const DESCRIPTION: &'static str = "var";
}

#[derive(Clone, Debug)]
pub struct BitData<T, B> {
    /// Registered Python bits.
    bits: Vec<B>,
    /// Maps Python bits to native type.
    indices: HashMap<B, T>,
    /// The bits registered, cached as a PyList.
    cached: OnceLock<Py<PyList>>,
}

impl<T, B> Default for BitData<T, B>
where
    T: From<BitType> + Copy,
    BitType: From<T>,
    B: for<'a> IntoPyObject<'a> + ShareableBit,
{
    fn default() -> Self {
        Self::new()
    }
}
impl<T, B> BitData<T, B>
where
    T: From<BitType> + Copy,
    BitType: From<T>,
    B: for<'a> IntoPyObject<'a> + ShareableBit,
{
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        BitData {
            bits: Vec::with_capacity(capacity),
            indices: HashMap::with_capacity(capacity),
            cached: OnceLock::new(),
        }
    }

    /// Gets the number of bits.
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// Gets a reference to the underlying vector of Python bits.
    #[inline]
    pub fn bits(&self) -> &Vec<B> {
        &self.bits
    }

    /// Gets a reference to the cached Python list, maintained by
    /// this instance.
    #[inline]
    pub fn cached(&self, py: Python) -> &Py<PyList> {
        self.cached
            .get_or_init(|| PyList::new(py, self.bits.clone()).unwrap().into())
    }

    /// Gets a reference to the cached Python list, even if not initialized.
    #[inline]
    pub fn cached_raw(&self) -> Option<&Py<PyList>> {
        self.cached.get()
    }

    /// Finds the native bit index of the given Python bit.
    #[inline]
    pub fn find(&self, bit: &B) -> Option<T> {
        self.indices.get(bit).copied()
    }

    /// Map the provided Python bits to their native indices.
    /// An error is returned if any bit is not registered.
    pub fn map_bits(&self, bits: impl IntoIterator<Item = B>) -> PyResult<impl Iterator<Item = T>> {
        let v: Result<Vec<_>, _> = bits
            .into_iter()
            .map(|b| {
                self.indices.get(&b).copied().ok_or_else(|| {
                    PyKeyError::new_err(format!("Bit {:?} has not been added to this circuit.", b))
                })
            })
            .collect();
        v.map(|x| x.into_iter())
    }

    /// Map the provided native indices to the corresponding Python
    /// bit instances.
    /// Panics if any of the indices are out of range.
    pub fn map_indices(&self, bits: &[T]) -> impl ExactSizeIterator<Item = &B> {
        let v: Vec<_> = bits.iter().map(|i| self.get(*i).unwrap()).collect();
        v.into_iter()
    }

    /// Gets the `SharableBit` corresponding to the given native
    /// bit index.
    #[inline]
    pub fn get(&self, index: T) -> Option<&B> {
        self.bits.get(<BitType as From<T>>::from(index) as usize)
    }

    ///Checks if the `SharableBit`` bit corresponding to the given native
    /// bit index.
    #[inline]
    pub fn contains(&self, key: &B) -> bool {
        self.indices.contains_key(key)
    }

    /// Adds a new Python bit.
    pub fn add(&mut self, bit: B, strict: bool) -> PyResult<T> {
        let idx: BitType = self.bits.len().try_into().map_err(|_| {
            PyRuntimeError::new_err(format!(
                "The number of {}s in the circuit has exceeded the maximum capacity",
                <B as ShareableBit>::DESCRIPTION,
            ))
        })?;
        // Dump the cache
        self.cached.take();
        if self.indices.try_insert(bit.clone(), idx.into()).is_ok() {
            self.bits.push(bit);
        } else if strict {
            return Err(PyValueError::new_err(format!(
                "Existing bit {:?} cannot be re-added in strict mode.",
                bit
            )));
        }
        Ok(idx.into())
    }

    pub fn remove_indices<I>(&mut self, indices: I) -> PyResult<()>
    where
        I: IntoIterator<Item = T>,
    {
        let mut indices_sorted: Vec<usize> = indices
            .into_iter()
            .map(|i| <BitType as From<T>>::from(i) as usize)
            .collect();
        indices_sorted.sort();
        self.cached.take();
        for index in indices_sorted.into_iter().rev() {
            let bit = self.bits.remove(index);
            self.indices.remove(&bit);
        }
        // Update indices.
        for (i, bit) in self.bits.iter().enumerate() {
            self.indices.insert(bit.clone(), (i as BitType).into());
        }
        Ok(())
    }

    /// Called during Python garbage collection, only!.
    /// Note: INVALIDATES THIS INSTANCE.
    pub fn dispose(&mut self) {
        self.indices.clear();
        self.bits.clear();
    }
}
