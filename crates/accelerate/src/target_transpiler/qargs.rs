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

#![allow(clippy::too_many_arguments)]

use std::{
    collections::hash_map::DefaultHasher,
    fmt::Display,
    hash::{Hash, Hasher},
};

use indexmap::{set::IntoIter, IndexSet};
use itertools::Itertools;
use pyo3::{
    exceptions::{PyKeyError, PyTypeError},
    prelude::*,
    pyclass,
    types::PySet,
};
use smallvec::{smallvec, IntoIter as SmallVecIntoIter, SmallVec};

use super::macro_rules::qargs_key_like_set_iterator;
use crate::nlayout::PhysicalQubit;
use hashbrown::HashSet;

pub type QargsTuple = SmallVec<[PhysicalQubit; 4]>;

#[derive(Debug, Clone, FromPyObject, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub enum QargsOrTuple {
    Qargs(Qargs),
    Tuple(QargsTuple),
}

impl QargsOrTuple {
    pub fn len(&self) -> usize {
        match self {
            Self::Tuple(tuple) => tuple.len(),
            Self::Qargs(qargs) => qargs.vec.len(),
        }
    }
}

impl QargsOrTuple {
    pub fn parse_qargs(self) -> Qargs {
        match self {
            QargsOrTuple::Qargs(qargs) => qargs,
            QargsOrTuple::Tuple(qargs) => Qargs::new(Some(qargs)),
        }
    }
}

/**
An iterator for the ``Qarg`` class.
*/
#[pyclass]
struct QargsIter {
    iter: SmallVecIntoIter<[PhysicalQubit; 4]>,
}

#[pymethods]
impl QargsIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PhysicalQubit> {
        slf.iter.next()
    }
}

qargs_key_like_set_iterator!(
    QargsSet,
    QargsSetIter,
    set,
    IntoIter<Option<Qargs>>,
    "",
    "qargs_set"
);

/**
   Hashable representation of a Qarg tuple in rust.

   Made to directly avoid conversions from a ``Vec`` structure in rust to a Python tuple.
*/
#[pyclass(sequence, module = "qiskit._accelerate.target")]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Qargs {
    pub vec: SmallVec<[PhysicalQubit; 4]>,
}

#[pymethods]
impl Qargs {
    #[new]
    pub fn new(qargs: Option<SmallVec<[PhysicalQubit; 4]>>) -> Self {
        match qargs {
            Some(qargs) => Qargs { vec: qargs },
            None => Qargs::default(),
        }
    }

    fn __len__(&self) -> usize {
        self.vec.len()
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<QargsIter>> {
        let iter = QargsIter {
            iter: slf.vec.clone().into_iter(),
        };
        Py::new(slf.py(), iter)
    }

    fn __hash__(slf: PyRef<'_, Self>) -> u64 {
        let mut hasher = DefaultHasher::new();
        slf.vec.hash(&mut hasher);
        hasher.finish()
    }

    fn __contains__(&self, obj: Bound<PyAny>) -> PyResult<bool> {
        if let Ok(obj) = obj.extract::<PhysicalQubit>() {
            Ok(self.vec.contains(&obj))
        } else {
            Ok(false)
        }
    }

    fn __getitem__(&self, obj: Bound<PyAny>) -> PyResult<PhysicalQubit> {
        if let Ok(index) = obj.extract::<usize>() {
            if let Some(item) = self.vec.get(index) {
                Ok(*item)
            } else {
                Err(PyKeyError::new_err(format!("Index {obj} is out of range.")))
            }
        } else {
            Err(PyTypeError::new_err(
                "Index type not supported.".to_string(),
            ))
        }
    }

    fn __getstate__(&self) -> PyResult<(QargsTuple,)> {
        Ok((self.vec.clone(),))
    }

    fn __setstate__(&mut self, py: Python<'_>, state: (PyObject,)) -> PyResult<()> {
        self.vec = state.0.extract::<QargsTuple>(py)?;
        Ok(())
    }

    fn __eq__(&self, other: Bound<PyAny>) -> bool {
        if let Ok(other) = other.extract::<Qargs>() {
            self.vec == other.vec
        } else if let Ok(other) = other.extract::<QargsTuple>() {
            self.vec == other
        } else {
            false
        }
    }

    fn __repr__(slf: PyRef<'_, Self>) -> String {
        let mut output = "(".to_owned();
        output.push_str(slf.vec.iter().map(|x| x.index()).join(", ").as_str());
        if slf.vec.len() < 2 {
            output.push(',');
        }
        output.push(')');
        output
    }
}

impl Display for Qargs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut output = "(".to_owned();
        output.push_str(self.vec.iter().map(|x| x.index()).join(", ").as_str());
        if self.vec.len() < 2 {
            output.push(',');
        }
        output.push(')');
        write!(f, "{}", output)
    }
}

impl Default for Qargs {
    fn default() -> Self {
        Self { vec: smallvec![] }
    }
}
