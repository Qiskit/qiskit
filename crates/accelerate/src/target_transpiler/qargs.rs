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

use hashbrown::{hash_set::IntoIter, HashSet};

use itertools::Itertools;
use pyo3::{
    exceptions::{PyKeyError, PyTypeError},
    prelude::*,
    pyclass,
    types::PySet,
};
use smallvec::{smallvec, IntoIter as SmallVecIntoIter, SmallVec};

use crate::nlayout::PhysicalQubit;

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

enum QargIterType {
    Qarg(SmallVecIntoIter<[PhysicalQubit; 4]>),
    QargSet(IntoIter<Option<Qargs>>),
}
/**
An iterator for the ``Qarg`` class.
*/
#[pyclass]
struct QargsIter {
    iter: QargIterType,
}

#[pymethods]
impl QargsIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PyObject> {
        match &mut slf.iter {
            QargIterType::Qarg(iter) => iter.next().map(|next| next.to_object(slf.py())),
            QargIterType::QargSet(iter) => iter.next().map(|next| next.into_py(slf.py())),
        }
    }
}

#[pyclass(sequence)]
#[derive(Debug, Clone)]
pub struct QargsSet {
    pub set: HashSet<Option<Qargs>>,
}

#[pymethods]
impl QargsSet {
    #[new]
    pub fn new(set: HashSet<Option<Qargs>>) -> Self {
        Self { set }
    }

    fn __eq__(slf: PyRef<Self>, other: Bound<PySet>) -> PyResult<bool> {
        for item in other.iter() {
            let qargs = if item.is_none() {
                None
            } else {
                Some(item.extract::<QargsOrTuple>()?.parse_qargs())
            };
            if !slf.set.contains(&qargs) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn __iter__(slf: PyRef<Self>) -> PyResult<Py<QargsIter>> {
        let iter = QargsIter {
            iter: QargIterType::QargSet(slf.set.clone().into_iter()),
        };
        Py::new(slf.py(), iter)
    }

    fn __getitem__(&self, obj: Bound<PyAny>) -> PyResult<Option<Qargs>> {
        let qargs = if obj.is_none() {
            None
        } else {
            Some(obj.extract::<QargsOrTuple>()?.parse_qargs())
        };
        if let Some(qargs) = self.set.get(&qargs) {
            Ok(qargs.to_owned())
        } else {
            Err(PyKeyError::new_err("{:} was not in QargSet."))
        }
    }

    fn __len__(slf: PyRef<Self>) -> usize {
        slf.set.len()
    }

    fn __repr__(slf: PyRef<Self>) -> String {
        let mut output = "qargs_set{".to_owned();
        output.push_str(
            slf.set
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
        output.push('}');
        output
    }
}

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
            iter: QargIterType::Qarg(slf.vec.clone().into_iter()),
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
