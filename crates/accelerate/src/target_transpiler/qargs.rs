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

use core::panic;

use indexmap::Equivalent;
use pyo3::{prelude::*, types::PyTuple, IntoPyObject};
use smallvec::SmallVec;

use crate::nlayout::PhysicalQubit;

pub type TargetQargs = SmallVec<[PhysicalQubit; 2]>;

/// Representation of quantum args for a [Target].
///
/// An instruction stored within a [Target] can have two different types
/// of qargs when specifying its properties:
/// - Global: If the instruction is a Variadic or can operate in any set
///   of qargs as long as they match the capacity of the instruction.
/// - Concrete: Specific combination of quantum args.
///
/// This enumeration represents these two conditions efficiently while
/// solving certain ownership issues that [Option] currently has.
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Qargs {
    #[default]
    Global,
    Concrete(TargetQargs),
}

impl From<Option<TargetQargs>> for Qargs {
    fn from(value: Option<TargetQargs>) -> Self {
        match value {
            Some(qargs) => Self::Concrete(qargs),
            None => Self::Global,
        }
    }
}

impl From<TargetQargs> for Qargs {
    fn from(value: TargetQargs) -> Self {
        Self::Concrete(value)
    }
}

impl Qargs {
    /// Returns a reference version of a qarg.
    pub fn as_ref(&self) -> QargsRef<'_> {
        match self {
            Qargs::Global => QargsRef::Global,
            Qargs::Concrete(small_vec) => QargsRef::Concrete(small_vec),
        }
    }

    /// Checks if the qargs in question are `Global`.
    pub fn is_global(&self) -> bool {
        matches!(self, Self::Global)
    }

    /// Checks if the qargs in question are `Concrete`.
    pub fn is_concrete(&self) -> bool {
        !self.is_global()
    }

    /// Returns an iterator of either zero or 1 step depending on whether
    /// the operation is global (0) or concrete (1)
    pub fn iter(&self) -> Iter<'_> {
        self.as_ref().iter()
    }

    /// Turns the qargs into an option view
    pub fn as_option(&self) -> Option<&[PhysicalQubit]> {
        self.as_ref().as_option()
    }

    /// Returns the enclosed qargs in the case of `Concrete` variant.
    ///
    /// This function may `panic!`, and its unsafe use is discouraged.
    pub fn unwrap(self) -> TargetQargs {
        match self {
            Self::Global => panic!("Attempted to unwrap a 'Global' variant of 'TargetQargs'"),
            Self::Concrete(small_vec) => small_vec,
        }
    }

    /// Returns the enclosed qargs in the case of `Concrete` variant, otherwise
    /// return an empty [SmallVec].
    pub fn unwrap_or_default(self) -> TargetQargs {
        match self {
            Self::Global => SmallVec::default(),
            Self::Concrete(small_vec) => small_vec,
        }
    }
}

impl IntoIterator for Qargs {
    type Item = TargetQargs;

    type IntoIter = IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter { inner: self }
    }
}

impl<'py> IntoPyObject<'py> for Qargs {
    type Target = PyAny;

    type Output = Bound<'py, PyAny>;

    type Error = PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Self::Global => Ok(py.None().into_bound(py)),
            Self::Concrete(small_vec) => Ok(PyTuple::new(py, small_vec)?.into_any()),
        }
    }
}

impl<'py> FromPyObject<'py> for Qargs {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let qargs: Option<TargetQargs> = ob.extract()?;
        Ok(qargs.into())
    }
}

impl Equivalent<Qargs> for QargsRef<'_> {
    fn equivalent(&self, key: &Qargs) -> bool {
        *self == key.as_ref()
    }
}

/// Reference representation of [TargetQargs].
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum QargsRef<'a> {
    Global,
    Concrete(&'a [PhysicalQubit]),
}

impl<'a> QargsRef<'a> {
    /// Checks if the qargs in question are `Global`.
    pub fn is_global(&self) -> bool {
        matches!(self, Self::Global)
    }

    /// Checks if the qargs in question are `Concrete`.
    pub fn is_concrete(&self) -> bool {
        !self.is_global()
    }

    /// Returns an iterator of either zero or 1 step depending on whether
    /// the operation is global (0) or concrete (1)
    pub fn iter(self) -> Iter<'a> {
        Iter { inner: self }
    }

    /// Turns the qargs into an option view
    pub fn as_option(&self) -> Option<&'a [PhysicalQubit]> {
        match self {
            QargsRef::Global => None,
            QargsRef::Concrete(qargs) => Some(qargs),
        }
    }

    /// Returns the enclosed qargs in the case of `Concrete` variant.
    ///
    /// This function may `panic!`, and its unsafe use is discouraged.
    pub fn unwrap(self) -> &'a [PhysicalQubit] {
        match self {
            Self::Global => panic!("Attempted to unwrap a 'Global' variant of 'TargetQargs'"),
            Self::Concrete(qargs) => qargs,
        }
    }

    /// Returns the enclosed qargs in the case of `Concrete` variant, otherwise
    /// return an empty slice.
    pub fn unwrap_or_default(self) -> &'a [PhysicalQubit] {
        match self {
            Self::Global => &[],
            Self::Concrete(qargs) => qargs,
        }
    }
}

impl<'a> From<Option<&'a [PhysicalQubit]>> for QargsRef<'a> {
    fn from(value: Option<&'a [PhysicalQubit]>) -> Self {
        match value {
            Some(qargs) => Self::Concrete(qargs),
            None => Self::Global,
        }
    }
}

impl<'a> From<&'a [PhysicalQubit]> for QargsRef<'a> {
    fn from(value: &'a [PhysicalQubit]) -> Self {
        Self::Concrete(value)
    }
}

impl<'py> IntoPyObject<'py> for QargsRef<'_> {
    type Target = PyAny;

    type Output = Bound<'py, PyAny>;

    type Error = PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Self::Global => Ok(py.None().into_bound(py)),
            Self::Concrete(small_vec) => Ok(PyTuple::new(py, small_vec)?.into_any()),
        }
    }
}

// Iterators

#[doc(hidden)]
/// Owning iterator for [TargetQargs].
pub struct IntoIter {
    inner: Qargs,
}

impl Iterator for IntoIter {
    type Item = TargetQargs;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            Qargs::Global => None,
            Qargs::Concrete(small_vec) => {
                let vec = std::mem::take(small_vec);
                self.inner = Qargs::Global;
                Some(vec)
            }
        }
    }
}

#[doc(hidden)]
/// Borrowed iterator for [TargetQargs]
pub struct Iter<'a> {
    inner: QargsRef<'a>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a [PhysicalQubit];

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner {
            QargsRef::Global => None,
            QargsRef::Concrete(small_vec) => {
                let vec = small_vec;
                self.inner = QargsRef::Global;
                Some(vec)
            }
        }
    }
}
