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

pub type Qargs = SmallVec<[PhysicalQubit; 2]>;

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
pub enum TargetQargs {
    #[default]
    Global,
    Concrete(Qargs),
}

impl From<Option<Qargs>> for TargetQargs {
    fn from(value: Option<Qargs>) -> Self {
        match value {
            Some(qargs) => Self::Concrete(qargs),
            None => Self::Global,
        }
    }
}

impl From<Qargs> for TargetQargs {
    fn from(value: Qargs) -> Self {
        Self::Concrete(value)
    }
}

impl TargetQargs {
    /// Returns a reference version of a qarg.
    pub fn as_ref(&self) -> TargetQargsRef<'_> {
        match self {
            TargetQargs::Global => TargetQargsRef::Global,
            TargetQargs::Concrete(small_vec) => TargetQargsRef::Concrete(small_vec),
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
    pub fn unwrap(self) -> Qargs {
        match self {
            Self::Global => panic!("Attempted to unwrap a 'Global' variant of 'TargetQargs'"),
            Self::Concrete(small_vec) => small_vec,
        }
    }

    /// Returns the enclosed qargs in the case of `Concrete` variant, otherwise
    /// return an empty [SmallVec].
    pub fn unwrap_or_default(self) -> Qargs {
        match self {
            Self::Global => SmallVec::default(),
            Self::Concrete(small_vec) => small_vec,
        }
    }
}

impl IntoIterator for TargetQargs {
    type Item = Qargs;

    type IntoIter = IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter { inner: self }
    }
}

impl<'py> IntoPyObject<'py> for TargetQargs {
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

impl<'py> FromPyObject<'py> for TargetQargs {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let qargs: Option<Qargs> = ob.extract()?;
        Ok(qargs.into())
    }
}

impl Equivalent<TargetQargs> for TargetQargsRef<'_> {
    fn equivalent(&self, key: &TargetQargs) -> bool {
        *self == key.as_ref()
    }
}

/// Reference representation of [TargetQargs].
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum TargetQargsRef<'a> {
    Global,
    Concrete(&'a [PhysicalQubit]),
}

impl<'a> TargetQargsRef<'a> {
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
            TargetQargsRef::Global => None,
            TargetQargsRef::Concrete(qargs) => Some(qargs),
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

impl<'a> From<Option<&'a [PhysicalQubit]>> for TargetQargsRef<'a> {
    fn from(value: Option<&'a [PhysicalQubit]>) -> Self {
        match value {
            Some(qargs) => Self::Concrete(qargs),
            None => Self::Global,
        }
    }
}

impl<'a> From<&'a [PhysicalQubit]> for TargetQargsRef<'a> {
    fn from(value: &'a [PhysicalQubit]) -> Self {
        Self::Concrete(value)
    }
}

impl<'py> IntoPyObject<'py> for TargetQargsRef<'_> {
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
    inner: TargetQargs,
}

impl Iterator for IntoIter {
    type Item = Qargs;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            TargetQargs::Global => None,
            TargetQargs::Concrete(small_vec) => {
                let vec = std::mem::take(small_vec);
                self.inner = TargetQargs::Global;
                Some(vec)
            }
        }
    }
}

#[doc(hidden)]
/// Borrowed iterator for [TargetQargs]
pub struct Iter<'a> {
    inner: TargetQargsRef<'a>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a [PhysicalQubit];

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner {
            TargetQargsRef::Global => None,
            TargetQargsRef::Concrete(small_vec) => {
                let vec = small_vec;
                self.inner = TargetQargsRef::Global;
                Some(vec)
            }
        }
    }
}
