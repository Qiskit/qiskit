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
    fmt::Debug,
    hash::Hash,
    sync::{atomic::AtomicU64, Arc},
};

use crate::register::{OwningRegisterInfo, RegisterInfo};
use pyo3::{prelude::*, types::PyDict, IntoPyObjectExt};

pub trait ShareableBit
where
    <Self as ShareableBit>::ExtraAttributes: Debug + Clone + PartialEq + Eq + PartialOrd + Hash,
{
    /// Struct defining any specific extra attributes for the bit.
    type ExtraAttributes;
    /// Literal description of the bit type.
    const DESCRIPTION: &'static str;

    /// Returns reference to the instance counter for the bit.
    fn anonymous_instances() -> &'static AtomicU64;
}

/// Counter for all existing anonymous Qubit instances.
static QUBIT_COUNTER: AtomicU64 = AtomicU64::new(0);
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Hash)]

/// Alias for shareable version of a Qubit, implements [ShareableBit] trait.
pub(crate) struct ShareableQubit;
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub(crate) struct QubitExtraInfo {
    is_ancilla: bool,
}

impl QubitExtraInfo {
    /// Creates a new instance.
    pub fn new(is_ancilla: bool) -> Self {
        Self { is_ancilla }
    }

    /// Check if the [ShareableQubit] is an ancilla bit.
    pub fn is_ancilla(&self) -> bool {
        self.is_ancilla
    }
}

impl From<bool> for QubitExtraInfo {
    fn from(value: bool) -> Self {
        QubitExtraInfo { is_ancilla: value }
    }
}

impl ShareableBit for ShareableQubit {
    type ExtraAttributes = QubitExtraInfo;
    const DESCRIPTION: &'static str = "qubit";
    fn anonymous_instances() -> &'static AtomicU64 {
        &QUBIT_COUNTER
    }
}

/// Alias for shareable version of a Clbit, implements [ShareableBit] trait.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub(crate) struct ShareableClbit;

impl ShareableBit for ShareableClbit {
    type ExtraAttributes = ();
    const DESCRIPTION: &'static str = "clbit";
    fn anonymous_instances() -> &'static AtomicU64 {
        &QUBIT_COUNTER
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub enum BitInfo<T: ShareableBit> {
    Owned {
        register: Arc<OwningRegisterInfo<T>>,
        index: u32,
    },
    Anonymous {
        /// Unique id for bit, derives from [ShareableBit::anonymous_instances]
        unique_id: u64,
        /// Data about the
        extra: T::ExtraAttributes,
    },
}

impl<T: ShareableBit> BitInfo<T> {
    /// Creates an instance of anonymous [BitInfo].
    pub fn new_anonymous(extra: <T as ShareableBit>::ExtraAttributes) -> Self {
        Self::Anonymous {
            unique_id: <T as ShareableBit>::anonymous_instances()
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            extra,
        }
    }

    /// Creates an instance of anonymous [BitInfo].
    pub fn new_owned(register: Arc<OwningRegisterInfo<T>>, index: u32) -> Self {
        Self::Owned { register, index }
    }
}

impl BitInfo<ShareableQubit> {
    pub fn is_ancilla(&self) -> bool {
        match self {
            BitInfo::Owned { register, .. } => register.is_ancilla(),
            BitInfo::Anonymous { extra, .. } => extra.is_ancilla,
        }
    }
}

impl<'py> IntoPyObject<'py> for BitInfo<ShareableQubit> {
    type Target = PyQubit;

    type Output = Bound<'py, PyQubit>;

    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let ancilla = self.is_ancilla();
        let base = PyQubit(self);
        match ancilla {
            true => Ok(Py::new(py, (PyAncillaQubit(base.clone()), base))?
                .into_bound(py)
                .into_super()),
            false => base.into_pyobject(py),
        }
    }
}

impl<'py> IntoPyObject<'py> for BitInfo<ShareableClbit> {
    type Target = PyClbit;

    type Output = Bound<'py, PyClbit>;

    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let bit: PyClbit = self.into();
        bit.into_pyobject(py)
    }
}

macro_rules! create_py_bit {
    ($name:ident, $natbit:ty, $pyname:literal, $pymodule:literal, $extra:expr, ) => {
        /// Implements a quantum bit
        #[pyclass(
                                                                            subclass,
                                                                            name = $pyname,
                                                                            module = $pymodule,
                                                                            eq,
                                                                            frozen,
                                                                            hash
                                                                        )]
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name(pub(crate) BitInfo<$natbit>);

        #[pymethods]
        impl $name {
            /// Creates a new bit.
            #[new]
            pub fn new() -> Self {
                Self(BitInfo::new_anonymous($extra))
            }

            pub fn __repr__(slf: Bound<Self>) -> PyResult<String> {
                let borrowed = slf.borrow();
                match &borrowed.0 {
                    BitInfo::Owned { register, index } => Ok(format!(
                        "{}({}, {})",
                        slf.get_type().qualname()?,
                        register,
                        index
                    )),
                    BitInfo::Anonymous { .. } => Ok(format!(
                        "<{} object at {:p}>",
                        slf.get_type().fully_qualified_name()?,
                        &slf
                    )),
                }
            }

            fn __copy__(slf: Bound<Self>) -> Bound<Self> {
                slf
            }

            fn __deepcopy__<'a>(
                slf: Bound<'a, Self>,
                _cache: Bound<'a, PyDict>,
            ) -> PyResult<Bound<'a, Self>> {
                let borrowed = slf.borrow();
                let py = slf.py();
                borrowed.clone().into_pyobject(py)
            }

            #[getter]
            fn _register(slf: PyRef<Self>) -> PyResult<Option<PyObject>> {
                match &slf.0 {
                    BitInfo::Owned { register, .. } => Ok(Some(
                        RegisterInfo::Owning(register.clone()).into_py_any(slf.py())?,
                    )),
                    BitInfo::Anonymous { .. } => Ok(None),
                }
            }

            #[getter]
            fn _index(slf: PyRef<Self>) -> Option<u32> {
                match &slf.0 {
                    BitInfo::Owned { index, .. } => Some(*index),
                    BitInfo::Anonymous { .. } => None,
                }
            }
        }

        impl $name {
            /// Creates a qubit that is owned by a register
            pub(crate) fn new_owned(
                register: Arc<OwningRegisterInfo<$natbit>>,
                index: u32,
            ) -> Self {
                Self(BitInfo::Owned { register, index })
            }
        }

        impl From<BitInfo<$natbit>> for $name {
            fn from(value: BitInfo<$natbit>) -> Self {
                $name(value)
            }
        }
    };
}

create_py_bit!(
    PyQubit,
    ShareableQubit,
    "Qubit",
    "qiskit.circuit.quantumcircuit",
    QubitExtraInfo { is_ancilla: false },
);

create_py_bit!(
    PyClbit,
    ShareableClbit,
    "Clbit",
    "qiskit.circuit.classicalcircuit",
    (),
);

/// A qubit used as ancillary qubit.
#[pyclass(
    extends=PyQubit,
    name = "AncillaQubit",
    module = "qiskit.circuit.quantumregister",
    eq,
    frozen,
    hash
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PyAncillaQubit(pub(crate) PyQubit);

#[pymethods]
impl PyAncillaQubit {
    /// Creates a new anonymous `AncillaQubit`.
    #[new]
    pub fn new() -> (Self, PyQubit) {
        let base = PyQubit(BitInfo::new_anonymous(QubitExtraInfo { is_ancilla: true }));
        (Self(base.clone()), base)
    }
}
