use std::{
    fmt::Debug,
    hash::Hash,
    sync::{atomic::AtomicU64, Arc},
};

use crate::register::OwningRegisterInfo;
use pyo3::prelude::*;

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
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]

/// Alias for shareable version of a Qubit, implements [ShareableBit] trait.
pub(crate) struct ShareableQubit;
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub(crate) struct QubitExtraInfo {
    is_ancilla: bool,
}

impl QubitExtraInfo {
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
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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

/// Implements a quantum bit
#[pyclass(
    subclass,
    name = "Qubit",
    module = "qiskit.circuit.quantumregister",
    eq,
    frozen,
    hash
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PyQubit(BitInfo<ShareableQubit>);

#[pymethods]
impl PyQubit {
    /// Creates a new Qubit.
    #[new]
    pub fn new() -> Self {
        Self(BitInfo::new_anonymous(QubitExtraInfo { is_ancilla: false }))
    }

    fn __repr__(slf: Bound<Self>) -> PyResult<String> {
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

    fn __deepcopy__(slf: Bound<Self>) -> PyResult<Bound<Self>> {
        let borrowed = slf.borrow();
        let py = slf.py();
        borrowed.clone().into_pyobject(py)
    }

    #[getter]
    fn _index(slf: PyRef<Self>) -> Option<u32> {
        match &slf.0 {
            BitInfo::Owned { index, .. } => Some(*index),
            BitInfo::Anonymous { .. } => None,
        }
    }
}

impl PyQubit {
    /// Creates a qubit that is owned by a register
    pub(crate) fn new_owned(register: Arc<OwningRegisterInfo<ShareableQubit>>, index: u32) -> Self {
        Self(BitInfo::Owned { register, index })
    }
}

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
pub struct PyAncillaQubit(PyQubit);

#[pymethods]
impl PyAncillaQubit {
    /// Creates a new anonymous `AncillaQubit`.
    #[new]
    pub fn new() -> (Self, PyQubit) {
        let base = PyQubit(BitInfo::new_anonymous(QubitExtraInfo { is_ancilla: true }));
        (Self(base.clone()), base)
    }
}

/// Implements a quantum bit
#[pyclass(
    subclass,
    name = "Clbit",
    module = "qiskit.circuit.classicalregister",
    eq,
    frozen,
    hash
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PyClbit(BitInfo<ShareableClbit>);

#[pymethods]
impl PyClbit {
    /// Creates a new Qubit.
    #[new]
    pub fn new() -> Self {
        Self(BitInfo::new_anonymous(()))
    }

    fn __repr__(slf: Bound<Self>) -> PyResult<String> {
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

    fn __deepcopy__(slf: Bound<Self>) -> PyResult<Bound<Self>> {
        let borrowed = slf.borrow();
        let py = slf.py();
        borrowed.clone().into_pyobject(py)
    }

    #[getter]
    fn _index(slf: PyRef<Self>) -> Option<u32> {
        match &slf.0 {
            BitInfo::Owned { index, .. } => Some(*index),
            BitInfo::Anonymous { .. } => None,
        }
    }
}

impl PyClbit {
    /// Creates a qubit that is owned by a register
    pub(crate) fn new_owned(register: Arc<OwningRegisterInfo<ShareableClbit>>, index: u32) -> Self {
        Self(BitInfo::Owned { register, index })
    }
}
