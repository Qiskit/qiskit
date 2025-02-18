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

use crate::{
    circuit_data::CircuitError,
    register::{
        ClassicalRegister, OwningRegisterInfo, PyAncillaRegister, PyClassicalRegister,
        PyQuantumRegister, PyRegister, QuantumRegister, RegisterInfo,
    },
};
use pyo3::{prelude::*, types::PyDict};

/// Counter for all existing anonymous Qubit instances.
static BIT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Alias for extra properties stored in a Qubit
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub struct QubitExtraInfo {
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

/// Main representation of the inner properties of a shareable `Bit` object.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub enum BitInfo {
    Owned {
        register: Arc<OwningRegisterInfo>,
        index: u32,
    },
    Anonymous {
        /// Unique id for bit, derives from [ShareableBit::anonymous_instances]
        unique_id: u64,
    },
}

impl BitInfo {
    /// Creates an instance of anonymous [BitInfo].
    pub fn new_anonymous() -> Self {
        Self::Anonymous {
            unique_id: BIT_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Creates an instance of anonymous [BitInfo].
    pub fn new_owned(register: Arc<OwningRegisterInfo>, index: u32) -> Self {
        Self::Owned { register, index }
    }
}

macro_rules! create_bit_object {
    ($name:ident, $extra:ty, $extra_exp:expr, $reg:tt) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
        pub struct $name {
            pub(crate) info: BitInfo,
            pub(crate) extra: $extra,
        }

        impl $name {
            /// Creates an instance of owned [QubitObject].
            pub fn new_anonymous(extra: $extra) -> Self {
                Self {
                    extra,
                    info: BitInfo::new_anonymous(),
                }
            }

            /// Creates an instance of owned [QubitObject].
            pub(crate) fn new_owned(
                register: Arc<OwningRegisterInfo>,
                index: u32,
                extra: $extra,
            ) -> Self {
                Self {
                    info: BitInfo::new_owned(register, index),
                    extra,
                }
            }

            /// Returns a reference to the owning register of the [QubitObject] if any exists.
            pub fn register(&self) -> Option<$reg> {
                match &self.info {
                    BitInfo::Owned { register, .. } => Some($reg {
                        data: RegisterInfo::Owning(register.clone()).into(),
                        extra: self.extra,
                    }),
                    BitInfo::Anonymous { .. } => None,
                }
            }

            /// Returns the index of the [QubitObject] within the owning register if any exists.
            pub fn index(&self) -> Option<u32> {
                match &self.info {
                    BitInfo::Owned { index, .. } => Some(*index),
                    _ => None,
                }
            }
        }
    };
}

create_bit_object! {ShareableQubit, QubitExtraInfo, QubitExtraInfo::new(false), QuantumRegister}

impl ShareableQubit {
    /// Check if the Qubit instance is ancillary.
    pub fn is_ancilla(&self) -> bool {
        self.extra.is_ancilla()
    }
}
create_bit_object! {ShareableClbit, (), (), ClassicalRegister}

impl<'py> IntoPyObject<'py> for ShareableQubit {
    type Target = PyQubit;

    type Output = Bound<'py, PyQubit>;

    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let ancilla = self.is_ancilla();
        let bit = PyBit(self.info.clone());
        let base = PyQubit(PyBit(self.info));
        match ancilla {
            true => Ok(Bound::new(
                py,
                PyClassInitializer::from(bit)
                    .add_subclass(base.clone())
                    .add_subclass(PyAncillaQubit(base.clone())),
            )?
            .into_super()),
            false => Bound::new(py, (base, bit)),
        }
    }
}

impl<'py> IntoPyObject<'py> for ShareableClbit {
    type Target = PyClbit;

    type Output = Bound<'py, PyClbit>;

    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let bit: PyBit = PyBit(self.info);
        Bound::new(py, (PyClbit(bit.clone()), bit))
    }
}

#[pyclass(
    subclass,
    name = "Bit",
    module = "qiskit.circuit.bit",
    eq,
    frozen,
    hash
)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub struct PyBit(pub(crate) BitInfo);

#[pymethods]
impl PyBit {
    #[new]
    #[pyo3(signature = (register = None, index = None))]
    fn new(register: Option<PyRegister>, index: Option<u32>) -> PyResult<Self> {
        match (register, index) {
            (Some(register), Some(index)) => {
                let RegisterInfo::Owning(owned) = register.0.as_ref() else {
                    return Err(CircuitError::new_err(
                        "The provided register for this bit was invalid.",
                    ));
                };
                if index as usize >= owned.len() {
                    return Err(CircuitError::new_err(format!(
                        "index must be under the size of the register: {index} was provided"
                    )));
                }
                Ok(Self(BitInfo::new_owned(owned.clone(), index)))
            }
            (None, None) => Ok(Self(BitInfo::new_anonymous())),
            _ => Err(CircuitError::new_err(
                "You should provide both a valid register and an index, not either or.".to_string(),
            )),
        }
    }

    pub fn __repr__(slf: Bound<Self>) -> PyResult<String> {
        let borrowed = slf.borrow();
        let reg = slf.getattr("_register")?;
        match &borrowed.0 {
            BitInfo::Owned { index, .. } => Ok(format!(
                "{}({}, {})",
                slf.get_type().qualname()?,
                reg.repr()?,
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
    fn _register(&self) -> Option<PyRegister> {
        match &self.0 {
            BitInfo::Owned { register, .. } => {
                Some(PyRegister(RegisterInfo::Owning(register.clone()).into()))
            }
            BitInfo::Anonymous { .. } => None,
        }
    }

    #[getter]
    fn _index(&self) -> Option<u32> {
        match &self.0 {
            BitInfo::Owned { index, .. } => Some(*index),
            BitInfo::Anonymous { .. } => None,
        }
    }
}

macro_rules! create_py_bit {
    ($name:ident, $natbit:ty, $pyname:literal, $pymodule:literal, $extra:expr, $pyreg:tt) => {
        /// Implements a quantum bit
        #[pyclass(
                                                            subclass,
                                                            name = $pyname,
                                                            module = $pymodule,
                                                            eq,
                                                            frozen,
                                                            hash,
                                                            extends=PyBit,
                                                        )]
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name(pub(crate) PyBit);

        #[pymethods]
        impl $name {
            /// Creates a new bit.
            #[new]
            #[pyo3(signature = (register = None, index = None))]
            fn new(register: Option<$pyreg>, index: Option<u32>) -> PyResult<(Self, PyBit)> {
                let inner = PyBit::new(register.map(|reg| reg.0), index)?;
                Ok((Self(inner.clone()), inner))
            }

            fn __deepcopy__<'a>(
                slf: Bound<'a, Self>,
                _cache: Bound<'a, PyDict>,
            ) -> PyResult<Bound<'a, Self>> {
                let borrowed = slf.borrow();
                let py = slf.py();
                Bound::new(py, (borrowed.clone(), borrowed.0.clone()))
            }

            #[getter]
            fn _register(slf: PyRef<Self>) -> PyResult<Option<Bound<$pyreg>>> {
                slf.as_super()
                    ._register()
                    .map(|reg| Bound::new(slf.py(), ($pyreg(reg.clone()), reg)))
                    .transpose()
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
    PyQuantumRegister
);

create_py_bit!(
    PyClbit,
    ShareableClbit,
    "Clbit",
    "qiskit.circuit.classicalcircuit",
    (),
    PyClassicalRegister
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
    #[pyo3(signature = (register = None, index = None))]
    fn new(
        register: Option<PyAncillaRegister>,
        index: Option<u32>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let (qubit, base) = PyQubit::new(register.map(|reg| reg.0), index)?;
        Ok(PyClassInitializer::from(base)
            .add_subclass(qubit.clone())
            .add_subclass(Self(qubit)))
    }
}
