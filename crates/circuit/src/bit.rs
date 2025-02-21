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
        ClassicalRegister, OwningRegisterInfo, PyClassicalRegister, PyQuantumRegister, PyRegister,
        QuantumRegister, RegisterInfo,
    },
};
use pyo3::{prelude::*, types::PyDict};

/// Counter for all existing anonymous Qubit instances.
static BIT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Alias for extra properties stored in a Bit, can work as an identifier.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub enum BitExtraInfo {
    Qubit { is_ancilla: bool },
    Clbit(),
}

/// Main representation of the inner properties of a shareable `Bit` object.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub enum BitInfo {
    Owned {
        register: Arc<OwningRegisterInfo>,
        index: u32,
        extra: Option<BitExtraInfo>,
    },
    Anonymous {
        /// Unique id for bit, derives from [ShareableBit::anonymous_instances]
        unique_id: u64,
        extra: Option<BitExtraInfo>,
    },
}

impl BitInfo {
    /// Creates an instance of anonymous [BitInfo].
    pub fn new_anonymous(extra: Option<BitExtraInfo>) -> Self {
        Self::Anonymous {
            unique_id: BIT_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            extra,
        }
    }

    /// Creates an instance of anonymous [BitInfo].
    pub fn new_owned(
        register: Arc<OwningRegisterInfo>,
        index: u32,
        extra: Option<BitExtraInfo>,
    ) -> Self {
        Self::Owned {
            register,
            index,
            extra,
        }
    }

    pub fn extra_info(&self) -> Option<&BitExtraInfo> {
        match self {
            BitInfo::Owned { extra, .. } => extra.as_ref(),
            BitInfo::Anonymous { extra, .. } => extra.as_ref(),
        }
    }

    pub fn is_owned(&self) -> bool {
        matches!(self, BitInfo::Owned { .. })
    }

    pub(crate) fn register(&self) -> Option<RegisterInfo> {
        match self {
            BitInfo::Owned { register, .. } => Some(RegisterInfo::Owning(register.clone())),
            BitInfo::Anonymous { .. } => None,
        }
    }
}

macro_rules! create_bit_object {
    ($name:ident, $extra:ty, $extra_exp:expr, $reg:tt) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name(pub(crate) BitInfo);

        impl $name {
            /// Creates an instance of owned [QubitObject].
            pub fn new_anonymous() -> Self {
                Self(BitInfo::new_anonymous(Some($extra_exp)))
            }

            // /// Creates an instance of owned [QubitObject].
            // pub(crate) fn new_owned(register: $reg, index: u32) -> Self {
            //     let RegisterInfo::Owning(owning) = register.data().as_ref() else {
            //         panic!("The provided register does not own its bits.")
            //     };
            //     Self {
            //         info: BitInfo::new_owned(owning.clone(), index, Some($extra_exp)),
            //         reg: Some(register),
            //     }
            // }

            /// Returns a reference to the owning register of the [QubitObject] if any exists.
            pub fn register(&self) -> Option<$reg> {
                self.0.register().map(|reg| $reg(reg.into()))
            }

            /// Returns the index of the [QubitObject] within the owning register if any exists.
            pub fn index(&self) -> Option<u32> {
                match &self.0 {
                    BitInfo::Owned { index, .. } => Some(*index),
                    _ => None,
                }
            }
        }
    };
}

create_bit_object! {ShareableQubit, BitExtraInfo, BitExtraInfo::Qubit{is_ancilla: false}, QuantumRegister}

impl ShareableQubit {
    /// Check if the Qubit instance is ancillary.
    pub fn is_ancilla(&self) -> bool {
        match self.0.extra_info() {
            Some(BitExtraInfo::Qubit { is_ancilla }) => *is_ancilla,
            _ => false,
        }
    }
}

create_bit_object! {ShareableClbit, (), BitExtraInfo::Clbit(), ClassicalRegister}

impl<'py> IntoPyObject<'py> for ShareableQubit {
    type Target = PyQubit;

    type Output = Bound<'py, PyQubit>;

    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let ancilla = self.is_ancilla();
        let bit = PyBit(self.0.clone());
        let base = PyQubit(self);
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

impl<'py> IntoPyObject<'py> for &'py ShareableQubit {
    type Target = PyQubit;

    type Output = Bound<'py, PyQubit>;

    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        self.clone().into_pyobject(py)
    }
}

impl<'py> FromPyObject<'py> for ShareableQubit {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(ob.downcast::<PyQubit>()?.get().0.clone())
    }
}

impl<'py> IntoPyObject<'py> for ShareableClbit {
    type Target = PyClbit;

    type Output = Bound<'py, PyClbit>;

    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let bit: PyBit = PyBit(self.0.clone());
        Bound::new(py, (PyClbit(self), bit))
    }
}

impl<'py> IntoPyObject<'py> for &'py ShareableClbit {
    type Target = PyClbit;

    type Output = Bound<'py, PyClbit>;

    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        self.clone().into_pyobject(py)
    }
}

impl<'py> FromPyObject<'py> for ShareableClbit {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(ob.downcast::<PyClbit>()?.get().0.clone())
    }
}

#[pyclass(subclass, name = "Bit", module = "qiskit.circuit.bit", frozen)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub struct PyBit(pub(crate) BitInfo);

#[pymethods]
impl PyBit {
    #[new]
    #[pyo3(signature = (register = None, index = None))]
    fn new(register: Option<PyRegister>, index: Option<u32>) -> PyResult<Self> {
        Self::inner_new(register, index, None)
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

    fn __hash__(slf: Bound<Self>) -> PyResult<isize> {
        let borrow_slf = slf.borrow();
        match &borrow_slf.0 {
            BitInfo::Owned { index, .. } => {
                (slf.get_type().name()?, slf.getattr("_register")?, *index)
                    .into_pyobject(slf.py())?
                    .hash()
            }
            BitInfo::Anonymous { unique_id, .. } => {
                (slf.get_type(), unique_id).into_pyobject(slf.py())?.hash()
            }
        }
    }

    fn __eq__(slf: Bound<Self>, other: Bound<Self>) -> PyResult<bool> {
        let borrow_slf = slf.borrow();
        let borrow_other = other.borrow();
        match (&borrow_slf.0, &borrow_other.0) {
            (BitInfo::Owned { .. }, BitInfo::Owned { .. }) => {
                Ok(slf.repr()?.to_string() == other.repr()?.to_string())
            }
            (
                BitInfo::Anonymous {
                    unique_id: uid0, ..
                },
                BitInfo::Anonymous {
                    unique_id: uid1, ..
                },
            ) => Ok(slf.is(&other) || (slf.get_type().eq(other.get_type())? && uid0 == uid1)),
            _ => Ok(false),
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
        match borrowed.0 {
            BitInfo::Owned { .. } => {
                let py = slf.py();
                borrowed.clone().into_pyobject(py)
            }
            BitInfo::Anonymous { .. } => Ok(slf),
        }
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

    fn __getnewargs__(slf: Bound<'_, Self>) -> PyResult<(Bound<'_, PyAny>, Bound<'_, PyAny>)> {
        Ok((slf.getattr("_register")?, slf.getattr("_index")?))
    }
}

impl PyBit {
    /// Quickly retrieves the inner `BitData` living in the `Bit`
    pub(crate) fn inner_new(
        register: Option<PyRegister>,
        index: Option<u32>,
        extra: Option<BitExtraInfo>,
    ) -> PyResult<Self> {
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
                Ok(Self(BitInfo::new_owned(owned.clone(), index, extra)))
            }
            (None, None) => Ok(Self(BitInfo::new_anonymous(extra))),
            _ => Err(CircuitError::new_err(
                "You should provide both a valid register and an index, not either or.".to_string(),
            )),
        }
    }
}

macro_rules! create_py_bit {
    ($name:ident, $natbit:tt, $pyname:literal, $pymodule:literal, $extra:expr, $pyreg:tt) => {
        /// Implements a quantum bit
        #[pyclass(
                                            subclass,
                                            name = $pyname,
                                            module = $pymodule,
                                            frozen,
                                            extends=PyBit,
                                        )]
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name(pub(crate) $natbit);

        #[pymethods]
        impl $name {
            /// Creates a new bit.
            #[new]
            #[pyo3(signature = (register = None, index = None))]
            fn new(register: Option<$pyreg>, index: Option<u32>) -> PyResult<(Self, PyBit)> {
                let inner = PyBit::inner_new(
                    register.clone().map(|reg| PyRegister(reg.data().clone())),
                    index,
                    Some($extra),
                )?;
                Ok((Self($natbit(inner.0.clone())), inner))
            }

            #[getter]
            fn _register(slf: PyRef<Self>) -> PyResult<Option<Bound<$pyreg>>> {
                slf.0
                    .register()
                    .map(|reg| reg.clone().into_pyobject(slf.py()))
                    .transpose()
            }

            fn __deepcopy__<'a>(
                slf: Bound<'a, Self>,
                _cache: Bound<'a, PyDict>,
            ) -> PyResult<Bound<'a, Self>> {
                let borrowed = slf.borrow();
                match borrowed.0 .0 {
                    BitInfo::Owned { .. } => {
                        let py = slf.py();
                        borrowed.0.clone().into_pyobject(py)
                    }
                    BitInfo::Anonymous { .. } => Ok(slf),
                }
            }
        }
    };
}

create_py_bit!(
    PyQubit,
    ShareableQubit,
    "Qubit",
    "qiskit.circuit.quantumcircuit",
    BitExtraInfo::Qubit { is_ancilla: false },
    PyQuantumRegister
);

impl PyQubit {
    /// Alternative constructor reserved for `AncillaQubit`
    fn new_ancilla(
        register: Option<PyQuantumRegister>,
        index: Option<u32>,
    ) -> PyResult<(Self, PyBit)> {
        let mut inner = PyBit::new(
            register.clone().map(|reg| PyRegister(reg.data().clone())),
            index,
        )?;
        match &mut inner.0 {
            BitInfo::Owned { extra, .. } => *extra = Some(BitExtraInfo::Qubit { is_ancilla: true }),
            BitInfo::Anonymous { extra, .. } => {
                *extra = Some(BitExtraInfo::Qubit { is_ancilla: true })
            }
        }

        Ok((Self(ShareableQubit(inner.0.clone())), inner))
    }
}

create_py_bit!(
    PyClbit,
    ShareableClbit,
    "Clbit",
    "qiskit.circuit.classicalregister",
    BitExtraInfo::Clbit(),
    PyClassicalRegister
);

/// A qubit used as ancillary qubit.
#[pyclass(
    extends=PyQubit,
    name = "AncillaQubit",
    module = "qiskit.circuit.quantumregister",
    frozen,
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PyAncillaQubit(pub(crate) PyQubit);

#[pymethods]
impl PyAncillaQubit {
    /// Creates a new anonymous `AncillaQubit`.
    #[new]
    #[pyo3(signature = (register = None, index = None))]
    fn new(
        register: Option<PyQuantumRegister>,
        index: Option<u32>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let (qubit, base) = PyQubit::new_ancilla(register, index)?;
        Ok(PyClassInitializer::from(base)
            .add_subclass(qubit.clone())
            .add_subclass(Self(qubit)))
    }
}
