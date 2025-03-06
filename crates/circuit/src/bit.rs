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

//! Definitions of the shareable bit types and registers.

use std::{
    cmp::Eq,
    fmt::{Debug, Display},
    hash::Hash,
    sync::{
        atomic::{AtomicU32, AtomicU64},
        Arc,
    },
};

use indexmap::IndexSet;
use pyo3::prelude::*;
use pyo3::{
    exceptions::PyValueError,
    types::{PyDict, PyList, PyString, PyTuple, PyType},
    IntoPyObjectExt, PyTypeInfo,
};

use crate::{
    circuit_data::CircuitError,
    dag_circuit::PyBitLocations,
    slice::{PySequenceIndex, PySequenceIndexError},
};

/// Describes a relationship between a bit and all the registers it belongs to
#[derive(Debug, Clone)]
pub struct BitLocations<R: Register> {
    pub(crate) index: u32,
    registers: Vec<(R, usize)>,
}

impl<R: Register + PartialEq> BitLocations<R> {
    /// Creates new instance of [BitLocations]
    pub fn new<T: IntoIterator<Item = (R, usize)>>(index: u32, registers: T) -> Self {
        Self {
            index,
            registers: registers.into_iter().collect(),
        }
    }

    /// Adds a register entry
    pub fn add_register(&mut self, register: R, index: usize) {
        self.registers.push((register, index))
    }

    /// Removes a register location on `O(n)`` time, where N is the number of
    /// registers in this entry.
    pub fn remove_register(&mut self, register: &R, index: usize) -> Option<(R, usize)> {
        for (idx, reg) in self.registers.iter().enumerate() {
            if (&reg.0, &reg.1) == (register, &index) {
                let res = self.registers.remove(idx);
                return Some(res);
            }
        }
        None
    }
}

impl<'py, R> IntoPyObject<'py> for BitLocations<R>
where
    R: Debug + Clone + Register + for<'a> IntoPyObject<'a>,
{
    type Target = PyBitLocations;
    type Output = Bound<'py, PyBitLocations>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        PyBitLocations::new(
            self.index as usize,
            self.registers
                .into_pyobject(py)?
                .downcast_into::<PyList>()?
                .unbind(),
        )
        .into_pyobject(py)
    }
}

impl<'py, R> FromPyObject<'py> for BitLocations<R>
where
    R: Debug + Clone + Register + for<'a> IntoPyObject<'a> + for<'a> FromPyObject<'a>,
{
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let ob_down = ob.downcast::<PyBitLocations>()?.borrow();
        Ok(Self {
            index: ob_down.index as u32,
            registers: ob_down.registers.extract(ob.py())?,
        })
    }
}

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
pub(crate) enum BitInfo {
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
        #[doc = concat!("Creates an instance of [", stringify!($name), "].")]
        pub struct $name(pub(crate) BitInfo);

        impl $name {
            #[doc = concat!("Creates an anonymous instance of [", stringify!($name), "].")]
            pub fn new_anonymous() -> Self {
                Self(BitInfo::new_anonymous(Some($extra_exp)))
            }

            #[doc = concat!("Returns a reference to the owning register of the [", stringify!($name), "] if any exists.")]
            pub fn register(&self) -> Option<$reg> {
                self.0.register().map(|reg| $reg(reg.into()))
            }

            #[doc = concat!("Returns the index of the [", stringify!($name), "] within the owning register if any exists.")]
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

    /// Creates an instance of ancilla qubit.
    pub fn new_anonymous_ancilla() -> Self {
        Self(BitInfo::new_anonymous(Some(BitExtraInfo::Qubit {
            is_ancilla: true,
        })))
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
        Ok(ob.downcast::<PyQubit>()?.borrow().0.clone())
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
        Ok(ob.downcast::<PyClbit>()?.borrow().0.clone())
    }
}

/// Implement a generic bit.
///
/// .. note::
///     This class should not be instantiated directly. This is just a superclass
///     for :class:`~.Clbit` and :class:`~.circuit.Qubit`.
#[pyclass(subclass, name = "Bit", module = "qiskit.circuit")]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub struct PyBit(pub(crate) BitInfo);

#[pymethods]
impl PyBit {
    /// Create a new generic bit.
    #[new]
    #[pyo3(signature = (register = None, index = None))]
    fn new(register: Option<PyRegister>, index: Option<u32>) -> PyResult<Self> {
        Self::inner_new(register, index, None)
    }

    /// Return the official string representing the bit.
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
                slf.repr()?.as_any().eq(other.repr()?)
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
        // Bits are immutable.
        slf
    }

    fn __deepcopy__<'a>(
        slf: Bound<'a, Self>,
        _cache: Bound<'a, PyDict>,
    ) -> PyResult<Bound<'a, Self>> {
        let borrowed = slf.borrow();
        match borrowed.0 {
            BitInfo::Owned { .. } => {
                // Only perform a clone of an owned bit because it needs special handling.
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

    fn __reduce__(slf: Bound<'_, Self>) -> PyResult<Bound<PyTuple>> {
        let borrowed = slf.borrow();
        match borrowed.0 {
            BitInfo::Owned { index, .. } => (
                slf.get_type(),
                (slf.getattr("_register")?.unbind(), Some(index)),
                None,
            ),
            BitInfo::Anonymous { unique_id, .. } => {
                (slf.get_type(), (slf.py().None(), None), Some(unique_id))
            }
        }
        .into_pyobject(slf.py())
    }

    #[pyo3(signature = (state = None))]
    fn __setstate__(slf: Bound<'_, Self>, state: Option<u64>) -> PyResult<()> {
        let mut borrowed_mut = slf.borrow_mut();
        let result = if let Some(state) = state {
            match &mut borrowed_mut.0 {
                BitInfo::Owned { .. } => Ok(()),
                BitInfo::Anonymous { unique_id, .. } => {
                    *unique_id = state;
                    Ok(())
                }
            }
        } else {
            Ok(())
        };
        drop(borrowed_mut);
        result
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
    ($name:ident, $natbit:tt, $pyname:literal, $pymodule:literal, $extra:expr, $pyreg:tt, $specifier:literal, $natreg:ty) => {
        #[doc = concat!("Implements a ", $specifier, " bit.")]
        #[rustfmt::skip] // Due to a bug in rustfmt, formatting is skipped in this line
        #[pyclass(
            subclass,
            name = $pyname,
            module = $pymodule,
            extends=PyBit,
        )]
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name(pub(crate) $natbit);

        #[pymethods]
        impl $name {
            #[doc = concat!("Creates a", stringify!($name), " .\n
            Args:\n
                register (", stringify!($natreg), "): Optional. A quantum register containing the bit.\n
                index (int): Optional. The index of the bit in its containing register.\n
            \n
            \n
            Raises:\n
                CircuitError: if the provided register is not a valid :class:`", stringify!($natreg), "`
            ")]
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

            #[pyo3(signature = (state = None))]
            fn __setstate__(slf: Bound<'_, Self>, state: Option<u64>) -> PyResult<()> {
                let mut borrowed_mut = slf.borrow_mut();
                if let Some(state) = state {
                    match &mut borrowed_mut.0 .0 {
                        BitInfo::Owned { .. } => (),
                        BitInfo::Anonymous { unique_id, .. } => {
                            *unique_id = state;
                        }
                    }
                }
                drop(borrowed_mut);
                PyBit::__setstate__(slf.into_super(), state)
            }
        }
    };
}

create_py_bit!(
    PyQubit,
    ShareableQubit,
    "Qubit",
    "qiskit.circuit",
    BitExtraInfo::Qubit { is_ancilla: false },
    PyQuantumRegister,
    "quantum",
    QuantumRegister
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
    "qiskit.circuit",
    BitExtraInfo::Clbit(),
    PyClassicalRegister,
    "classical",
    ClassicalRegister
);

/// A qubit used as ancillary qubit.
#[pyclass(
    extends=PyQubit,
    name = "AncillaQubit",
    module = "qiskit.circuit",
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

pub trait Register {
    /// The type of bit stored by the [Register]
    type Bit;

    /// Returns the size of the [Register].
    fn len(&self) -> usize;
    /// Checks if the [Register] is empty.
    fn is_empty(&self) -> bool;
    /// Returns the name of the [Register].
    fn name(&self) -> &str;
    /// Checks if a bit exists within the [Register].
    fn contains(&self, bit: &Self::Bit) -> bool;
    /// Return an iterator over all the bits in the register
    fn bits(&self) -> impl ExactSizeIterator<Item = Self::Bit>;
    /// Gets a bit by index
    fn get(&self, index: usize) -> Option<Self::Bit>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RegisterInfo {
    Owning(Arc<OwningRegisterInfo>),
    Alias {
        name: String,
        bits: Box<IndexSet<BitInfo>>,
        extra: Option<BitExtraInfo>,
    },
}

// Custom implmentation of Hash to disregard the `IndexMap` used for Alias.
impl Hash for RegisterInfo {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            RegisterInfo::Owning(owning_register_info) => owning_register_info.hash(state),
            RegisterInfo::Alias { name, bits, extra } => {
                (core::mem::discriminant(self), name, bits.len(), extra).hash(state)
            }
        }
    }
}

impl RegisterInfo {
    /// Creates a Register whose bits are owned by its instance
    pub fn new_owning(name: String, size: u32, extra: Option<BitExtraInfo>) -> Self {
        // When creating `Owning` register, we don't need to create the `BitInfo`
        // instances, they can be entirely derived from `self`.
        Self::Owning(Arc::new(OwningRegisterInfo { name, size, extra }))
    }

    /// Creates a Register whose bits already exist.
    pub fn new_alias(
        name: String,
        bits: Box<IndexSet<BitInfo>>,
        extra: Option<BitExtraInfo>,
    ) -> Self {
        Self::Alias { name, bits, extra }
    }

    /// A reference to the register's name
    pub fn name(&self) -> &str {
        match self {
            RegisterInfo::Owning(owning_register_info) => owning_register_info.name.as_str(),
            RegisterInfo::Alias { name, .. } => name.as_str(),
        }
    }

    /// Returns the size of the register.
    pub fn len(&self) -> usize {
        match self {
            RegisterInfo::Owning(owning_register_info) => owning_register_info.size as usize,
            RegisterInfo::Alias { bits, .. } => bits.len(),
        }
    }

    /// Returns whether the register is empty or not.
    pub fn is_empty(&self) -> bool {
        match self {
            RegisterInfo::Owning(owning_register_info) => owning_register_info.is_empty(),
            RegisterInfo::Alias { bits, .. } => bits.is_empty(),
        }
    }

    /// Returns an iterator over the bits within the circuit
    pub(crate) fn bits(&self) -> Box<dyn ExactSizeIterator<Item = BitInfo> + '_> {
        match self {
            RegisterInfo::Owning(owning_register_info) => {
                Box::new((0..owning_register_info.size).map(|bit| BitInfo::Owned {
                    register: owning_register_info.clone(),
                    index: bit,
                    extra: owning_register_info.extra,
                }))
            }
            RegisterInfo::Alias { bits, .. } => Box::new(bits.iter().cloned()),
        }
    }

    /// Checks if a bit is contained within the register
    pub(crate) fn contains(&self, bit: &BitInfo) -> bool {
        match self {
            RegisterInfo::Owning(owning_register_info) => match bit {
                BitInfo::Owned {
                    register, index, ..
                } => register == owning_register_info && *index < owning_register_info.size,
                BitInfo::Anonymous { .. } => false,
            },
            RegisterInfo::Alias { bits, .. } => bits.contains(bit),
        }
    }

    pub(crate) fn get(&self, index: usize) -> Option<BitInfo> {
        match self {
            RegisterInfo::Owning(owning_register_info) => {
                if index < owning_register_info.size as usize {
                    Some(BitInfo::new_owned(
                        owning_register_info.clone(),
                        index.try_into().expect(
                            "The register you tried to index from has exceeded its size limit",
                        ),
                        owning_register_info.extra,
                    ))
                } else {
                    None
                }
            }
            RegisterInfo::Alias { bits, .. } => bits.get_index(index).cloned(),
        }
    }
}

/// Contains the informaion for a register that owns the bits it contains.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub struct OwningRegisterInfo {
    name: String,
    size: u32,
    extra: Option<BitExtraInfo>,
}

impl OwningRegisterInfo {
    /// A reference to the register's name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the size of the register.
    pub fn len(&self) -> usize {
        self.size as usize
    }

    /// Returns whether the register is empty or not.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

// Create rust native Register types.
macro_rules! create_register_object {
    ($name:ident, $bit:tt, $extra_exp:expr, $counter:ident, $prefix:literal) => {
        static $counter: AtomicU32 = AtomicU32::new(0);

        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name(pub(crate) Arc<RegisterInfo>);

        impl $name {
            fn name_parse(name: Option<String>) -> String {
                if let Some(name) = name {
                    name
                } else {
                    format!(
                        "{}{}",
                        $prefix,
                        $counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    )
                }
            }

            /// Creates a Register whose bits are owned by its instance
            pub fn new_owning(name: Option<String>, size: u32) -> Self {
                // When creating `Owning` register, we don't need to create the `BitInfo`
                // instances, they can be entirely derived from `self`.
                Self(
                    RegisterInfo::new_owning(Self::name_parse(name), size, Some($extra_exp)).into(),
                )
            }

            /// Creates a Register whose bits already exist.
            pub fn new_alias(name: Option<String>, bits: IndexSet<$bit>) -> Self {
                Self(
                    RegisterInfo::new_alias(
                        Self::name_parse(name),
                        Box::new(bits.into_iter().map(|bit| bit.0).collect()),
                        Some($extra_exp),
                    )
                    .into(),
                )
            }
        }

        impl $name {
            /// Provide a counted reference to the inner data of the register
            pub(crate) fn data(&self) -> &Arc<RegisterInfo> {
                &self.0
            }

            pub(crate) fn get_instance_count() -> u32 {
                $counter.load(std::sync::atomic::Ordering::Relaxed)
            }
        }

        impl Register for $name {
            type Bit = $bit;
            /// A reference to the register's name
            fn name(&self) -> &str {
                self.0.name()
            }

            /// Returns the size of the register.
            fn len(&self) -> usize {
                self.0.len()
            }

            /// Returns whether the register is empty.
            fn is_empty(&self) -> bool {
                self.0.is_empty()
            }

            /// Returns an iterator over the bits within the circuit
            fn bits(&self) -> impl ExactSizeIterator<Item = <Self as Register>::Bit> {
                self.0.bits().map(|bit| $bit(bit))
            }

            /// Checks if a bit is contained within the register
            fn contains(&self, bit: &<$name as Register>::Bit) -> bool {
                self.0.contains(&bit.0)
            }

            /// Gets a bit via index, return None if not present
            fn get(&self, index: usize) -> Option<$bit> {
                self.0.get(index).map(|bit| $bit(bit))
            }
        }
    };
}

create_register_object! {QuantumRegister, ShareableQubit, BitExtraInfo::Qubit{is_ancilla: false}, QREG_COUNTER, "q"}
impl QuantumRegister {
    /// Check if the [QuantumRegister] instance is ancillary.
    pub fn is_ancilla(&self) -> bool {
        match self.0.as_ref() {
            RegisterInfo::Owning(owning_register_info) => match owning_register_info.extra {
                Some(BitExtraInfo::Qubit { is_ancilla }) => is_ancilla,
                _ => false,
            },
            RegisterInfo::Alias { extra, .. } => match extra {
                Some(BitExtraInfo::Qubit { is_ancilla }) => *is_ancilla,
                _ => false,
            },
        }
    }

    /// Creates a Register whose bits are owned by its instance
    pub fn new_ancilla_owning(name: Option<String>, size: u32) -> Self {
        // When creating `Owning` register, we don't need to create the `BitInfo`
        // instances, they can be entirely derived from `self`.
        Self(
            RegisterInfo::new_owning(
                Self::name_parse(name),
                size,
                Some(BitExtraInfo::Qubit { is_ancilla: true }),
            )
            .into(),
        )
    }

    /// Creates a Register whose bits already exist.
    pub fn new_ancilla_alias(name: Option<String>, bits: IndexSet<ShareableQubit>) -> Self {
        Self(
            RegisterInfo::new_alias(
                Self::name_parse(name),
                Box::new(bits.into_iter().map(|bit| bit.0).collect()),
                Some(BitExtraInfo::Qubit { is_ancilla: true }),
            )
            .into(),
        )
    }
}
create_register_object! {ClassicalRegister, ShareableClbit, BitExtraInfo::Clbit(), CREG_COUNTER, "c"}

impl Display for QuantumRegister {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let identifier = match self.is_ancilla() {
            true => "AncillaRegister",
            false => "QuantumRegister",
        };
        write!(f, "{}({}, '{}')", identifier, self.len(), self.name())
    }
}

impl Display for ClassicalRegister {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ClassicalRegister({}, '{}')", self.len(), self.name())
    }
}

// Fast path conversion to avoid using python types.
impl<'py> IntoPyObject<'py> for QuantumRegister {
    type Target = PyQuantumRegister;
    type Output = Bound<'py, PyQuantumRegister>;
    type Error = PyErr;
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let is_ancilla = self.is_ancilla();
        let q_reg = PyQuantumRegister(self.clone());
        match is_ancilla {
            true => {
                let ancilla = PyAncillaRegister(q_reg.clone());
                let initializer = PyClassInitializer::from(PyRegister(self.data().clone()))
                    .add_subclass(q_reg)
                    .add_subclass(ancilla);
                Ok(Bound::new(py, initializer)?.into_super())
            }
            false => Bound::new(py, (q_reg, PyRegister(self.data().clone()))),
        }
    }
}

// Fast path conversion to avoid using extracting into python types.
impl<'py> FromPyObject<'py> for QuantumRegister {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(ob.downcast::<PyQuantumRegister>()?.borrow().0.clone())
    }
}

// Fast path conversion to avoid using python types.
impl<'py> IntoPyObject<'py> for ClassicalRegister {
    type Target = PyClassicalRegister;
    type Output = Bound<'py, PyClassicalRegister>;
    type Error = PyErr;
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Bound::new(
            py,
            (
                PyClassicalRegister(self.clone()),
                PyRegister(self.data().clone()),
            ),
        )
    }
}

// Fast path conversion to avoid using extracting into python types.
impl<'py> FromPyObject<'py> for ClassicalRegister {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(ob.downcast::<PyClassicalRegister>()?.borrow().0.clone())
    }
}

/// Counter for base register types.
static REG_COUNTER: AtomicU32 = AtomicU32::new(0);
/// Implement a generic register.
///
/// .. note::
///     This class should not be instantiated directly. This is just a superclass
///     for :class:`~.ClassicalRegister` and :class:`~.QuantumRegister`.
#[pyclass(
    name = "Register",
    module = "qiskit.circuit",
    subclass,
    frozen,
    sequence
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PyRegister(pub(crate) Arc<RegisterInfo>);

#[pymethods]
impl PyRegister {
    /// Create a new generic register.
    ///
    /// Either the ``size`` or the ``bits`` argument must be provided. If
    /// ``size`` is not None, the register will be pre-populated with bits of the
    /// correct type.
    ///
    /// Args:
    ///     size (int): Optional. The number of bits to include in the register.
    ///     name (str): Optional. The name of the register. If not provided, a
    ///         unique name will be auto-generated from the register type.
    ///     bits (list[Bit]): Optional. A list of Bit() instances to be used to
    ///         populate the register.
    ///
    /// Raises:
    ///     CircuitError: if both the ``size`` and ``bits`` arguments are
    ///         provided, or if neither are.
    ///     CircuitError: if ``size`` is not valid.
    ///     CircuitError: if ``name`` is not a valid name according to the
    ///         OpenQASM spec.
    ///     CircuitError: if ``bits`` contained duplicated bits.
    ///     CircuitError: if ``bits`` contained bits of an incorrect type.
    ///     CircuitError: if ``bits`` exceeds the possible capacity for a register.
    #[new]
    #[pyo3(signature = (size = None, name = None, bits = None))]
    pub fn new(
        size: Option<Bound<PyAny>>,
        name: Option<Bound<PyAny>>,
        bits: Option<Bound<PyAny>>,
    ) -> PyResult<Self> {
        let name_parse = |name: Option<String>| -> String {
            if let Some(name) = name {
                name
            } else {
                format!(
                    "reg{}",
                    REG_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                )
            }
        };

        let (size, name) = Self::inner_parse_new(&size, &name, &bits)?;

        let register = if let Some(bits) = bits {
            let Ok(bits_set): PyResult<IndexSet<BitInfo>> = bits
                .try_iter()?
                .map(|bit| -> PyResult<BitInfo> { bit?.extract::<PyBit>().map(|b| b.0) })
                .collect()
            else {
                return Err(CircuitError::new_err(format!(
                    "Provided bits did not all match register type. bits={}",
                    bits.repr()?
                )));
            };
            if bits_set.len() != size as usize {
                return Err(CircuitError::new_err(
                    "Register bits must not be duplicated.",
                ));
            }
            RegisterInfo::new_alias(name_parse(name), bits_set.into(), None)
        } else {
            RegisterInfo::new_owning(name_parse(name), size, None)
        };

        Ok(Self(register.into()))
    }

    /// Return the official string representing the register.
    pub fn __repr__(slf: Bound<Self>) -> PyResult<String> {
        let borrowed = slf.borrow();
        match borrowed.0.as_ref() {
            RegisterInfo::Owning(owning_register_info) => Ok(format!(
                "{}({}, '{}')",
                slf.get_type().qualname()?,
                owning_register_info.len(),
                owning_register_info.name()
            )),
            RegisterInfo::Alias { name, bits, .. } => Ok(format!(
                "{}({}, '{}')",
                slf.get_type().qualname()?,
                bits.len(),
                name,
            )),
        }
    }

    fn __contains__(&self, bit: &PyBit) -> bool {
        self.0.contains(&bit.0)
    }

    /// Get the register name
    #[getter]
    pub fn name(&self) -> &str {
        self.0.name()
    }

    /// Get the register size
    #[getter]
    pub fn size(&self) -> usize {
        self.0.len()
    }

    /// Arg:
    ///     bit_type (Qubit or Clbit): a constructor type return element/s.
    ///     key (int or slice or list): index of the bit to be retrieved.
    ///
    /// Returns:
    ///     Qubit or Clbit or list(Qubit) or list(Clbit): a Qubit or Clbit instance if
    ///     key is int. If key is a slice, returns a list of these instances.
    ///
    /// Raises:
    ///     CircuitError: if the `key` is not an integer or not in the range `(0, self.size)`.
    fn __getitem__<'py>(slf: PyRef<'py, Self>, key: SliceOrList<'py>) -> PyResult<PyObject> {
        let py = slf.py();
        match &key {
            SliceOrList::Slice(py_sequence_index) => {
                let sequence = py_sequence_index.with_len(slf.size())?;
                match sequence {
                    crate::slice::SequenceIndex::Int(_) => {
                        slf.getitem_inner(key)?.next().map(PyBit).into_py_any(py)
                    }
                    _ => Ok(PyList::new(py, slf.getitem_inner(key)?.map(PyBit))?
                        .into_any()
                        .unbind()),
                }
            }
            _ => Ok(PyList::new(py, slf.getitem_inner(key)?.map(PyBit))?
                .into_any()
                .unbind()),
        }
    }

    /// Find the index of the provided bit within this register.
    fn index(slf: Bound<'_, Self>, bit: Bound<PyBit>) -> PyResult<usize> {
        let err = || -> PyResult<usize> {
            Err(PyValueError::new_err(format!(
                "Bit {} not found in Register {}.",
                bit,
                slf.as_super().repr()?,
            )))
        };
        let slf_borrowed = slf.borrow();
        let bit_borrowed = bit.borrow();
        let bit_inner = &bit_borrowed.0;
        match bit_inner {
            BitInfo::Owned { index, .. } => {
                if slf_borrowed.0.contains(bit_inner) {
                    Ok((*index) as usize)
                } else {
                    err()
                }
            }
            BitInfo::Anonymous { .. } => match slf_borrowed.0.as_ref() {
                RegisterInfo::Owning(..) => err(),
                RegisterInfo::Alias { bits, .. } => {
                    bits.get_index_of(bit_inner)
                        .ok_or(PyValueError::new_err(format!(
                            "Bit {} not found in Register {}.",
                            bit,
                            slf.repr()?,
                        )))
                }
            },
        }
    }

    /// Make object hashable, based on the name and size to hash.
    fn __hash__(slf: Bound<'_, Self>) -> PyResult<isize> {
        let borrowed = slf.borrow();
        (slf.get_type(), borrowed.name(), borrowed.size())
            .into_bound_py_any(slf.py())?
            .hash()
    }

    /// Two Registers are the same if they are of the same type
    /// (i.e. quantum/classical), and have the same name and size. Additionally,
    /// if either Register contains new-style bits, the bits in both registers
    /// will be checked for pairwise equality. If two registers are equal,
    /// they will have behave identically when specified as circuit args.
    ///
    /// Args:
    ///     other (Register): other Register
    ///
    /// Returns:
    ///     bool: `self` and `other` are equal.
    fn __eq__(slf: Bound<'_, Self>, other: Bound<'_, Self>) -> PyResult<bool> {
        if slf.is(&other) {
            return Ok(true);
        }
        let slf_borrow = slf.borrow();
        let other_borrow = other.borrow();
        Ok(slf.get_type().eq(other.get_type())?
            && slf.repr()?.to_string().eq(&other.repr()?.to_string())
            && slf_borrow.eq(&other_borrow))
    }

    fn __reduce__(slf: Bound<'_, Self>) -> PyResult<Bound<PyTuple>> {
        let borrowed = slf.borrow();
        let ty = slf.get_type();
        let args = match borrowed.0.as_ref() {
            RegisterInfo::Owning(reg) => (Some(reg.size), Some(reg.name.clone()), None),
            RegisterInfo::Alias { name, .. } => (
                None,
                Some(name.to_string()),
                Some(PyList::type_object(slf.py()).call1((&slf,))?),
            ),
        };
        (ty, args).into_pyobject(slf.py())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Bound<'_, pyo3::types::PyIterator>> {
        return PyList::new(slf.py(), slf.iter().map(PyBit))?.try_iter();
    }

    /// Return register size.
    fn __len__(slf: PyRef<'_, Self>) -> usize {
        slf.size()
    }

    /// Prefix to use for auto naming.
    #[classattr]
    fn prefix<'py>() -> &'py str {
        "reg"
    }

    /// Bit type stored in the register.
    #[classattr]
    fn bit_type(py: Python) -> Bound<PyType> {
        PyBit::type_object(py)
    }

    /// Counter for the number of instances in this class.
    #[classattr]
    fn instances_count() -> u32 {
        REG_COUNTER.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl PyRegister {
    /// Correctly performs extraction of size and name during creation of a register
    pub fn inner_parse_new(
        size: &Option<Bound<PyAny>>,
        name: &Option<Bound<PyAny>>,
        bits: &Option<Bound<PyAny>>,
    ) -> PyResult<(u32, Option<String>)> {
        if (size.is_none() && bits.is_none()) || (size.is_some() && bits.is_some()) {
            return Err(CircuitError::new_err(format!("Exactly one of the size or bits arguments can be provided. Provided size={:?} bits={:?}.", size, bits)));
        }

        let size: u32 = if let Some(bits) = bits.as_ref() {
            bits.len()?.try_into().map_err(|_| CircuitError::new_err(format!("The amount of bits provided exceeds the capacity of the register. Current size {}", bits.len().unwrap_or_default())))?
        } else {
            let size = size.as_ref().unwrap();
            let valid_size: isize = if let Ok(valid) = size.extract() {
                valid
            } else if let Ok(almost_valid) = size.extract::<f64>() {
                let valid: isize = (almost_valid - almost_valid.floor() == 0.0).then_some(almost_valid as isize).ok_or( CircuitError::new_err(format!(
                    "Register size must be an integer or castable to an integer. {} '{}' was provided",
                    size.get_type().name()?,
                    size.repr()?
                )))?;
                valid
            } else {
                return Err(CircuitError::new_err(format!(
                    "Register size must be an integer or castable to an integer. {} '{}' was provided",
                    size.get_type().name()?,
                    size.repr()?
                )));
            };
            if valid_size < 0 {
                return Err(CircuitError::new_err(format!(
                    "Register size must be non-negative. {} '{}' was provided",
                    size.get_type().name()?,
                    size.repr()?
                )));
            }

            let Ok(valid_size) = valid_size.try_into() else {
                return Err(CircuitError::new_err(format!(
                    "Register size exceeds possible allocated capacity. {} '{}' was provided",
                    size.get_type().name()?,
                    size.repr()?
                )));
            };

            valid_size
        };

        let Ok(name) = name
            .as_ref()
            .map(|name| -> PyResult<String> {
                PyString::type_object(name.py())
                    .call1((name,))?
                    .extract::<String>()
            })
            .transpose()
        else {
            return Err(CircuitError::new_err("The circuit name should be castable to a string (or None for autogenerate a name)."));
        };

        Ok((size, name))
    }

    /// Inner function for [PyRegister::__getitem__] to process indexing more efficiently and
    /// allow reuse in subclasses.
    fn getitem_inner(
        &self,
        key: SliceOrList<'_>,
    ) -> PyResult<Box<dyn ExactSizeIterator<Item = BitInfo>>> {
        match &key {
            SliceOrList::Slice(py_sequence_index) => {
                let sequence = py_sequence_index.with_len(self.size())?;
                match sequence {
                    crate::slice::SequenceIndex::Int(idx) => {
                        let Some(bit) = self.0.get(idx) else {
                            return Err(CircuitError::new_err("register index out of range"));
                        };
                        Ok(Box::new(std::iter::once(bit)))
                    }
                    _ => {
                        let result: Vec<BitInfo> = key
                            .iter_with_size(self.size())?
                            .map(|idx| -> PyResult<BitInfo> {
                                self.0
                                    .get(idx)
                                    .ok_or(CircuitError::new_err("register index out of range"))
                            })
                            .collect::<PyResult<_>>()?;
                        Ok(Box::new(result.into_iter()))
                    }
                }
            }
            SliceOrList::List(_) => {
                let result: Vec<BitInfo> = key
                    .iter_with_size(self.size())?
                    .map(|idx| -> PyResult<BitInfo> {
                        self.0
                            .get(idx)
                            .ok_or(CircuitError::new_err("register index out of range"))
                    })
                    .collect::<PyResult<_>>()?;
                Ok(Box::new(result.into_iter()))
            }
        }
    }

    fn iter(&self) -> impl ExactSizeIterator<Item = BitInfo> + '_ {
        (0..self.size()).map(|bit| self.0.get(bit).unwrap())
    }
}

/// Correctly extracts a Slice or a Vec from Python.
#[derive(FromPyObject)]
enum SliceOrList<'py> {
    Slice(PySequenceIndex<'py>),
    List(Vec<isize>),
}

impl SliceOrList<'_> {
    pub fn iter_with_size(&self, size: usize) -> PyResult<Box<dyn Iterator<Item = usize>>> {
        match self {
            SliceOrList::Slice(py_sequence_index) => {
                let sequence = py_sequence_index.with_len(size);
                match sequence {
                    Ok(sequence) => Ok(Box::new(sequence.iter())),
                    Err(e) => Err(e.into()),
                }
            }
            SliceOrList::List(items) => {
                let items: Vec<usize> = items
                    .iter()
                    .copied()
                    .map(|idx| {
                        if idx.is_negative() {
                            size.checked_sub(idx.unsigned_abs())
                                .ok_or(PySequenceIndexError::OutOfRange.into())
                        } else {
                            Ok(idx as usize)
                        }
                    })
                    .collect::<PyResult<_>>()?;
                Ok(Box::new(items.into_iter()))
            }
        }
    }
}

macro_rules! create_py_register {
    ($name:ident, $nativereg:tt, $pybit:tt, $nativebit:tt, $pyname:literal, $pymodule:literal, $extra:expr, $prefix:literal, $specifier: literal) => {
#[rustfmt::skip] // Due to a bug in rustfmt, formatting is skipped in this line
        #[doc = concat!("Implement a ", $specifier, " register.")]
        #[pyclass(
            name = $pyname,
            module = $pymodule,
            subclass,
            extends = PyRegister,
            frozen,
            sequence,
        )]
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name(pub(crate) $nativereg);

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (size = None, name = None, bits = None))]
            pub fn new(
                size: Option<Bound<PyAny>>,
                name: Option<Bound<PyAny>>,
                bits: Option<Bound<PyAny>>,
            ) -> PyResult<(Self, PyRegister)> {
                let (size, name) = PyRegister::inner_parse_new(&size, &name, &bits)?;

                let register = if let Some(bits) = bits {
                    let Ok(bits_set): PyResult<IndexSet<$nativebit>> = bits
                        .try_iter()?
                        .map(|bit| -> PyResult<$nativebit> { bit?.extract::<$nativebit>() })
                        .collect()
                    else {
                        return Err(CircuitError::new_err(format!(
                            "Provided bits did not all match register type. bits={}",
                            bits.repr()?
                        )));
                    };
                    if bits_set.len() != size as usize {
                        return Err(CircuitError::new_err(
                            "Register bits must not be duplicated.",
                        ));
                    }
                    $nativereg::new_alias(name, bits_set)
                } else {
                    $nativereg::new_owning(name, size)
                };
                let inner_reg = register.data().clone();
                Ok((Self(register), PyRegister(inner_reg)))
            }

            fn __getitem__<'py>(
                slf: PyRef<'py, Self>,
                key: SliceOrList<'py>,
            ) -> PyResult<PyObject> {
                let py = slf.py();
                match &key {
                    SliceOrList::Slice(py_sequence_index) => {
                        let sequence = py_sequence_index
                            .with_len(slf.as_super().size().try_into().unwrap())?;
                        match sequence {
                            crate::slice::SequenceIndex::Int(_) => {
                                slf.getitem_inner(key)?.next().into_py_any(py)
                            }
                            _ => Ok(PyList::new(py, slf.getitem_inner(key)?)?
                                .into_any()
                                .unbind()),
                        }
                    }
                    _ => Ok(PyList::new(py, slf.getitem_inner(key)?)?
                        .into_any()
                        .unbind()),
                }
            }

            fn __contains__(&self, bit: $nativebit) -> bool {
                self.0.contains(&bit)
            }

            fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Bound<'_, pyo3::types::PyIterator>> {
                let list: Vec<$nativebit> = slf.0.bits().collect();
                list.into_pyobject(slf.py())?.try_iter()
            }

            #[classattr]
            fn prefix<'py>() -> &'py str {
                $prefix
            }

            #[classattr]
            fn bit_type(py: Python) -> Bound<PyType> {
                $pybit::type_object(py)
            }

            #[classattr]
            fn instances_count() -> u32 {
                $nativereg::get_instance_count()
            }
        }

        impl $name {
            /// Provide a reference to the inner data of the register
            pub(crate) fn data(&self) -> &Arc<RegisterInfo> {
                self.0.data()
            }

            /// Inner function for [PyRegister::__getnewargs__] to ensure serialization can be
            /// preserved between register types.
            fn getitem_inner(
                &self,
                key: SliceOrList<'_>,
            ) -> PyResult<Box<dyn ExactSizeIterator<Item = $nativebit>>> {
                match &key {
                    SliceOrList::Slice(py_sequence_index) => {
                        let sequence = py_sequence_index.with_len(self.0.len())?;
                        match sequence {
                            crate::slice::SequenceIndex::Int(idx) => {
                                let Some(bit) = self.0.get(idx) else {
                                    return Err(CircuitError::new_err(
                                        "register index out of range",
                                    ));
                                };
                                Ok(Box::new(std::iter::once(bit)))
                            }
                            _ => {
                                let result: Vec<$nativebit> = key
                                    .iter_with_size(self.0.len())?
                                    .map(|idx| -> PyResult<$nativebit> {
                                        self.0.get(idx).ok_or(CircuitError::new_err(
                                            "register index out of range",
                                        ))
                                    })
                                    .collect::<PyResult<_>>()?;
                                Ok(Box::new(result.into_iter()))
                            }
                        }
                    }
                    SliceOrList::List(_) => {
                        let result: Vec<$nativebit> = key
                            .iter_with_size(self.0.len())?
                            .map(|idx| -> PyResult<$nativebit> {
                                self.0
                                    .get(idx)
                                    .ok_or(CircuitError::new_err("register index out of range"))
                            })
                            .collect::<PyResult<_>>()?;
                        Ok(Box::new(result.into_iter()))
                    }
                }
            }
        }
    };
}

create_py_register! {
    PyQuantumRegister,
    QuantumRegister,
    PyQubit,
    ShareableQubit,
    "QuantumRegister",
    "qiskit.circuit",
    BitExtraInfo::Qubit { is_ancilla: false },
    "q",
    "quantum"
}

impl PyQuantumRegister {
    fn new_ancilla(
        size: Option<Bound<PyAny>>,
        name: Option<Bound<PyAny>>,
        bits: Option<Bound<PyAny>>,
    ) -> PyResult<(Self, PyRegister)> {
        let name_parse = |name: Option<String>| -> String {
            if let Some(name) = name {
                name
            } else {
                format!(
                    "{}{}",
                    'a',
                    AREG_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                )
            }
        };
        let (size, name) = PyRegister::inner_parse_new(&size, &name, &bits)?;
        let register = if let Some(bits) = bits {
            let Ok(bits_set): PyResult<IndexSet<ShareableQubit>> = bits
                .try_iter()?
                .map(|bit| -> PyResult<ShareableQubit> { bit?.extract::<ShareableQubit>() })
                .collect()
            else {
                return Err(CircuitError::new_err(format!(
                    "Provided bits did not all match register type. bits={}",
                    bits.repr()?
                )));
            };
            if bits_set.len() != size as usize {
                return Err(CircuitError::new_err(
                    "Register bits must not be duplicated.",
                ));
            }
            QuantumRegister::new_ancilla_alias(Some(name_parse(name)), bits_set)
        } else {
            QuantumRegister::new_ancilla_owning(name, size)
        };
        let inner_reg = register.data().clone();
        Ok((Self(register), PyRegister(inner_reg)))
    }
}

create_py_register! {
    PyClassicalRegister,
    ClassicalRegister,
    PyClbit,
    ShareableClbit,
    "ClassicalRegister",
    "qiskit.circuit",
    BitExtraInfo::Clbit(),
    "c",
    "classical"
}

static AREG_COUNTER: AtomicU32 = AtomicU32::new(0);

/// Implement an ancilla register.
#[pyclass(
    name = "AncillaRegister",
    module = "qiskit.circuit",
    frozen,
    extends=PyQuantumRegister
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PyAncillaRegister(pub(crate) PyQuantumRegister);

#[pymethods]
impl PyAncillaRegister {
    #[new]
    #[pyo3(signature = (size = None, name = None, bits = None))]
    pub fn new(
        size: Option<Bound<PyAny>>,
        name: Option<Bound<PyAny>>,
        bits: Option<Bound<PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let (reg, base) = PyQuantumRegister::new_ancilla(size, name, bits)?;
        Ok(PyClassInitializer::from(base)
            .add_subclass(reg.clone())
            .add_subclass(Self(reg)))
    }

    #[classattr]
    fn prefix<'py>() -> &'py str {
        "a"
    }

    #[classattr]
    fn bit_type(py: Python) -> Bound<PyType> {
        PyAncillaQubit::type_object(py)
    }

    #[classattr]
    fn instances_count() -> u32 {
        AREG_COUNTER.load(std::sync::atomic::Ordering::Relaxed)
    }
}
