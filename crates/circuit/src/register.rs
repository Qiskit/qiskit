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

use crate::{
    bit::{
        BitExtraInfo, BitInfo, PyAncillaQubit, PyBit, PyClbit, PyQubit, ShareableClbit,
        ShareableQubit,
    },
    circuit_data::CircuitError,
    slice::{PySequenceIndex, PySequenceIndexError},
};
use indexmap::IndexSet;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyList, PyString, PyTuple, PyType},
    IntoPyObjectExt, PyTypeInfo,
};
use std::{
    cmp::Eq,
    fmt::Display,
    hash::Hash,
    sync::{atomic::AtomicU32, Arc},
};

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
            pub fn new_owning(
                name: Option<String>,
                size: u32,
                extra: Option<BitExtraInfo>,
            ) -> Self {
                // When creating `Owning` register, we don't need to create the `BitInfo`
                // instances, they can be entirely derived from `self`.
                Self(RegisterInfo::new_owning(Self::name_parse(name), size, extra).into())
            }

            /// Creates a Register whose bits already exist.
            pub fn new_alias(
                name: Option<String>,
                bits: Box<IndexSet<BitInfo>>,
                extra: Option<BitExtraInfo>,
            ) -> Self {
                Self(RegisterInfo::new_alias(Self::name_parse(name), bits, extra).into())
            }

            /// A reference to the register's name
            pub fn name(&self) -> &str {
                self.0.name()
            }

            /// Returns the size of the register.
            pub fn len(&self) -> usize {
                self.0.len()
            }

            /// Returns whether the register is empty.
            pub fn is_empty(&self) -> bool {
                self.0.is_empty()
            }

            /// Returns an iterator over the bits within the circuit
            pub fn bits(&self) -> impl ExactSizeIterator<Item = $bit> + '_ {
                self.0.bits().map(|bit| $bit(bit))
            }

            /// Checks if a bit is contained within the register
            pub fn contains(&self, bit: &$bit) -> bool {
                self.0.contains(&bit.0)
            }

            /// Gets a bit via index, return None if not present
            pub fn get(&self, index: usize) -> Option<$bit> {
                self.0.get(index).map(|bit| $bit(bit))
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

impl<'py> FromPyObject<'py> for QuantumRegister {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(ob.downcast::<PyQuantumRegister>()?.borrow().0.clone())
    }
}

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

impl<'py> FromPyObject<'py> for ClassicalRegister {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(ob.downcast::<PyClassicalRegister>()?.borrow().0.clone())
    }
}

static REG_COUNTER: AtomicU32 = AtomicU32::new(0);
#[pyclass(
    name = "Register",
    module = "qiskit.circuit.register",
    subclass,
    frozen,
    sequence
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PyRegister(pub(crate) Arc<RegisterInfo>);

#[pymethods]
impl PyRegister {
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

    /// Visual representation of a Register
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
    fn __getitem__<'py>(slf: PyRef<'py, Self>, key: SliceOrInt<'py>) -> PyResult<PyObject> {
        let py = slf.py();
        match &key {
            SliceOrInt::Slice(py_sequence_index) => {
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

    fn __hash__(slf: Bound<'_, Self>) -> PyResult<isize> {
        let borrowed = slf.borrow();
        (slf.get_type(), borrowed.name(), borrowed.size())
            .into_bound_py_any(slf.py())?
            .hash()
    }

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

    fn __len__(slf: PyRef<'_, Self>) -> usize {
        slf.size()
    }

    #[classattr]
    fn prefix<'py>() -> &'py str {
        "reg"
    }

    #[classattr]
    fn bit_type(py: Python) -> Bound<PyType> {
        PyBit::type_object(py)
    }

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

    /// Inner function for [PyRegister::__getnewargs__] to ensure serialization can be
    /// preserved between register types.
    fn getitem_inner(
        &self,
        key: SliceOrInt<'_>,
    ) -> PyResult<Box<dyn ExactSizeIterator<Item = BitInfo>>> {
        match &key {
            SliceOrInt::Slice(py_sequence_index) => {
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
            SliceOrInt::List(_) => {
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
enum SliceOrInt<'py> {
    Slice(PySequenceIndex<'py>),
    List(Vec<isize>),
}

impl SliceOrInt<'_> {
    pub fn iter_with_size(&self, size: usize) -> PyResult<Box<dyn Iterator<Item = usize>>> {
        match self {
            SliceOrInt::Slice(py_sequence_index) => {
                let sequence = py_sequence_index.with_len(size);
                match sequence {
                    Ok(sequence) => Ok(Box::new(sequence.iter())),
                    Err(e) => Err(e.into()),
                }
            }
            SliceOrInt::List(items) => {
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
    ($name:ident, $nativereg:tt, $pybit:tt, $nativebit:tt, $pyname:literal, $pymodule:literal, $extra:expr, $prefix:literal) => {
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
                    let Ok(bits_set): PyResult<IndexSet<BitInfo>> = bits
                        .try_iter()?
                        .map(|bit| -> PyResult<BitInfo> {
                            bit?.extract::<$pybit>().map(|b| b.0 .0)
                        })
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
                    $nativereg::new_alias(name, bits_set.into(), Some($extra))
                } else {
                    $nativereg::new_owning(name, size, Some($extra))
                };
                let inner_reg = register.data().clone();
                Ok((Self(register), PyRegister(inner_reg)))
            }

            fn __getitem__<'py>(slf: PyRef<'py, Self>, key: SliceOrInt<'py>) -> PyResult<PyObject> {
                let py = slf.py();
                match &key {
                    SliceOrInt::Slice(py_sequence_index) => {
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
                key: SliceOrInt<'_>,
            ) -> PyResult<Box<dyn ExactSizeIterator<Item = $nativebit>>> {
                match &key {
                    SliceOrInt::Slice(py_sequence_index) => {
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
                    SliceOrInt::List(_) => {
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
    "qiskit.circuit.quantumregister",
    BitExtraInfo::Qubit { is_ancilla: false },
    "q"
}

impl PyQuantumRegister {
    fn new_ancilla(
        size: Option<Bound<PyAny>>,
        name: Option<Bound<PyAny>>,
        bits: Option<Bound<PyList>>,
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
        if (size.is_none() && bits.is_none()) || (size.is_some() && bits.is_some()) {
            return Err(CircuitError::new_err(format!("Exactly one of the size or bits arguments can be provided. Provided size={:?} bits={:?}.", size, bits)));
        }

        let size: u32 = if let Some(bits) = bits.as_ref() {
            bits.len().try_into().map_err(|_| CircuitError::new_err(format!("The amount of bits provided exceeds the capacity of the register. Current size {}", bits.len())))?
        } else {
            let Ok(valid_size): PyResult<isize> = size.as_ref().unwrap().extract() else {
                return Err(CircuitError::new_err(format!(
                    "Register size must be an integer. {} '{}' was provided",
                    size.as_ref().unwrap().get_type().name()?,
                    size.as_ref().unwrap().repr()?
                )));
            };
            if valid_size < 0 {
                return Err(CircuitError::new_err(format!(
                    "Register size must be non-negative. {} '{}' was provided",
                    size.as_ref().unwrap().get_type().name()?,
                    size.as_ref().unwrap().repr()?
                )));
            }

            let Ok(valid_size) = valid_size.abs().try_into() else {
                return Err(CircuitError::new_err(format!(
                    "Register size exceeds possible allocated capacity. {} '{}' was provided",
                    size.as_ref().unwrap().get_type().name()?,
                    size.as_ref().unwrap().repr()?
                )));
            };

            valid_size
        };

        let Ok(name): PyResult<Option<String>> =
            name.as_ref().map(|name| name.extract()).transpose()
        else {
            return Err(CircuitError::new_err("The circuit name should be castable to a string (or None for autogenerate a name)."));
        };
        let register = if let Some(bits) = bits {
            let Ok(bits_set): PyResult<IndexSet<BitInfo>> = bits
                .try_iter()?
                .map(|bit| -> PyResult<BitInfo> { bit?.extract::<ShareableQubit>().map(|b| b.0) })
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
            QuantumRegister::new_alias(
                Some(name_parse(name)),
                bits_set.into(),
                Some(BitExtraInfo::Qubit { is_ancilla: true }),
            )
        } else {
            QuantumRegister::new_owning(name, size, Some(BitExtraInfo::Qubit { is_ancilla: true }))
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
    "qiskit.circuit.classicalregister",
    BitExtraInfo::Clbit(),
    "c"
}

static AREG_COUNTER: AtomicU32 = AtomicU32::new(0);

#[pyclass(
    name = "AncillaRegister",
    module = "qiskit.circuit.quantumregister",
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
        bits: Option<Bound<PyList>>,
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
