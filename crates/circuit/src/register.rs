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
        BitInfo, PyAncillaQubit, PyBit, PyClbit, PyQubit, QubitExtraInfo, ShareableClbit,
        ShareableQubit,
    },
    circuit_data::CircuitError,
    slice::PySequenceIndex,
};
use indexmap::IndexSet;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyList, IntoPyObjectExt};
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
    },
}

// Custom implmentation of Hash to disregard the `IndexMap` used for Alias.
impl Hash for RegisterInfo {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            RegisterInfo::Owning(owning_register_info) => owning_register_info.hash(state),
            RegisterInfo::Alias { name, bits } => {
                (core::mem::discriminant(self), name, bits.len()).hash(state)
            }
        }
    }
}

impl RegisterInfo {
    /// Creates a Register whose bits are owned by its instance
    pub(crate) fn new_owning(name: String, size: u32) -> Self {
        // When creating `Owning` register, we don't need to create the `BitInfo`
        // instances, they can be entirely derived from `self`.
        Self::Owning(Arc::new(OwningRegisterInfo { name, size }))
    }

    /// Creates a Register whose bits already exist.
    pub(crate) fn new_alias(name: String, bits: Box<IndexSet<BitInfo>>) -> Self {
        Self::Alias { name, bits }
    }

    /// A reference to the register's name
    pub(crate) fn name(&self) -> &str {
        match self {
            RegisterInfo::Owning(owning_register_info) => owning_register_info.name.as_str(),
            RegisterInfo::Alias { name, .. } => name.as_str(),
        }
    }

    /// Returns the size of the register.
    pub(crate) fn len(&self) -> usize {
        match self {
            RegisterInfo::Owning(owning_register_info) => owning_register_info.size as usize,
            RegisterInfo::Alias { name: _, bits } => bits.len(),
        }
    }

    /// Returns an iterator over the bits within the circuit
    pub(crate) fn bits(&self) -> Box<dyn ExactSizeIterator<Item = BitInfo> + '_> {
        match self {
            RegisterInfo::Owning(owning_register_info) => {
                Box::new((0..owning_register_info.size).map(|bit| BitInfo::Owned {
                    register: owning_register_info.clone(),
                    index: bit,
                }))
            }
            RegisterInfo::Alias { bits, .. } => Box::new(bits.iter().cloned()),
        }
    }

    /// Checks if a bit is contained within the register
    pub(crate) fn contains(&self, bit: &BitInfo) -> bool {
        match self {
            RegisterInfo::Owning(owning_register_info) => match bit {
                BitInfo::Owned { register, index } => {
                    register == owning_register_info && *index < owning_register_info.size
                }
                BitInfo::Anonymous { .. } => false,
            },
            RegisterInfo::Alias { bits, .. } => bits.contains(bit),
        }
    }

    /// Owning register if any
    pub fn register(&self) -> Option<&Arc<OwningRegisterInfo>> {
        match self {
            RegisterInfo::Owning(reg) => Some(reg),
            RegisterInfo::Alias { .. } => None,
        }
    }
}

/// Contains the informaion for a register that owns the bits it contains.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub struct OwningRegisterInfo {
    name: String,
    size: u32,
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
}

macro_rules! create_register_object {
    ($name:ident, $bit:tt, $extra:ty, $extra_exp:expr) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name {
            pub(crate) data: Arc<RegisterInfo>,
            pub(crate) extra: $extra,
        }

        impl $name {
            /// Creates a Register whose bits are owned by its instance
            pub(crate) fn new_owning(name: String, size: u32, extra: $extra) -> Self {
                // When creating `Owning` register, we don't need to create the `BitInfo`
                // instances, they can be entirely derived from `self`.
                Self {
                    data: RegisterInfo::new_owning(name, size).into(),
                    extra,
                }
            }

            /// Creates a Register whose bits already exist.
            pub(crate) fn new_alias(
                name: String,
                bits: Box<IndexSet<BitInfo>>,
                extra: $extra,
            ) -> Self {
                Self {
                    data: RegisterInfo::new_alias(name, bits).into(),
                    extra,
                }
            }

            /// A reference to the register's name
            pub(crate) fn name(&self) -> &str {
                self.data.name()
            }

            /// Returns the size of the register.
            pub(crate) fn len(&self) -> usize {
                self.data.len()
            }

            /// Returns an iterator over the bits within the circuit
            pub(crate) fn bits(&self) -> Box<dyn ExactSizeIterator<Item = $bit> + '_> {
                match self.data.as_ref() {
                    RegisterInfo::Owning(owning_register_info) => {
                        Box::new((0..owning_register_info.size).map(|bit| {
                            $bit::new_owned(owning_register_info.clone(), bit, self.extra)
                        }))
                    }
                    RegisterInfo::Alias { bits, .. } => {
                        Box::new(bits.iter().cloned().map(|bit| $bit {
                            info: bit,
                            extra: self.extra,
                        }))
                    }
                }
            }

            /// Checks if a bit is contained within the register
            pub(crate) fn contains(&self, bit: &$bit) -> bool {
                self.data.contains(&bit.info)
            }
        }
    };
}

create_register_object! {QuantumRegister, ShareableQubit, QubitExtraInfo, QubitExtraInfo::new(false)}
impl QuantumRegister {
    /// Check if the [QuantumRegister] instance is ancillary.
    pub fn is_ancilla(&self) -> bool {
        self.extra.is_ancilla()
    }
}
create_register_object! {ClassicalRegister, ShareableClbit, (), ()}

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
        let reg = PyRegister(self.data);
        let inner = PyQuantumRegister(reg.clone());
        match is_ancilla {
            true => {
                let initializer = PyClassInitializer::from(reg)
                    .add_subclass(inner.clone())
                    .add_subclass(PyAncillaRegister(inner));
                Ok(Bound::new(py, initializer)?.into_super())
            }
            false => Bound::new(py, (inner, reg)),
        }
    }
}

impl<'py> IntoPyObject<'py> for ClassicalRegister {
    type Target = PyClassicalRegister;

    type Output = Bound<'py, PyClassicalRegister>;

    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let base = PyRegister(self.data);
        Bound::new(py, (PyClassicalRegister(base.clone()), base))
    }
}

static REG_COUNTER: AtomicU32 = AtomicU32::new(0);
#[pyclass(
    name = "Register",
    module = "qiskit.circuit.register",
    frozen,
    eq,
    hash,
    subclass
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PyRegister(pub(crate) Arc<RegisterInfo>);

#[pymethods]
impl PyRegister {
    #[new]
    #[pyo3(signature = (size = None, name = None, bits = None))]
    pub fn new(
        size: Option<u32>,
        name: Option<String>,
        bits: Option<Vec<PyBit>>,
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

        if (size.is_none() && bits.is_none()) || (size.is_some() && bits.is_some()) {
            return Err(CircuitError::new_err(format!("Exactly one of the size or bits arguments can be provided. Provided size={:?} bits={:?}.", size, bits)));
        }

        let size: u32 = if let Some(bits) = bits.as_ref() {
            bits.len().try_into().map_err(|_| CircuitError::new_err(format!("The amount of bits provided exceeds the capacity of the register. Current size {}", bits.len())))?
        } else {
            size.unwrap_or_default()
        };

        let register = if let Some(bits) = bits {
            let bits_set: IndexSet<BitInfo> = bits.into_iter().map(|bit| bit.0).collect();
            if bits_set.len() != size as usize {
                return Err(CircuitError::new_err(
                    "Register bits must not be duplicated.",
                ));
            }
            RegisterInfo::new_alias(name_parse(name), bits_set.into())
        } else {
            RegisterInfo::new_owning(name_parse(name), size)
        };

        Ok(Self(register.into()))
    }

    pub fn __repr__(slf: Bound<Self>) -> PyResult<String> {
        let borrowed = slf.borrow();
        match borrowed.0.as_ref() {
            RegisterInfo::Owning(owning_register_info) => Ok(format!(
                "{}({}, {})",
                slf.get_type().qualname()?,
                owning_register_info.len(),
                owning_register_info.name()
            )),
            RegisterInfo::Alias { name, bits } => Ok(format!(
                "{}({}, {})",
                slf.get_type().qualname()?,
                bits.len(),
                name,
            )),
        }
    }

    fn __contains__<'py>(&self, bit: &PyBit) -> bool {
        self.0.contains(&bit.0)
    }

    /// Get the register name
    #[getter]
    pub fn name(&self) -> &str {
        self.0.name()
    }

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

    /// Raises:
    ///     CircuitError: if the `key` is not an integer or not in the range `(0, self.size)`.
    fn __getitem__<'py>(slf: PyRef<'py, Self>, key: SliceOrInt<'py>) -> PyResult<PyObject> {
        let py = slf.py();
        match &key {
            SliceOrInt::Slice(py_sequence_index) => {
                let sequence = py_sequence_index.with_len(slf.size().try_into().unwrap())?;
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
        let slf_borrowed = slf.get();
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

    fn __getnewargs__(
        slf: PyRef<'_, Self>,
    ) -> PyResult<(Option<usize>, String, Option<Bound<'_, PyList>>)> {
        let (size, name, bits) = slf.reduce();
        Ok((
            size,
            name,
            bits.map(|elements| PyList::new(slf.py(), elements))
                .transpose()?,
        ))
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Bound<'_, pyo3::types::PyIterator>> {
        return PyList::new(slf.py(), slf.iter().map(PyBit))?.try_iter();
    }

    #[classattr]
    fn prefix<'py>() -> &'py str {
        "reg"
    }
}

impl PyRegister {
    fn getitem_inner(
        &self,
        key: SliceOrInt<'_>,
    ) -> PyResult<Box<dyn ExactSizeIterator<Item = BitInfo>>> {
        match key {
            SliceOrInt::Slice(py_sequence_index) => {
                let sequence = py_sequence_index.with_len(self.size())?;
                match sequence {
                    crate::slice::SequenceIndex::Int(idx) => match self.0.as_ref() {
                        RegisterInfo::Owning(owning_register_info) => {
                            if self.size() < idx {
                                return Err(CircuitError::new_err("register index out of range"));
                            }
                            Ok(Box::new(std::iter::once(BitInfo::new_owned(
                                owning_register_info.clone(),
                                idx as u32,
                            ))))
                        }
                        RegisterInfo::Alias { bits, .. } => Ok(Box::new(std::iter::once(
                            bits.get_index(idx)
                                .cloned()
                                .ok_or(CircuitError::new_err("register index out of range"))?,
                        ))),
                    },
                    _ => match self.0.as_ref() {
                        RegisterInfo::Owning(owning_register_info) => {
                            let result: Vec<BitInfo> = sequence
                                .iter()
                                .map(|idx| -> PyResult<BitInfo> {
                                    if idx < self.size() {
                                        Ok(BitInfo::new_owned(
                                            owning_register_info.clone(),
                                            idx as u32,
                                        ))
                                    } else {
                                        Err(CircuitError::new_err("register index out of range"))
                                    }
                                })
                                .collect::<PyResult<_>>()?;
                            Ok(Box::new(result.into_iter()))
                        }
                        RegisterInfo::Alias { bits, .. } => {
                            let result: Vec<BitInfo> = sequence
                                .iter()
                                .map(|idx| -> PyResult<BitInfo> {
                                    bits.get_index(idx)
                                        .cloned()
                                        .ok_or(CircuitError::new_err("register index out of range"))
                                })
                                .collect::<PyResult<_>>()?;
                            Ok(Box::new(result.into_iter()))
                        }
                    },
                }
            }
            SliceOrInt::List(vec) => match self.0.as_ref() {
                RegisterInfo::Owning(owning_register_info) => {
                    let result: Vec<BitInfo> = vec
                        .iter()
                        .map(|idx| -> PyResult<BitInfo> {
                            if idx < &self.size() {
                                Ok(BitInfo::new_owned(
                                    owning_register_info.clone(),
                                    *idx as u32,
                                ))
                            } else {
                                Err(CircuitError::new_err("register index out of range"))
                            }
                        })
                        .collect::<PyResult<_>>()?;
                    Ok(Box::new(result.into_iter()))
                }
                RegisterInfo::Alias { bits, .. } => {
                    let result: Vec<BitInfo> = vec
                        .iter()
                        .copied()
                        .map(|idx| -> PyResult<BitInfo> {
                            bits.get_index(idx)
                                .cloned()
                                .ok_or(CircuitError::new_err("register index out of range"))
                        })
                        .collect::<PyResult<_>>()?;
                    Ok(Box::new(result.into_iter()))
                }
            },
        }
    }

    fn reduce(
        &self,
    ) -> (
        Option<usize>,
        String,
        Option<impl ExactSizeIterator<Item = PyBit> + '_>,
    ) {
        match self.0.as_ref() {
            RegisterInfo::Owning(..) => (Some(self.0.len()), self.name().to_string(), None),
            RegisterInfo::Alias { bits, .. } => (
                None,
                self.name().to_string(),
                Some(bits.iter().cloned().map(PyBit)),
            ),
        }
    }

    fn iter<'py>(&'py self) -> Box<dyn ExactSizeIterator<Item = BitInfo> + 'py> {
        match self.0.as_ref() {
            RegisterInfo::Owning(owning_register_info) => Box::new(
                (0..self.size() as u32)
                    .map(|bit| BitInfo::new_owned(owning_register_info.clone(), bit)),
            ),
            RegisterInfo::Alias { bits, .. } => Box::new(bits.iter().cloned()),
        }
    }
}

/// Correctly extracts a slice or a vec from python.
#[derive(FromPyObject)]
enum SliceOrInt<'py> {
    Slice(PySequenceIndex<'py>),
    List(Vec<usize>),
}

macro_rules! create_py_register {
    ($name:ident, $pybit:tt, $nativebit:tt, $pyname:literal, $pymodule:literal, $counter:ident, $extra:expr, $prefix:literal) => {
        static $counter: AtomicU32 = AtomicU32::new(0);

        #[pyclass(
                                                                            name = $pyname,
                                                                            module = $pymodule,
                                                                            frozen,
                                                                            eq,
                                                                            hash,
                                                                            subclass,
                                                                            extends = PyRegister,
                                                                        )]
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name(pub(crate) PyRegister);

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (size = None, name = None, bits = None))]
            pub fn new(
                size: Option<u32>,
                name: Option<String>,
                bits: Option<Vec<$pybit>>,
            ) -> PyResult<(Self, PyRegister)> {
                let name_parse = |name: Option<String>| -> String {
                    if let Some(name) = name {
                        name
                    } else {
                        format!(
                            "{}{}",
                            $prefix,
                            $counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                        )
                    }
                };

                let reg = PyRegister::new(
                    size,
                    Some(name_parse(name)),
                    bits.map(|bits| bits.into_iter().map(|bit| bit.0).collect()),
                )?;
                Ok((Self(reg.clone()), reg))
            }

            fn __getitem__<'py>(slf: PyRef<'py, Self>, key: SliceOrInt<'py>) -> PyResult<PyObject> {
                let py = slf.py();
                match &key {
                    SliceOrInt::Slice(py_sequence_index) => {
                        let sequence = py_sequence_index
                            .with_len(slf.as_super().size().try_into().unwrap())?;
                        match sequence {
                            crate::slice::SequenceIndex::Int(_) => slf
                                .as_super()
                                .getitem_inner(key)?
                                .next()
                                .map(|bit| $nativebit {
                                    info: bit,
                                    extra: $extra,
                                })
                                .into_py_any(py),
                            _ => Ok(PyList::new(
                                py,
                                slf.as_super().getitem_inner(key)?.map(|bit| $nativebit {
                                    info: bit,
                                    extra: $extra,
                                }),
                            )?
                            .into_any()
                            .unbind()),
                        }
                    }
                    _ => Ok(PyList::new(
                        py,
                        slf.as_super().getitem_inner(key)?.map(|bit| $nativebit {
                            info: bit,
                            extra: $extra,
                        }),
                    )?
                    .into_any()
                    .unbind()),
                }
            }

            fn __contains__(&self, bit: &$pybit) -> bool {
                self.0.__contains__(&bit.0)
            }

            fn __getnewargs__(
                slf: PyRef<'_, Self>,
            ) -> PyResult<(Option<usize>, String, Option<Bound<'_, PyList>>)> {
                let py = slf.py();
                let (size, name, bits) = slf.as_super().reduce();
                let list = if let Some(bits) = bits {
                    let list = PyList::empty(py);
                    for bit in bits {
                        let bound_bit = $nativebit {
                            info: bit.0,
                            extra: $extra,
                        }
                        .into_pyobject(slf.py())?;
                        list.append(bound_bit)?;
                    }
                    Some(list)
                } else {
                    None
                };
                Ok((size, name, list))
            }

            fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Bound<'_, pyo3::types::PyIterator>> {
                let list: Vec<$nativebit> = slf
                    .as_super()
                    .iter()
                    .map(|bit| $nativebit {
                        info: bit,
                        extra: $extra,
                    })
                    .collect();
                list.into_pyobject(slf.py())?.try_iter()
            }

            fn __len__(slf: PyRef<'_, Self>) -> usize {
                slf.as_super().size()
            }

            #[classattr]
            fn prefix<'py>() -> &'py str {
                $prefix
            }
        }
    };
}

create_py_register! {
    PyQuantumRegister,
    PyQubit,
    ShareableQubit,
    "QuantumRegister",
    "qiskit.circuit.quantumregister",
    QREG_COUNTER,
    QubitExtraInfo::new(false),
    "q"
}

create_py_register! {
    PyClassicalRegister,
    PyClbit,
    ShareableClbit,
    "ClassicalRegister",
    "qiskit.circuit.classicalregister",
    CREG_COUNTER,
    (),
    "c"
}

static AREG_COUNTER: AtomicU32 = AtomicU32::new(0);

#[pyclass(
    name = "AncillaRegister",
    module = "qiskit.circuit.quantumregister",
    frozen,
    eq,
    hash,
    extends=PyQuantumRegister
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PyAncillaRegister(pub(crate) PyQuantumRegister);

#[pymethods]
impl PyAncillaRegister {
    #[new]
    #[pyo3(signature = (size = None, name = None, bits = None))]
    pub fn new(
        size: Option<u32>,
        name: Option<String>,
        bits: Option<Vec<PyAncillaQubit>>,
    ) -> PyResult<PyClassInitializer<Self>> {
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
        let (reg, base) = PyQuantumRegister::new(
            size,
            Some(name_parse(name)),
            bits.map(|bits| bits.into_iter().map(|bit| bit.0).collect()),
        )?;
        Ok(PyClassInitializer::from(base)
            .add_subclass(reg.clone())
            .add_subclass(Self(reg)))
    }

    fn __getitem__<'py>(slf: PyRef<'py, Self>, key: SliceOrInt<'py>) -> PyResult<PyObject> {
        let py = slf.py();
        match &key {
            SliceOrInt::Slice(py_sequence_index) => {
                let sequence = py_sequence_index
                    .with_len(slf.as_super().as_super().size().try_into().unwrap())?;
                match sequence {
                    crate::slice::SequenceIndex::Int(_) => slf
                        .as_super()
                        .as_super()
                        .getitem_inner(key)?
                        .next()
                        .map(|bit| ShareableQubit {
                            info: bit,
                            extra: QubitExtraInfo::new(true),
                        })
                        .into_py_any(py),
                    _ => Ok(PyList::new(
                        py,
                        slf.as_super()
                            .as_super()
                            .getitem_inner(key)?
                            .map(|bit| ShareableQubit {
                                info: bit,
                                extra: QubitExtraInfo::new(true),
                            }),
                    )?
                    .into_any()
                    .unbind()),
                }
            }
            _ => Ok(PyList::new(
                py,
                slf.as_super()
                    .as_super()
                    .getitem_inner(key)?
                    .map(|bit| ShareableQubit {
                        info: bit,
                        extra: QubitExtraInfo::new(true),
                    }),
            )?
            .into_any()
            .unbind()),
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Bound<'_, pyo3::types::PyIterator>> {
        let list: Vec<ShareableQubit> = slf
            .as_super()
            .as_super()
            .iter()
            .map(|bit| ShareableQubit {
                info: bit,
                extra: QubitExtraInfo::new(true),
            })
            .collect();
        list.into_pyobject(slf.py())?.try_iter()
    }

    fn __getnewargs__(
        slf: PyRef<'_, Self>,
    ) -> PyResult<(Option<usize>, String, Option<Bound<'_, PyList>>)> {
        let py = slf.py();
        let (size, name, bits) = slf.as_super().as_super().reduce();
        let list = if let Some(bits) = bits {
            let list = PyList::empty(py);
            for bit in bits {
                let bound_bit = ShareableQubit {
                    info: bit.0,
                    extra: QubitExtraInfo::new(true),
                }
                .into_pyobject(slf.py())?;
                list.append(bound_bit)?;
            }
            Some(list)
        } else {
            None
        };
        Ok((size, name, list))
    }

    #[classattr]
    fn prefix<'py>() -> &'py str {
        "a"
    }
}
