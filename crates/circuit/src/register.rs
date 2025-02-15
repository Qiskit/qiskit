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
        BitInfo, PyClbit, PyQubit, QubitExtraInfo, ShareableBit, ShareableClbit, ShareableQubit,
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
pub(crate) enum RegisterInfo<T: ShareableBit + Hash + Eq + PartialOrd> {
    Owning(Arc<OwningRegisterInfo<T>>),
    Alias {
        name: String,
        bits: Box<IndexSet<BitInfo<T>>>,
        extra: <T as ShareableBit>::ExtraAttributes,
    },
}

impl<T: ShareableBit + Hash + Eq + PartialOrd> Hash for RegisterInfo<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            RegisterInfo::Owning(owning_register_info) => owning_register_info.hash(state),
            RegisterInfo::Alias { name, bits, extra } => {
                (core::mem::discriminant(self), name, bits.len(), extra).hash(state)
            }
        }
    }
}

impl<T: ShareableBit + Clone + Hash + Eq + PartialOrd> RegisterInfo<T> {
    /// Creates a Register whose bits are owned by its instance
    pub fn new_owning(name: String, size: u32, extra: T::ExtraAttributes) -> Self {
        // When creating `Owning` register, we don't need to create the `BitInfo`
        // instances, they can be entirely derived from `self`.
        Self::Owning(Arc::new(OwningRegisterInfo { name, size, extra }))
    }

    /// Creates a Register whose bits already exist.
    pub fn new_alias(
        name: String,
        bits: Box<IndexSet<BitInfo<T>>>,
        extra: T::ExtraAttributes,
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
            RegisterInfo::Alias {
                name: _,
                bits,
                extra: _,
            } => bits.len(),
        }
    }

    /// Returns an iterator over the bits within the circuit
    pub fn bits(&self) -> Box<dyn ExactSizeIterator<Item = BitInfo<T>> + '_> {
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
    pub fn contains(&self, bit: &BitInfo<T>) -> bool {
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
}

/// Contains the informaion for a register that owns the bits it contains.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub struct OwningRegisterInfo<T: ShareableBit> {
    name: String,
    size: u32,
    extra: <T as ShareableBit>::ExtraAttributes,
}

impl<T: ShareableBit> OwningRegisterInfo<T> {
    /// A reference to the register's name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the size of the register.
    pub fn len(&self) -> usize {
        self.size as usize
    }
}

impl OwningRegisterInfo<ShareableQubit> {
    /// Checks if the register contains ancilla qubits.
    pub fn is_ancilla(&self) -> bool {
        self.extra.is_ancilla()
    }
}

impl Display for OwningRegisterInfo<ShareableQubit> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let identifier = match self.is_ancilla() {
            true => "AncillaRegister",
            false => "QuantumRegister",
        };
        write!(f, "{}({}, {})", identifier, self.name(), self.len())
    }
}

impl Display for OwningRegisterInfo<ShareableClbit> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ClassicalRegister({}, {})", self.name(), self.len())
    }
}

impl<'py> IntoPyObject<'py> for RegisterInfo<ShareableQubit> {
    type Target = PyQuantumRegister;

    type Output = Bound<'py, PyQuantumRegister>;

    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        // TODO: Implement conversion to ancilla qubit
        PyQuantumRegister(self).into_pyobject(py)
    }
}

impl<'py> IntoPyObject<'py> for RegisterInfo<ShareableClbit> {
    type Target = PyClassicalRegister;

    type Output = Bound<'py, PyClassicalRegister>;

    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        // TODO: Implement conversion to ancilla qubit
        PyClassicalRegister(self).into_pyobject(py)
    }
}

impl Display for RegisterInfo<ShareableQubit> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegisterInfo::Owning(owning_register_info) => write!(f, "{}", owning_register_info),
            RegisterInfo::Alias { name, bits, extra } => {
                let identifier = match extra.is_ancilla() {
                    true => "AncillaRegister",
                    false => "QuantumRegister",
                };
                write!(f, "{}({}, {})", identifier, name, bits.len())
            }
        }
    }
}

impl Display for RegisterInfo<ShareableClbit> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegisterInfo::Owning(owning_register_info) => write!(f, "{}", owning_register_info),
            RegisterInfo::Alias { .. } => {
                write!(f, "ClassicalRegister({}, {})", self.name(), self.len())
            }
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
    ($name:ident, $pybit:ty, $nativebit:ty, $pyname:literal, $pymodule:literal, $counter:ident, $extra:expr, $prefix: literal) => {
        static $counter: AtomicU32 = AtomicU32::new(0);

        #[pyclass(
            name = $pyname,
            module = $pymodule,
            frozen,
            eq,
            hash,
            subclass
        )]
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name(RegisterInfo<$nativebit>);

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (size = None, name = None, bits = None))]
            pub fn new(
                size: Option<u32>,
                name: Option<String>,
                bits: Option<Vec<$pybit>>,
            ) -> PyResult<Self> {
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

                if (size.is_none() && bits.is_none()) || (size.is_some() && bits.is_some()) {
                    return Err(CircuitError::new_err(format!("Exactly one of the size or bits arguments can be provided. Provided size={:?} bits={:?}.", size, bits)));
                }

                let size: u32 = if let Some(bits) = bits.as_ref() {
                    bits.len().try_into().map_err(|_| CircuitError::new_err(format!("The amount of bits provided exceeds the capacity of the register. Current size {}", bits.len())))?
                } else {
                    size.unwrap_or_default()
                };

                let register = if let Some(bits) = bits {
                    let bits_set: IndexSet<BitInfo<$nativebit>> =
                        bits.into_iter().map(|bit| bit.0).collect();
                    if bits_set.len() != size as usize {
                        return Err(CircuitError::new_err(
                            "Register bits must not be duplicated.",
                        ));
                    }
                    RegisterInfo::new_alias(
                        name_parse(name),
                        bits_set.into(),
                        $extra,
                    )
                } else {
                    RegisterInfo::new_owning(name_parse(name), size, $extra)
                };

                Ok(Self(register))
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

            fn __repr__(&self) -> String {
                self.0.to_string()
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
                match key {
                    SliceOrInt::Slice(py_sequence_index) => {
                        let sequence = py_sequence_index.with_len(slf.size().try_into().unwrap())?;
                        match sequence {
                            crate::slice::SequenceIndex::Int(idx) => match &slf.0 {
                                RegisterInfo::Owning(owning_register_info) => {
                                    if slf.size() < idx {
                                        return Err(CircuitError::new_err("register index out of range"));
                                    }
                                    Ok(<$pybit>::new_owned(owning_register_info.clone(), idx as u32)
                                        .into_py_any(py)?)
                                }
                                RegisterInfo::Alias { bits, .. } => bits
                                    .get_index(idx)
                                    .cloned()
                                    .ok_or(CircuitError::new_err("register index out of range"))?
                                    .into_py_any(py),
                            },
                            _ => match &slf.0 {
                                RegisterInfo::Owning(owning_register_info) => {
                                    let result: Vec<$pybit> = sequence
                                        .iter()
                                        .map(|idx| -> PyResult<$pybit> {
                                            if idx < slf.size() {
                                                Ok(<$pybit>::new_owned(
                                                    owning_register_info.clone(),
                                                    idx as u32,
                                                ))
                                            } else {
                                                Err(CircuitError::new_err("register index out of range"))
                                            }
                                        })
                                        .collect::<PyResult<_>>()?;
                                    result.into_py_any(py)
                                }
                                RegisterInfo::Alias { bits, .. } => {
                                    let result: Vec<$pybit> = sequence
                                        .iter()
                                        .map(|idx| -> PyResult<$pybit> {
                                            bits.get_index(idx)
                                                .map(|bit| bit.clone().into())
                                                .ok_or(CircuitError::new_err("register index out of range"))
                                        })
                                        .collect::<PyResult<_>>()?;
                                    result.into_py_any(py)
                                }
                            },
                        }
                    }
                    SliceOrInt::List(vec) => match &slf.0 {
                        RegisterInfo::Owning(owning_register_info) => {
                            let result: Vec<$pybit> = vec
                                .iter()
                                .map(|idx| -> PyResult<$pybit> {
                                    if idx < &slf.size() {
                                        Ok(<$pybit>::new_owned(
                                            owning_register_info.clone(),
                                            *idx as u32,
                                        ))
                                    } else {
                                        Err(CircuitError::new_err("register index out of range"))
                                    }
                                })
                                .collect::<PyResult<_>>()?;
                            result.into_py_any(py)
                        }
                        RegisterInfo::Alias { bits, .. } => {
                            let result: Vec<$pybit> = vec
                                .iter()
                                .copied()
                                .map(|idx| -> PyResult<$pybit> {
                                    bits.get_index(idx)
                                        .map(|bit| bit.clone().into())
                                        .ok_or(CircuitError::new_err("register index out of range"))
                                })
                                .collect::<PyResult<_>>()?;
                            result.into_py_any(py)
                        }
                    },
                }
            }

            fn __contains__<'py>(slf: PyRef<'py, Self>, bit: &$pybit) -> bool {
                slf.0.contains(&bit.0)
            }

            /// Find the index of the provided bit within this register.
            fn index<'py>(slf: PyRef<'py, Self>, bit: Bound<$pybit>) -> PyResult<usize> {
                let err = || -> PyErr {
                    PyValueError::new_err(format!(
                        "Bit {} not found in Register {}.",
                        bit,
                        slf.__repr__()
                    ))
                };
                let bit_borrowed = bit.get();
                let bit_inner = &bit_borrowed.0;
                match bit_inner {
                    BitInfo::Owned { index, .. } => {
                        if slf.0.contains(bit_inner) {
                            Ok((*index) as usize)
                        } else {
                            Err(err())
                        }
                    }
                    BitInfo::Anonymous { .. } => match &slf.0 {
                        RegisterInfo::Owning(..) => Err(err()),
                        RegisterInfo::Alias { bits, .. } => bits.get_index_of(bit_inner).ok_or(err()),
                    },
                }
            }

            fn __getnewargs__<'py>(
                slf: PyRef<'py, Self>,
            ) -> PyResult<(Option<usize>, String, Option<Bound<'py, PyList>>)> {
                match &slf.0 {
                    RegisterInfo::Owning(..) => Ok((Some(slf.0.len()), slf.name().to_string(), None)),
                    RegisterInfo::Alias { bits, .. } => Ok((
                        None,
                        slf.name().to_string(),
                        Some(PyList::new(slf.py(), bits.iter().cloned())?),
                    )),
                }
            }

            fn __iter__<'py>(slf: PyRef<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyIterator>> {
                match &slf.0 {
                    RegisterInfo::Owning(owning_register_info) => {
                        return PyList::new(
                            slf.py(),
                            (0..slf.size() as u32)
                                .map(|bit| <$pybit>::new_owned(owning_register_info.clone(), bit)),
                        )?
                        .try_iter()
                    }
                    RegisterInfo::Alias { bits, .. } => {
                        return PyList::new(slf.py(), bits.iter().cloned())?.try_iter()
                    }
                }
            }


        }
        impl From<RegisterInfo<$nativebit>> for $name {
            fn from(value: RegisterInfo<$nativebit>) -> Self {
                Self(value)
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
