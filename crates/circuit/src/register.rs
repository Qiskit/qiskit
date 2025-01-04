use hashbrown::HashMap;
use indexmap::IndexSet;
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    intern,
    prelude::*,
    types::{PyAnyMethods, PyList, PySet},
    FromPyObject, PyTypeInfo,
};
use std::{
    hash::{Hash, Hasher},
    sync::{Mutex, OnceLock},
};

use crate::{
    bit::{PyBit, PyClbit, PyQubit},
    circuit_data::CircuitError,
    interner::{Interned, Interner},
    slice::PySequenceIndex,
    Clbit, Qubit,
};

static REGISTER_INSTANCE_COUNTER: Mutex<u32> = Mutex::new(0);

/// This represents the hash value of a Register according to the register's
/// name and number of qubits.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RegisterAsKey {
    Register((String, u32)),
    Quantum((String, u32)),
    Classical((String, u32)),
}

impl RegisterAsKey {
    #[inline]
    pub fn reduce(&self) -> (&str, u32) {
        match self {
            RegisterAsKey::Register((name, num_qubits)) => (name.as_str(), *num_qubits),
            RegisterAsKey::Quantum((name, num_qubits)) => (name.as_str(), *num_qubits),
            RegisterAsKey::Classical((name, num_qubits)) => (name.as_str(), *num_qubits),
        }
    }

    #[inline]
    pub fn name(&self) -> &str {
        match self {
            RegisterAsKey::Register((name, _)) => name.as_str(),
            RegisterAsKey::Quantum((name, _)) => name.as_str(),
            RegisterAsKey::Classical((name, _)) => name.as_str(),
        }
    }

    #[inline]
    pub fn index(&self) -> u32 {
        match self {
            RegisterAsKey::Register((_, idx)) => *idx,
            RegisterAsKey::Quantum((_, idx)) => *idx,
            RegisterAsKey::Classical((_, idx)) => *idx,
        }
    }

    #[inline]
    pub fn type_identifier(&self) -> &str {
        match self {
            RegisterAsKey::Register(_) => "Register",
            RegisterAsKey::Quantum(_) => "QuantumRegister",
            RegisterAsKey::Classical(_) => "ClassicalRegister",
        }
    }
}

impl<'py> FromPyObject<'py> for RegisterAsKey {
    fn extract_bound(ob: &pyo3::Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        if ob.is_instance(&PyRegister::type_object_bound(ob.py()))? {
            let (name, num_qubits) = (
                ob.getattr(intern!(ob.py(), "name"))?.extract()?,
                ob.len()? as u32,
            );
            if ob.downcast::<PyClassicalRegister>().is_ok() {
                return Ok(RegisterAsKey::Classical((name, num_qubits)));
            } else if ob.downcast::<PyQuantumRegister>().is_ok() {
                return Ok(RegisterAsKey::Quantum((name, num_qubits)));
            } else {
                return Ok(RegisterAsKey::Register((name, num_qubits)));
            }
        }
        Err(PyTypeError::new_err(
            "The provided argument was not a register.",
        ))
    }
}
/// Described the desired behavior of a Register.
pub trait Register {
    /// The type of bit stored by the [Register]
    type Bit;

    /// Returns the size of the [Register].
    fn len(&self) -> usize;
    /// Checks if a bit exists within the [Register].
    fn contains(&self, bit: Self::Bit) -> bool;
    /// Finds the local index of a certain bit within [Register].
    fn find_index(&self, bit: Self::Bit) -> Option<u32>;
    /// Return an iterator over all the bits in the register
    fn bits(&self) -> impl ExactSizeIterator<Item = Self::Bit>;
}

macro_rules! create_register {
    ($name:ident, $bit:ty, $counter:ident, $prefix:literal) => {
        static $counter: Mutex<u32> = Mutex::new(0);

        #[derive(Debug, Clone, Eq)]
        pub struct $name {
            register: IndexSet<<$name as Register>::Bit>,
            name: String,
        }

        impl $name {
            pub fn new(size: usize, name: Option<String>) -> Self {
                let name = if let Some(name) = name {
                    name
                } else {
                    let count = if let Ok(ref mut count) = $counter.try_lock() {
                        let curr = **count;
                        **count += 1;
                        curr
                    } else {
                        panic!("Could not access register counter.")
                    };
                    format!("{}{}", $prefix, count)
                };
                Self {
                    register: (0..size).map(|bit| <$bit>::new(bit)).collect(),
                    name,
                }
            }
        }

        impl Register for $name {
            type Bit = $bit;

            fn len(&self) -> usize {
                self.register.len()
            }

            fn contains(&self, bit: Self::Bit) -> bool {
                self.register.contains(&bit)
            }

            fn find_index(&self, bit: Self::Bit) -> Option<u32> {
                self.register.get_index_of(&bit).map(|idx| idx as u32)
            }

            fn bits(&self) -> impl ExactSizeIterator<Item = Self::Bit> {
                self.register.iter().copied()
            }
        }

        impl Hash for $name {
            fn hash<H: Hasher>(&self, state: &mut H) {
                (self.name.as_str(), self.len()).hash(state);
            }
        }

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.register.len() == other.register.len() && self.name == other.name
            }
        }
    };
}

create_register!(QuantumRegister, Qubit, QREG_COUNTER, "qr");
create_register!(ClassicalRegister, Clbit, CREG_COUNTER, "cr");

/// Represents a collection of registers of a certain type within a circuit.
#[derive(Debug, Clone)]
pub(crate) struct CircuitRegistry<T: Register + Clone, P> {
    registry: Interner<T>,
    /// Python cache for registers
    python_cache: HashMap<Interned<T>, OnceLock<Py<P>>>,
}

impl<T: Register + Hash + Eq + Clone, P> CircuitRegistry<T, P> {
    pub fn add_register(&mut self, register: T) -> Interned<T> {
        self.registry.insert_owned(register)
    }

    /// Retreives the index of a register if it exists within a registry.
    pub fn find_index(&self, register: &T) -> Option<Interned<T>> {
        self.registry.get_interned(register)
    }

    /// Checks if a register exists within a circuit
    pub fn contains(&self, register: &T) -> bool {
        self.registry.contains(register)
    }
}

#[derive(FromPyObject)]
enum SliceOrInt<'py> {
    Slice(PySequenceIndex<'py>),
    List(Vec<usize>),
}

/// Python representation of a generic register
#[derive(Debug, Clone)]
#[pyclass(name = "Register", module = "qiskit.circuit.register", subclass)]
pub struct PyRegister {
    /// Bits are stored in Python-space.
    bits: Vec<Py<PyBit>>,
    /// Name of the register in question
    #[pyo3(get)]
    name: String,
    /// Size of the register
    #[pyo3(get)]
    size: u32,
    /// Mapping of the hash value of each bit and their index in the register.
    bit_indices: HashMap<isize, u32>,
}

#[pymethods]
impl PyRegister {
    #[new]
    pub fn new(
        py: Python,
        mut size: Option<u32>,
        mut name: Option<String>,
        bits: Option<Vec<Py<PyBit>>>,
    ) -> PyResult<Self> {
        if (size.is_none(), bits.is_none()) == (false, false) || (size.is_some() && bits.is_some())
        {
            return Err(
                CircuitError::new_err(
                    format!("Exactly one of the size or bits arguments can be provided. Provided size={:?} bits={:?}.", size, bits)
                )
            );
        }
        if let Some(bits) = bits.as_ref() {
            size = Some(bits.len() as u32);
        }
        if name.is_none() {
            let count = if let Ok(ref mut count) = REGISTER_INSTANCE_COUNTER.try_lock() {
                let curr = **count;
                **count += 1;
                curr
            } else {
                panic!("Could not access register counter.")
            };
            name = Some(format!("{}{}", "reg", count));
        }
        if let Some(bits) = bits {
            if size != Some(PySet::new_bound(py, bits.iter())?.len() as u32) {
                return Err(CircuitError::new_err(format!(
                    "Register bits must not be duplicated. bits={:?}",
                    bits
                )));
            }
            let bit_indices: HashMap<isize, u32> = bits
                .iter()
                .enumerate()
                .flat_map(|(idx, obj)| -> PyResult<(isize, u32)> {
                    Ok((obj.bind(py).hash()?, idx as u32))
                })
                .collect();
            Ok(Self {
                bits,
                name: name.unwrap(),
                size: size.unwrap(),
                bit_indices,
            })
        } else {
            let name = name.unwrap();
            let size = size.unwrap();
            let bits: Vec<Py<PyBit>> = (0..size)
                .map(|idx| {
                    Py::new(
                        py,
                        PyBit::new(
                            Some(RegisterAsKey::Register((name.clone(), size))),
                            Some(idx),
                        )
                        .unwrap(),
                    )
                    .unwrap()
                })
                .collect();
            let bit_indices: HashMap<isize, u32> = bits
                .iter()
                .enumerate()
                .flat_map(|(idx, obj)| -> PyResult<(isize, u32)> {
                    Ok((obj.bind(py).hash()?, idx as u32))
                })
                .collect();
            Ok(Self {
                bits,
                name,
                size,
                bit_indices,
            })
        }
    }

    fn __repr__(slf: Bound<Self>) -> PyResult<String> {
        let borrowed = slf.borrow();
        Ok(format!(
            "{}({}, '{}')",
            slf.get_type().name()?,
            borrowed.size,
            borrowed.name,
        ))
    }

    fn __len__(&self) -> usize {
        self.size as usize
    }

    fn __getitem__(&self, py: Python, key: SliceOrInt) -> PyResult<PyObject> {
        match key {
            SliceOrInt::Slice(py_sequence_index) => {
                let sequence = py_sequence_index.with_len(self.size.try_into().unwrap())?;
                match sequence {
                    crate::slice::SequenceIndex::Int(idx) => {
                        Ok(self.bits[idx].clone_ref(py).into_any())
                    }
                    _ => Ok(PyList::new_bound(
                        py,
                        sequence.iter().map(|idx| self.bits[idx].clone_ref(py)),
                    )
                    .into()),
                }
            }
            SliceOrInt::List(vec) => {
                if vec.iter().max() < Some(&(self.size as usize)) {
                    Ok(
                        PyList::new_bound(py, vec.iter().map(|idx| self.bits[*idx].clone_ref(py)))
                            .into(),
                    )
                } else {
                    Err(CircuitError::new_err("Register index is our of range"))
                }
            }
        }
    }

    fn __contains__(&self, bit: &Bound<PyBit>) -> PyResult<bool> {
        Ok(self.bit_indices.contains_key(&bit.hash()?))
    }

    fn index(slf: Bound<Self>, bit: &Bound<PyBit>) -> PyResult<u32> {
        let borrowed = slf.borrow();
        if borrowed.__contains__(bit)? {
            Ok(borrowed.bit_indices[&bit.hash()?])
        } else {
            Err(PyValueError::new_err(format!(
                "Bit {} not found in Register {}.",
                bit.repr()?,
                slf.repr()?,
            )))
        }
    }

    fn __iter__<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyIterator>> {
        PyList::new_bound(py, self.bits.iter().map(|obj| obj.clone_ref(py)))
            .into_any()
            .iter()
    }

    fn __getnewargs__(&self, py: Python) -> (Option<u32>, String, PyObject) {
        (
            None,
            self.name.clone(),
            self.bits
                .iter()
                .map(|bit| bit.clone_ref(py))
                .collect::<Vec<_>>()
                .into_py(py),
        )
    }

    fn __getstate__(&self, py: Python) -> (String, u32, PyObject) {
        (
            self.name.clone(),
            self.size,
            self.bits
                .iter()
                .map(|bit| bit.clone_ref(py))
                .collect::<Vec<_>>()
                .into_py(py),
        )
    }

    fn __setstate__(&mut self, py: Python, state: (String, u32, PyObject)) -> PyResult<()> {
        self.name = state.0;
        self.size = state.1;
        self.bits = state.2.extract(py)?;
        self.bit_indices = self
            .bits
            .iter()
            .enumerate()
            .flat_map(|(idx, obj)| -> PyResult<(isize, u32)> {
                Ok((obj.bind(py).hash()?, idx as u32))
            })
            .collect();
        Ok(())
    }

    fn __eq__(slf: Bound<Self>, other: Bound<Self>) -> PyResult<bool> {
        if slf.is(&other) {
            return Ok(true);
        }

        let self_borrowed = slf.borrow();
        let other_borrowed = other.borrow();

        Ok(slf.get_type().eq(other.get_type())?
            && slf.repr()?.to_string() == other.repr()?.to_string()
            && self_borrowed
                .bits
                .iter()
                .zip(other_borrowed.bits.iter())
                .filter_map(|(bit, other)| -> Option<bool> {
                    let borrowed_bit = bit.borrow(slf.py());
                    let borrowed_other = other.borrow(slf.py());
                    match (borrowed_bit.is_new(), borrowed_other.is_new()) {
                        (false, false) => None,
                        _ => Some(bit.is(other)),
                    }
                })
                .all(|bool| bool))
    }
}

#[derive(Debug, Clone)]
#[pyclass(name="QuantumRegister", module="qiskit.circuit.quantumregister", extends=PyRegister)]
pub struct PyQuantumRegister();

#[pymethods]
impl PyQuantumRegister {
    #[new]
    pub fn new(
        py: Python,
        mut size: Option<u32>,
        mut name: Option<String>,
        mut bits: Option<Vec<Py<PyQubit>>>,
    ) -> PyResult<(Self, PyRegister)> {
        if (size.is_none(), bits.is_none()) == (false, false) || (size.is_some() && bits.is_some())
        {
            return Err(
                    CircuitError::new_err(
                        format!("Exactly one of the size or bits arguments can be provided. Provided size={:?} bits={:?}.", size, bits)
                    )
                );
        }
        if name.is_none() {
            // This line is the reason we cannot turn this into a macro-rule
            let count = if let Ok(ref mut count) = QREG_COUNTER.try_lock() {
                let curr = **count;
                **count += 1;
                curr
            } else {
                panic!("Could not access register counter.")
            };
            name = Some(format!("{}{}", "q", count));
        }
        if bits.is_none() && size.is_some() {
            bits = Some(
                (0..size.unwrap())
                    .map(|idx| {
                        Py::new(
                            py,
                            PyQubit::py_new(
                                Some(RegisterAsKey::Quantum((
                                    name.clone().unwrap(),
                                    size.unwrap(),
                                ))),
                                Some(idx),
                            )
                            .unwrap(),
                        )
                        .unwrap()
                    })
                    .collect(),
            );
            size = None;
        }
        Ok((
            Self(),
            PyRegister::new(
                py,
                size,
                name,
                bits.map(|vec| {
                    vec.into_iter()
                        .map(|ob| {
                            <pyo3::Bound<'_, pyo3::PyAny> as Clone>::clone(&ob.into_bound(py))
                                .downcast_into()
                                .unwrap()
                                .into()
                        })
                        .collect()
                }),
            )?,
        ))
    }
}

#[derive(Debug, Clone)]
#[pyclass(name="ClassicalRegister", module="qiskit.circuit.classicalregister", extends=PyRegister)]
pub struct PyClassicalRegister();

#[pymethods]
impl PyClassicalRegister {
    #[new]
    pub fn new(
        py: Python,
        mut size: Option<u32>,
        mut name: Option<String>,
        mut bits: Option<Vec<Py<PyClbit>>>,
    ) -> PyResult<(Self, PyRegister)> {
        if (size.is_none(), bits.is_none()) == (false, false) || (size.is_some() && bits.is_some())
        {
            return Err(
                    CircuitError::new_err(
                        format!("Exactly one of the size or bits arguments can be provided. Provided size={:?} bits={:?}.", size, bits)
                    )
                );
        }
        if bits.is_none() && size.is_some() {
            bits = Some(
                (0..size.unwrap())
                    .map(|idx| {
                        Py::new(
                            py,
                            PyClbit::py_new(
                                Some(RegisterAsKey::Quantum((
                                    name.clone().unwrap(),
                                    size.unwrap(),
                                ))),
                                Some(idx),
                            )
                            .unwrap(),
                        )
                        .unwrap()
                    })
                    .collect(),
            );
            size = None;
        }
        if name.is_none() {
            let count = if let Ok(ref mut count) = CREG_COUNTER.try_lock() {
                let curr = **count;
                **count += 1;
                curr
            } else {
                panic!("Could not access register counter.")
            };
            name = Some(format!("{}{}", "c", count));
        }
        Ok((
            Self(),
            PyRegister::new(
                py,
                size,
                name,
                bits.map(|vec| {
                    vec.into_iter()
                        .map(|ob| {
                            <pyo3::Bound<'_, pyo3::PyAny> as Clone>::clone(&ob.into_bound(py))
                                .downcast_into()
                                .unwrap()
                                .into()
                        })
                        .collect()
                }),
            )?,
        ))
    }
}
