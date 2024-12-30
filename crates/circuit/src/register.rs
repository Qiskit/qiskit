use std::{hash::{DefaultHasher, Hash, Hasher}, ops::Index};
use indexmap::IndexSet;
use pyo3::{intern, types::PyAnyMethods, FromPyObject};

use crate::{Clbit, Qubit};

/// This represents the hash value of a Register according to the register's
/// name and number of qubits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegisterAsKey(u64);

impl RegisterAsKey {
    pub fn new(name: Option<&str>, num_qubits: u32) -> Self {
        let mut hasher = DefaultHasher::default();
        (name, num_qubits).hash(&mut hasher);
        Self(hasher.finish())
    }
}

impl<'py> FromPyObject<'py> for RegisterAsKey {
    fn extract_bound(ob: &pyo3::Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        let (name, num_qubits) = (
            ob.getattr(intern!(ob.py(), "name"))?
                .extract::<Option<String>>()?,
            ob.getattr(intern!(ob.py(), "num_qubits"))?.extract()?,
        );
        Ok(RegisterAsKey::new(
            name.as_ref().map(|x| x.as_str()),
            num_qubits,
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
    fn bits(&self) -> impl ExactSizeIterator<Item=Self::Bit>;
}

macro_rules! create_register {
    ($name:ident, $bit:ty) => {
        #[derive(Debug, Clone, Eq)]
        pub struct $name {
            register: IndexSet<<$name as Register>::Bit>,
            name: Option<String>,
        }
        
        impl $name {
            pub fn new(size: usize, name: Option<String>) -> Self {
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

            fn bits(&self) -> impl ExactSizeIterator<Item=Self::Bit> {
                self.register.iter().copied()
            }
        }
        
        impl Hash for $name {
            fn hash<H: Hasher>(&self, state: &mut H) {
                (self.name.as_ref(), self.len()).hash(state);
            }
        }

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.register.len() == other.register.len() && self.name == other.name
            }
        }
    };
}

create_register!(QuantumRegister, Qubit);
create_register!(ClassicalRegister, Clbit);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegistryIndex(u32);

impl From<usize> for RegistryIndex {
    fn from(value: usize) -> Self {
        Self(value.try_into().expect("Index falls out of range"))
    }
}

impl From<u32> for RegistryIndex {
    fn from(value: u32) -> Self {
        Self(value)
    }
}
/// Represents a collection of registers of a certain type within a circuit.
#[derive(Debug, Clone)]
pub(crate) struct CircuitRegistry<T: Register> {
    registry: IndexSet<T>,
}

impl<T: Register> Index<RegistryIndex> for CircuitRegistry<T> {
    type Output = T;

    fn index(&self, index: RegistryIndex) -> &Self::Output {
        &self.registry[index.0 as usize]
    }
}

impl<T: Register + Hash + Eq> CircuitRegistry<T> {
    /// Retreives the index of a register if it exists within a registry.
    pub fn find_index(&self, register: &T) -> Option<RegistryIndex> {
        self.registry.get_index_of(register).map(RegistryIndex::from)
    }
}
