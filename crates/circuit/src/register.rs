use indexmap::IndexSet;
use pyo3::{intern, types::PyAnyMethods, FromPyObject};
use std::{
    hash::{Hash, Hasher},
    sync::Mutex,
};

use crate::{
    interner::{Interned, Interner},
    Clbit, Qubit,
};

static REGISTER_INSTANCE_COUNTER: Mutex<u32> = Mutex::new(0);
static PREFIX: &str = "reg";

/// This represents the hash value of a Register according to the register's
/// name and number of qubits.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegisterAsKey(String, u32);

impl RegisterAsKey {
    pub fn new(name: &str, num_qubits: u32) -> Self {
        Self(name.to_string(), num_qubits)
    }

    pub fn reduce(&self) -> (&str, u32) {
        (self.0.as_str(), self.1)
    }
}

impl<'py> FromPyObject<'py> for RegisterAsKey {
    fn extract_bound(ob: &pyo3::Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        let (name, num_qubits) = (
            ob.getattr(intern!(ob.py(), "name"))?.extract()?,
            ob.len()? as u32,
        );
        Ok(RegisterAsKey(name, num_qubits))
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
    ($name:ident, $bit:ty) => {
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
                    let count = if let Ok(ref mut count) = REGISTER_INSTANCE_COUNTER.try_lock() {
                        let curr = **count;
                        **count += 1;
                        curr
                    } else {
                        panic!("Could not access register counter.")
                    };
                    format!("{}{}", PREFIX, count)
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

create_register!(QuantumRegister, Qubit);
create_register!(ClassicalRegister, Clbit);

/// Represents a collection of registers of a certain type within a circuit.
#[derive(Debug, Clone)]
pub(crate) struct CircuitRegistry<T: Register + Clone> {
    registry: Interner<T>,
}

impl<T: Register + Hash + Eq + Clone> CircuitRegistry<T> {
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
