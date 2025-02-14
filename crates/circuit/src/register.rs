use crate::bit::{BitInfo, ShareableBit, ShareableClbit, ShareableQubit};
use pyo3::prelude::*;
use std::{fmt::Display, sync::Arc};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub(crate) enum RegisterInfo<T: ShareableBit> {
    Owning(Arc<OwningRegisterInfo<T>>),
    Alias {
        name: String,
        bits: Box<[BitInfo<T>]>,
        extra: <T as ShareableBit>::ExtraAttributes,
    },
}

impl<T: ShareableBit + Clone> RegisterInfo<T> {
    /// Creates a Register whose bits are owned by its instance
    pub fn new_owning(name: String, size: u32, extra: T::ExtraAttributes) -> Self {
        // When creating `Owning` register, we don't need to create the `BitInfo`
        // instances, they can be entirely derived from `self`.
        Self::Owning(Arc::new(OwningRegisterInfo { name, size, extra }))
    }

    /// Creates a Register whose bits already exist.
    pub fn new_alias(name: String, bits: Box<[BitInfo<T>]>, extra: T::ExtraAttributes) -> Self {
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
