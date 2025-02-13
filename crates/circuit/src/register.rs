use std::sync::Arc;

use crate::bit::{BitInfo, SharableBit};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub(crate) enum RegisterInfo<T: SharableBit> {
    Owning(Arc<OwningRegisterInfo<T>>),
    Alias {
        name: String,
        bits: Box<[BitInfo<T>]>,
        extra: <T as SharableBit>::ExtraAttributes,
    },
}

impl<T: SharableBit + Clone> RegisterInfo<T> {
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

    /// Returns the size of said register.
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
            RegisterInfo::Alias {
                name: _,
                bits,
                extra: _,
            } => Box::new(bits.iter().cloned()),
        }
    }
}

/// Contains the informaion for a register that owns the bits it contains.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub(crate) struct OwningRegisterInfo<T: SharableBit> {
    name: String,
    size: u32,
    extra: <T as SharableBit>::ExtraAttributes,
}
