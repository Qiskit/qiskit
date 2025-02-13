use std::{
    fmt::Debug,
    hash::Hash,
    sync::{atomic::AtomicU64, Arc},
};

use crate::register::OwningRegisterInfo;

pub trait SharableBit
where
    <Self as SharableBit>::ExtraAttributes: Debug + Clone + PartialEq + Eq + PartialOrd + Hash,
{
    /// Struct defining any specific extra attributes for the bit.
    type ExtraAttributes;
    /// Literal description of the bit type.
    const DESCRIPTION: &'static str;

    /// Returns reference to the instance counter for the bit.
    fn anonymous_instances() -> &'static AtomicU64;
}

/// Counter for all existing anonymous Qubit instances.
static QUBIT_COUNTER: AtomicU64 = AtomicU64::new(0);
#[derive(Copy, Clone, PartialEq, Eq, Hash)]

/// Alias for sharable version of a Qubit, implements [SharableBit] trait.
struct ShareableQubit;
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Hash)]
struct QubitExtraInfo {
    is_ancilla: bool,
}

impl From<bool> for QubitExtraInfo {
    fn from(value: bool) -> Self {
        QubitExtraInfo { is_ancilla: value }
    }
}

impl SharableBit for ShareableQubit {
    type ExtraAttributes = QubitExtraInfo;
    const DESCRIPTION: &'static str = "qubit";
    fn anonymous_instances() -> &'static AtomicU64 {
        &QUBIT_COUNTER
    }
}

/// Alias for sharable version of a Clbit, implements [SharableBit] trait.
struct ShareableClbit;
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ClbitExtraInfo();

impl SharableBit for ShareableClbit {
    type ExtraAttributes = ClbitExtraInfo;
    const DESCRIPTION: &'static str = "clbit";
    fn anonymous_instances() -> &'static AtomicU64 {
        &QUBIT_COUNTER
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub enum BitInfo<T: SharableBit> {
    Owned {
        register: Arc<OwningRegisterInfo<T>>,
        index: u32,
    },
    Anonymous {
        /// Unique id for bit, derives from [SharableBit::anonymous_instances]
        unique_id: u64,
        /// Data about the
        extra: T::ExtraAttributes,
    },
}

impl<T: SharableBit> BitInfo<T> {
    /// Creates an instance of anonymous [BitInfo].
    pub fn new_anonymous(extra: <T as SharableBit>::ExtraAttributes) -> Self {
        Self::Anonymous {
            unique_id: <T as SharableBit>::anonymous_instances()
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            extra,
        }
    }
}

impl BitInfo<ShareableQubit> {
    pub fn is_ancilla(&self) -> bool {
        match self {
            BitInfo::Owned { register, index } => todo!(),
            BitInfo::Anonymous { unique_id, extra } => extra.is_ancilla,
        }
    }
}
