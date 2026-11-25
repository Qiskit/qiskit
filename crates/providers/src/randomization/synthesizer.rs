// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Synthesizers for virtual gates.

use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use thiserror::Error;

use crate::program::ProgramFunction;

use super::flags::BitFlag;
use super::partition::Partition;
use super::virtual_type::VirtualType;

/// Why a virtual gate could not be synthesized.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SynthesizerError {
    #[error("this synthesizer cannot express {0:?}")]
    UnsupportedType(VirtualType),
}

/// A synthesizer.
pub trait Synthesizer: Any + Debug + Send + Sync {
    /// The virtual types this synthesizer can express.
    fn accepts(&self) -> &[VirtualType];

    /// How the synthesis factorizes over the qubits.
    fn factorization(&self) -> &Partition<u32>;

    /// The program function that synthesizes `virtual_type`.
    fn program_function(
        &self,
        virtual_type: VirtualType,
    ) -> Result<ProgramFunction, SynthesizerError>;

    /// How many qubits the synthesizer spans.
    fn num_qubits(&self) -> u32 {
        self.factorization().num_elements() as u32
    }

    /// How many tensor factors the synthesizer makes.
    fn num_factors(&self) -> usize {
        self.factorization().num_parts()
    }

    /// The qubits `factor` is synthesized on, or `None` if there is no such factor.
    fn factor_qubits(&self, factor: u32) -> Option<&[u32]> {
        self.factorization().part(factor as usize)
    }
}

impl dyn Synthesizer + 'static {
    /// Cast to a concrete synthesizer, or `None` if it is not of that type.
    pub fn downcast_ref<T: Synthesizer + 'static>(&self) -> Option<&T> {
        let self_as_any: &dyn Any = self;
        self_as_any.downcast_ref()
    }
}

/// Synthesize each qubit's virtual gate as `rz`-`sx`-`rz`-`sx`-`rz`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RzSx {
    factorization: Partition<u32>,
}

impl RzSx {
    pub fn new(num_qubits: u32) -> Self {
        RzSx {
            factorization: Partition::singletons(0..num_qubits)
                .expect("a range of qubits has no repeats"),
        }
    }
}

impl Synthesizer for RzSx {
    fn accepts(&self) -> &[VirtualType] {
        VirtualType::FLAGS
    }

    fn factorization(&self) -> &Partition<u32> {
        &self.factorization
    }

    fn program_function(
        &self,
        _virtual_type: VirtualType,
    ) -> Result<ProgramFunction, SynthesizerError> {
        todo!("synthesizing to rz and sx is not implemented yet")
    }
}

/// Synthesize each qubit's virtual gate as `rz`-`rx`-`rz`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RzRx {
    factorization: Partition<u32>,
}

impl RzRx {
    pub fn new(num_qubits: u32) -> Self {
        RzRx {
            factorization: Partition::singletons(0..num_qubits)
                .expect("a range of qubits has no repeats"),
        }
    }
}

impl Synthesizer for RzRx {
    fn accepts(&self) -> &[VirtualType] {
        VirtualType::FLAGS
    }

    fn factorization(&self) -> &Partition<u32> {
        &self.factorization
    }

    fn program_function(
        &self,
        _virtual_type: VirtualType,
    ) -> Result<ProgramFunction, SynthesizerError> {
        todo!("synthesizing to rz and rx is not implemented yet")
    }
}

/// The name of a built-in synthesizer, as carried by an annotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SynthesizerKind {
    RzSx,
    RzRx,
}

impl SynthesizerKind {
    /// The number of qubits one factor acts on.
    pub fn num_qubits(&self) -> u32 {
        match self {
            Self::RzSx | Self::RzRx => 1,
        }
    }

    /// Resolve against the qubits it synthesizes over.
    pub fn resolve(&self, num_qubits: u32) -> Arc<dyn Synthesizer> {
        match self {
            Self::RzSx => Arc::new(RzSx::new(num_qubits)),
            Self::RzRx => Arc::new(RzRx::new(num_qubits)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a_synthesizer_puts_one_factor_on_each_qubit() {
        for kind in [SynthesizerKind::RzSx, SynthesizerKind::RzRx] {
            let synthesizer = kind.resolve(3);
            assert_eq!(synthesizer.num_qubits(), 3, "{kind:?}");
            assert_eq!(synthesizer.num_factors(), 3, "{kind:?}");
            assert!(synthesizer.factorization().is_singletons(), "{kind:?}");
            for factor in 0..3 {
                assert_eq!(synthesizer.factor_qubits(factor), Some([factor].as_slice()));
            }
            assert_eq!(synthesizer.factor_qubits(3), None);
            // One factor is as wide as the kind says.
            assert_eq!(
                synthesizer.factor_qubits(0).unwrap().len() as u32,
                kind.num_qubits()
            );
        }
    }

    #[test]
    fn test_a_synthesizer_accepts_what_it_can_express() {
        let synthesizer = SynthesizerKind::RzSx.resolve(1);
        assert_eq!(synthesizer.accepts(), VirtualType::FLAGS);
    }

    #[test]
    fn test_a_synthesizer_can_be_cast_back_to_what_it_is() {
        let synthesizer = SynthesizerKind::RzRx.resolve(1);
        assert!(synthesizer.downcast_ref::<RzRx>().is_some());
        assert!(synthesizer.downcast_ref::<RzSx>().is_none());
    }
}
