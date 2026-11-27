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

//! Pauli-Lindblad noise.

use crate::program::ProgramFunction;
use crate::randomization::partition::Partition;
use crate::randomization::virtual_type::VirtualType;
use crate::tensor::Dim;

use super::Distribution;

/// A characterized Pauli-Lindblad noise map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Noise {
    /// The characterization this map was taken from.
    pub reference: Box<str>,
    /// How the map is modified before sampling.
    pub modifier: Option<Box<str>>,
    factorization: Partition<u32>,
}

impl Noise {
    /// A map factorizing as `factorization`.
    pub fn new(
        reference: impl Into<Box<str>>,
        modifier: Option<Box<str>>,
        factorization: Partition<u32>,
    ) -> Option<Self> {
        if !factorization.is_singletons() {
            return None;
        }
        Some(Noise {
            reference: reference.into(),
            modifier,
            factorization,
        })
    }
}

impl Distribution for Noise {
    fn virtual_type(&self) -> VirtualType {
        VirtualType::Pauli
    }

    fn factorization(&self) -> &Partition<u32> {
        &self.factorization
    }

    fn program_function(&self, _batch: &[Dim]) -> ProgramFunction {
        todo!("sampling a Pauli-Lindblad map is not implemented yet")
    }
}
