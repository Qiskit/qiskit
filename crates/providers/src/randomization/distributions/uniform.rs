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

//! Distributions drawn uniformly.

use crate::program::ProgramFunction;
use crate::randomization::partition::Partition;
use crate::randomization::virtual_type::VirtualType;

use super::Distribution;

/// Uniform over the Pauli group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniformPauli {
    factorization: Partition<u32>,
}

impl UniformPauli {
    /// Uniform Paulis factorizing as `factorization`.
    ///
    /// Every part has to be one qubit index.
    pub fn new(factorization: Partition<u32>) -> Option<Self> {
        if !factorization.is_singletons() {
            return None;
        }
        Some(UniformPauli { factorization })
    }
}

impl Distribution for UniformPauli {
    fn virtual_type(&self) -> VirtualType {
        VirtualType::Pauli
    }

    fn factorization(&self) -> &Partition<u32> {
        &self.factorization
    }

    fn program_function(&self) -> ProgramFunction {
        todo!("drawing a uniform Pauli is not implemented yet")
    }
}

/// Haar-random single-qubit unitaries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HaarU2 {
    factorization: Partition<u32>,
}

impl HaarU2 {
    /// Haar-random unitaries factorizing as `factorization` says, or `None` if it cannot.
    ///
    /// Every part has to be one qubit index.
    pub fn new(factorization: Partition<u32>) -> Option<Self> {
        if !factorization.is_singletons() {
            return None;
        }
        Some(HaarU2 { factorization })
    }
}

impl Distribution for HaarU2 {
    fn virtual_type(&self) -> VirtualType {
        VirtualType::U2
    }

    fn factorization(&self) -> &Partition<u32> {
        &self.factorization
    }

    fn program_function(&self) -> ProgramFunction {
        todo!("drawing a Haar-random U2 is not implemented yet")
    }
}
