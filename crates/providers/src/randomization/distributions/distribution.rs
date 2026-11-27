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

//! The trait every distribution implements.

use std::any::Any;
use std::fmt::Debug;

use crate::program::ProgramFunction;
use crate::randomization::partition::Partition;
use crate::randomization::virtual_type::VirtualType;

/// A distribution.
pub trait Distribution: Any + Debug + Send + Sync {
    /// The algebraic type of the sampled virtual gates.
    fn virtual_type(&self) -> VirtualType;

    /// How a sample factorizes over the qubits.
    fn factorization(&self) -> &Partition<u32>;

    /// The sampling program function.
    fn program_function(&self) -> ProgramFunction;

    /// How many qubits the distribution is drawn over.
    fn num_qubits(&self) -> u32 {
        self.factorization().num_elements() as u32
    }

    /// How many tensor factors a sample has.
    fn num_factors(&self) -> usize {
        self.factorization().num_parts()
    }

    /// The qubits `factor` is drawn on, or `None` if there is no such factor.
    fn factor_qubits(&self, factor: u32) -> Option<&[u32]> {
        self.factorization().part(factor as usize)
    }
}

impl dyn Distribution + 'static {
    /// Cast to a concrete distribution, or `None` if it is not of that type.
    pub fn downcast_ref<T: Distribution + 'static>(&self) -> Option<&T> {
        let self_as_any: &dyn Any = self;
        self_as_any.downcast_ref()
    }
}
