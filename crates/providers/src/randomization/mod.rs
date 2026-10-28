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

//! Generate randomizations of quantum circuits and programs.

pub mod annotations;
pub mod emissions;
pub mod partition;
pub mod virtual_type;

<<<<<<< HEAD
use virtual_type::VirtualType;

/// How to synthesize virtual randomizations.
=======
/// The synthesizer to use for virtual gates.
>>>>>>> randomization-annotations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SynthesizerType {
    RzSx,
    RzRx,
}

impl SynthesizerType {
<<<<<<< HEAD
    /// How many qubits one synthesis covers.
=======
    /// The number of qubits the synthesized gate acts on.
>>>>>>> randomization-annotations
    pub fn num_qubits(&self) -> u32 {
        match self {
            Self::RzSx | Self::RzRx => 1,
        }
    }
<<<<<<< HEAD

    /// Whether a virtual gate of this type is one this can express.
    ///
    /// Both of these decompose an arbitrary single-qubit unitary, so both accept everything a
    /// single-qubit distribution can draw.
    pub fn accepts(&self, _virtual_type: VirtualType) -> bool {
        match self {
            Self::RzSx | Self::RzRx => true,
        }
    }
}

/// A named distribution to sample virtual gates from.
=======
}

/// The distribution to sample virtual gates from.
>>>>>>> randomization-annotations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DistributionType {
    UniformPauli,
    BalancedUniformPauli,
    UniformPauliSubset,
    UniformC1,
    UniformLocalC1,
    HaarU2,
}

impl DistributionType {
<<<<<<< HEAD
    /// The algebraic type of the virtual gates drawn from this, which decides what a draw may be
    /// propagated through and which synthesizers can express it.
    pub fn virtual_type(&self) -> VirtualType {
        match self {
            Self::UniformPauli | Self::BalancedUniformPauli | Self::UniformPauliSubset => {
                VirtualType::Pauli
            }
            Self::UniformC1 | Self::UniformLocalC1 => VirtualType::C1,
            Self::HaarU2 => VirtualType::U2,
        }
    }

    /// How many qubits one sample covers.
=======
    /// The number qubits the sampled virtual gate acts on.
>>>>>>> randomization-annotations
    pub fn num_qubits(&self) -> u32 {
        match self {
            Self::UniformPauli
            | Self::BalancedUniformPauli
            | Self::UniformPauliSubset
            | Self::UniformC1
            | Self::HaarU2 => 1,
            Self::UniformLocalC1 => 2,
        }
    }
}

/// Which way virtual state flows temporally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    Forward,
    Backward,
}
