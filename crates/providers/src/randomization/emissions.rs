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

//! Emit and collect operations.

use std::sync::Arc;

use uuid::Uuid;

use qiskit_circuit::operations::{CustomOperation, Operation};
use qiskit_circuit::packed_instruction::PackedInstruction;

use super::virtual_type::VirtualType;
use super::{Direction, DistributionType, SynthesizerType};

/// The name of the emit operation.
pub const EMIT_NAME: &str = "emit";

/// The name of the collect operation.
pub const COLLECT_NAME: &str = "collect";

/// A distribution.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Distribution {
    Noise {
        reference: Arc<str>,
        modifier: Option<Arc<str>>,
        slot: u32,
    },
    Sample(DistributionType),
}

impl Distribution {
    /// The number of qubits.
    pub fn num_qubits(&self) -> u32 {
        match self {
            Self::Noise { .. } => 1,
            Self::Sample(distribution) => distribution.num_qubits(),
        }
    }

    /// The virtual type.
    pub fn virtual_type(&self) -> VirtualType {
        match self {
            Self::Noise { .. } => VirtualType::Pauli,
            Self::Sample(distribution) => distribution.virtual_type(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Emit {
    pub direction: Option<Direction>,
    pub distribution: Distribution,
    pub uuid: Uuid,
    pub adjoint: bool,
}

impl Operation for Emit {
    fn name(&self) -> &str {
        EMIT_NAME
    }

    fn num_qubits(&self) -> u32 {
        self.distribution.num_qubits()
    }

    fn num_clbits(&self) -> u32 {
        0
    }

    fn num_params(&self) -> u32 {
        0
    }

    fn directive(&self) -> bool {
        true
    }
}

impl CustomOperation for Emit {
    fn is_unitary(&self) -> bool {
        false
    }
}

pub type CollectStep = (Vec<u32>, PackedInstruction);

#[derive(Debug, Clone)]
pub struct Collect {
    pub synthesizer: SynthesizerType,
    pub steps: Vec<CollectStep>,
}

impl PartialEq for Collect {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl Collect {
    pub fn new(synthesizer: SynthesizerType) -> Self {
        Collect {
            synthesizer,
            steps: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    pub fn accepts(&self, virtual_type: VirtualType) -> bool {
        self.synthesizer.accepts(virtual_type)
    }
}

impl Operation for Collect {
    fn name(&self) -> &str {
        COLLECT_NAME
    }

    fn num_qubits(&self) -> u32 {
        self.synthesizer.num_qubits()
    }

    fn num_clbits(&self) -> u32 {
        0
    }

    fn num_params(&self) -> u32 {
        0
    }

    fn directive(&self) -> bool {
        true
    }
}

impl CustomOperation for Collect {
    fn is_unitary(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(slot: u32) -> Distribution {
        Distribution::Noise {
            reference: "layer_1".into(),
            modifier: None,
            slot,
        }
    }

    fn emission(distribution: Distribution, uuid: Uuid) -> Emit {
        Emit {
            direction: Some(Direction::Forward),
            distribution,
            uuid,
            adjoint: false,
        }
    }

    #[test]
    fn test_an_emission_is_as_wide_as_one_uuid() {
        // Not as wide as the box it came from: a box wider than its distribution is covered by
        // several emissions, which is what lets each be absorbed on its own wire.
        let pauli = emission(
            Distribution::Sample(DistributionType::UniformPauli),
            Uuid::new_v4(),
        );
        assert_eq!(pauli.num_qubits(), 1);
        let local = emission(
            Distribution::Sample(DistributionType::UniformLocalC1),
            Uuid::new_v4(),
        );
        assert_eq!(local.num_qubits(), 2);
        assert_eq!(emission(noise(3), Uuid::new_v4()).num_qubits(), 1);
    }

    #[test]
    fn test_emissions_of_one_sample_share_a_uuid_and_differ_otherwise() {
        let uuid = Uuid::new_v4();
        let distribution = Distribution::Sample(DistributionType::UniformPauli);
        let near = Emit {
            direction: Some(Direction::Forward),
            distribution: distribution.clone(),
            uuid,
            adjoint: false,
        };
        let far = Emit {
            adjoint: true,
            ..near.clone()
        };
        assert_eq!(near.uuid, far.uuid);
        assert_ne!(near, far);
    }

    #[test]
    fn test_two_uuids_from_one_distribution_are_independent() {
        let distribution = Distribution::Sample(DistributionType::UniformPauli);
        let first = emission(distribution.clone(), Uuid::new_v4());
        let second = emission(distribution, Uuid::new_v4());
        assert_ne!(first.uuid, second.uuid);
        assert_ne!(first, second);
    }

    #[test]
    fn test_noise_slots_are_told_apart_by_their_slot() {
        let uuid = Uuid::new_v4();
        let first = emission(noise(0), uuid);
        let second = emission(noise(1), uuid);
        // One sample of one map, read at two slots.
        assert_eq!(first.uuid, second.uuid);
        assert_ne!(first, second);
    }

    #[test]
    fn test_a_distributions_virtual_type_decides_what_can_carry_it() {
        // Noise is Pauli however it was uuidn, which is what lets it propagate and be absorbed like
        // any other Pauli.
        assert_eq!(noise(0).virtual_type(), VirtualType::Pauli);
        assert_eq!(
            Distribution::Sample(DistributionType::HaarU2).virtual_type(),
            VirtualType::U2
        );
    }

    #[test]
    fn test_a_collector_starts_empty_and_as_wide_as_its_synthesizer() {
        let collector = Collect::new(SynthesizerType::RzSx);
        assert!(collector.is_empty());
        assert_eq!(collector.num_qubits(), 1);
        assert_eq!(collector.name(), COLLECT_NAME);
    }

    #[test]
    fn test_a_collector_accepts_what_its_synthesizer_can_express() {
        let collector = Collect::new(SynthesizerType::RzSx);
        for virtual_type in [VirtualType::Pauli, VirtualType::C1, VirtualType::U2] {
            assert!(collector.accepts(virtual_type), "{virtual_type:?}");
        }
    }

    #[test]
    fn test_neither_operation_is_unitary() {
        assert!(!emission(noise(0), Uuid::new_v4()).is_unitary());
        assert!(!Collect::new(SynthesizerType::RzRx).is_unitary());
    }
}
