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
use super::{Direction, DistributionKind, SynthesizerKind};

/// The name of the emit operation.
pub const EMIT_NAME: &str = "emit";

/// The name of the collect operation.
pub const COLLECT_NAME: &str = "collect";

/// A distribution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Distribution {
    Noise {
        reference: Box<str>,
        modifier: Option<Box<str>>,
        num_qubits: u32,
    },
    Sample(DistributionKind),
}

impl Distribution {
    /// How many tensor factors each sample decomposes into.
    pub fn num_factors(&self) -> usize {
        match self {
            Self::Noise { num_qubits, .. } => *num_qubits as usize,
            Self::Sample(distribution) => distribution.num_factors(),
        }
    }

    /// The width of the virtual gate of `factor`, or `None` if it has no such factor.
    pub fn factor_qubits(&self, factor: u32) -> Option<u32> {
        match self {
            // Every factor of a noise map is one qubit's local Pauli.
            Self::Noise { num_qubits, .. } => (factor < *num_qubits).then_some(1),
            Self::Sample(distribution) => distribution
                .factorization()
                .part(factor as usize)
                .map(|part| part.len() as u32),
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

/// A sample of a distribution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Draw {
    /// Unique identifier of the sample.
    pub uuid: Uuid,
    /// What it was drawn from.
    pub distribution: Distribution,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Emit {
    pub direction: Option<Direction>,
    /// The sample from which to select the factor.
    pub draw: Arc<Draw>,
    /// Which factor of the distribution this emit corresponds to.
    pub factor: u32,
    /// Whether to invert the sample before composing it.
    pub adjoint: bool,
}

impl Operation for Emit {
    fn name(&self) -> &str {
        EMIT_NAME
    }

    fn num_qubits(&self) -> u32 {
        self.draw
            .distribution
            .factor_qubits(self.factor)
            .expect("an emission places a factor its distribution has")
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
    /// How gathered gates are synthesized.
    pub synthesizer: SynthesizerKind,
    pub steps: Vec<CollectStep>,
}

impl PartialEq for Collect {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl Collect {
    pub fn new(synthesizer: SynthesizerKind) -> Self {
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

    /// A Pauli-Lindblad map over `width` qubits factorizes into `width` factors.
    fn noise(width: u32) -> Distribution {
        Distribution::Noise {
            reference: "layer_1".into(),
            modifier: None,
            num_qubits: width,
        }
    }

    fn emission(distribution: Distribution, uuid: Uuid) -> Emit {
        emission_at(distribution, uuid, 0)
    }

    fn emission_at(distribution: Distribution, uuid: Uuid, factor: u32) -> Emit {
        Emit {
            direction: Some(Direction::Forward),
            draw: Arc::new(Draw { uuid, distribution }),
            factor,
            adjoint: false,
        }
    }

    #[test]
    fn test_an_emission_is_as_wide_as_one_factor() {
        let pauli = emission(
            Distribution::Sample(DistributionKind::UniformPauli),
            Uuid::new_v4(),
        );
        assert_eq!(pauli.num_qubits(), 1);
        assert_eq!(emission(noise(3), Uuid::new_v4()).num_qubits(), 1);
    }

    #[test]
    fn test_emissions_of_one_sample_share_a_uuid_and_differ_otherwise() {
        let uuid = Uuid::new_v4();
        let distribution = Distribution::Sample(DistributionKind::UniformPauli);
        let near = emission(distribution, uuid);
        let far = Emit {
            adjoint: true,
            ..near.clone()
        };
        assert_eq!(near.draw.uuid, far.draw.uuid);
        assert_ne!(near, far);
    }

    #[test]
    fn test_two_uuids_from_one_distribution_are_independent() {
        let first = emission(
            Distribution::Sample(DistributionKind::UniformPauli),
            Uuid::new_v4(),
        );
        let second = emission(
            Distribution::Sample(DistributionKind::UniformPauli),
            Uuid::new_v4(),
        );
        assert_ne!(first.draw.uuid, second.draw.uuid);
        assert_ne!(first, second);
    }

    #[test]
    fn test_noise_emits_are_told_apart_by_their_factor() {
        let uuid = Uuid::new_v4();
        let draw = Arc::new(Draw {
            uuid,
            distribution: noise(2),
        });
        let first = Emit {
            direction: Some(Direction::Forward),
            draw: Arc::clone(&draw),
            factor: 0,
            adjoint: false,
        };
        let second = Emit {
            factor: 1,
            ..first.clone()
        };
        assert_eq!(first.draw.uuid, second.draw.uuid);
        assert_eq!(first.draw.distribution, second.draw.distribution);
        assert_ne!(first, second);
    }

    #[test]
    fn test_one_draw_is_one_draw_however_it_was_built() {
        let uuid = Uuid::new_v4();
        let built = |factor| Emit {
            direction: Some(Direction::Forward),
            draw: Arc::new(Draw {
                uuid,
                distribution: noise(2),
            }),
            factor,
            adjoint: false,
        };
        let separate = [built(0), built(1)];
        assert!(!Arc::ptr_eq(&separate[0].draw, &separate[1].draw));
        assert_eq!(separate[0].draw, separate[1].draw);

        let shared_draw = Arc::new(Draw {
            uuid,
            distribution: noise(2),
        });
        let shared = Emit {
            direction: Some(Direction::Forward),
            draw: Arc::clone(&shared_draw),
            factor: 0,
            adjoint: false,
        };
        assert_eq!(separate[0], shared);
    }

    #[test]
    fn test_distribution_virtual_type() {
        assert_eq!(noise(1).virtual_type(), VirtualType::Pauli);
        assert_eq!(
            Distribution::Sample(DistributionKind::HaarU2).virtual_type(),
            VirtualType::U2
        );
    }

    #[test]
    fn test_a_collector_starts_empty_and_as_wide_as_its_synthesizer() {
        let collector = Collect::new(SynthesizerKind::RzSx);
        assert!(collector.is_empty());
        assert_eq!(collector.num_qubits(), 1);
        assert_eq!(collector.name(), COLLECT_NAME);
    }

    #[test]
    fn test_a_collector_accepts_what_its_synthesizer_can_express() {
        let collector = Collect::new(SynthesizerKind::RzSx);
        for virtual_type in [VirtualType::Pauli, VirtualType::C1, VirtualType::U2] {
            assert!(collector.accepts(virtual_type), "{virtual_type:?}");
        }
    }

    #[test]
    fn test_neither_operation_is_unitary() {
        assert!(!emission(noise(0), Uuid::new_v4()).is_unitary());
        assert!(!Collect::new(SynthesizerKind::RzRx).is_unitary());
    }
}
