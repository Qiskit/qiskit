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

//! Distributions to sample virtual gates from.

mod distribution;
mod noise;
mod uniform;

use std::sync::Arc;

use super::partition::Partition;
use super::virtual_type::VirtualType;

pub use distribution::Distribution;
pub use noise::Noise;
pub use uniform::{HaarU2, UniformPauli};

/// The kind of distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DistributionKind {
    UniformPauli,
    // BalancedUniformPauli,
    // UniformPauliSubset,
    // UniformC1,
    HaarU2,
    // UniformLocalC1,
}

impl DistributionKind {
    /// The algebraic type of the sampled virtual gates.
    pub fn virtual_type(&self) -> VirtualType {
        match self {
            Self::UniformPauli => VirtualType::Pauli,
            // Self::BalancedUniformPauli | Self::UniformPauliSubset => VirtualType::Pauli,
            // Self::UniformC1 | Self::UniformLocalC1 => VirtualType::C1,
            Self::HaarU2 => VirtualType::U2,
        }
    }

    /// Resolve over `factorization`, or `None` if this kind cannot factorize that way.
    pub fn resolve(&self, factorization: Partition<u32>) -> Option<Arc<dyn Distribution>> {
        match self {
            Self::UniformPauli => UniformPauli::new(factorization)
                .map(|pauli| Arc::new(pauli) as Arc<dyn Distribution>),
            Self::HaarU2 => {
                HaarU2::new(factorization).map(|haar| Arc::new(haar) as Arc<dyn Distribution>)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn singletons(num_qubits: u32) -> Partition<u32> {
        Partition::singletons(0..num_qubits).expect("a range of qubits has no repeats")
    }

    fn noise(num_qubits: u32) -> Arc<dyn Distribution> {
        Noise::new("layer_1", None, singletons(num_qubits))
            .map(|noise| Arc::new(noise) as Arc<dyn Distribution>)
            .expect("noise resolves over singletons")
    }

    fn resolved(kind: DistributionKind, num_qubits: u32) -> Arc<dyn Distribution> {
        kind.resolve(singletons(num_qubits))
            .expect("a kind resolves over singletons")
    }

    fn all(num_qubits: u32) -> [Arc<dyn Distribution>; 3] {
        [
            resolved(DistributionKind::UniformPauli, num_qubits),
            resolved(DistributionKind::HaarU2, num_qubits),
            noise(num_qubits),
        ]
    }

    #[test]
    fn test_a_kind_knows_the_type_of_gate_it_names() {
        assert_eq!(
            DistributionKind::UniformPauli.virtual_type(),
            VirtualType::Pauli
        );
        assert_eq!(DistributionKind::HaarU2.virtual_type(), VirtualType::U2);
        assert_eq!(noise(1).virtual_type(), VirtualType::Pauli);
    }

    #[test]
    fn test_resolving_keeps_the_type_its_kind_names() {
        for kind in [DistributionKind::UniformPauli, DistributionKind::HaarU2] {
            assert_eq!(
                resolved(kind, 2).virtual_type(),
                kind.virtual_type(),
                "{kind:?}"
            );
        }
    }

    #[test]
    fn test_every_distribution_draws_independently_across_factors() {
        for distribution in all(3) {
            assert_eq!(distribution.num_qubits(), 3, "{distribution:?}");
            assert_eq!(distribution.num_factors(), 3, "{distribution:?}");
            assert!(
                distribution.factorization().is_singletons(),
                "{distribution:?}"
            );
            for factor in 0..3 {
                assert_eq!(
                    distribution.factor_qubits(factor),
                    Some([factor].as_slice())
                );
            }
            assert_eq!(distribution.factor_qubits(3), None);
        }
    }

    #[test]
    fn test_a_distribution_over_no_qubits_has_no_factors() {
        for distribution in all(0) {
            assert_eq!(distribution.num_qubits(), 0, "{distribution:?}");
            assert_eq!(distribution.num_factors(), 0, "{distribution:?}");
            assert_eq!(distribution.factor_qubits(0), None);
        }
    }

    #[test]
    fn test_a_distribution_can_be_cast_back_to_what_it_is() {
        let distribution = noise(2);
        let map = distribution
            .downcast_ref::<Noise>()
            .expect("a noise map resolves to Noise");
        assert_eq!(&*map.reference, "layer_1");
        assert_eq!(map.modifier, None);
        assert!(distribution.downcast_ref::<UniformPauli>().is_none());
    }

    #[test]
    fn test_a_kind_that_cannot_factorize_that_way_declines() {
        let paired = Partition::new([vec![0, 1]]).expect("a partition the test means");
        assert!(
            DistributionKind::HaarU2.resolve(paired.clone()).is_none(),
            "Haar over a pair is not Haar over U(2)"
        );
    }

    #[test]
    fn test_a_kind_resolves_over_the_factorization_it_is_given_rather_than_a_width() {
        let permuted = Partition::new([vec![2], vec![0], vec![1]]).expect("a partition");
        let distribution = resolved_over(DistributionKind::UniformPauli, permuted);
        assert_eq!(distribution.num_qubits(), 3);
        assert_eq!(distribution.factor_qubits(0), Some([2].as_slice()));
        assert_eq!(distribution.factor_qubits(1), Some([0].as_slice()));
    }

    fn resolved_over(
        kind: DistributionKind,
        factorization: Partition<u32>,
    ) -> Arc<dyn Distribution> {
        kind.resolve(factorization).expect("the kind accepts it")
    }
}
