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

//! Annotations for randomization.

use std::sync::Arc;

use thiserror::Error;

use qiskit_circuit::annotation::Annotation;

use super::distributions::{Distribution, Noise};
use super::partition::Partition;
use super::{DistributionKind, SynthesizerKind};

/// The namespace for randomization annotations.
pub(crate) const NAMESPACE: &str = "randomization";

pub type Result<T> = std::result::Result<T, InvalidAnnotationError>;

/// Errors from invalid annotation combinations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum InvalidAnnotationError {
    #[error("duplicate randomization annotation on a single box")]
    DuplicateAnnotation,
    /// Without a twirl there is no collect for the injected sample to resolve at.
    #[error("a noise injection needs a twirl on the same box")]
    InjectionWithoutTwirl,
}

/// Which side of a box its easy gates get composed into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DressingSide {
    Left,
    Right,
}

/// Where noise is injected in the box.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InjectionSite {
    Before,
    Between,
    After,
}

/// Which randomization annotation this is, or `None` if it is not.
pub fn annotation_kind(annotation: &dyn Annotation) -> Option<RandomizationAnnotation<'_>> {
    if annotation.namespace() != NAMESPACE {
        return None;
    }
    if let Some(twirl) = annotation.downcast_ref::<Twirl>() {
        Some(RandomizationAnnotation::Twirl(twirl))
    } else {
        annotation
            .downcast_ref::<InjectNoise>()
            .map(RandomizationAnnotation::InjectNoise)
    }
}

/// A randomization annotation borrowed from the box carrying it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RandomizationAnnotation<'a> {
    Twirl(&'a Twirl),
    InjectNoise(&'a InjectNoise),
}

/// Randomize this box by conjugating it with gates sampled from a distribution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Twirl {
    pub distribution: DistributionKind,
    pub dressing: DressingSide,
    pub decomposition: SynthesizerKind,
}

impl Annotation for Twirl {
    fn namespace(&self) -> &str {
        NAMESPACE
    }
}

/// Inject Pauli-Lindblad noise into this box.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InjectNoise {
    pub reference: String,
    pub modifier: Option<String>,
    pub site: InjectionSite,
}

impl InjectNoise {
    /// The noise map this injects, resolved over `factorization`, or `None` if it cannot
    /// factorize that way.
    ///
    /// The only way to reach a [`Noise`] distribution.
    pub fn resolve(&self, factorization: Partition<u32>) -> Option<Arc<dyn Distribution>> {
        Noise::new(
            self.reference.as_str(),
            self.modifier.as_deref().map(Box::from),
            factorization,
        )
        .map(|noise| Arc::new(noise) as Arc<dyn Distribution>)
    }
}

impl Annotation for InjectNoise {
    fn namespace(&self) -> &str {
        NAMESPACE
    }
}

/// Randomization annotations on a box.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RandomizationAnnotations<'a> {
    pub twirl: Option<&'a Twirl>,
    pub inject_noise: Option<&'a InjectNoise>,
}

impl<'a> RandomizationAnnotations<'a> {
    /// Return validated annotations.
    ///
    /// A box carries at most one of each kind and an injection needs a twirl beside it.
    pub fn new(annotations: &'a [Arc<dyn Annotation>]) -> Result<Self> {
        let mut resolved = Self::default();
        for annotation in annotations {
            let Some(kind) = annotation_kind(annotation.as_ref()) else {
                continue;
            };
            let already_set = match kind {
                RandomizationAnnotation::Twirl(twirl) => resolved.twirl.replace(twirl).is_some(),
                RandomizationAnnotation::InjectNoise(injection) => {
                    resolved.inject_noise.replace(injection).is_some()
                }
            };
            if already_set {
                return Err(InvalidAnnotationError::DuplicateAnnotation);
            }
        }
        if resolved.inject_noise.is_some() && resolved.twirl.is_none() {
            return Err(InvalidAnnotationError::InjectionWithoutTwirl);
        }

        Ok(resolved)
    }
}

#[cfg(test)]
mod tests {
    use super::super::virtual_type::VirtualType;
    use super::*;

    fn erased(annotation: impl Annotation + 'static) -> Arc<dyn Annotation> {
        Arc::new(annotation)
    }

    fn twirl(distribution: DistributionKind) -> Arc<dyn Annotation> {
        erased(Twirl {
            distribution,
            dressing: DressingSide::Left,
            decomposition: SynthesizerKind::RzSx,
        })
    }

    fn noise(site: InjectionSite) -> Arc<dyn Annotation> {
        erased(InjectNoise {
            reference: "layer_1".to_string(),
            modifier: None,
            site,
        })
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Foreign;

    impl Annotation for Foreign {
        fn namespace(&self) -> &str {
            "someone.else"
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Impostor;

    impl Annotation for Impostor {
        fn namespace(&self) -> &str {
            NAMESPACE
        }
    }

    #[test]
    fn test_injected_noise_resolves_to_a_map_over_the_qubits_it_covers() {
        let annotation = InjectNoise {
            reference: "layer_1".to_string(),
            modifier: Some("twice".to_string()),
            site: InjectionSite::After,
        };
        let distribution = annotation
            .resolve(Partition::singletons(0..3).expect("a range of qubits has no repeats"))
            .expect("noise resolves over singletons");
        assert_eq!(distribution.virtual_type(), VirtualType::Pauli);
        assert_eq!(distribution.num_qubits(), 3);
        assert_eq!(distribution.num_factors(), 3);

        let map = distribution
            .downcast_ref::<Noise>()
            .expect("injected noise resolves to Noise");
        assert_eq!(&*map.reference, "layer_1");
        assert_eq!(map.modifier.as_deref(), Some("twice"));
    }

    #[test]
    fn test_a_twirl_resolves_to_what_it_declares() {
        let annotations = [twirl(DistributionKind::UniformPauli)];
        let resolved = RandomizationAnnotations::new(&annotations).unwrap();
        let twirl = resolved.twirl.expect("the twirl resolves");
        assert_eq!(twirl.distribution, DistributionKind::UniformPauli);
        assert_eq!(twirl.dressing, DressingSide::Left);
        assert_eq!(twirl.decomposition, SynthesizerKind::RzSx);
        assert!(resolved.inject_noise.is_none());
    }

    #[test]
    fn test_a_box_with_foreign_annotations_on_it_resolves_to_nothing() {
        let annotations = [erased(Foreign)];
        let resolved = RandomizationAnnotations::new(&annotations).unwrap();
        assert_eq!(resolved, RandomizationAnnotations::default());
        assert!(resolved.twirl.is_none() && resolved.inject_noise.is_none());
    }

    #[test]
    fn test_an_unannotated_box_resolves_to_nothing() {
        assert_eq!(
            RandomizationAnnotations::new(&[]).unwrap(),
            RandomizationAnnotations::default()
        );
    }

    #[test]
    fn test_noise_resolves_beside_a_twirl() {
        let annotations = [
            twirl(DistributionKind::UniformPauli),
            noise(InjectionSite::Between),
        ];
        let resolved = RandomizationAnnotations::new(&annotations).unwrap();
        assert!(resolved.twirl.is_some());
        let injection = resolved.inject_noise.expect("the injection resolves");
        assert_eq!(injection.reference, "layer_1");
        assert_eq!(injection.site, InjectionSite::Between);
        assert_eq!(injection.modifier, None);
    }

    #[test]
    fn test_noise_without_a_twirl_to_resolve_at_is_refused() {
        for site in [
            InjectionSite::Before,
            InjectionSite::Between,
            InjectionSite::After,
        ] {
            assert_eq!(
                RandomizationAnnotations::new(&[noise(site)]).unwrap_err(),
                InvalidAnnotationError::InjectionWithoutTwirl,
                "{site:?}"
            );
            assert_eq!(
                RandomizationAnnotations::new(&[erased(Foreign), noise(site)]).unwrap_err(),
                InvalidAnnotationError::InjectionWithoutTwirl,
                "{site:?}"
            );
        }
    }

    #[test]
    fn test_two_of_one_kind_on_a_box_is_refused() {
        assert_eq!(
            RandomizationAnnotations::new(&[
                twirl(DistributionKind::UniformPauli),
                twirl(DistributionKind::HaarU2),
            ])
            .unwrap_err(),
            InvalidAnnotationError::DuplicateAnnotation
        );
        assert_eq!(
            RandomizationAnnotations::new(&[
                twirl(DistributionKind::UniformPauli),
                noise(InjectionSite::Before),
                noise(InjectionSite::After),
            ])
            .unwrap_err(),
            InvalidAnnotationError::DuplicateAnnotation
        );
    }

    #[test]
    fn test_claiming_the_namespace_is_not_enough() {
        let annotations = [erased(Impostor), twirl(DistributionKind::UniformPauli)];
        let resolved = RandomizationAnnotations::new(&annotations).unwrap();
        assert_eq!(
            resolved.twirl.map(|twirl| twirl.distribution),
            Some(DistributionKind::UniformPauli)
        );
        assert!(
            RandomizationAnnotations::new(&[erased(Impostor), erased(Impostor)])
                .unwrap()
                .twirl
                .is_none()
        );
    }

    #[test]
    fn test_a_foreign_annotation_rides_through() {
        let annotations = [
            erased(Foreign),
            twirl(DistributionKind::UniformPauli),
            erased(Foreign),
        ];
        let resolved = RandomizationAnnotations::new(&annotations).unwrap();
        assert!(resolved.twirl.is_some());
    }

    #[test]
    fn test_the_annotations_share_one_flat_namespace() {
        for annotation in [
            twirl(DistributionKind::UniformPauli),
            noise(InjectionSite::After),
        ] {
            assert_eq!(annotation.namespace(), NAMESPACE);
        }
    }
}
