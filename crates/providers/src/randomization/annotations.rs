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
//!
//! [`Twirl`] and [`InjectNoise`] implement [`Annotation`]. [`resolve_annotations`] resolves a box's
//!  set of them into one [`ResolvedAnnotations`], applying the rules that say which combinations mean
//! anything.

use std::sync::Arc;

use hashbrown::HashSet;
use thiserror::Error;

use qiskit_circuit::annotation::Annotation;

/// The namespace every annotation here declares.
pub(crate) const NAMESPACE: &str = "randomization";

pub type Result<T> = std::result::Result<T, RandomizationError>;

/// Why a box's annotations do not resolve.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RandomizationError {
    #[error("duplicate annotation type {0:?} on a single box")]
    DuplicateAnnotation(AnnotationKind),
    #[error("InjectNoise requires a Twirl on the same box")]
    InjectNoiseWithoutTwirl,
}

/// How to synthesize virtual randomizations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SynthesizerType {
    RzSx,
    RzRx,
}

impl SynthesizerType {
    /// How many qubits one synthesis covers.
    pub fn num_qubits(&self) -> u32 {
        match self {
            Self::RzSx | Self::RzRx => 1,
        }
    }
}

/// A named distribution to sample virtual gates from.
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
    /// How many qubits one sample covers.
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

/// Which edge of a box its easy gates compose into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dressing {
    Left,
    Right,
}

/// Which side of a box's hard content an injection sits on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InjectionSite {
    Before,
    After,
}

/// Which annotation this is, or `None` if randomization does not own it.
pub fn annotation_kind(annotation: &dyn Annotation) -> Option<AnnotationKind> {
    if annotation.namespace() != NAMESPACE {
        return None;
    }
    if annotation.downcast_ref::<Twirl>().is_some() {
        Some(AnnotationKind::Twirl)
    } else if annotation.downcast_ref::<InjectNoise>().is_some() {
        Some(AnnotationKind::InjectNoise)
    } else {
        None
    }
}

/// Annotation type discriminant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnnotationKind {
    Twirl,
    InjectNoise,
}

/// Randomize this box by conjugating it with gates sampled from a distribution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Twirl {
    pub distribution: DistributionType,
    pub dressing: Dressing,
    pub decomposition: SynthesizerType,
}

/// Inject sampled noise into this box from a Pauli-Lindblad map the caller supplies.
///
/// `reference` names the map and `modifier` names an input to optionally scale its rates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InjectNoise {
    pub reference: String,
    pub modifier: Option<String>,
    pub site: InjectionSite,
}

impl Annotation for Twirl {
    fn namespace(&self) -> &str {
        NAMESPACE
    }
}

impl Annotation for InjectNoise {
    fn namespace(&self) -> &str {
        NAMESPACE
    }
}

/// What a box's annotations resolve to once the vocabulary's rules have been applied.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResolvedAnnotations {
    pub twirl: Option<Twirl>,
    pub inject_noise: Option<InjectNoise>,
}

impl ResolvedAnnotations {
    /// Whether this box produces an emission.
    pub fn is_emitting(&self) -> bool {
        self.twirl.is_some()
    }

    /// Which edge this box's easy gates compose into.
    pub fn dressing(&self) -> Option<Dressing> {
        self.twirl.as_ref().map(|twirl| twirl.dressing)
    }

    /// How the collectors of this box synthesize what they compose.
    pub fn synthesizer(&self) -> Option<SynthesizerType> {
        self.twirl.as_ref().map(|twirl| twirl.decomposition)
    }
}

/// Resolve a box's annotations, applying the vocabulary's validation rules.
pub fn resolve_annotations(annotations: &[Arc<dyn Annotation>]) -> Result<ResolvedAnnotations> {
    // `seen` is a membership test only, never iterated, so its `HashSet` cannot leak an ordering
    // into anything downstream.
    let mut seen: HashSet<AnnotationKind> = HashSet::new();
    for annotation in annotations {
        let Some(kind) = annotation_kind(annotation.as_ref()) else {
            continue;
        };
        if !seen.insert(kind) {
            return Err(RandomizationError::DuplicateAnnotation(kind));
        }
    }

    if seen.contains(&AnnotationKind::InjectNoise) && !seen.contains(&AnnotationKind::Twirl) {
        return Err(RandomizationError::InjectNoiseWithoutTwirl);
    }

    let mut resolved = ResolvedAnnotations::default();
    for annotation in annotations {
        let annotation = annotation.as_ref();
        if let Some(twirl) = annotation.downcast_ref::<Twirl>() {
            resolved.twirl = Some(twirl.clone());
        } else if let Some(injection) = annotation.downcast_ref::<InjectNoise>() {
            resolved.inject_noise = Some(injection.clone());
        }
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Erase a value to the trait object a box actually stores.
    fn erased(annotation: impl Annotation + 'static) -> Arc<dyn Annotation> {
        Arc::new(annotation)
    }

    fn twirl(distribution: DistributionType) -> Arc<dyn Annotation> {
        erased(Twirl {
            distribution,
            dressing: Dressing::Left,
            decomposition: SynthesizerType::RzSx,
        })
    }

    fn noise(site: InjectionSite) -> Arc<dyn Annotation> {
        erased(InjectNoise {
            reference: "layer_1".to_string(),
            modifier: None,
            site,
        })
    }

    /// An annotation from outside this vocabulary.
    #[derive(Debug, Clone, PartialEq)]
    struct Foreign;

    impl Annotation for Foreign {
        fn namespace(&self) -> &str {
            "someone.else"
        }
    }

    /// Something claiming the "randomization" namespace.
    #[derive(Debug, Clone, PartialEq)]
    struct Impostor;

    impl Annotation for Impostor {
        fn namespace(&self) -> &str {
            NAMESPACE
        }
    }

    #[test]
    fn test_a_twirl_resolves_to_a_dressing_and_a_synthesizer() {
        let resolved = resolve_annotations(&[twirl(DistributionType::UniformPauli)]).unwrap();
        assert!(resolved.is_emitting());
        assert_eq!(resolved.dressing(), Some(Dressing::Left));
        assert_eq!(resolved.synthesizer(), Some(SynthesizerType::RzSx));
        assert!(resolved.inject_noise.is_none());
    }

    #[test]
    fn test_a_box_with_nothing_of_ours_on_it_resolves_to_nothing() {
        let resolved = resolve_annotations(&[erased(Foreign)]).unwrap();
        assert!(!resolved.is_emitting());
        assert_eq!(resolved.dressing(), None);
        assert_eq!(resolved.synthesizer(), None);
        assert_eq!(resolved, ResolvedAnnotations::default());
    }

    #[test]
    fn test_an_unannotated_box_resolves_to_nothing() {
        assert_eq!(
            resolve_annotations(&[]).unwrap(),
            ResolvedAnnotations::default()
        );
    }

    #[test]
    fn test_noise_resolves_beside_a_twirl() {
        let resolved = resolve_annotations(&[
            twirl(DistributionType::UniformPauli),
            noise(InjectionSite::Before),
        ])
        .unwrap();
        assert!(resolved.twirl.is_some());
        let injection = resolved.inject_noise.expect("the injection resolves");
        assert_eq!(injection.reference, "layer_1");
        assert_eq!(injection.site, InjectionSite::Before);
        assert_eq!(injection.modifier, None);
    }

    #[test]
    fn test_noise_without_a_twirl_is_refused() {
        assert_eq!(
            resolve_annotations(&[noise(InjectionSite::After)]).unwrap_err(),
            RandomizationError::InjectNoiseWithoutTwirl
        );
    }

    #[test]
    fn test_two_of_one_kind_on_a_box_is_refused() {
        assert_eq!(
            resolve_annotations(&[
                twirl(DistributionType::UniformPauli),
                twirl(DistributionType::HaarU2),
            ])
            .unwrap_err(),
            RandomizationError::DuplicateAnnotation(AnnotationKind::Twirl)
        );
    }

    #[test]
    fn test_claiming_the_namespace_is_not_enough_to_be_one_of_ours() {
        assert_eq!(annotation_kind(&Impostor), None);
        let resolved =
            resolve_annotations(&[erased(Impostor), twirl(DistributionType::UniformPauli)])
                .unwrap();
        assert!(resolved.is_emitting());
    }

    #[test]
    fn test_a_foreign_annotation_rides_through() {
        assert_eq!(annotation_kind(&Foreign), None);
        let resolved = resolve_annotations(&[
            erased(Foreign),
            twirl(DistributionType::UniformPauli),
            erased(Foreign),
        ])
        .unwrap();
        assert!(resolved.is_emitting());
    }

    #[test]
    fn test_the_annotations_share_one_flat_namespace() {
        for annotation in [
            twirl(DistributionType::UniformPauli),
            noise(InjectionSite::Before),
        ] {
            assert_eq!(annotation.namespace(), NAMESPACE);
        }
    }

    #[test]
    fn test_only_local_c1_covers_two_qubits() {
        for distribution in [
            DistributionType::UniformPauli,
            DistributionType::BalancedUniformPauli,
            DistributionType::UniformPauliSubset,
            DistributionType::UniformC1,
            DistributionType::HaarU2,
        ] {
            assert_eq!(distribution.num_qubits(), 1, "{distribution:?}");
        }
        assert_eq!(DistributionType::UniformLocalC1.num_qubits(), 2);
    }

    #[test]
    fn test_a_synthesizer_covers_one_qubit() {
        for synthesizer in [SynthesizerType::RzSx, SynthesizerType::RzRx] {
            assert_eq!(synthesizer.num_qubits(), 1, "{synthesizer:?}");
        }
    }
}
