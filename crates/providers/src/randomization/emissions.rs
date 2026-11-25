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

use std::ops::Range;
use std::sync::Arc;

use thiserror::Error;

use qiskit_circuit::operations::{CustomOperation, Operation};
use qiskit_circuit::packed_instruction::PackedInstruction;

use super::Direction;
use super::distributions::Distribution;
use super::synthesizer::Synthesizer;

/// The name of the emit operation.
pub const EMIT_NAME: &str = "emit";

/// The name of the collect operation.
pub const COLLECT_NAME: &str = "collect";

/// Why an emit could not be built.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EmitError {
    #[error("an emit must place at least one factor")]
    NoFactors,
    #[error("an emit contains factor {0} more than once")]
    DuplicateFactor(u32),
    #[error("factor {factor} is not one of the distribution's {num_factors}")]
    UnknownFactor { factor: u32, num_factors: usize },
    #[error("a regrouping must place each of the emit's factors exactly once")]
    IncompleteRegrouping,
}

/// An emit operation.
///
/// This operation corresponds to inserting samples from `distribution` as virtual gates on
/// qubits. These gates are moved in `direction` through other operations until they can be
/// implemented in a [`Collect`].
#[derive(Debug, Clone)]
pub struct Emit {
    pub direction: Direction,
    /// The distribution that this emit's samples are drawn from.
    pub distribution: Arc<dyn Distribution>,
    /// Which factors of the distribution this emit places.
    factors: Vec<u32>,
    /// Whether to invert the sample before composing it.
    pub adjoint: bool,
}

impl Emit {
    /// Place `factors` of `distribution`, in qubit order.
    pub fn new(
        direction: Direction,
        distribution: Arc<dyn Distribution>,
        factors: impl IntoIterator<Item = u32>,
        adjoint: bool,
    ) -> Result<Self, EmitError> {
        let factors: Vec<u32> = factors.into_iter().collect();
        if factors.is_empty() {
            return Err(EmitError::NoFactors);
        }
        let num_factors = distribution.num_factors();
        for (placed, &factor) in factors.iter().enumerate() {
            if factor as usize >= num_factors {
                return Err(EmitError::UnknownFactor {
                    factor,
                    num_factors,
                });
            }
            if factors[..placed].contains(&factor) {
                return Err(EmitError::DuplicateFactor(factor));
            }
        }
        Ok(Emit {
            direction,
            distribution,
            factors,
            adjoint,
        })
    }

    /// The factors this emit contains.
    pub fn factors(&self) -> &[u32] {
        &self.factors
    }

    /// Whether this and `other` are drawn from the same distribution.
    pub fn same_sample(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.distribution, &other.distribution)
    }

    /// How wide each factor is.
    pub fn factor_widths(&self) -> impl Iterator<Item = u32> + '_ {
        self.factors.iter().map(|&factor| {
            self.distribution
                .factor_qubits(factor)
                .expect("an emit places factors its distribution has")
                .len() as u32
        })
    }

    /// Where each placed factor sits in this emit's qargs.
    pub fn qubit_offsets(&self) -> Vec<Range<u32>> {
        let mut offset = 0;
        self.factor_widths()
            .map(|width| {
                let start = offset;
                offset += width;
                start..offset
            })
            .collect()
    }

    /// The offsets into this emit's qargs that `factors` take, in the order given, or `None`
    /// if it does not place one of them.
    pub fn offsets_of(&self, factors: &[u32]) -> Option<Vec<u32>> {
        let offsets = self.qubit_offsets();
        let mut taken = Vec::new();
        for &factor in factors {
            let at = self.factors.iter().position(|&own| own == factor)?;
            taken.extend(offsets[at].clone());
        }
        Some(taken)
    }

    /// Break this emit into `groups`.
    ///
    /// Together the groups must place each of this emit's factors exactly once. Every piece
    /// shares this emit's sample and [`Emit::offsets_of`] gives the qargs each one takes.
    pub fn regroup(&self, groups: &[Vec<u32>]) -> Result<Vec<Emit>, EmitError> {
        let mut seen = vec![false; self.factors.len()];
        let mut pieces = Vec::with_capacity(groups.len());
        for group in groups {
            for &factor in group {
                let at = self
                    .factors
                    .iter()
                    .position(|&own| own == factor)
                    .ok_or(EmitError::IncompleteRegrouping)?;
                if std::mem::replace(&mut seen[at], true) {
                    return Err(EmitError::DuplicateFactor(factor));
                }
            }
            pieces.push(Emit::new(
                self.direction,
                Arc::clone(&self.distribution),
                group.iter().copied(),
                self.adjoint,
            )?);
        }
        if !seen.iter().all(|&placed| placed) {
            return Err(EmitError::IncompleteRegrouping);
        }
        Ok(pieces)
    }

    /// Break this emit all the way apart, one piece per factor.
    pub fn explode(&self) -> Vec<Emit> {
        let groups: Vec<Vec<u32>> = self.factors.iter().map(|&factor| vec![factor]).collect();
        self.regroup(&groups)
            .expect("an emit's own factors regroup it")
    }
}

/// Two emits are equal when they contain the same factors of the same sample the same way.
impl PartialEq for Emit {
    fn eq(&self, other: &Self) -> bool {
        self.same_sample(other)
            && self.direction == other.direction
            && self.factors == other.factors
            && self.adjoint == other.adjoint
    }
}

impl Eq for Emit {}

impl Operation for Emit {
    fn name(&self) -> &str {
        EMIT_NAME
    }

    /// As wide as the factors it places, in total.
    fn num_qubits(&self) -> u32 {
        self.factor_widths().sum()
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

/// Why an instruction could not be collected.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CollectError {
    #[error("an instruction on {expected} qubits was given {given}")]
    WrongWidth { expected: u32, given: usize },
    #[error("qubit {0} is not one of the collect's")]
    UnknownQubit(u32),
    #[error("an instruction acts on qubit {0} twice")]
    DuplicateQubit(u32),
}

/// Gather and synthesize virtual gates over a span of qubits.
#[derive(Debug, Clone)]
pub struct Collect {
    /// How gathered gates are synthesized.
    pub synthesizer: Arc<dyn Synthesizer>,
    /// The gathered instructions in multiplication order.
    instructions: Vec<PackedInstruction>,
    /// Where each instruction's row of `qubit_idxs` ends, after a leading zero.
    offsets: Vec<u32>,
    /// The qubit indices each instruction acts on.
    qubit_idxs: Vec<u32>,
}

impl PartialEq for Collect {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl Collect {
    /// A collect with nothing gathered yet, spanning whatever its synthesizer does.
    pub fn new(synthesizer: Arc<dyn Synthesizer>) -> Self {
        Collect {
            synthesizer,
            instructions: Vec::new(),
            offsets: vec![0],
            qubit_idxs: Vec::new(),
        }
    }

    /// Gather `instruction` acting on the collect's `qubit_idxs`.
    pub fn push(
        &mut self,
        qubit_idxs: impl IntoIterator<Item = u32>,
        instruction: PackedInstruction,
    ) -> Result<(), CollectError> {
        let qubit_idxs: Vec<u32> = qubit_idxs.into_iter().collect();
        let expected = instruction.op.num_qubits();
        if qubit_idxs.len() != expected as usize {
            return Err(CollectError::WrongWidth {
                expected,
                given: qubit_idxs.len(),
            });
        }
        let span = self.num_qubits();
        for (seen, &qubit) in qubit_idxs.iter().enumerate() {
            if qubit >= span {
                return Err(CollectError::UnknownQubit(qubit));
            }
            if qubit_idxs[..seen].contains(&qubit) {
                return Err(CollectError::DuplicateQubit(qubit));
            }
        }
        self.qubit_idxs.extend_from_slice(&qubit_idxs);
        self.offsets.push(self.qubit_idxs.len() as u32);
        self.instructions.push(instruction);
        Ok(())
    }

    /// How many instructions have been gathered.
    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    /// Whether nothing has been gathered.
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }

    /// The instruction gathered at `step` with the qubit_idxs it acts on.
    pub fn step(&self, step: usize) -> Option<(&[u32], &PackedInstruction)> {
        let start = *self.offsets.get(step)? as usize;
        let end = *self.offsets.get(step + 1)? as usize;
        Some((&self.qubit_idxs[start..end], &self.instructions[step]))
    }

    /// Every gathered instruction with the qubit_idxs it acts on, in order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&[u32], &PackedInstruction)> {
        (0..self.instructions.len()).map(|step| {
            let start = self.offsets[step] as usize;
            let end = self.offsets[step + 1] as usize;
            (&self.qubit_idxs[start..end], &self.instructions[step])
        })
    }
}

impl Operation for Collect {
    fn name(&self) -> &str {
        COLLECT_NAME
    }

    /// As wide as its synthesizer spans, whatever it has gathered.
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
    use qiskit_circuit::operations::StandardGate;

    use super::super::distributions::Noise;
    use super::super::partition::Partition;
    use super::super::{DistributionKind, SynthesizerKind};
    use super::*;

    fn gate(gate: StandardGate) -> PackedInstruction {
        PackedInstruction::from_standard_gate(gate, None, Default::default())
    }

    fn singletons(width: u32) -> Partition<u32> {
        Partition::singletons(0..width).expect("a range of qubits has no repeats")
    }

    fn noise(width: u32) -> Arc<dyn Distribution> {
        Noise::new("layer_1", None, singletons(width))
            .map(|noise| Arc::new(noise) as Arc<dyn Distribution>)
            .expect("noise resolves over singletons")
    }

    fn paulis(width: u32) -> Arc<dyn Distribution> {
        DistributionKind::UniformPauli
            .resolve(singletons(width))
            .expect("uniform Paulis resolve over singletons")
    }

    fn emit(distribution: Arc<dyn Distribution>, factors: impl IntoIterator<Item = u32>) -> Emit {
        Emit::new(Direction::Forward, distribution, factors, false).expect("a valid emit")
    }

    fn collect(kind: SynthesizerKind, num_qubits: u32) -> Collect {
        Collect::new(kind.resolve(num_qubits))
    }

    #[test]
    fn test_an_emit_is_as_wide_as_the_factors_it_places() {
        assert_eq!(emit(paulis(1), [0]).num_qubits(), 1);
        assert_eq!(emit(noise(3), [0]).num_qubits(), 1);
        assert_eq!(emit(noise(3), [0, 2]).num_qubits(), 2);
        assert_eq!(emit(noise(3), [0, 1, 2]).num_qubits(), 3);
    }

    #[test]
    fn test_an_emit_lays_its_factors_out_in_the_order_it_was_given() {
        let emit = emit(noise(4), [3, 0, 2]);
        assert_eq!(emit.factors(), [3, 0, 2]);
        assert_eq!(emit.factor_widths().collect::<Vec<_>>(), [1, 1, 1]);
        assert_eq!(emit.qubit_offsets(), [0..1, 1..2, 2..3]);
        assert_eq!(emit.num_qubits(), 3);
    }

    #[test]
    fn test_a_layer_of_independent_paulis_is_one_emit() {
        let emit = emit(paulis(5), 0..5);
        assert_eq!(emit.num_qubits(), 5);
        assert_eq!(emit.factors(), [0, 1, 2, 3, 4]);
        let per_qubit = emit.explode();
        assert_eq!(per_qubit.len(), 5);
        for (qubit, piece) in per_qubit.iter().enumerate() {
            assert!(emit.same_sample(piece));
            assert_eq!(piece.num_qubits(), 1);
            assert_eq!(
                emit.offsets_of(piece.factors())
                    .expect("a piece of this emit"),
                [qubit as u32]
            );
        }
    }

    #[test]
    fn test_the_order_of_placed_factors_is_meaningful() {
        let sample = noise(2);
        let forward = Emit::new(Direction::Forward, Arc::clone(&sample), [0, 1], false).unwrap();
        let reversed = Emit::new(Direction::Forward, Arc::clone(&sample), [1, 0], false).unwrap();
        assert!(forward.same_sample(&reversed));
        assert_ne!(forward, reversed);
    }

    #[test]
    fn test_an_emit_must_place_factors_its_distribution_has() {
        let sample = noise(2);
        assert_eq!(
            Emit::new(Direction::Forward, Arc::clone(&sample), [], false),
            Err(EmitError::NoFactors)
        );
        assert_eq!(
            Emit::new(Direction::Forward, Arc::clone(&sample), [0, 0], false),
            Err(EmitError::DuplicateFactor(0))
        );
        assert_eq!(
            Emit::new(Direction::Forward, Arc::clone(&sample), [2], false),
            Err(EmitError::UnknownFactor {
                factor: 2,
                num_factors: 2
            })
        );
        // An empty noise map has no factors, so nothing can emit one.
        assert_eq!(
            Emit::new(Direction::Forward, noise(0), [0], false),
            Err(EmitError::UnknownFactor {
                factor: 0,
                num_factors: 0
            })
        );
    }

    #[test]
    fn test_exploding_keeps_every_piece_in_the_one_sample() {
        let emit = emit(noise(3), [0, 1, 2]);
        let pieces = emit.explode();
        assert_eq!(pieces.len(), 3);
        for piece in &pieces {
            assert!(emit.same_sample(piece));
            let qubits = emit
                .offsets_of(piece.factors())
                .expect("a piece of this emit");
            assert_eq!(piece.num_qubits(), qubits.len() as u32);
        }
        let placed: Vec<&[u32]> = pieces.iter().map(|piece| piece.factors()).collect();
        assert_eq!(placed, [[0].as_slice(), [1].as_slice(), [2].as_slice()]);
    }

    #[test]
    fn test_the_qargs_a_factor_takes_follow_the_layout_not_its_number() {
        let emit = emit(noise(4), [3, 0, 2]);
        assert_eq!(emit.offsets_of(&[3]).unwrap(), [0]);
        assert_eq!(emit.offsets_of(&[0]).unwrap(), [1]);
        assert_eq!(emit.offsets_of(&[2]).unwrap(), [2]);
        assert_eq!(emit.offsets_of(&[2, 3]).unwrap(), [2, 0]);
        // Factor 1 is the distribution's, but this emit does not place it.
        assert_eq!(emit.offsets_of(&[1]), None);
    }

    #[test]
    fn test_a_regrouping_splits_off_only_what_it_names() {
        let emit = emit(noise(3), [0, 1, 2]);
        let pieces = emit.regroup(&[vec![0, 2], vec![1]]).unwrap();
        assert_eq!(pieces.len(), 2);
        assert_eq!(pieces[0].factors(), [0, 2]);
        assert_eq!(pieces[0].num_qubits(), 2);
        assert_eq!(emit.offsets_of(pieces[0].factors()).unwrap(), [0, 2]);
        assert_eq!(pieces[1].factors(), [1]);
        assert_eq!(emit.offsets_of(pieces[1].factors()).unwrap(), [1]);
        assert!(emit.same_sample(&pieces[0]));
        assert!(emit.same_sample(&pieces[1]));
    }

    #[test]
    fn test_a_regrouping_must_place_every_factor_exactly_once() {
        let whole = emit(noise(3), [0, 1, 2]);
        let partial = emit(noise(3), [0, 1]);
        assert_eq!(
            whole.regroup(&[vec![0, 1]]),
            Err(EmitError::IncompleteRegrouping)
        );
        assert_eq!(
            whole.regroup(&[vec![0, 0], vec![1, 2]]),
            Err(EmitError::DuplicateFactor(0))
        );
        assert_eq!(
            partial.regroup(&[vec![0, 1], vec![2]]),
            Err(EmitError::IncompleteRegrouping),
            "factor 2 is the distribution's but this emit does not place it"
        );
    }

    #[test]
    fn test_emits_of_one_sample_share_it_and_differ_otherwise() {
        let sample = paulis(1);
        let near = Emit::new(Direction::Forward, Arc::clone(&sample), [0], false).unwrap();
        let far = Emit::new(Direction::Forward, Arc::clone(&sample), [0], true).unwrap();
        assert!(near.same_sample(&far));
        assert_ne!(near, far);
    }

    #[test]
    fn test_emits_are_told_apart_by_the_way_they_flow() {
        let sample = paulis(1);
        let forward = Emit::new(Direction::Forward, Arc::clone(&sample), [0], false).unwrap();
        let backward = Emit::new(Direction::Backward, Arc::clone(&sample), [0], false).unwrap();
        assert!(forward.same_sample(&backward));
        assert_ne!(forward, backward);
    }

    #[test]
    fn test_noise_emits_are_told_apart_by_their_factors() {
        let sample = noise(2);
        let first = Emit::new(Direction::Forward, Arc::clone(&sample), [0], false).unwrap();
        let second = Emit::new(Direction::Forward, Arc::clone(&sample), [1], false).unwrap();
        assert!(first.same_sample(&second));
        assert_ne!(first, second);
    }

    #[test]
    fn test_two_samples_from_one_distribution_are_independent() {
        let first = emit(paulis(1), [0]);
        let second = emit(paulis(1), [0]);
        assert!(!first.same_sample(&second));
        assert_ne!(first, second);
        // What they were drawn from is the same; the samples are not.
        assert_eq!(
            first.distribution.virtual_type(),
            second.distribution.virtual_type()
        );
        assert_eq!(
            first.distribution.factorization(),
            second.distribution.factorization()
        );
    }

    #[test]
    fn test_sharing_the_allocation_is_the_only_way_to_share_a_sample() {
        // Identical in every field, but each built its own sample.
        let separate = [emit(noise(2), [0, 1]), emit(noise(2), [0, 1])];
        assert!(!separate[0].same_sample(&separate[1]));
        assert_eq!(
            separate[0].distribution.factorization(),
            separate[1].distribution.factorization()
        );
        assert_ne!(separate[0], separate[1]);

        let sample = noise(2);
        let shared = |factor| {
            Emit::new(Direction::Forward, Arc::clone(&sample), [factor], false)
                .expect("a valid factor")
        };
        assert!(shared(0).same_sample(&shared(1)));
        assert_eq!(shared(0), shared(0));
        assert_ne!(shared(0), shared(1));
    }

    #[test]
    fn test_a_collect_starts_empty_and_as_wide_as_it_spans() {
        let collect = collect(SynthesizerKind::RzSx, 3);
        assert!(collect.is_empty());
        assert_eq!(collect.len(), 0);
        assert_eq!(collect.num_qubits(), 3);
        assert_eq!(collect.name(), COLLECT_NAME);
        assert!(collect.step(0).is_none());
        assert_eq!(collect.iter().len(), 0);
    }

    #[test]
    fn test_a_collect_gathers_instructions_with_the_qubits_they_act_on() {
        let mut collect = collect(SynthesizerKind::RzSx, 3);
        collect.push([0], gate(StandardGate::H)).unwrap();
        collect.push([2, 1], gate(StandardGate::CX)).unwrap();
        assert_eq!(collect.len(), 2);
        assert!(!collect.is_empty());

        let (qubits, instruction) = collect.step(0).expect("a gathered step");
        assert_eq!(qubits, [0]);
        assert_eq!(instruction.op.num_qubits(), 1);
        let (qubits, instruction) = collect.step(1).expect("a gathered step");
        assert_eq!(qubits, [2, 1]);
        assert_eq!(instruction.op.num_qubits(), 2);
        assert!(collect.step(2).is_none());

        let rows: Vec<&[u32]> = collect.iter().map(|(qubits, _)| qubits).collect();
        assert_eq!(rows, [[0].as_slice(), [2, 1].as_slice()]);
    }

    #[test]
    fn test_a_collect_refuses_what_it_cannot_place_and_keeps_nothing() {
        let mut collect = collect(SynthesizerKind::RzSx, 2);
        assert_eq!(
            collect.push([0, 1], gate(StandardGate::H)),
            Err(CollectError::WrongWidth {
                expected: 1,
                given: 2
            })
        );
        assert_eq!(
            collect.push([2], gate(StandardGate::H)),
            Err(CollectError::UnknownQubit(2))
        );
        assert_eq!(
            collect.push([1, 1], gate(StandardGate::CX)),
            Err(CollectError::DuplicateQubit(1))
        );
        assert!(collect.is_empty());
        assert_eq!(collect.num_qubits(), 2);
    }

    #[test]
    fn test_neither_operation_is_unitary() {
        assert!(!emit(noise(1), [0]).is_unitary());
        assert!(!collect(SynthesizerKind::RzRx, 1).is_unitary());
    }
}
