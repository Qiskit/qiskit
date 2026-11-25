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

use crate::ops::{Cast, UniformInt};
use crate::program::ProgramFunction;
use crate::randomization::partition::Partition;
use crate::randomization::virtual_type::VirtualType;
use crate::tensor::{DType, Dim, PrngKind, TensorType};

use super::Distribution;

/// How many Paulis one qubit has up to phase.
const NUM_PAULIS: i64 = 4;

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

    /// Draw one Pauli per factor.
    ///
    /// The sample's last axis indexes the factors. A draw is a `u8` in `0..4`. Bit 0 is the `z`
    /// bit and bit 1 is the `x` bit. So `0b00` is the identity and `0b11` is `Y`. `quantum_info`
    /// encodes its sparse Pauli terms the same way.
    ///
    /// Every factor draws from its own substream of its own key. [`UniformInt`] places those
    /// substreams so a factor's draw depends on neither the number of factors nor the batch.
    fn program_function(&self, batch: &[Dim]) -> ProgramFunction {
        let mut function = ProgramFunction::new();
        let mut shape = batch.to_vec();
        // The factors broadcast into this axis, which right-aligns them past the batch.
        shape.push(Dim::Fixed(1));
        let key = function.add_parameter(TensorType {
            dtype: DType::Key(PrngKind::Philox4x32),
            shape,
        });
        let draw = UniformInt::new(vec![self.num_factors()], 0, NUM_PAULIS - 1)
            .expect("0 does not exceed the highest Pauli code");
        let drawn = function
            .add_op(draw, &[key])
            .expect("a uniform draw takes one key and this function has one");
        let paulis = function
            .add_op(Cast::new(self.virtual_type().dtype()), &drawn)
            .expect("an integer draw casts to the dtype a Pauli is stored as");
        function
            .add_result(paulis[0])
            .expect("the cast produces a value of this function");
        function
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

    fn program_function(&self, _batch: &[Dim]) -> ProgramFunction {
        todo!("drawing a Haar-random U2 is not implemented yet")
    }
}

#[cfg(test)]
mod tests {
    use ndarray::IxDyn;

    use super::*;
    use crate::tensor::{KeyTensor, Tensor};

    fn paulis(num_qubits: u32) -> UniformPauli {
        let factorization =
            Partition::singletons(0..num_qubits).expect("a range of qubits has no repeats");
        UniformPauli::new(factorization).expect("uniform Paulis resolve over singletons")
    }

    fn key_type(shape: Vec<Dim>) -> TensorType {
        TensorType {
            dtype: DType::Key(PrngKind::Philox4x32),
            shape,
        }
    }

    fn keys(seeds: &[u32]) -> Tensor {
        let mut words = Vec::with_capacity(seeds.len() * 2);
        for &seed in seeds {
            words.push(seed);
            words.push(0x5eed);
        }
        let shape = IxDyn(&[seeds.len(), 1, 2]);
        let words = ndarray::ArrayD::from_shape_vec(shape, words)
            .expect("two words make one philox key")
            .into_shared();
        Tensor::Key(KeyTensor::new(PrngKind::Philox4x32, words).expect("keys of two words"))
    }

    fn sample(function: &ProgramFunction, seeds: &[u32]) -> Vec<u8> {
        let mut outputs = function.eval(&[keys(seeds)]).expect("the draw evaluates");
        assert_eq!(outputs.len(), 1, "a sample is one tensor");
        let Tensor::U8(sample) = outputs.pop().expect("one output") else {
            panic!("a Pauli sample is stored as u8")
        };
        sample.iter().copied().collect()
    }

    fn drawn(num_qubits: u32, seed: u32) -> Vec<u8> {
        let function = paulis(num_qubits).program_function(&[Dim::Fixed(1)]);
        sample(&function, &[seed])
    }

    #[test]
    fn test_the_function_takes_a_key_and_returns_a_pauli_per_factor() {
        let signature = paulis(3).program_function(&[]).signature();
        assert_eq!(signature.inputs, vec![key_type(vec![Dim::Fixed(1)])]);
        assert_eq!(
            signature.outputs,
            vec![TensorType {
                dtype: VirtualType::Pauli.dtype(),
                shape: vec![Dim::Fixed(3)],
            }]
        );
    }

    #[test]
    fn test_a_batch_becomes_the_leading_axes_of_the_sample() {
        let signature = paulis(4).program_function(&[Dim::Fixed(8)]).signature();
        assert_eq!(
            signature.inputs,
            vec![key_type(vec![Dim::Fixed(8), Dim::Fixed(1)])]
        );
        assert_eq!(
            signature.outputs[0].shape,
            vec![Dim::Fixed(8), Dim::Fixed(4)]
        );
    }

    #[test]
    fn test_several_batch_axes_all_lead() {
        let batch = [Dim::Fixed(2), Dim::Fixed(3)];
        let signature = paulis(4).program_function(&batch).signature();
        assert_eq!(
            signature.inputs,
            vec![key_type(vec![Dim::Fixed(2), Dim::Fixed(3), Dim::Fixed(1)])]
        );
        assert_eq!(
            signature.outputs[0].shape,
            vec![Dim::Fixed(2), Dim::Fixed(3), Dim::Fixed(4)]
        );
    }

    #[test]
    fn test_a_batch_axis_may_be_bounded_rather_than_fixed() {
        let signature = paulis(4)
            .program_function(&[Dim::Bounded { max: 8 }])
            .signature();
        assert_eq!(
            signature.inputs,
            vec![key_type(vec![Dim::Bounded { max: 8 }, Dim::Fixed(1)])]
        );
        assert_eq!(
            signature.outputs[0].shape,
            vec![Dim::Bounded { max: 8 }, Dim::Fixed(4)]
        );
    }

    #[test]
    fn test_the_sample_is_stored_as_the_virtual_type_says() {
        let signature = paulis(1).program_function(&[]).signature();
        assert_eq!(signature.outputs[0].dtype, DType::U8);
    }

    #[test]
    fn test_every_drawn_pauli_is_one_of_the_four() {
        let sample = drawn(64, 7);
        assert_eq!(sample.len(), 64);
        assert!(
            sample.iter().all(|&pauli| i64::from(pauli) < NUM_PAULIS),
            "a draw outside 0..4 is not a Pauli code, got {sample:?}"
        );
    }

    #[test]
    fn test_all_four_paulis_are_drawn_over_enough_factors() {
        let sample = drawn(256, 11);
        for pauli in 0..NUM_PAULIS as u8 {
            assert!(
                sample.contains(&pauli),
                "{pauli} never came up over 256 uniform draws"
            );
        }
    }

    #[test]
    fn test_a_distribution_over_no_qubits_draws_an_empty_sample() {
        let function = paulis(0).program_function(&[]);
        assert_eq!(function.signature().outputs[0].shape, vec![Dim::Fixed(0)]);
        assert_eq!(drawn(0, 3), Vec::<u8>::new());
    }

    #[test]
    fn test_one_key_always_draws_the_same_sample() {
        assert_eq!(drawn(8, 42), drawn(8, 42));
    }

    #[test]
    fn test_different_keys_draw_different_samples() {
        assert_ne!(drawn(16, 1), drawn(16, 2));
    }

    #[test]
    fn test_a_factor_draws_the_same_whatever_the_width() {
        let wide = drawn(32, 19);
        for width in 1..32u32 {
            assert_eq!(
                drawn(width, 19),
                wide[..width as usize],
                "factor draws moved at width {width}"
            );
        }
    }

    #[test]
    fn test_each_batch_position_draws_from_its_own_key() {
        let seeds = [5, 6, 7];
        let function = paulis(16).program_function(&[Dim::Fixed(3)]);
        let batched = sample(&function, &seeds);
        assert_eq!(batched.len(), 48);
        for (position, &seed) in seeds.iter().enumerate() {
            assert_eq!(
                &batched[position * 16..][..16],
                drawn(16, seed),
                "batch position {position} did not draw its own key's sample"
            );
        }
    }

    #[test]
    fn test_a_factor_draws_the_same_whatever_the_batch() {
        let seeds = [5, 6, 7];
        let narrow = paulis(16).program_function(&[Dim::Fixed(1)]);
        let wide = paulis(16).program_function(&[Dim::Fixed(3)]);
        let batched = sample(&wide, &seeds);
        assert_eq!(sample(&narrow, &seeds[..1]), &batched[..16]);
    }
}
