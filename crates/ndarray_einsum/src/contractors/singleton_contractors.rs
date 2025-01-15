// Copyright 2019 Jared Samet
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Contains the specific implementations of `SingletonContractor` and `SingletonViewer` that
//! represent the base-case ways to contract or simplify a single tensor.
//!
//! All the structs here perform perform some combination of
//! permutation of the input axes (e.g. `ijk->jki`), diagonalization across repeated but
//! un-summed axes (e.g. `ii->i`),
//! and summation across axes not present in the output index list (e.g. `ijk->j`).

use ndarray::prelude::*;
use ndarray::LinalgScalar;

use super::{SingletonContractor, SingletonViewer};
use crate::{Contraction, SizedContraction};

/// Returns a view or clone of the input tensor.
///
/// Example: `ij->ij`
#[derive(Clone, Debug)]
pub struct Identity {}

impl Identity {
    pub fn new(_sc: &SizedContraction) -> Self {
        Identity {}
    }
}

impl<A> SingletonViewer<A> for Identity {
    fn view_singleton<'a, 'b>(&self, tensor: &'b ArrayViewD<'a, A>) -> ArrayViewD<'b, A>
    where
        'a: 'b,
        A: Clone + LinalgScalar,
    {
        tensor.view()
    }
}

impl<A> SingletonContractor<A> for Identity {
    fn contract_singleton<'a, 'b>(&self, tensor: &'b ArrayViewD<'a, A>) -> ArrayD<A>
    where
        'a: 'b,
        A: Clone + LinalgScalar,
    {
        tensor.to_owned()
    }
}

/// Permutes the axes of the input tensor and returns a view or clones the elements.
///
/// Example: `ij->ji`
#[derive(Clone, Debug)]
pub struct Permutation {
    permutation: Vec<usize>,
}

impl Permutation {
    pub fn new(sc: &SizedContraction) -> Self {
        let SizedContraction {
            contraction:
                Contraction {
                    ref operand_indices,
                    ref output_indices,
                    ..
                },
            ..
        } = sc;

        assert_eq!(operand_indices.len(), 1);
        assert_eq!(operand_indices[0].len(), output_indices.len());

        let mut permutation = Vec::new();
        for &c in output_indices.iter() {
            permutation.push(operand_indices[0].iter().position(|&x| x == c).unwrap());
        }

        Permutation { permutation }
    }

    pub fn from_indices(permutation: &[usize]) -> Self {
        Permutation {
            permutation: permutation.to_vec(),
        }
    }
}

impl<A> SingletonViewer<A> for Permutation {
    fn view_singleton<'a, 'b>(&self, tensor: &'b ArrayViewD<'a, A>) -> ArrayViewD<'b, A>
    where
        'a: 'b,
        A: Clone + LinalgScalar,
    {
        tensor.view().permuted_axes(IxDyn(&self.permutation))
    }
}

impl<A> SingletonContractor<A> for Permutation {
    fn contract_singleton<'a, 'b>(&self, tensor: &'b ArrayViewD<'a, A>) -> ArrayD<A>
    where
        'a: 'b,
        A: Clone + LinalgScalar,
    {
        tensor
            .view()
            .permuted_axes(IxDyn(&self.permutation))
            .to_owned()
    }
}

/// Sums across the elements of the input tensor that don't appear in the output tensor.
///
/// Example: `ij->i`
#[derive(Clone, Debug)]
pub struct Summation {
    adjusted_axis_list: Vec<usize>,
}

impl Summation {
    pub fn new(sc: &SizedContraction) -> Self {
        let output_indices = &sc.contraction.output_indices;
        let input_indices = &sc.contraction.operand_indices[0];

        Summation::from_sizes(
            output_indices.len(),
            input_indices.len() - output_indices.len(),
        )
    }

    fn from_sizes(start_index: usize, num_summed_axes: usize) -> Self {
        assert!(num_summed_axes >= 1);
        let adjusted_axis_list = (0..num_summed_axes).map(|_| start_index).collect();

        Summation { adjusted_axis_list }
    }
}

impl<A> SingletonContractor<A> for Summation {
    fn contract_singleton<'a, 'b>(&self, tensor: &'b ArrayViewD<'a, A>) -> ArrayD<A>
    where
        'a: 'b,
        A: Clone + LinalgScalar,
    {
        let mut result = tensor.sum_axis(Axis(self.adjusted_axis_list[0]));
        for &axis in self.adjusted_axis_list[1..].iter() {
            result = result.sum_axis(Axis(axis));
        }
        result
    }
}

/// Returns the elements of the input tensor where all instances of the repeated indices are equal to one another.
/// Optionally permutes the axes of the tensor as well.
///
/// Examples:
///
/// 1. `ii->i`
/// 2. `iij->ji`
#[derive(Clone, Debug)]
pub struct Diagonalization {
    input_to_output_mapping: Vec<usize>,
    output_shape: Vec<usize>,
}

impl Diagonalization {
    pub fn new(sc: &SizedContraction) -> Self {
        let SizedContraction {
            contraction:
                Contraction {
                    ref operand_indices,
                    ref output_indices,
                    ..
                },
            ref output_size,
        } = sc;
        assert_eq!(operand_indices.len(), 1);

        let mut adjusted_output_indices = output_indices.clone();
        let mut input_to_output_mapping = Vec::new();
        for &c in operand_indices[0].iter() {
            let current_length = adjusted_output_indices.len();
            match adjusted_output_indices.iter().position(|&x| x == c) {
                Some(pos) => {
                    input_to_output_mapping.push(pos);
                }
                None => {
                    adjusted_output_indices.push(c);
                    input_to_output_mapping.push(current_length);
                }
            }
        }
        let output_shape = adjusted_output_indices
            .iter()
            .map(|c| output_size[c])
            .collect();

        Diagonalization {
            input_to_output_mapping,
            output_shape,
        }
    }
}

impl<A> SingletonViewer<A> for Diagonalization {
    fn view_singleton<'a, 'b>(&self, tensor: &'b ArrayViewD<'a, A>) -> ArrayViewD<'b, A>
    where
        'a: 'b,
        A: Clone + LinalgScalar,
    {
        // Construct the stride array on the fly by enumerating (idx, stride) from strides() and
        // adding stride to self.which_index_is_this
        let mut strides = vec![0; self.output_shape.len()];
        for (idx, &stride) in tensor.strides().iter().enumerate() {
            assert!(stride > 0);
            strides[self.input_to_output_mapping[idx]] += stride as usize;
        }

        // Output shape we want is already stored in self.output_shape
        // let t = ArrayView::from_shape(IxDyn(&[3]).strides(IxDyn(&[4])), &sl).unwrap();
        let data_slice = tensor.as_slice_memory_order().unwrap();
        ArrayView::from_shape(
            IxDyn(&self.output_shape).strides(IxDyn(&strides)),
            data_slice,
        )
        .unwrap()
    }
}

impl<A> SingletonContractor<A> for Diagonalization {
    fn contract_singleton<'a, 'b>(&self, tensor: &'b ArrayViewD<'a, A>) -> ArrayD<A>
    where
        'a: 'b,
        A: Clone + LinalgScalar,
    {
        // We're only using this method if the tensor is not contiguous
        // Clones twice as a result
        let cloned_tensor: ArrayD<A> =
            Array::from_shape_vec(tensor.raw_dim(), tensor.iter().cloned().collect()).unwrap();
        self.view_singleton(&cloned_tensor.view()).into_owned()
    }
}

/// Permutes the elements of the input tensor and sums across elements that don't appear in the output.
///
/// Example: `ijk->kj`
#[derive(Clone, Debug)]
pub struct PermutationAndSummation {
    permutation: Permutation,
    summation: Summation,
}

impl PermutationAndSummation {
    pub fn new(sc: &SizedContraction) -> Self {
        let mut output_order: Vec<usize> = Vec::new();

        for &output_char in sc.contraction.output_indices.iter() {
            let input_pos = sc.contraction.operand_indices[0]
                .iter()
                .position(|&input_char| input_char == output_char)
                .unwrap();
            output_order.push(input_pos);
        }
        for (i, &input_char) in sc.contraction.operand_indices[0].iter().enumerate() {
            if !sc
                .contraction
                .output_indices
                .iter()
                .any(|&output_char| output_char == input_char)
            {
                output_order.push(i);
            }
        }

        let permutation = Permutation::from_indices(&output_order);
        let summation = Summation::new(sc);

        PermutationAndSummation {
            permutation,
            summation,
        }
    }
}

impl<A> SingletonContractor<A> for PermutationAndSummation {
    fn contract_singleton<'a, 'b>(&self, tensor: &'b ArrayViewD<'a, A>) -> ArrayD<A>
    where
        'a: 'b,
        A: Clone + LinalgScalar,
    {
        let permuted_singleton = self.permutation.view_singleton(tensor);
        self.summation.contract_singleton(&permuted_singleton)
    }
}

/// Returns the elements of the input tensor where all instances of the repeated indices are equal
/// to one another, optionally permuting the axes, and sums across indices that don't appear in the output.
///
/// Examples:
///
/// 1. `iijk->ik` (Diagonalizes the `i` axes and sums over `j`)
/// 2. `jijik->ki` (Diagonalizes `i` and `j` and sums over `j` after diagonalization)
#[derive(Clone, Debug)]
pub struct DiagonalizationAndSummation {
    diagonalization: Diagonalization,
    summation: Summation,
}

impl DiagonalizationAndSummation {
    pub fn new(sc: &SizedContraction) -> Self {
        let diagonalization = Diagonalization::new(sc);
        let summation = Summation::from_sizes(
            sc.contraction.output_indices.len(),
            diagonalization.output_shape.len() - sc.contraction.output_indices.len(),
        );

        DiagonalizationAndSummation {
            diagonalization,
            summation,
        }
    }
}

impl<A> SingletonContractor<A> for DiagonalizationAndSummation {
    fn contract_singleton<'a, 'b>(&self, tensor: &'b ArrayViewD<'a, A>) -> ArrayD<A>
    where
        'a: 'b,
        A: Clone + LinalgScalar,
    {
        // We can only do Diagonalization directly as a view if all the strides are
        // positive and if the tensor is contiguous. We can't know this just from
        // looking at the SizedContraction; we need the actual tensor that will
        // be operated on. So this needs to get checked at "runtime".
        //
        // If either condition fails, we use the contract_singleton version to
        // create a new tensor and view() that intermediate result.
        let contracted_singleton;
        let viewed_singleton = if tensor.as_slice_memory_order().is_some()
            && tensor.strides().iter().all(|&stride| stride > 0)
        {
            self.diagonalization.view_singleton(tensor)
        } else {
            contracted_singleton = self.diagonalization.contract_singleton(tensor);
            contracted_singleton.view()
        };

        self.summation.contract_singleton(&viewed_singleton)
    }
}
