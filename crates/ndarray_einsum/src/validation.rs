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

//! Contains functions and structs related to parsing an `einsum`-formatted string
//!
//! This module has the implementation of `Contraction` and `SizedContraction`. `SizedContraction`
//! is used throughout the library to store the details of a full contraction (corresponding
//! to a string supplied by the caller) or a mini-contraction (corresponding to a simplification of
//! a single tensor or a pairwise contraction between two tensors) produced by the optimizer in order
//! to perform the full contraction.
//!
//!
use crate::{
    generate_optimized_order, ArrayLike, ContractionOrder, EinsumPath, OptimizationMethod,
};
use hashbrown::{HashMap, HashSet};
use lazy_static::lazy_static;
use ndarray::prelude::*;
use ndarray::LinalgScalar;
use regex::Regex;

/// The result of running an `einsum`-formatted string through the regex.
#[derive(Debug)]
struct EinsumParse {
    operand_indices: Vec<String>,
    output_indices: Option<String>,
}

/// A `Contraction` contains the result of parsing an `einsum`-formatted string.
///
/// ```
/// # use ndarray_einsum::*;
/// let contraction = Contraction::new("ij,jk->ik").unwrap();
/// assert_eq!(contraction.operand_indices, vec![vec!['i', 'j'], vec!['j', 'k']]);
/// assert_eq!(contraction.output_indices, vec!['i', 'k']);
/// assert_eq!(contraction.summation_indices, vec!['j']);
///
/// let contraction = Contraction::new("ij,jk").unwrap();
/// assert_eq!(contraction.operand_indices, vec![vec!['i', 'j'], vec!['j', 'k']]);
/// assert_eq!(contraction.output_indices, vec!['i', 'k']);
/// assert_eq!(contraction.summation_indices, vec!['j']);
/// ```
#[derive(Debug, Clone)]
pub struct Contraction {
    /// A vector with as many elements as input operands, where each
    /// member of the vector is a `Vec<char>` with each char representing the label for
    /// each axis of the operand.
    pub operand_indices: Vec<Vec<char>>,

    /// Specifies which axes the resulting tensor will contain
    // (corresponding to axes in one or more of the input operands).
    pub output_indices: Vec<char>,

    /// Contains the axes that will be summed over (a.k.a contracted) by the operation.
    pub summation_indices: Vec<char>,
}

impl Contraction {
    /// Validates and creates a `Contraction` from an `einsum`-formatted string.
    pub fn new(input_string: &str) -> Result<Self, &'static str> {
        let p = parse_einsum_string(input_string).ok_or("Invalid string")?;
        Contraction::from_parse(&p)
    }

    /// If output_indices has been specified in the parse (i.e. explicit case),
    /// e.g. "ij,jk->ik", simply converts the strings to `Vec<char>`s and passes
    /// them to Contraction::from_indices. If the output indices haven't been specified,
    /// e.g. "ij,jk", figures out which ones aren't duplicated and hence summed over,
    /// sorts them alphabetically, and uses those as the output indices.
    fn from_parse(parse: &EinsumParse) -> Result<Self, &'static str> {
        let requested_output_indices: Vec<char> = match &parse.output_indices {
            Some(s) => s.chars().collect(),
            _ => {
                // Handle implicit case, e.g. nothing to the right of the arrow
                let mut input_indices = HashMap::new();
                for c in parse.operand_indices.iter().flat_map(|s| s.chars()) {
                    *input_indices.entry(c).or_insert(0) += 1;
                }

                let mut unique_indices: Vec<char> = input_indices
                    .iter()
                    .filter(|(_, &v)| v == 1)
                    .map(|(&k, _)| k)
                    .collect();
                unique_indices.sort();
                unique_indices
            }
        };

        let operand_indices: Vec<Vec<char>> = parse
            .operand_indices
            .iter()
            .map(|x| x.chars().collect::<Vec<char>>())
            .collect();
        Contraction::from_indices(&operand_indices, &requested_output_indices)
    }

    /// Validates and creates a `Contraction` from a slice of `Vec<char>`s containing
    /// the operand indices, and a slice of `char` containing the desired output indices.
    fn from_indices(
        operand_indices: &[Vec<char>],
        output_indices: &[char],
    ) -> Result<Self, &'static str> {
        let mut input_char_counts = HashMap::new();
        for &c in operand_indices.iter().flat_map(|operand| operand.iter()) {
            *input_char_counts.entry(c).or_insert(0) += 1;
        }

        let mut distinct_output_indices = HashMap::new();
        for &c in output_indices.iter() {
            *distinct_output_indices.entry(c).or_insert(0) += 1;
        }
        for (&c, &n) in distinct_output_indices.iter() {
            // No duplicates
            if n > 1 {
                return Err("Requested output has duplicate index");
            }

            // Must be in inputs
            if !input_char_counts.contains_key(&c) {
                return Err("Requested output contains an index not found in inputs");
            }
        }

        let mut summation_indices: Vec<char> = input_char_counts
            .keys()
            .filter(|&c| !distinct_output_indices.contains_key(c))
            .cloned()
            .collect();
        summation_indices.sort();

        let cloned_operand_indices: Vec<Vec<char>> = operand_indices.to_vec();

        Ok(Contraction {
            operand_indices: cloned_operand_indices,
            output_indices: output_indices.to_vec(),
            summation_indices,
        })
    }
}

/// Alias for `HashMap<char, usize>`. Contains the axis lengths for all indices in the contraction.
/// Contrary to the name, does not only hold the sizes for output indices.
pub type OutputSize = HashMap<char, usize>;

/// Enables `OutputSize::from_contraction_and_shapes()`
trait OutputSizeMethods {
    fn from_contraction_and_shapes(
        contraction: &Contraction,
        operand_shapes: &[Vec<usize>],
    ) -> Result<OutputSize, &'static str>;
}
impl OutputSizeMethods for OutputSize {
    /// Build the HashMap containing the axis lengths
    fn from_contraction_and_shapes(
        contraction: &Contraction,
        operand_shapes: &[Vec<usize>],
    ) -> Result<Self, &'static str> {
        // Check that len(operand_indices) == len(operands)
        if contraction.operand_indices.len() != operand_shapes.len() {
            return Err(
                "number of operands in contraction does not match number of operands supplied",
            );
        }

        let mut index_lengths: OutputSize = HashMap::new();

        for (indices, operand_shape) in contraction.operand_indices.iter().zip(operand_shapes) {
            // Check that len(operand_indices[i]) == len(operands[i].shape())
            if indices.len() != operand_shape.len() {
                return Err(
                    "number of indices in one or more operands does not match dimensions of operand",
                );
            }

            // Check that whenever there are multiple copies of an index,
            // operands[i].shape()[m] == operands[j].shape()[n]
            for (&c, &n) in indices.iter().zip(operand_shape) {
                let existing_n = index_lengths.entry(c).or_insert(n);
                if *existing_n != n {
                    return Err("repeated index with different size");
                }
            }
        }

        Ok(index_lengths)
    }
}

/// A `SizedContraction` contains a `Contraction` as well as a `HashMap<char, usize>`
/// specifying the axis lengths for each index in the contraction.
///
/// Note that output_size is a misnomer (to be changed); it contains all the axis lengths,
/// including the ones that will be contracted (i.e. not just the ones in
/// contraction.output_indices).
#[derive(Debug, Clone)]
pub struct SizedContraction {
    pub contraction: Contraction,
    pub output_size: OutputSize,
}

impl SizedContraction {
    /// Creates a new SizedContraction based on a subset of the operand indices and/or output
    /// indices. Not intended for general use; used internally in the crate when compiling
    /// a multi-tensor contraction into a set of tensor simplifications (a.k.a. singleton
    /// contractions) and pairwise contractions.
    ///
    /// ```
    /// # use ndarray_einsum::*;
    /// # use ndarray::prelude::*;
    /// let m1: Array3<f64> = Array::zeros((2, 2, 3));
    /// let m2: Array2<f64> = Array::zeros((3, 4));
    /// let sc = SizedContraction::new("iij,jk->ik", &[&m1, &m2]).unwrap();
    /// let lhs_simplification = sc.subset(&[vec!['i','i','j']], &['i','j']).unwrap();
    /// let diagonalized_m1 = lhs_simplification.contract_operands(&[&m1]);
    /// assert_eq!(diagonalized_m1.shape(), &[2, 3]);
    /// ```
    pub fn subset(
        &self,
        new_operand_indices: &[Vec<char>],
        new_output_indices: &[char],
    ) -> Result<Self, &'static str> {
        // Make sure all chars in new_operand_indices are in self
        let all_operand_indices: HashSet<char> = new_operand_indices
            .iter()
            .flat_map(|operand| operand.iter())
            .cloned()
            .collect();
        if all_operand_indices
            .iter()
            .any(|c| !self.output_size.contains_key(c))
        {
            return Err("Character found in new_operand_indices but not in self.output_size");
        }

        // Validate what they asked for and compute summation_indices
        let new_contraction = Contraction::from_indices(new_operand_indices, new_output_indices)?;

        // Clone output_size, omitting unused characters
        let new_output_size: OutputSize = self
            .output_size
            .iter()
            .filter(|(&k, _)| all_operand_indices.contains(&k))
            .map(|(&k, &v)| (k, v))
            .collect();

        Ok(SizedContraction {
            contraction: new_contraction,
            output_size: new_output_size,
        })
    }

    fn from_contraction_and_shapes(
        contraction: &Contraction,
        operand_shapes: &[Vec<usize>],
    ) -> Result<Self, &'static str> {
        let output_size = OutputSize::from_contraction_and_shapes(contraction, operand_shapes)?;

        Ok(SizedContraction {
            contraction: contraction.clone(),
            output_size,
        })
    }

    /// Create a SizedContraction from an already-created Contraction and a list
    /// of operands.
    /// ```
    /// # use ndarray_einsum::*;
    /// # use ndarray::prelude::*;
    /// let m1: Array2<f64> = Array::zeros((2, 3));
    /// let m2: Array2<f64> = Array::zeros((3, 4));
    /// let c = Contraction::new("ij,jk->ik").unwrap();
    /// let sc = SizedContraction::from_contraction_and_operands(&c, &[&m1, &m2]).unwrap();
    /// assert_eq!(sc.output_size[&'i'], 2);
    /// assert_eq!(sc.output_size[&'k'], 4);
    /// assert_eq!(sc.output_size[&'j'], 3);
    /// ```
    pub fn from_contraction_and_operands<A>(
        contraction: &Contraction,
        operands: &[&dyn ArrayLike<A>],
    ) -> Result<Self, &'static str> {
        let operand_shapes = get_operand_shapes(operands);

        SizedContraction::from_contraction_and_shapes(contraction, &operand_shapes)
    }

    /// Create a SizedContraction from an `einsum`-formatted input string and a slice
    /// of `Vec<usize>`s containing the shapes of each operand.
    /// ```
    /// # use ndarray_einsum::*;
    /// # use ndarray::prelude::*;
    /// let sc = SizedContraction::from_string_and_shapes(
    ///     "ij,jk->ik",
    ///     &[vec![2, 3], vec![3, 4]]
    /// ).unwrap();
    /// assert_eq!(sc.output_size[&'i'], 2);
    /// assert_eq!(sc.output_size[&'k'], 4);
    /// assert_eq!(sc.output_size[&'j'], 3);
    /// ```
    pub fn from_string_and_shapes(
        input_string: &str,
        operand_shapes: &[Vec<usize>],
    ) -> Result<Self, &'static str> {
        let contraction = validate(input_string)?;
        SizedContraction::from_contraction_and_shapes(&contraction, operand_shapes)
    }

    /// Create a SizedContraction from an `einsum`-formatted input string and a list
    /// of operands.
    ///
    /// ```
    /// # use ndarray_einsum::*;
    /// # use ndarray::prelude::*;
    /// let m1: Array2<f64> = Array::zeros((2, 3));
    /// let m2: Array2<f64> = Array::zeros((3, 4));
    /// let sc = SizedContraction::new("ij,jk->ik", &[&m1, &m2]).unwrap();
    /// assert_eq!(sc.output_size[&'i'], 2);
    /// assert_eq!(sc.output_size[&'k'], 4);
    /// assert_eq!(sc.output_size[&'j'], 3);
    /// ```
    pub fn new<A>(
        input_string: &str,
        operands: &[&dyn ArrayLike<A>],
    ) -> Result<Self, &'static str> {
        let operand_shapes = get_operand_shapes(operands);

        SizedContraction::from_string_and_shapes(input_string, &operand_shapes)
    }

    /// Perform the contraction on a set of operands.
    ///
    /// ```
    /// # use ndarray_einsum::*;
    /// # use ndarray::prelude::*;
    /// let m1: Array2<f64> = Array::zeros((2, 3));
    /// let m2: Array2<f64> = Array::zeros((3, 4));
    /// let out: ArrayD<f64> = Array::zeros((2, 4)).into_dyn();
    /// let sc = validate_and_size("ij,jk->ik", &[&m1, &m2]).unwrap();
    /// assert_eq!(sc.contract_operands(&[&m1, &m2]), out);
    /// ```
    pub fn contract_operands<A: Clone + LinalgScalar>(
        &self,
        operands: &[&dyn ArrayLike<A>],
    ) -> ArrayD<A> {
        let cpc = EinsumPath::new(self);
        cpc.contract_operands(operands)
    }

    /// Show as an `einsum`-formatted string.
    ///
    /// ```
    /// # use ndarray_einsum::*;
    /// # use ndarray::prelude::*;
    /// let m1: Array2<f64> = Array::zeros((2, 3));
    /// let m2: Array2<f64> = Array::zeros((3, 4));
    /// let sc = validate_and_size("ij,jk", &[&m1, &m2]).unwrap();
    /// assert_eq!(sc.as_einsum_string(), "ij,jk->ik");
    /// ```
    pub fn as_einsum_string(&self) -> String {
        assert!(!self.contraction.operand_indices.is_empty());
        let mut s: String = self.contraction.operand_indices[0]
            .iter()
            .cloned()
            .collect();
        for op in self.contraction.operand_indices[1..].iter() {
            s.push(',');
            for &c in op.iter() {
                s.push(c);
            }
        }
        s.push_str("->");
        for &c in self.contraction.output_indices.iter() {
            s.push(c);
        }
        s
    }
}

/// Runs an input string through a regex and convert it to an EinsumParse.
fn parse_einsum_string(input_string: &str) -> Option<EinsumParse> {
    lazy_static! {
        // Unwhitespaced version:
        // ^([a-z]+)((?:,[a-z]+)*)(?:->([a-z]*))?$
        static ref RE: Regex = Regex::new(r"(?x)
            ^
            (?P<first_operand>[a-z]+)
            (?P<more_operands>(?:,[a-z]+)*)
            (?:->(?P<output>[a-z]*))?
            $
            ").unwrap();
    }
    let captures = RE.captures(input_string)?;
    let mut operand_indices = Vec::new();
    let output_indices = captures.name("output").map(|s| String::from(s.as_str()));

    operand_indices.push(String::from(&captures["first_operand"]));
    for s in captures["more_operands"].split(',').skip(1) {
        operand_indices.push(String::from(s));
    }

    Some(EinsumParse {
        operand_indices,
        output_indices,
    })
}

/// Wrapper around [Contraction::new()](struct.Contraction.html#method.new).
pub fn validate(input_string: &str) -> Result<Contraction, &'static str> {
    Contraction::new(input_string)
}

/// Returns a vector holding one `Vec<usize>` for each operand.
fn get_operand_shapes<A>(operands: &[&dyn ArrayLike<A>]) -> Vec<Vec<usize>> {
    operands
        .iter()
        .map(|operand| Vec::from(operand.into_dyn_view().shape()))
        .collect()
}

/// Wrapper around [SizedContraction::new()](struct.SizedContraction.html#method.new).
pub fn validate_and_size<A>(
    input_string: &str,
    operands: &[&dyn ArrayLike<A>],
) -> Result<SizedContraction, &'static str> {
    SizedContraction::new(input_string, operands)
}

/// Create a [SizedContraction](struct.SizedContraction.html) and then optimize the order in which pairs of inputs will be contracted.
pub fn validate_and_optimize_order<A>(
    input_string: &str,
    operands: &[&dyn ArrayLike<A>],
    optimization_strategy: OptimizationMethod,
) -> Result<ContractionOrder, &'static str> {
    let sc = validate_and_size(input_string, operands)?;
    Ok(generate_optimized_order(&sc, optimization_strategy))
}
