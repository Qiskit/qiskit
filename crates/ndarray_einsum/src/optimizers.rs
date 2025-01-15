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

//! Methods to produce a `ContractionOrder`, specifying what order in which to perform pairwise contractions between tensors
//! in order to perform the full contraction.
use crate::SizedContraction;
use hashbrown::HashSet;

/// Either an input operand or an intermediate result
#[derive(Debug, Clone)]
pub enum OperandNumber {
    Input(usize),
    IntermediateResult(usize),
}

/// Which two tensors to contract
#[derive(Debug, Clone)]
pub struct OperandNumPair {
    pub lhs: OperandNumber,
    pub rhs: OperandNumber,
}

/// A single pairwise contraction between two input operands, an input operand and an intermediate
/// result, or two intermediate results.
#[derive(Debug, Clone)]
pub struct Pair {
    /// The contraction to be performed
    pub sized_contraction: SizedContraction,

    /// Which two tensors to contract
    pub operand_nums: OperandNumPair,
}

/// The order in which to contract pairs of tensors and the specific contractions to be performed between the pairs.
///
/// Either a singleton contraction, in the case of a single input operand, or a list of pair contractions,
/// given two or more input operands
#[derive(Debug, Clone)]
pub enum ContractionOrder {
    /// If there's only one input operand, this is simply a clone of the original SizedContraction
    Singleton(SizedContraction),

    /// If there are two or more input operands, this is a vector of pairwise contractions between
    /// input operands and/or intermediate results from prior contractions.
    Pairs(Vec<Pair>),
}

/// Strategy for optimizing the contraction. The only currently supported options are "Naive" and "Reverse".
///
/// TODO: Figure out whether this should be done with traits
#[derive(Debug)]
pub enum OptimizationMethod {
    /// Contracts each pair of tensors in the order given in the input and uses the intermediate
    /// result as the LHS of the next contraction.
    Naive,

    /// Contracts each pair of tensors in the reverse of the order given in the input and uses the
    /// intermediate result as the LHS of the next contraction. Only implemented to help test
    /// that this is actually functioning properly.
    Reverse,

    /// (Not yet supported) Something like [this](https://optimized-einsum.readthedocs.io/en/latest/greedy_path.html)
    Greedy,

    /// (Not yet supported) Something like [this](https://optimized-einsum.readthedocs.io/en/latest/optimal_path.html)
    Optimal,

    /// (Not yet supported) Something like [this](https://optimized-einsum.readthedocs.io/en/latest/branching_path.html)
    Branch,
}

/// Returns a set of all the indices in any of the remaining operands or in the output
fn get_remaining_indices(operand_indices: &[Vec<char>], output_indices: &[char]) -> HashSet<char> {
    let mut result: HashSet<char> = HashSet::new();
    for &c in operand_indices.iter().flat_map(|s| s.iter()) {
        result.insert(c);
    }
    for &c in output_indices.iter() {
        result.insert(c);
    }
    result
}

/// Returns a set of all the indices in the LHS or the RHS
fn get_existing_indices(lhs_indices: &[char], rhs_indices: &[char]) -> HashSet<char> {
    let mut result: HashSet<char> = lhs_indices.iter().cloned().collect();
    for &c in rhs_indices.iter() {
        result.insert(c);
    }
    result
}

/// Returns a permuted version of `sized_contraction`, specified by `tensor_order`
fn generate_permuted_contraction(
    sized_contraction: &SizedContraction,
    tensor_order: &[usize],
) -> SizedContraction {
    // Reorder the operands of the SizedContraction and clone everything else
    assert_eq!(
        sized_contraction.contraction.operand_indices.len(),
        tensor_order.len()
    );
    let mut new_operand_indices = Vec::new();
    for &i in tensor_order {
        new_operand_indices.push(sized_contraction.contraction.operand_indices[i].clone());
    }
    sized_contraction
        .subset(
            &new_operand_indices,
            &sized_contraction.contraction.output_indices,
        )
        .unwrap()
}

/// Generates a mini-contraction corresponding to `lhs_operand_indices`,`rhs_operand_indices`->`output_indices`
fn generate_sized_contraction_pair(
    lhs_operand_indices: &[char],
    rhs_operand_indices: &[char],
    output_indices: &[char],
    orig_contraction: &SizedContraction,
) -> SizedContraction {
    orig_contraction
        .subset(
            &[lhs_operand_indices.to_vec(), rhs_operand_indices.to_vec()],
            output_indices,
        )
        .unwrap()
}

/// Generate the actual path consisting of all the mini-contractions. Currently always
/// contracts two input operands and then repeatedly uses the result as the LHS of the
/// next pairwise contraction.
fn generate_path(sized_contraction: &SizedContraction, tensor_order: &[usize]) -> ContractionOrder {
    // Generate the actual path consisting of all the mini-contractions.
    //
    // TODO: Take a &[OperandNumPair] instead of &[usize]
    // and Keep track of the intermediate results

    // Make a reordered full SizedContraction in the order specified by the called
    let permuted_contraction = generate_permuted_contraction(sized_contraction, tensor_order);

    match permuted_contraction.contraction.operand_indices.len() {
        1 => {
            // If there's only one input tensor, make a single-step path consisting of a
            // singleton contraction (operand_nums = None).
            ContractionOrder::Singleton(permuted_contraction.clone())
        }
        2 => {
            // If there's exactly two input tensors, make a single-step path consisting
            // of a pair contraction (operand_nums = Some(OperandNumPair)).
            let sc = generate_sized_contraction_pair(
                &permuted_contraction.contraction.operand_indices[0],
                &permuted_contraction.contraction.operand_indices[1],
                &permuted_contraction.contraction.output_indices,
                &permuted_contraction,
            );
            let operand_num_pair = OperandNumPair {
                lhs: OperandNumber::Input(tensor_order[0]),
                rhs: OperandNumber::Input(tensor_order[1]),
            };
            let only_step = Pair {
                sized_contraction: sc,
                operand_nums: operand_num_pair,
            };
            ContractionOrder::Pairs(vec![only_step])
        }
        _ => {
            // If there's three or more input tensors, we have some work to do.

            let mut steps = Vec::new();
            // In the main body of the loop, output_indices will contain the result of the prior pair
            // contraction. Initialize it to the elements of the first LHS tensor so that we can
            // clone it on the first go-around as well as all the later ones.
            let mut output_indices = permuted_contraction.contraction.operand_indices[0].clone();

            for idx_of_lhs in 0..(permuted_contraction.contraction.operand_indices.len() - 1) {
                // lhs_indices is either the first tensor (on the first iteration of the loop)
                // or the output from the previous step.
                let lhs_indices = output_indices.clone();

                // rhs_indices is always the next tensor.
                let idx_of_rhs = idx_of_lhs + 1;
                let rhs_indices = &permuted_contraction.contraction.operand_indices[idx_of_rhs];

                // existing_indices and remaining_indices are only needed to figure out
                // what output_indices will be for this step.
                //
                // existing_indices consists of the indices in either the LHS or the RHS tensor
                // for this step.
                //
                // remaining_indices consists of the indices in all the elements after the RHS
                // tensor or in the outputs.
                //
                // The output indices we want is the intersection of the two (unless this is
                // the RHS is the last operand, in which case it's just the output indices).
                //
                // For example, say the string is "ij,jk,kl,lm->im".
                // First iteration:
                //      lhs = [i,j]
                //      rhs = [j,k]
                //      existing = {i,j,k}
                //      remaining = {k,l,m,i} (the i is used in the final output so we need to
                //      keep it around)
                //      output = {i,k}
                //      Mini-contraction: ij,jk->ik
                // Second iteration:
                //      lhs = [i,k]
                //      rhs = [k,l]
                //      existing = {i,k,l}
                //      remaining = {l,m,i}
                //      output = {i,l}
                //      Mini-contraction: ik,kl->il
                // Third (and final) iteration:
                //      lhs = [i,l]
                //      rhs = [l,m]
                //      (Short-circuit) output = {i,m}
                //      Mini-contraction: il,lm->im
                output_indices =
                    if idx_of_rhs == (permuted_contraction.contraction.operand_indices.len() - 1) {
                        // Used up all the operands; just return output
                        permuted_contraction.contraction.output_indices.clone()
                    } else {
                        let existing_indices = get_existing_indices(&lhs_indices, rhs_indices);
                        let remaining_indices = get_remaining_indices(
                            &permuted_contraction.contraction.operand_indices[(idx_of_rhs + 1)..],
                            &permuted_contraction.contraction.output_indices,
                        );
                        existing_indices
                            .intersection(&remaining_indices)
                            .cloned()
                            .collect()
                    };

                // Phew, now make the mini-contraction.
                let sc = generate_sized_contraction_pair(
                    &lhs_indices,
                    rhs_indices,
                    &output_indices,
                    &permuted_contraction,
                );

                let operand_nums = if idx_of_lhs == 0 {
                    OperandNumPair {
                        lhs: OperandNumber::Input(tensor_order[idx_of_lhs]), // tensor_order[0]
                        rhs: OperandNumber::Input(tensor_order[idx_of_rhs]), // tensor_order[1]
                    }
                } else {
                    OperandNumPair {
                        lhs: OperandNumber::IntermediateResult(idx_of_lhs - 1),
                        rhs: OperandNumber::Input(tensor_order[idx_of_rhs]),
                    }
                };
                steps.push(Pair {
                    sized_contraction: sc,
                    operand_nums,
                });
            }

            ContractionOrder::Pairs(steps)
        }
    }
}

/// Contracts the first two operands, then contracts the result with the third operand, etc.
fn naive_order(sized_contraction: &SizedContraction) -> Vec<usize> {
    (0..sized_contraction.contraction.operand_indices.len()).collect()
}

/// Contracts the last two operands, then contracts the result with the third-to-last operand, etc.
fn reverse_order(sized_contraction: &SizedContraction) -> Vec<usize> {
    (0..sized_contraction.contraction.operand_indices.len())
        .rev()
        .collect()
}

// TODO: Maybe this should take a function pointer from &SizedContraction -> Vec<usize>?
/// Given a `SizedContraction` and an optimization strategy, returns an order in which to
/// perform pairwise contractions in order to produce the final result
pub fn generate_optimized_order(
    sized_contraction: &SizedContraction,
    strategy: OptimizationMethod,
) -> ContractionOrder {
    let tensor_order = match strategy {
        OptimizationMethod::Naive => naive_order(sized_contraction),
        OptimizationMethod::Reverse => reverse_order(sized_contraction),
        _ => panic!("Unsupported optimization method"),
    };
    generate_path(sized_contraction, &tensor_order)
}
