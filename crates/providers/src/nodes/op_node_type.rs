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

//! The contract every node type satisfies.

use crate::tensor::{Tensor, TensorType};

/// The [`OpNodeType::namespace`] of every node type Qiskit defines.
pub const QISKIT: &str = "qiskit";

/// One step in a program function: a typed mapping from tensors to tensors.
///
/// A node declares how many operands it takes ([`Self::arity`]) and how to derive its result
/// types from theirs ([`Self::infer_output_types`]). Operands and results are flat and
/// positional, and a node instance is monomorphic: given operand types it accepts, its result
/// types are determined.
///
/// The boolean [`Self::has_builtin_eval`] specifies whether [`Self::eval`] contains an implementation.
pub trait OpNodeType {
    /// The error this node reports for a rejected operand type or a failed evaluation.
    type Error;

    /// The name of this program node.
    fn name(&self) -> &str;

    /// The namespace this program node belongs to.
    fn namespace(&self) -> &str;

    /// The namespace and name as one string.
    fn full_name(&self) -> String {
        format!("{}.{}", self.namespace(), self.name())
    }

    /// The number of operand tensors this node consumes.
    fn arity(&self) -> usize;

    /// Whether [`Self::eval`] contains an implementation.
    fn has_builtin_eval(&self) -> bool;

    /// Infer the types of this node's results from the types of its operands.
    ///
    /// This runs when the node is added to a program function, and is therefore what makes a
    /// built function well-typed: an operand type this node cannot accept must be reported as an
    /// error here, naming the operand position at fault.
    ///
    /// # Panics
    ///
    /// May panic if `inputs.len()` is not [`Self::arity`].
    fn infer_output_types(&self, inputs: &[TensorType]) -> Result<Vec<TensorType>, Self::Error>;

    /// Evaluate this node on `args`, returning one tensor per result.
    ///
    /// The returned tensors match, in count and type, what [`Self::infer_output_types`] promised
    /// for the corresponding operand types. An error is a last resort: a division node, for
    /// instance, returns non-finite values for a zero divisor rather than failing, because some
    /// of the data from elsewhere in the program may still be usable. A node whose
    /// [`Self::has_builtin_eval`] is false always returns an error.
    ///
    /// # Panics
    ///
    /// May panic if `args` does not match the operand types this node was type-checked against.
    fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error>;
}
