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

//! Defines the contract of atomic units within a quantum program.

use crate::tensor::{Tensor, TensorType};

/// The [`ProgramOp::namespace`] of every op Qiskit defines.
pub const QISKIT: &str = "qiskit";

/// An atomic operation in a quantum program: a typed mapping from tensors to tensors.
///
/// An op declares how many operands it takes ([`Self::arity`]) and how to derive its result types
/// from prospective input types ([`Self::infer_output_types`]). Operands and results are flat and
/// positional, and inference is monomorphic: given operand types the op accepts, its result types
/// are determined.
///
/// An op may have a payload of its own, such as "which axis" information in
/// [`Mean`](crate::ops::Mean), quantum circuit instances in [`ShotLoop`](crate::ops::ShotLoop), or
/// a tensor in [`Constant`](crate::ops::Constant).
///
/// An op can optionally implement [`Self::eval`] to explicitly perform the tensor manipulation
/// that it represents, declaring its choice to do so or not in [`Self::has_builtin_eval`].
/// `QuantumProgram` offers call options to enable externally-defined evaluations.
///
/// An op defined outside this crate lives in its own [`Self::namespace`] and is treated like
/// any other.
pub trait ProgramOp {
    /// The error this op reports for a rejected operand type or a failed evaluation.
    type Error;

    /// The name of this op within its namespace, `add` for instance.
    fn name(&self) -> &str;

    /// The namespace this op belongs to, [`QISKIT`] for one Qiskit defines.
    fn namespace(&self) -> &str;

    /// The namespace and name as one string, `qiskit.add` for instance.
    ///
    /// Backends dispatch on this value.
    fn full_name(&self) -> String {
        format!("{}.{}", self.namespace(), self.name())
    }

    /// The number of operand tensors this op consumes.
    fn arity(&self) -> usize;

    /// Whether [`Self::eval`] contains an implementation.
    fn has_builtin_eval(&self) -> bool;

    /// Infer the types of this op's results from the types of its operands.
    ///
    /// This runs when the op is added to a program function, and is the primary mechanism
    /// to ensure all quantum programs and the functions they contain are well-defined at
    /// all times. The inferred type returned by this method becomes the value type checked by
    /// subsequent ops.
    ///
    /// When several ops happen to share output type inference rules, they are typically made
    /// common in [`tensor::rules`](crate::tensor::rules). For example, binary arithmetic operations
    /// share the same broadcasting and type promotion rules.
    ///
    /// # Panics
    ///
    /// May panic if `inputs.len()` is not [`Self::arity`].
    fn infer_output_types(&self, inputs: &[TensorType]) -> Result<Vec<TensorType>, Self::Error>;

    /// Evaluate this op on `args`, returning one tensor per result.
    ///
    /// The returned tensors match, in count and type, what [`Self::infer_output_types`] promised
    /// for the corresponding operand types. Run time errors should be a last resort: a division op
    /// returns non-finite values for a zero divisor rather than failing, because data from
    /// elsewhere in the program may still be usable. An op whose [`Self::has_builtin_eval`] is
    /// false always returns an error.
    ///
    /// # Panics
    ///
    /// May panic if `args` does not correspond with a non-erroring input to
    /// [`Self::infer_output_types`].
    fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error>;
}

/// The error of an op whose type inference or evaluation fails, type-erased.
pub type BoxedOpError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// An owned [`ErasedProgramOp`].
pub type BoxedProgramOp = Box<dyn ErasedProgramOp>;

/// A type-erased [`ProgramOp`].
pub trait ErasedProgramOp:
    ProgramOp<Error = BoxedOpError> + Send + Sync + sealed::Clonable
{
}

impl<O> ErasedProgramOp for O where
    O: ProgramOp<Error = BoxedOpError> + Clone + Send + Sync + 'static
{
}

impl ToOwned for dyn ErasedProgramOp {
    type Owned = BoxedProgramOp;

    fn to_owned(&self) -> Self::Owned {
        self.clone_dyn()
    }
}

mod sealed {
    use super::{BoxedOpError, BoxedProgramOp, ProgramOp};

    /// Copying an op through a trait object.
    ///
    /// [`Clone`] is not dyn-compatible, because it returns `Self`. This trait is sealed and blanket
    /// implemented, so an implementor supplies nothing but `Clone`.
    #[diagnostic::on_unimplemented(
        message = "Clone is required to store {Self} in a program function",
        note = "Consider annotating {Self} with `#[derive(Clone)]`"
    )]
    pub trait Clonable {
        fn clone_dyn(&self) -> BoxedProgramOp;
    }

    impl<O> Clonable for O
    where
        O: ProgramOp<Error = BoxedOpError> + Clone + Send + Sync + 'static,
    {
        fn clone_dyn(&self) -> BoxedProgramOp {
            Box::new(self.clone())
        }
    }
}

/// Erase `op`'s error type, so that a program function can store it.
pub(crate) fn erase<O>(op: O) -> BoxedProgramOp
where
    O: ProgramOp + Clone + Send + Sync + 'static,
    O::Error: std::error::Error + Send + Sync + 'static,
{
    #[derive(Clone)]
    struct Erased<O>(O);

    impl<O> ProgramOp for Erased<O>
    where
        O: ProgramOp,
        O::Error: std::error::Error + Send + Sync + 'static,
    {
        type Error = BoxedOpError;
        fn name(&self) -> &str {
            self.0.name()
        }
        fn namespace(&self) -> &str {
            self.0.namespace()
        }
        // Forwarded rather than left to the default, so that an op overriding it keeps its
        // own spelling once it is stored in a function.
        fn full_name(&self) -> String {
            self.0.full_name()
        }
        fn arity(&self) -> usize {
            self.0.arity()
        }
        fn has_builtin_eval(&self) -> bool {
            self.0.has_builtin_eval()
        }
        fn infer_output_types(
            &self,
            inputs: &[TensorType],
        ) -> Result<Vec<TensorType>, BoxedOpError> {
            self.0
                .infer_output_types(inputs)
                .map_err(|e| Box::new(e) as BoxedOpError)
        }
        fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, BoxedOpError> {
            self.0.eval(args).map_err(|e| Box::new(e) as BoxedOpError)
        }
    }

    Box::new(Erased(op))
}
