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

/// Destructure an op's operands into one binding each, panicking if the count is wrong.
///
/// The panic should be unreachable when the op is part of a
/// [`ProgramFunction`](crate::ProgramFunction), because static analysis is done while inserting
/// ops.
#[macro_export]
macro_rules! unpack_operands {
    ($op:expr, $operands:expr, [$($name:ident),+ $(,)?]) => {
        let operands = $operands;
        let [$($name),+] = operands else {
            panic!(
                "{} expects {} operand(s), got {}",
                $op.full_name(),
                $op.arity(),
                operands.len()
            )
        };
    };
}

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
pub trait ErasedProgramOp: std::any::Any + Send + Sync + sealed::Clonable {
    fn name(&self) -> &str;
    fn namespace(&self) -> &str;
    fn full_name(&self) -> String;
    fn arity(&self) -> usize;
    fn has_builtin_eval(&self) -> bool;
    fn infer_output_types(&self, inputs: &[TensorType]) -> Result<Vec<TensorType>, BoxedOpError>;
    fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, BoxedOpError>;
}

impl<O> ErasedProgramOp for O
where
    O: ProgramOp + Clone + Send + Sync + 'static,
    O::Error: std::error::Error + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        ProgramOp::name(self)
    }
    fn namespace(&self) -> &str {
        ProgramOp::namespace(self)
    }

    fn full_name(&self) -> String {
        ProgramOp::full_name(self)
    }

    fn arity(&self) -> usize {
        ProgramOp::arity(self)
    }

    fn has_builtin_eval(&self) -> bool {
        ProgramOp::has_builtin_eval(self)
    }

    fn infer_output_types(&self, inputs: &[TensorType]) -> Result<Vec<TensorType>, BoxedOpError> {
        ProgramOp::infer_output_types(self, inputs).map_err(|error| Box::new(error) as BoxedOpError)
    }

    fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, BoxedOpError> {
        ProgramOp::eval(self, args).map_err(|error| Box::new(error) as BoxedOpError)
    }
}

impl dyn ErasedProgramOp + 'static {
    /// Downcast a type-erased program op into the specific program op it actually is.
    pub fn downcast_ref<O: ProgramOp + 'static>(&self) -> Option<&O> {
        (self as &dyn std::any::Any).downcast_ref()
    }
}

impl ToOwned for dyn ErasedProgramOp {
    type Owned = BoxedProgramOp;

    fn to_owned(&self) -> Self::Owned {
        self.clone_dyn()
    }
}

mod sealed {
    use super::{BoxedProgramOp, ProgramOp};

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
        O: ProgramOp + Clone + Send + Sync + 'static,
        O::Error: std::error::Error + Send + Sync + 'static,
    {
        fn clone_dyn(&self) -> BoxedProgramOp {
            Box::new(self.clone())
        }
    }
}

#[cfg(test)]
mod test {
    use crate::ops::{Add, ProgramOp};
    use crate::tensor::Tensor;

    #[test]
    #[should_panic(expected = "qiskit.add expects 2 operand(s), got 1")]
    fn test_unpack_operands_names_the_op_and_the_arity_it_declares() {
        let _ = Add.eval(&[Tensor::from([1.0_f64])]);
    }
}
