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

use super::error::{MathNodeError, check_axis, check_dtype};
use super::inference::{elementwise_binary, elementwise_unary, reduce};
use super::{OpNodeType, QISKIT};
use crate::tensor::{DType, Tensor, TensorType};
use ndarray::Axis;

/// Whether a dtype is `Bit`, the only one a bitwise operation is defined for.
fn is_bit(dtype: DType) -> bool {
    dtype == DType::Bit
}

/// Generate a [`OpNodeType`] struct for an elementwise binary bitwise operation on `Bit` tensors.
///
/// `Bit` promotes only to itself, so the promoted dtype is `Bit` exactly when both operands are it.
/// Shapes broadcast, as they do for arithmetic.
macro_rules! bitwise_binary_node {
    ($name:ident, $node_name:literal, $eval_fn:expr) => {
        #[doc = concat!("Elementwise `", $node_name, "` of two `Bit` tensors of identical shape.")]
        #[derive(Clone)]
        pub struct $name;

        impl OpNodeType for $name {
            type Error = MathNodeError;

            fn name(&self) -> &str {
                $node_name
            }
            fn namespace(&self) -> &str {
                QISKIT
            }
            fn arity(&self) -> usize {
                2
            }
            fn has_builtin_eval(&self) -> bool {
                true
            }
            fn infer_output_types(
                &self,
                inputs: &[TensorType],
            ) -> Result<Vec<TensorType>, Self::Error> {
                crate::unpack_operands!(self, inputs, [x, y]);
                Ok(vec![elementwise_binary(x, y, is_bit)?])
            }
            fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
                crate::unpack_operands!(self, args, [x, y]);
                check_dtype(0, x.dtype(), DType::Bit)?;
                check_dtype(1, y.dtype(), DType::Bit)?;
                Ok(vec![$eval_fn(x, y)?])
            }
        }
    };
}

bitwise_binary_node!(BitwiseAnd, "bitwise_and", Tensor::bitand_tensor);
bitwise_binary_node!(BitwiseOr, "bitwise_or", Tensor::bitor_tensor);
bitwise_binary_node!(BitwiseXor, "bitwise_xor", Tensor::bitxor_tensor);

/// Elementwise bitwise NOT of a `Bit` tensor.
#[derive(Clone)]
pub struct BitwiseNot;

impl OpNodeType for BitwiseNot {
    type Error = MathNodeError;

    fn name(&self) -> &str {
        "bitwise_not"
    }
    fn namespace(&self) -> &str {
        QISKIT
    }
    fn arity(&self) -> usize {
        1
    }
    fn has_builtin_eval(&self) -> bool {
        true
    }
    fn infer_output_types(&self, inputs: &[TensorType]) -> Result<Vec<TensorType>, Self::Error> {
        crate::unpack_operands!(self, inputs, [x]);
        Ok(vec![elementwise_unary(x, is_bit)?])
    }
    fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
        crate::unpack_operands!(self, args, [x]);
        check_dtype(0, x.dtype(), DType::Bit)?;
        let Tensor::Bit(arr) = x else {
            unreachable!("dtype checked above")
        };
        Ok(vec![Tensor::Bit(arr.mapv(|b| b ^ 1).into_shared())])
    }
}

/// XOR-reduction of a `Bit` tensor along a specified axis, removing that axis.
///
/// The parity of a sequence of bits is 1 if an odd number of bits are 1, and 0 otherwise,
/// which is equivalent to XOR-folding the sequence. The output has one fewer dimension than
/// the input, with the reduction axis removed.
#[derive(Clone)]
pub struct Parity {
    axis: usize,
}

impl Parity {
    /// Construct a `Parity` node that reduces along `axis`.
    pub fn new(axis: usize) -> Self {
        Self { axis }
    }
}

impl OpNodeType for Parity {
    type Error = MathNodeError;

    fn name(&self) -> &str {
        "parity"
    }
    fn namespace(&self) -> &str {
        QISKIT
    }
    fn arity(&self) -> usize {
        1
    }
    fn has_builtin_eval(&self) -> bool {
        true
    }
    fn infer_output_types(&self, inputs: &[TensorType]) -> Result<Vec<TensorType>, Self::Error> {
        crate::unpack_operands!(self, inputs, [x]);
        // Folding bits with exclusive-or leaves them bits, so the result dtype is the operand's.
        Ok(vec![reduce(x, self.axis, is_bit, |dtype| dtype)?])
    }
    fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
        crate::unpack_operands!(self, args, [x]);
        check_dtype(0, x.dtype(), DType::Bit)?;
        check_axis(self.axis, x.shape().len())?;
        let Tensor::Bit(arr) = x else {
            unreachable!("dtype checked above")
        };
        Ok(vec![Tensor::Bit(
            arr.fold_axis(Axis(self.axis), 0u8, |&acc, &b| acc ^ b)
                .into_shared(),
        )])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dim;
    use ndarray::{arr1, arr2};

    fn bit(data: &[u8]) -> Tensor {
        Tensor::Bit(arr1(data).into_dyn().into_shared())
    }

    /// The type of a 1-D `Bit` tensor of `len` elements.
    fn bit_1d(len: usize) -> TensorType {
        TensorType {
            dtype: DType::Bit,
            shape: vec![Dim::Fixed(len)],
        }
    }

    #[test]
    fn test_bitwise_and() {
        let result = BitwiseAnd
            .eval(&[bit(&[1, 0, 1, 1]), bit(&[1, 1, 0, 1])])
            .unwrap();
        assert_eq!(result, vec![bit(&[1, 0, 0, 1])]);
    }

    #[test]
    fn test_bitwise_or() {
        let result = BitwiseOr
            .eval(&[bit(&[1, 0, 1, 0]), bit(&[0, 1, 0, 1])])
            .unwrap();
        assert_eq!(result, vec![bit(&[1, 1, 1, 1])]);
    }

    #[test]
    fn test_bitwise_xor() {
        let result = BitwiseXor
            .eval(&[bit(&[1, 0, 1, 1]), bit(&[1, 1, 0, 1])])
            .unwrap();
        assert_eq!(result, vec![bit(&[0, 1, 1, 0])]);
    }

    #[test]
    fn test_bitwise_not() {
        let result = BitwiseNot.eval(&[bit(&[1, 0, 1, 0])]).unwrap();
        assert_eq!(result, vec![bit(&[0, 1, 0, 1])]);
    }

    #[test]
    fn test_parity_axis0() {
        // [[1,0,1],[0,1,1],[0,0,0]] axis 0 → [1, 1, 0]
        let x = Tensor::Bit(
            arr2(&[[1u8, 0, 1], [0, 1, 1], [0, 0, 0]])
                .into_dyn()
                .into_shared(),
        );
        let result = Parity::new(0).eval(&[x]).unwrap();
        assert_eq!(result, vec![bit(&[1, 1, 0])]);
    }

    #[test]
    fn test_infer_output_types_forwards_the_operand_shape() {
        assert_eq!(
            BitwiseAnd
                .infer_output_types(&[bit_1d(4), bit_1d(4)])
                .unwrap(),
            vec![bit_1d(4)]
        );
        assert_eq!(
            BitwiseNot.infer_output_types(&[bit_1d(4)]).unwrap(),
            vec![bit_1d(4)]
        );
    }

    #[test]
    fn test_bitwise_not_forwards_a_bounded_axis() {
        // Unary and elementwise, so no size is needed to forward the axis.
        let bounded = TensorType {
            dtype: DType::Bit,
            shape: vec![Dim::Bounded { max: 8 }],
        };
        assert_eq!(
            BitwiseNot
                .infer_output_types(std::slice::from_ref(&bounded))
                .unwrap(),
            vec![bounded]
        );
    }

    #[test]
    fn test_parity_removes_the_reduced_axis() {
        let ty = TensorType {
            dtype: DType::Bit,
            shape: vec![Dim::Bounded { max: 4000 }, Dim::Fixed(3)],
        };
        assert_eq!(
            Parity::new(0).infer_output_types(&[ty]).unwrap(),
            vec![bit_1d(3)]
        );
    }

    #[test]
    fn test_a_non_bit_operand_is_rejected_naming_both_dtypes() {
        // `Bit` is the bottom of the promotion lattice, so two operands promote to `Bit` exactly
        // when both are `Bit`. Anything else lands on a dtype no bitwise operation implements.
        let f64_1 = TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Fixed(1)],
        };
        let err = BitwiseAnd
            .infer_output_types(&[bit_1d(1), f64_1])
            .unwrap_err();
        assert_eq!(
            err,
            MathNodeError::UnsupportedPromotion {
                lhs: DType::Bit,
                rhs: DType::F64,
                dtype: DType::F64,
            }
        );
        assert_eq!(
            err.to_string(),
            "operands of dtype Bit and F64 promote to F64, which is not supported"
        );
    }

    #[test]
    fn test_bitwise_not_names_the_operand_it_rejects() {
        // Unary, so there is no promotion to speak of and the offending operand can be named.
        let f64_1 = TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Fixed(1)],
        };
        assert_eq!(
            BitwiseNot.infer_output_types(&[f64_1]).unwrap_err(),
            MathNodeError::UnsupportedDType {
                operand: 0,
                dtype: DType::F64,
            }
        );
    }

    #[test]
    fn test_eval_rejects_a_wrong_dtype() {
        let err = BitwiseAnd
            .eval(&[Tensor::from([1.0_f64]), bit(&[1])])
            .unwrap_err();
        assert_eq!(
            err,
            MathNodeError::WrongDType {
                operand: 0,
                expected: DType::Bit,
                actual: DType::F64,
            }
        );
    }

    #[test]
    fn test_eval_broadcasts_its_operands() {
        assert_eq!(
            BitwiseAnd.eval(&[bit(&[1, 0, 1]), bit(&[1])]).unwrap(),
            vec![bit(&[1, 0, 1])]
        );
        assert_eq!(
            BitwiseAnd.eval(&[bit(&[1]), bit(&[1, 0, 1])]).unwrap(),
            vec![bit(&[1, 0, 1])],
            "broadcasting is symmetric, unlike ndarray's own operators"
        );
    }

    #[test]
    fn test_eval_rejects_shapes_that_do_not_broadcast() {
        let err = BitwiseAnd
            .eval(&[bit(&[1, 0, 1]), bit(&[1, 0])])
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "shapes [3] and [2] are not broadcast-compatible"
        );
    }

    #[test]
    fn test_parity_axis_out_of_bounds_errors() {
        assert_eq!(
            Parity::new(1).infer_output_types(&[bit_1d(3)]).unwrap_err(),
            MathNodeError::InvalidAxis { axis: 1, ndim: 1 }
        );
        assert_eq!(
            Parity::new(1).eval(&[bit(&[1, 0, 1])]).unwrap_err(),
            MathNodeError::InvalidAxis { axis: 1, ndim: 1 }
        );
    }
}
