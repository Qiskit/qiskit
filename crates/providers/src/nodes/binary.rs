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

use super::error::MathNodeError;
use super::inference::{elementwise_binary, promoted_dtype};
use super::{OpNodeType, QISKIT};
use crate::tensor::{DType, Tensor, TensorType};

/// Generate a [`OpNodeType`] struct for an elementwise binary operation.
///
/// These nodes coerce, so the dtype `$eval_fn` computes in is the one the operands promote to, and
/// `$accepts` is what admits that dtype rather than either operand's.
macro_rules! elementwise_binary_node {
    ($name:ident, $node_name:literal, $eval_fn:expr, $accepts:expr) => {
        #[doc = concat!(
                    "Elementwise `",
                    $node_name,
                    "` of two tensors, promoting their dtypes and broadcasting their shapes."
                )]
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
                let [x, y] = inputs else {
                    panic!(
                        "{} expects 2 operands, got {}",
                        self.full_name(),
                        inputs.len()
                    )
                };
                Ok(vec![elementwise_binary(x, y, $accepts)?])
            }
            fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
                let [x, y] = args else {
                    panic!(
                        "{} expects 2 operands, got {}",
                        self.full_name(),
                        args.len()
                    )
                };
                // Coerce to the dtype inference promised, so that the tensors agree with the type
                // the node was given when it was added. A cast to the dtype a tensor already has
                // is free.
                let dtype = promoted_dtype(x.dtype(), y.dtype(), $accepts)?;
                let (x, y) = (x.clone().cast(dtype), y.clone().cast(dtype));
                Ok(vec![$eval_fn(&x, &y)?])
            }
        }
    };
}

/// Every dtype but `Bit`, which the arithmetic operators do not implement.
fn numeric(dtype: DType) -> bool {
    dtype != DType::Bit
}

/// Every real dtype, since a remainder is not defined for a complex number.
fn real(dtype: DType) -> bool {
    !matches!(dtype, DType::Bit | DType::C64 | DType::C128)
}

elementwise_binary_node!(Add, "add", Tensor::add_tensor, numeric);
elementwise_binary_node!(Subtract, "subtract", Tensor::sub_tensor, numeric);
elementwise_binary_node!(Multiply, "multiply", Tensor::mul_tensor, numeric);
elementwise_binary_node!(Divide, "divide", Tensor::div_tensor, numeric);
elementwise_binary_node!(Remainder, "remainder", Tensor::rem_tensor, real);
elementwise_binary_node!(Power, "power", Tensor::pow, numeric);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Dim, TensorError};

    /// The type of a 1-D `F64` tensor of `len` elements.
    fn f64_1d(len: usize) -> TensorType {
        TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Fixed(len)],
        }
    }

    #[test]
    fn test_add_same_dtype() {
        let result = Add
            .eval(&[
                Tensor::from([1.0_f64, 2.0, 3.0]),
                Tensor::from([4.0_f64, 5.0, 6.0]),
            ])
            .unwrap();
        assert_eq!(result, vec![Tensor::from([5.0_f64, 7.0, 9.0])]);
    }

    #[test]
    fn test_subtract() {
        let result = Subtract
            .eval(&[
                Tensor::from([5.0_f64, 6.0, 7.0]),
                Tensor::from([1.0_f64, 2.0, 3.0]),
            ])
            .unwrap();
        assert_eq!(result, vec![Tensor::from([4.0_f64, 4.0, 4.0])]);
    }

    #[test]
    fn test_multiply() {
        let result = Multiply
            .eval(&[
                Tensor::from([2.0_f64, 3.0, 4.0]),
                Tensor::from([10.0_f64, 10.0, 10.0]),
            ])
            .unwrap();
        assert_eq!(result, vec![Tensor::from([20.0_f64, 30.0, 40.0])]);
    }

    #[test]
    fn test_divide() {
        let result = Divide
            .eval(&[
                Tensor::from([10.0_f64, 9.0, 8.0]),
                Tensor::from([2.0_f64, 3.0, 4.0]),
            ])
            .unwrap();
        assert_eq!(result, vec![Tensor::from([5.0_f64, 3.0, 2.0])]);
    }

    #[test]
    fn test_remainder() {
        let result = Remainder
            .eval(&[
                Tensor::from([7.0_f64, 8.0, 9.0]),
                Tensor::from([3.0_f64, 3.0, 3.0]),
            ])
            .unwrap();
        assert_eq!(result, vec![Tensor::from([1.0_f64, 2.0, 0.0])]);
    }

    #[test]
    fn test_power() {
        let result = Power
            .eval(&[
                Tensor::from([2.0_f64, 3.0, 4.0]),
                Tensor::from([3.0_f64, 2.0, 1.0]),
            ])
            .unwrap();
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        for (a, b) in arr.as_slice().unwrap().iter().zip(&[8.0_f64, 9.0, 4.0]) {
            assert!(approx::abs_diff_eq!(a, b, epsilon = 1e-12));
        }
    }

    #[test]
    fn test_infer_output_types_forwards_the_operand_type() {
        assert_eq!(
            Add.infer_output_types(&[f64_1d(3), f64_1d(3)]).unwrap(),
            vec![f64_1d(3)]
        );
    }

    #[test]
    fn test_infer_output_types_promotes_differing_dtypes() {
        let f32_3 = TensorType {
            dtype: DType::F32,
            shape: vec![Dim::Fixed(3)],
        };
        assert_eq!(
            Add.infer_output_types(&[f64_1d(3), f32_3]).unwrap(),
            vec![f64_1d(3)]
        );
    }

    #[test]
    fn test_infer_output_types_broadcasts_differing_shapes() {
        assert_eq!(
            Add.infer_output_types(&[f64_1d(3), f64_1d(1)]).unwrap(),
            vec![f64_1d(3)]
        );
    }

    #[test]
    fn test_infer_output_types_rejects_shapes_that_do_not_broadcast() {
        let err = Add.infer_output_types(&[f64_1d(3), f64_1d(4)]).unwrap_err();
        assert_eq!(
            err,
            MathNodeError::Tensor(TensorError::DimShapeMismatch {
                lhs: vec![Dim::Fixed(3)],
                rhs: vec![Dim::Fixed(4)],
            })
        );
        assert_eq!(
            err.to_string(),
            "shapes [3] and [4] are not broadcast-compatible"
        );
    }

    #[test]
    fn test_infer_output_types_rejects_two_bounded_axes() {
        // Two bounded axes have equal types without having equal sizes, so this node cannot pair
        // their elements up. A bounded axis meeting a fixed 1 is fine, since only one size is in
        // question there.
        let bounded = TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Bounded { max: 8 }],
        };
        assert_eq!(
            Add.infer_output_types(&[bounded.clone(), bounded.clone()])
                .unwrap_err(),
            MathNodeError::Tensor(TensorError::DynamicDim {
                shape: bounded.shape.clone(),
            })
        );
        assert_eq!(
            Add.infer_output_types(&[bounded.clone(), f64_1d(1)])
                .unwrap(),
            vec![bounded]
        );
    }

    #[test]
    fn test_eval_broadcasts_and_promotes_to_match_the_inferred_type() {
        assert_eq!(
            Add.eval(&[Tensor::from([1.0_f64, 2.0]), Tensor::from([10.0_f32])])
                .unwrap(),
            vec![Tensor::from([11.0_f64, 12.0])]
        );
    }

    #[test]
    fn test_a_dtype_the_operation_cannot_compute_is_rejected_when_inferring() {
        // `add` has no `Bit` implementation and `remainder` no complex one, so accepting either
        // would let a function type-check and then fail as it ran. The check is against the dtype
        // the operands promote to, which is the one the operation would compute in.
        let bit = TensorType {
            dtype: DType::Bit,
            shape: vec![Dim::Fixed(2)],
        };
        let err = Add
            .infer_output_types(&[bit.clone(), bit.clone()])
            .unwrap_err();
        assert_eq!(
            err,
            MathNodeError::UnsupportedPromotion {
                lhs: DType::Bit,
                rhs: DType::Bit,
                dtype: DType::Bit,
            }
        );
        assert_eq!(
            err.to_string(),
            "operands of dtype Bit and Bit promote to Bit, which is not supported"
        );

        let c128 = TensorType {
            dtype: DType::C128,
            shape: vec![Dim::Fixed(2)],
        };
        assert_eq!(
            Remainder
                .infer_output_types(&[c128.clone(), c128.clone()])
                .unwrap_err(),
            MathNodeError::UnsupportedPromotion {
                lhs: DType::C128,
                rhs: DType::C128,
                dtype: DType::C128,
            }
        );
        // Which dtypes are accepted is per operation: `add` implements the complex ones.
        assert_eq!(
            Add.infer_output_types(&[c128.clone(), c128.clone()])
                .unwrap(),
            vec![c128]
        );
    }

    #[test]
    fn test_a_bit_operand_is_accepted_where_it_promotes_to_something_computable() {
        // A `Bit` operand is only a problem when the other one is also `Bit`: alongside an `F64` it
        // promotes to `F64`, which `add` implements.
        let bit = TensorType {
            dtype: DType::Bit,
            shape: vec![Dim::Fixed(3)],
        };
        assert_eq!(
            Add.infer_output_types(&[bit, f64_1d(3)]).unwrap(),
            vec![f64_1d(3)]
        );
    }
}
