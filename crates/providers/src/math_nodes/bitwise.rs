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

use crate::data_tree::DataTree;
use crate::program_node::{CallInputError, ProgramNode};
use crate::tensor::{DType, DTypeLike, Tensor, TensorType, broadcast_shape};
use crate::unpack_tensor_args;
use ndarray::Axis;
use std::sync::LazyLock;

/// Shared input type spec for binary bitwise nodes
static INPUT_TYPES: LazyLock<DataTree<TensorType>> = LazyLock::new(|| {
    let mut types = DataTree::with_capacity(2);
    types.insert_leaf(
        "x",
        TensorType {
            dtype: DTypeLike::Concrete(DType::Bit),
            shape: vec![],
            broadcastable: true,
        },
    );
    types.insert_leaf(
        "y",
        TensorType {
            dtype: DTypeLike::Concrete(DType::Bit),
            shape: vec![],
            broadcastable: true,
        },
    );
    types
});

/// A single broadcastable `Bit` leaf — used for unary inputs and all bitwise outputs.
static LEAF_TYPE: LazyLock<DataTree<TensorType>> = LazyLock::new(|| {
    DataTree::new_leaf(TensorType {
        dtype: DTypeLike::Concrete(DType::Bit),
        shape: vec![],
        broadcastable: true,
    })
});

/// Construct an `UnexpectedDType` error for a slice element that did not match
/// the schema's required dtype.
fn unexpected_dtype(key: &str, actual: &Tensor) -> CallInputError {
    CallInputError::UnexpectedDType {
        key: key.into(),
        expected: DType::Bit.to_string(),
        actual: actual.dtype(),
    }
}

/// Generate a [`ProgramNode`] struct for an elementwise binary bitwise operation on `Bit` tensors.
macro_rules! bitwise_binary_node {
    ($name:ident, $node_name:literal, $call_fn:expr) => {
        #[doc = concat!("Elementwise `", $node_name, "` of two broadcastable `Bit` tensors.")]
        pub struct $name;

        impl ProgramNode for $name {
            type CallError = super::MathNodeError;

            fn name(&self) -> &'static str {
                $node_name
            }
            fn namespace(&self) -> &'static str {
                "math"
            }
            fn input_types(&self) -> &DataTree<TensorType> {
                &INPUT_TYPES
            }
            fn output_types(&self) -> &DataTree<TensorType> {
                &LEAF_TYPE
            }
            fn implements_call(&self) -> bool {
                true
            }
            fn call_flat(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::CallError> {
                unpack_tensor_args!(args, [x, y]);
                let Tensor::Bit(x_arr) = x else {
                    return Err(unexpected_dtype("x", x).into());
                };
                let Tensor::Bit(y_arr) = y else {
                    return Err(unexpected_dtype("y", y).into());
                };
                broadcast_shape(x_arr.shape(), y_arr.shape())?;
                Ok(vec![Tensor::Bit($call_fn(x_arr, y_arr).into_shared())])
            }
        }
    };
}

bitwise_binary_node!(BitwiseAnd, "bitwise_and", |x, y| x & y);
bitwise_binary_node!(BitwiseOr, "bitwise_or", |x, y| x | y);
bitwise_binary_node!(BitwiseXor, "bitwise_xor", |x, y| x ^ y);

/// Elementwise bitwise NOT of a broadcastable `Bit` tensor.
pub struct BitwiseNot;

impl ProgramNode for BitwiseNot {
    type CallError = super::MathNodeError;

    fn name(&self) -> &'static str {
        "bitwise_not"
    }
    fn namespace(&self) -> &'static str {
        "math"
    }
    fn input_types(&self) -> &DataTree<TensorType> {
        &LEAF_TYPE
    }
    fn output_types(&self) -> &DataTree<TensorType> {
        &LEAF_TYPE
    }
    fn implements_call(&self) -> bool {
        true
    }
    fn call_flat(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::CallError> {
        unpack_tensor_args!(args, [x]);
        let Tensor::Bit(arr) = x else {
            return Err(unexpected_dtype("", x).into());
        };
        Ok(vec![Tensor::Bit(arr.mapv(|b| b ^ 1).into_shared())])
    }
}

/// XOR-reduction of a `Bit` tensor along a specified axis, removing that axis.
///
/// The parity of a sequence of bits is 1 if an odd number of bits are 1, and 0 otherwise,
/// which is equivalent to XOR-folding the sequence. The output has one fewer dimension than
/// the input, with the reduction axis removed.
pub struct Parity {
    axis: usize,
}

impl Parity {
    /// Construct a `Parity` node that reduces along `axis`.
    pub fn new(axis: usize) -> Self {
        Self { axis }
    }
}

impl ProgramNode for Parity {
    type CallError = super::MathNodeError;

    fn name(&self) -> &'static str {
        "parity"
    }
    fn namespace(&self) -> &'static str {
        "math"
    }
    fn input_types(&self) -> &DataTree<TensorType> {
        &LEAF_TYPE
    }
    fn output_types(&self) -> &DataTree<TensorType> {
        &LEAF_TYPE
    }
    fn implements_call(&self) -> bool {
        true
    }
    fn call_flat(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::CallError> {
        unpack_tensor_args!(args, [x]);
        let Tensor::Bit(arr) = x else {
            return Err(unexpected_dtype("", x).into());
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
    use crate::math_nodes::MathNodeError;
    use crate::program_node::{CallError, CallInputError, ProgramNodeExt};
    use ndarray::{arr1, arr2};

    fn bit(data: &[u8]) -> Tensor {
        Tensor::Bit(arr1(data).into_dyn().into_shared())
    }

    #[test]
    fn test_bitwise_and() {
        let result = BitwiseAnd
            .call_flat(&[bit(&[1, 0, 1, 1]), bit(&[1, 1, 0, 1])])
            .unwrap();
        let Tensor::Bit(arr) = &result[0] else {
            panic!("expected Bit leaf");
        };
        assert_eq!(arr.as_slice().unwrap(), &[1, 0, 0, 1]);
    }

    #[test]
    fn test_bitwise_or() {
        let result = BitwiseOr
            .call_flat(&[bit(&[1, 0, 1, 0]), bit(&[0, 1, 0, 1])])
            .unwrap();
        let Tensor::Bit(arr) = &result[0] else {
            panic!("expected Bit leaf");
        };
        assert_eq!(arr.as_slice().unwrap(), &[1, 1, 1, 1]);
    }

    #[test]
    fn test_bitwise_xor() {
        let result = BitwiseXor
            .call_flat(&[bit(&[1, 0, 1, 1]), bit(&[1, 1, 0, 1])])
            .unwrap();
        let Tensor::Bit(arr) = &result[0] else {
            panic!("expected Bit leaf");
        };
        assert_eq!(arr.as_slice().unwrap(), &[0, 1, 1, 0]);
    }

    #[test]
    fn test_bitwise_and_broadcasts() {
        // shape [3] & shape [1] -> shape [3]
        let result = BitwiseAnd.call_flat(&[bit(&[1, 0, 1]), bit(&[1])]).unwrap();
        let Tensor::Bit(arr) = &result[0] else {
            panic!("expected Bit leaf");
        };
        assert_eq!(arr.as_slice().unwrap(), &[1, 0, 1]);
    }

    #[test]
    fn test_bitwise_not() {
        let result = BitwiseNot.call_flat(&[bit(&[1, 0, 1, 0])]).unwrap();
        let Tensor::Bit(arr) = &result[0] else {
            panic!("expected Bit leaf");
        };
        assert_eq!(arr.as_slice().unwrap(), &[0, 1, 0, 1]);
    }

    #[test]
    fn test_parity_axis0() {
        // [[1,0,1],[0,1,1],[0,0,0]] axis 0 → [1, 1, 0]
        let x = Tensor::Bit(
            arr2(&[[1u8, 0, 1], [0, 1, 1], [0, 0, 0]])
                .into_dyn()
                .into_shared(),
        );
        let result = Parity::new(0).call_flat(&[x]).unwrap();
        let Tensor::Bit(arr) = &result[0] else {
            panic!("expected Bit leaf");
        };
        assert_eq!(arr.as_slice().unwrap(), &[1, 1, 0]);
    }

    #[test]
    fn test_bitwise_and_wrong_dtype_errors() {
        let err = BitwiseAnd
            .call_flat(&[Tensor::from([1.0_f64]), bit(&[1])])
            .unwrap_err();
        assert_eq!(
            err,
            MathNodeError::Input(CallInputError::UnexpectedDType {
                key: "x".to_string(),
                expected: "Bit".to_string(),
                actual: DType::F64,
            })
        );
    }

    #[test]
    fn test_bitwise_and_wrong_arity_errors() {
        let err = BitwiseAnd.call_flat(&[bit(&[1, 0])]).unwrap_err();
        assert_eq!(
            err,
            MathNodeError::Input(CallInputError::WrongArity {
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn test_bitwise_not_wrong_arity_errors() {
        let err = BitwiseNot
            .call_flat(&[bit(&[1, 0]), bit(&[0, 1])])
            .unwrap_err();
        assert_eq!(
            err,
            MathNodeError::Input(CallInputError::WrongArity {
                expected: 1,
                actual: 2,
            })
        );
    }

    #[test]
    fn test_bitwise_and_shape_mismatch_errors() {
        let err = BitwiseAnd
            .call_flat(&[bit(&[1, 0, 1]), bit(&[1, 0, 1, 1])])
            .unwrap_err();
        assert_eq!(
            err,
            MathNodeError::Tensor(crate::tensor::TensorError::ShapeMismatch {
                lhs: vec![3],
                rhs: vec![4],
            })
        );
    }

    #[test]
    fn test_call_branch_where_leaf_expected_errors() {
        let mut tree = DataTree::new();
        tree.insert_leaf("x", bit(&[1, 0]));
        let err = BitwiseNot.call(&tree).unwrap_err();
        assert!(matches!(
            err,
            CallError::<MathNodeError>::Input(CallInputError::ExpectedLeaf {
                ref key,
            }) if key.is_empty()
        ));
    }
}
