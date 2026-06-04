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
use crate::program_node::ProgramNode;
use crate::tensor::{DTypeLike, Tensor, TensorType, promotion};
use crate::unpack_tensor_args;
use std::sync::LazyLock;

/// Shared input type spec for all elementwise binary nodes: two broadcastable tensors `x` and `y`.
static INPUT_TYPES: LazyLock<DataTree<TensorType>> = LazyLock::new(|| {
    let mut types = DataTree::with_capacity(2);
    types.insert_leaf(
        "x",
        TensorType {
            dtype: DTypeLike::Var("x".into()),
            shape: vec![],
            broadcastable: true,
        },
    );
    types.insert_leaf(
        "y",
        TensorType {
            dtype: DTypeLike::Var("y".into()),
            shape: vec![],
            broadcastable: true,
        },
    );
    types
});

/// Shared output type spec for all elementwise binary nodes: a single tensor of the promoted dtype.
static OUTPUT_TYPES: LazyLock<DataTree<TensorType>> = LazyLock::new(|| {
    DataTree::new_leaf(TensorType {
        dtype: DTypeLike::Promotion(
            vec![DTypeLike::Var("x".into()), DTypeLike::Var("y".into())].into(),
        ),
        shape: vec![],
        broadcastable: true,
    })
});

/// Generate a [`ProgramNode`] struct for an elementwise binary operation.
macro_rules! elementwise_binary_node {
    ($name:ident, $node_name:literal, $call_fn:expr) => {
        #[doc = concat!("Elementwise `", $node_name, "` of two broadcastable tensors.")]
        pub struct $name;

        impl ProgramNode for $name {
            type CallError = super::MathNodeError;

            fn name(&self) -> &'static str {
                $node_name
            }
            fn namespace(&self) -> &'static str {
                "qiskit"
            }
            fn input_types(&self) -> &DataTree<TensorType> {
                &INPUT_TYPES
            }
            fn output_types(&self) -> &DataTree<TensorType> {
                &OUTPUT_TYPES
            }
            fn implements_call(&self) -> bool {
                true
            }
            fn call_flat(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::CallError> {
                unpack_tensor_args!(args, [x, y]);
                let out_dtype = promotion(x.dtype(), y.dtype());
                Ok(vec![$call_fn(
                    &x.cast_ref(out_dtype),
                    &y.cast_ref(out_dtype),
                )?])
            }
        }
    };
}

elementwise_binary_node!(Add, "add", Tensor::add_tensor);
elementwise_binary_node!(Subtract, "subtract", Tensor::sub_tensor);
elementwise_binary_node!(Multiply, "multiply", Tensor::mul_tensor);
elementwise_binary_node!(Divide, "divide", Tensor::div_tensor);
elementwise_binary_node!(Remainder, "remainder", Tensor::rem_tensor);
elementwise_binary_node!(Power, "power", Tensor::pow);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math_nodes::MathNodeError;
    use crate::program_node::{CallError, CallInputError, ProgramNodeExt};
    use crate::tensor::{DType, Tensor};

    #[test]
    fn test_add_same_dtype() {
        let result = Add
            .call_flat(&[
                Tensor::from([1.0_f64, 2.0, 3.0]),
                Tensor::from([4.0_f64, 5.0, 6.0]),
            ])
            .unwrap();
        assert_eq!(result.len(), 1);
        let Tensor::F64(arr) = &result[0] else {
            panic!("expected f64")
        };
        assert_eq!(arr.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_promotes_dtype() {
        let result = Add
            .call_flat(&[Tensor::from([1.0_f32, 2.0]), Tensor::from([3.0_f64, 4.0])])
            .unwrap();
        assert_eq!(result[0].dtype(), DType::F64);
        let Tensor::F64(arr) = &result[0] else {
            panic!("expected f64")
        };
        assert_eq!(arr.as_slice().unwrap(), &[4.0, 6.0]);
    }

    #[test]
    fn test_add_broadcasts_2d_with_1d() {
        use ndarray::arr2;
        let x = Tensor::F64(
            arr2(&[[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]])
                .into_dyn()
                .into_shared(),
        );
        let y = Tensor::from([10.0_f64, 20.0, 30.0]);
        let result = Add.call_flat(&[x, y]).unwrap();
        let Tensor::F64(arr) = &result[0] else {
            panic!("expected f64")
        };
        let expected = arr2(&[[11.0_f64, 22.0, 33.0], [14.0, 25.0, 36.0]])
            .into_dyn()
            .into_shared();
        assert_eq!(arr, &expected);
    }

    #[test]
    fn test_subtract() {
        let result = Subtract
            .call_flat(&[
                Tensor::from([5.0_f64, 6.0, 7.0]),
                Tensor::from([1.0_f64, 2.0, 3.0]),
            ])
            .unwrap();
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        assert_eq!(arr.as_slice().unwrap(), &[4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_multiply() {
        let result = Multiply
            .call_flat(&[
                Tensor::from([2.0_f64, 3.0, 4.0]),
                Tensor::from([10.0_f64, 10.0, 10.0]),
            ])
            .unwrap();
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        assert_eq!(arr.as_slice().unwrap(), &[20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_divide() {
        let result = Divide
            .call_flat(&[
                Tensor::from([10.0_f64, 9.0, 8.0]),
                Tensor::from([2.0_f64, 3.0, 4.0]),
            ])
            .unwrap();
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        assert_eq!(arr.as_slice().unwrap(), &[5.0, 3.0, 2.0]);
    }

    #[test]
    fn test_remainder() {
        let result = Remainder
            .call_flat(&[
                Tensor::from([7.0_f64, 8.0, 9.0]),
                Tensor::from([3.0_f64, 3.0, 3.0]),
            ])
            .unwrap();
        let Tensor::F64(arr) = &result[0] else {
            panic!()
        };
        assert_eq!(arr.as_slice().unwrap(), &[1.0, 2.0, 0.0]);
    }

    #[test]
    fn test_power() {
        let result = Power
            .call_flat(&[
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
    fn test_call_missing_input_errors() {
        let mut tree = DataTree::new();
        tree.insert_leaf("x", Tensor::from([1.0_f64]));
        let err = Add.call(&tree).unwrap_err();
        assert!(matches!(
            err,
            CallError::<MathNodeError>::Input(CallInputError::MissingInput {
                ref key,
            }) if key == "y"
        ));
    }

    #[test]
    fn test_call_branch_where_leaf_expected_errors() {
        let mut tree = DataTree::new();
        tree.insert_leaf("x", Tensor::from([1.0_f64]));
        tree.insert_branch("y", DataTree::new());
        let err = Add.call(&tree).unwrap_err();
        assert!(matches!(
            err,
            CallError::<MathNodeError>::Input(CallInputError::ExpectedLeaf {
                ref key,
            }) if key == "y"
        ));
    }

    #[test]
    fn test_add_wrong_arity_errors() {
        let err = Add.call_flat(&[Tensor::from([1.0_f64])]).unwrap_err();
        assert_eq!(
            err,
            MathNodeError::Input(CallInputError::WrongArity {
                expected: 2,
                actual: 1,
            })
        );
    }
}
