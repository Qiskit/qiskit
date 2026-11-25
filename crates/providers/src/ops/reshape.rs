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

//! The op that rearranges a tensor's axes without moving its elements.

use super::error::MathOpError;
use super::{ProgramOp, QISKIT};
use crate::tensor::{Tensor, TensorType, require_static};

/// One axis of a [`Reshape`] target.
///
/// An axis is either a size the caller gives, or one left to be worked out from the operand's
/// element count. This is not a [`Dim`](crate::tensor::Dim): a dimension describes an axis a tensor
/// *has*, where a reshape target describes an axis a caller *wants*, and "infer this one" is only
/// meaningful in the latter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// An axis of exactly this size.
    Size(usize),
    /// An axis whose size is whatever makes the element count work out.
    ///
    /// NumPy spells this `-1`. At most one axis of a target may be inferred, since two unknowns
    /// cannot be recovered from one product.
    Infer,
}

/// Rearrange a tensor's axes, keeping its elements and their order.
///
/// A reshape moves no data: it reinterprets the same elements, read in row-major order, under a new
/// shape. The target must therefore hold exactly as many elements as the operand, so
/// `Bit[4, 5]` may become `Bit[20]` or `Bit[2, 10]` but not `Bit[4, 4]`.
///
/// One target axis may be [`Axis::Infer`], which takes whatever size makes the count work out; this
/// is NumPy's `-1`. Appending an axis of size 1 is the common case, and is how a mask computed over
/// one register is lined up against data of a wider one: NumPy spells that `mask[..., None]`, which
/// is a reshape from `Bit[n]` to `Bit[n, 1]`.
///
/// An operand with a [`Dim::Bounded`](crate::tensor::Dim::Bounded) axis is rejected. Reshaping needs
/// the operand's element count to check the target against, and a bounded axis means that count is
/// not known until the program runs.
#[derive(Clone, Debug)]
pub struct Reshape {
    target: Vec<Axis>,
}

impl Reshape {
    /// Construct a `Reshape` op producing `target`.
    ///
    /// At most one axis may be [`Axis::Infer`].
    pub fn new(target: impl IntoIterator<Item = Axis>) -> Self {
        Self {
            target: target.into_iter().collect(),
        }
    }

    /// Construct a `Reshape` op producing a target of wholly known sizes.
    pub fn sized(target: impl IntoIterator<Item = usize>) -> Self {
        Self::new(target.into_iter().map(Axis::Size))
    }

    /// Return the target shape this op reshapes to.
    pub fn target(&self) -> &[Axis] {
        &self.target
    }

    /// Resolve the target against an operand holding `elements` elements.
    ///
    /// An inferred axis takes the size that makes the target hold exactly `elements`, and the
    /// element counts must agree exactly either way.
    fn resolve(&self, elements: usize) -> Result<Vec<usize>, MathOpError> {
        let inferred = self
            .target
            .iter()
            .filter(|axis| **axis == Axis::Infer)
            .count();
        if inferred > 1 {
            return Err(MathOpError::AmbiguousReshape { axes: inferred });
        }
        // The product of the axes the caller did give, which an inferred axis divides into.
        let given: usize = self
            .target
            .iter()
            .filter_map(|axis| match axis {
                Axis::Size(size) => Some(*size),
                Axis::Infer => None,
            })
            .product();

        let mismatch = || MathOpError::ReshapeElementCount {
            target: self.target.clone(),
            elements,
        };
        let size = if inferred == 0 {
            if given != elements {
                return Err(mismatch());
            }
            0
        } else {
            // A zero product would make the division meaningless, and only an empty operand can
            // be reshaped through one.
            if given == 0 {
                if elements != 0 {
                    return Err(mismatch());
                }
                0
            } else {
                if elements % given != 0 {
                    return Err(mismatch());
                }
                elements / given
            }
        };

        Ok(self
            .target
            .iter()
            .map(|axis| match axis {
                Axis::Size(given) => *given,
                Axis::Infer => size,
            })
            .collect())
    }
}

impl ProgramOp for Reshape {
    type Error = MathOpError;

    fn name(&self) -> &str {
        "reshape"
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
    fn describe(&self) -> Option<String> {
        let axes: Vec<String> = self
            .target
            .iter()
            .map(|axis| match axis {
                Axis::Size(size) => size.to_string(),
                Axis::Infer => "-1".to_owned(),
            })
            .collect();
        Some(format!("target=[{}]", axes.join(", ")))
    }
    fn infer_output_types(&self, inputs: &[TensorType]) -> Result<Vec<TensorType>, Self::Error> {
        crate::unpack_operands!(self, inputs, [x]);
        // A bounded axis leaves the element count unknown, so there is nothing to check the target
        // against. This mirrors how `broadcast_dims_to` refuses a bounded operand.
        let shape = require_static(&x.shape)?;
        let resolved = self.resolve(shape.iter().product())?;
        Ok(vec![TensorType {
            dtype: x.dtype,
            shape: resolved
                .into_iter()
                .map(crate::tensor::Dim::Fixed)
                .collect(),
            ..Default::default()
        }])
    }
    fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
        crate::unpack_operands!(self, args, [x]);
        let resolved = self.resolve(x.shape().iter().product())?;
        Ok(vec![x.reshape(&resolved)?])
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::tensor::{DType, Dim, TensorError};

    /// Return a `TensorType` of `dtype` over fixed axes `shape`.
    fn ty(dtype: DType, shape: &[usize]) -> TensorType {
        TensorType {
            dtype,
            shape: shape.iter().copied().map(Dim::Fixed).collect(),
        },
        ..Default::default()
    }

    /// Return the single output type of reshaping an `F64` operand of `shape` to `target`.
    fn inferred(target: Reshape, shape: &[usize]) -> Result<TensorType, MathOpError> {
        target
            .infer_output_types(&[ty(DType::F64, shape)])
            .map(|mut types| types.remove(0))
    }

    /// Return a 1-D `F64` tensor counting up to `len`.
    fn range(len: usize) -> Tensor {
        Tensor::from((0..len).map(|i| i as f64).collect::<Vec<_>>().as_slice())
    }

    #[test]
    fn test_reshape_full_name_and_arity() {
        let op = Reshape::sized([2, 3]);
        assert_eq!(op.full_name(), "qiskit.reshape");
        assert_eq!(op.arity(), 1);
        assert!(op.has_builtin_eval());
        assert_eq!(op.target(), [Axis::Size(2), Axis::Size(3)]);
    }

    #[test]
    fn test_infer_output_types_is_the_target_and_keeps_the_dtype() {
        assert_eq!(
            inferred(Reshape::sized([6, 4]), &[2, 3, 4]).unwrap(),
            ty(DType::F64, &[6, 4])
        );
        assert_eq!(
            inferred(Reshape::sized([24]), &[2, 3, 4]).unwrap(),
            ty(DType::F64, &[24]),
            "flattening is just a reshape to one axis"
        );
        assert_eq!(
            Reshape::sized([4])
                .infer_output_types(&[ty(DType::Bit, &[2, 2])])
                .unwrap(),
            vec![ty(DType::Bit, &[4])],
            "the dtype passes through untouched"
        );
    }

    #[test]
    fn test_infer_output_types_appends_an_axis_of_size_one() {
        // The case the epic needs: `mask[..., None]` lines a mask up against wider data.
        assert_eq!(
            inferred(Reshape::sized([5, 1]), &[5]).unwrap(),
            ty(DType::F64, &[5, 1])
        );
    }

    #[test]
    fn test_infer_output_types_works_out_an_inferred_axis() {
        assert_eq!(
            inferred(Reshape::new([Axis::Infer, Axis::Size(4)]), &[2, 3, 4]).unwrap(),
            ty(DType::F64, &[6, 4]),
            "24 elements over an axis of 4 leaves 6"
        );
        assert_eq!(
            inferred(Reshape::new([Axis::Size(4), Axis::Infer]), &[2, 3, 4]).unwrap(),
            ty(DType::F64, &[4, 6]),
            "the inferred axis may be anywhere"
        );
        assert_eq!(
            inferred(Reshape::new([Axis::Infer]), &[2, 3, 4]).unwrap(),
            ty(DType::F64, &[24]),
            "a lone inferred axis flattens"
        );
    }

    #[test]
    fn test_infer_output_types_rejects_a_target_of_the_wrong_size() {
        // A reshape moves no data, so the counts have to agree exactly.
        assert!(matches!(
            inferred(Reshape::sized([4, 4]), &[2, 3, 4]).unwrap_err(),
            MathOpError::ReshapeElementCount { elements: 24, .. }
        ));
        assert!(matches!(
            inferred(Reshape::new([Axis::Infer, Axis::Size(5)]), &[2, 3, 4]).unwrap_err(),
            MathOpError::ReshapeElementCount { elements: 24, .. }
        ));
    }

    #[test]
    fn test_infer_output_types_rejects_more_than_one_inferred_axis() {
        // Two unknowns cannot be recovered from one product.
        assert!(matches!(
            inferred(Reshape::new([Axis::Infer, Axis::Infer]), &[4]).unwrap_err(),
            MathOpError::AmbiguousReshape { axes: 2 }
        ));
    }

    #[test]
    fn test_infer_output_types_rejects_a_bounded_operand() {
        // Without the operand's element count there is nothing to check the target against.
        let bounded = TensorType {
            dtype: DType::Bit,
            shape: vec![Dim::Bounded { max: 20 }],
            ..Default::default()
        };
        assert!(matches!(
            Reshape::sized([20])
                .infer_output_types(&[bounded])
                .unwrap_err(),
            MathOpError::Tensor(TensorError::DynamicDim { .. })
        ));
    }

    #[test]
    fn test_an_empty_operand_reshapes_only_through_an_empty_target() {
        assert_eq!(
            inferred(Reshape::sized([0, 3]), &[0]).unwrap(),
            ty(DType::F64, &[0, 3])
        );
        assert!(
            matches!(
                inferred(Reshape::sized([0, 3]), &[6]).unwrap_err(),
                MathOpError::ReshapeElementCount { elements: 6, .. }
            ),
            "a target holding nothing cannot hold six elements"
        );
        // An inferred axis beside a zero-sized one has no single answer, so the count must be zero.
        assert_eq!(
            inferred(Reshape::new([Axis::Size(0), Axis::Infer]), &[0]).unwrap(),
            ty(DType::F64, &[0, 0])
        );
    }

    #[test]
    fn test_eval_keeps_the_elements_in_row_major_order() {
        let out = Reshape::sized([2, 3]).eval(&[range(6)]).unwrap().remove(0);
        assert_eq!(out.shape(), &[2, 3]);
        let expected = Tensor::F64(
            ndarray::arr2(&[[0.0_f64, 1.0, 2.0], [3.0, 4.0, 5.0]])
                .into_dyn()
                .into_shared(),
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn test_eval_round_trips_through_a_flattening() {
        let x = Tensor::F64(
            ndarray::arr2(&[[1.0_f64, 2.0], [3.0, 4.0]])
                .into_dyn()
                .into_shared(),
        );
        let flat = Reshape::new([Axis::Infer])
            .eval(std::slice::from_ref(&x))
            .unwrap()
            .remove(0);
        assert_eq!(flat, Tensor::from([1.0_f64, 2.0, 3.0, 4.0]));
        let back = Reshape::sized([2, 2]).eval(&[flat]).unwrap().remove(0);
        assert_eq!(back, x, "reshaping back recovers the operand");
    }

    #[test]
    fn test_eval_appends_an_axis_without_moving_data() {
        let out = Reshape::sized([3, 1]).eval(&[range(3)]).unwrap().remove(0);
        assert_eq!(out.shape(), &[3, 1]);
        assert_eq!(
            out,
            Tensor::F64(
                ndarray::arr2(&[[0.0_f64], [1.0], [2.0]])
                    .into_dyn()
                    .into_shared()
            )
        );
    }

    #[test]
    fn test_eval_keeps_the_dtype() {
        let bits = Tensor::Bit(ndarray::arr1(&[1_u8, 0, 1, 1]).into_dyn().into_shared());
        let out = Reshape::sized([2, 2]).eval(&[bits]).unwrap().remove(0);
        assert_eq!(out.dtype(), DType::Bit);
        assert_eq!(
            out,
            Tensor::Bit(ndarray::arr2(&[[1_u8, 0], [1, 1]]).into_dyn().into_shared())
        );
    }

    #[test]
    fn test_eval_rejects_a_target_of_the_wrong_size() {
        assert!(matches!(
            Reshape::sized([5]).eval(&[range(6)]).unwrap_err(),
            MathOpError::ReshapeElementCount { elements: 6, .. }
        ));
    }

    #[test]
    fn test_eval_result_satisfies_the_inferred_type() {
        // The invariant `ProgramFunction::eval` checks: what an op produces must satisfy what it
        // said it would.
        for (op, shape) in [
            (Reshape::sized([2, 3]), vec![6]),
            (Reshape::new([Axis::Infer, Axis::Size(2)]), vec![3, 2]),
            (Reshape::sized([6, 1]), vec![2, 3]),
        ] {
            let operand = ty(DType::F64, &shape);
            let inferred = op.infer_output_types(&[operand]).unwrap();
            let elements: usize = shape.iter().product();
            let out = op.eval(&[range(elements)]).unwrap();
            assert!(
                out[0].matches(&inferred[0]),
                "{:?} does not satisfy {:?}",
                out[0].tensor_type(),
                inferred[0]
            );
        }
    }

    #[test]
    fn test_describe_renders_the_target() {
        assert_eq!(
            Reshape::sized([5, 1]).describe().as_deref(),
            Some("target=[5, 1]")
        );
        assert_eq!(
            Reshape::new([Axis::Infer, Axis::Size(4)])
                .describe()
                .as_deref(),
            Some("target=[-1, 4]"),
            "an inferred axis reads back as NumPy spells it"
        );
    }
}
