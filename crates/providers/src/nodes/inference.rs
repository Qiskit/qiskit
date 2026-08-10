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

//! How operand types become result types, written once per family of node.
//!
//! A node type names one of these rather than spelling out its own inference, contributing only the
//! dtypes it accepts and, where a family allows it, how its result dtype follows from its operand's.
//! The number of distinct rules in the catalogue is therefore the number of functions here, which is
//! what keeps two nodes in the same family from drifting apart.
//!
//! The rules coerce: an elementwise operation promotes its operand dtypes and broadcasts their
//! shapes rather than demanding that a caller insert conversions. Nothing is lost by that, because a
//! node records both its operand types and its result types, so the explicit form remains
//! reconstructible from a built program.

use super::error::MathNodeError;
use crate::tensor::rules::{broadcast_dims, promotion};
use crate::tensor::{DType, TensorType};

/// Which dtypes a node admits.
///
/// A node accepts only what its evaluation covers, so that a dtype it cannot compute is rejected
/// when the node is added rather than when it runs.
pub(super) type Accepts = fn(DType) -> bool;

/// How a node's result dtype follows from its operand's.
pub(super) type ResultDType = fn(DType) -> DType;

/// Admit every dtype.
pub(super) fn any(_dtype: DType) -> bool {
    true
}

/// Type an elementwise operation over two operands: dtypes promote, shapes broadcast.
///
/// The accepted set is checked against the *promoted* dtype rather than each operand's, because that
/// is the dtype the operation will actually compute in. So `add` admits a `Bit` operand alongside an
/// `F64` one, which promote to `F64`, while refusing two `Bit` operands, which do not promote to
/// anything it implements.
pub(super) fn elementwise_binary(
    x: &TensorType,
    y: &TensorType,
    accepts: Accepts,
) -> Result<TensorType, MathNodeError> {
    let dtype = promotion(x.dtype, y.dtype);
    if !accepts(dtype) {
        return Err(MathNodeError::UnsupportedPromotion {
            lhs: x.dtype,
            rhs: y.dtype,
            dtype,
        });
    }
    Ok(TensorType {
        dtype,
        shape: broadcast_dims(&x.shape, &y.shape)?,
    })
}

/// Type an elementwise operation over one operand: dtype and shape forwarded untouched.
///
/// Being elementwise and unary, this needs no size for any axis, so a bounded one passes straight
/// through.
pub(super) fn elementwise_unary(
    x: &TensorType,
    accepts: Accepts,
) -> Result<TensorType, MathNodeError> {
    check_accepts(x.dtype, accepts)?;
    Ok(x.clone())
}

/// Type a reduction along `axis`, which the result does not have.
///
/// The result dtype is the family's own function of the operand's, not a promotion: averaging a bit
/// tensor yields a float, which no promotion rule could express.
pub(super) fn reduce(
    x: &TensorType,
    axis: usize,
    accepts: Accepts,
    result_dtype: ResultDType,
) -> Result<TensorType, MathNodeError> {
    check_accepts(x.dtype, accepts)?;
    Ok(TensorType {
        dtype: result_dtype(x.dtype),
        shape: super::error::reduced_shape(axis, &x.shape)?,
    })
}

/// Validate that a single operand's dtype is admitted.
fn check_accepts(dtype: DType, accepts: Accepts) -> Result<(), MathNodeError> {
    if !accepts(dtype) {
        return Err(MathNodeError::UnsupportedDType { operand: 0, dtype });
    }
    Ok(())
}

/// The dtype an elementwise binary operation will compute in, for an evaluation that has tensors in
/// hand rather than types.
///
/// Evaluation coerces its operands to this before computing, which is what makes the tensors agree
/// with the types inference promised.
pub(super) fn promoted_dtype(x: DType, y: DType, accepts: Accepts) -> Result<DType, MathNodeError> {
    let dtype = promotion(x, y);
    if !accepts(dtype) {
        return Err(MathNodeError::UnsupportedPromotion {
            lhs: x,
            rhs: y,
            dtype,
        });
    }
    Ok(dtype)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::tensor::{Dim, TensorError};

    /// A `TensorType` of `dtype` over fixed axes `shape`.
    fn ty(dtype: DType, shape: &[usize]) -> TensorType {
        TensorType {
            dtype,
            shape: shape.iter().copied().map(Dim::Fixed).collect(),
        }
    }

    /// Admit only the floating-point dtypes.
    fn floats(dtype: DType) -> bool {
        matches!(dtype, DType::F32 | DType::F64)
    }

    #[test]
    fn test_elementwise_binary_promotes_the_dtypes_and_broadcasts_the_shapes() {
        assert_eq!(
            elementwise_binary(&ty(DType::Bit, &[3, 1]), &ty(DType::F32, &[4]), any).unwrap(),
            ty(DType::F32, &[3, 4])
        );
        assert_eq!(
            elementwise_binary(&ty(DType::I32, &[2]), &ty(DType::U32, &[2]), any).unwrap(),
            ty(DType::I64, &[2])
        );
    }

    #[test]
    fn test_elementwise_binary_admits_the_promoted_dtype_rather_than_each_operand() {
        assert_eq!(
            elementwise_binary(&ty(DType::Bit, &[2]), &ty(DType::F64, &[2]), floats).unwrap(),
            ty(DType::F64, &[2]),
            "a Bit operand beside a float promotes to a float"
        );
        let err =
            elementwise_binary(&ty(DType::Bit, &[2]), &ty(DType::Bit, &[2]), floats).unwrap_err();
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
            "operands of dtype Bit and Bit promote to Bit, which is not supported",
            "both operand dtypes and the promotion they were refused for are named"
        );
    }

    #[test]
    fn test_elementwise_binary_reports_shapes_that_do_not_broadcast() {
        assert_eq!(
            elementwise_binary(&ty(DType::F64, &[3]), &ty(DType::F64, &[4]), any).unwrap_err(),
            MathNodeError::Tensor(TensorError::DimShapeMismatch {
                lhs: vec![Dim::Fixed(3)],
                rhs: vec![Dim::Fixed(4)],
            })
        );
    }

    #[test]
    fn test_elementwise_unary_keeps_the_operand_type() {
        let x = TensorType {
            dtype: DType::C64,
            shape: vec![Dim::Bounded { max: 10 }, Dim::Fixed(2)],
        };
        assert_eq!(elementwise_unary(&x, any).unwrap(), x);

        let err = elementwise_unary(&x, floats).unwrap_err();
        assert_eq!(
            err,
            MathNodeError::UnsupportedDType {
                operand: 0,
                dtype: DType::C64,
            }
        );
        assert_eq!(err.to_string(), "operand 0: dtype C64 is not supported");
    }

    #[test]
    fn test_reduce_drops_the_axis_and_maps_the_dtype() {
        let x = TensorType {
            dtype: DType::Bit,
            shape: vec![Dim::Fixed(4), Dim::Bounded { max: 10 }, Dim::Fixed(2)],
        };
        assert_eq!(
            reduce(&x, 1, any, |_| DType::F64).unwrap(),
            ty(DType::F64, &[4, 2])
        );
        assert_eq!(
            reduce(&x, 0, any, |dtype| dtype).unwrap(),
            TensorType {
                dtype: DType::Bit,
                shape: vec![Dim::Bounded { max: 10 }, Dim::Fixed(2)],
            },
            "an axis the reduction keeps is unchanged, bound and all"
        );
    }

    #[test]
    fn test_reduce_reports_an_axis_the_operand_does_not_have() {
        let err = reduce(&ty(DType::F64, &[2, 3]), 2, any, |dtype| dtype).unwrap_err();
        assert_eq!(err, MathNodeError::InvalidAxis { axis: 2, ndim: 2 });
        assert_eq!(
            err.to_string(),
            "axis 2 is out of bounds for tensor with 2 dimension(s)"
        );
    }

    #[test]
    fn test_promoted_dtype_is_the_dtype_elementwise_binary_infers() {
        for (lhs, rhs) in [
            (DType::Bit, DType::F64),
            (DType::I32, DType::U64),
            (DType::F32, DType::C64),
        ] {
            assert_eq!(
                promoted_dtype(lhs, rhs, any).unwrap(),
                elementwise_binary(&ty(lhs, &[2]), &ty(rhs, &[2]), any)
                    .unwrap()
                    .dtype
            );
        }
        assert_eq!(
            promoted_dtype(DType::Bit, DType::Bit, floats).unwrap_err(),
            MathNodeError::UnsupportedPromotion {
                lhs: DType::Bit,
                rhs: DType::Bit,
                dtype: DType::Bit,
            },
            "a promotion the node does not admit is refused at evaluation time too"
        );
    }
}
