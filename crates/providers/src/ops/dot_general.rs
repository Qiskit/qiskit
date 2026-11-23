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

//! General tensor contraction, following StableHLO `dot_general`.

use super::error::MathOpError;
use super::inference::promoted_dtype;
use super::{ProgramOp, QISKIT};
use crate::tensor::{DType, Dim, Tensor, TensorType};

/// Which axes of each operand play which role in a contraction.
///
/// The lists are matched by position: `lhs_batch[i]` pairs with `rhs_batch[i]`, and `lhs_contract[i]`
/// pairs with `rhs_contract[i]`. Every axis that is not listed is a *free* axis.
///
/// | role     | in the result | effect                                               |
/// |----------|---------------|------------------------------------------------------|
/// | batch    | kept, once    | the pair is the same index, it is not multiplied out |
/// | contract | removed       | multiplied and summed over                           |
/// | free     | kept          | passes through                                       |
///
/// The result has the batch axes (in the order of `lhs_batch`), then the free axes of the left
/// operand, then the free axes of the right operand:
///
/// ```text
/// lhs (4, 6, 2)    lhs_batch [1]    lhs_contract [2]     free: axis 0 (size 4)
/// rhs (6, 2, 5)    rhs_batch [0]    rhs_contract [1]     free: axis 2 (size 5)
/// result (6, 4, 5)                  batch 6, then lhs free 4, then rhs free 5
/// ```
#[derive(Clone, Debug)]
pub struct DotDims {
    /// Batch axes of the left operand.
    pub lhs_batch: Vec<usize>,
    /// Batch axes of the right operand.
    pub rhs_batch: Vec<usize>,
    /// Contracted axes of the left operand.
    pub lhs_contract: Vec<usize>,
    /// Contracted axes of the right operand.
    pub rhs_contract: Vec<usize>,
}

impl DotDims {
    /// Return the shape of the result for operands of shapes `lhs` and `rhs`, or an error if the axis
    /// lists do not fit them. Paired axes must be fixed and of equal size, and nothing is broadcast.
    pub(crate) fn output_shape(&self, lhs: &[Dim], rhs: &[Dim]) -> Result<Vec<Dim>, MathOpError> {
        let _ = (lhs, rhs);
        todo!()
    }
}

/// Admit the dtypes this op computes in: floating point, complex, and signed and unsigned integers.
fn numeric(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::F32
            | DType::F64
            | DType::C64
            | DType::C128
            | DType::I8
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::U8
            | DType::U16
            | DType::U32
            | DType::U64
    )
}

/// Contract two tensors over chosen axes, keeping chosen batch axes.
///
/// This follows the semantics of StableHLO and JAX's `dot_general`. See [`DotDims`] for the axis
/// lists. The operand dtypes are promoted to a common dtype, which is the dtype of the result.
/// Floating point, complex and integer dtypes are accepted. Complex values are multiplied, not
/// conjugated, and integer results wrap on overflow.
#[derive(Clone, Debug)]
pub struct DotGeneral(pub DotDims);

impl ProgramOp for DotGeneral {
    type Error = MathOpError;

    fn name(&self) -> &str {
        "dot_general"
    }
    fn namespace(&self) -> &str {
        QISKIT
    }
    fn arity(&self) -> usize {
        2
    }
    fn has_builtin_eval(&self) -> bool {
        false
    }
    fn infer_output_types(&self, inputs: &[TensorType]) -> Result<Vec<TensorType>, Self::Error> {
        crate::unpack_operands!(self, inputs, [x, y]);
        let dtype = promoted_dtype(x.dtype, y.dtype, numeric)?;
        let shape = self.0.output_shape(&x.shape, &y.shape)?;
        Ok(vec![TensorType { dtype, shape }])
    }
    fn eval(&self, _args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
        Err(MathOpError::NoBuiltinEval)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// A tensor type with fixed sizes.
    fn ty(dtype: DType, sizes: &[usize]) -> TensorType {
        TensorType {
            dtype,
            shape: sizes.iter().map(|&n| Dim::Fixed(n)).collect(),
        }
    }

    /// A matrix product: contract the last axis of the left operand with the first of the right one.
    fn matrix_product() -> DotGeneral {
        DotGeneral(DotDims {
            lhs_batch: vec![],
            rhs_batch: vec![],
            lhs_contract: vec![1],
            rhs_contract: vec![0],
        })
    }

    #[test]
    fn test_metadata() {
        let op = matrix_product();
        assert_eq!(op.full_name(), "qiskit.dot_general");
        assert_eq!(op.arity(), 2);
        assert!(!op.has_builtin_eval());
        assert!(matches!(op.eval(&[]), Err(MathOpError::NoBuiltinEval)));
    }

    #[test]
    fn test_numeric_dtypes() {
        use DType::*;
        for dtype in [F32, F64, C64, C128, I8, I16, I32, I64, U8, U16, U32, U64] {
            assert!(numeric(dtype), "{dtype}");
        }
        assert!(!numeric(Bit));
    }

    #[test]
    fn test_two_bit_operands_are_rejected_before_the_shapes_are_looked_at() {
        let result = matrix_product()
            .infer_output_types(&[ty(DType::Bit, &[2, 2]), ty(DType::Bit, &[2, 2])]);
        assert!(matches!(
            result.unwrap_err(),
            MathOpError::UnsupportedPromotion {
                dtype: DType::Bit,
                ..
            }
        ));
    }
}
