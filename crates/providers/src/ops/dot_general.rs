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

use super::error::{MathOpError, check_axis};
use super::inference::promoted_dtype;
use super::{ProgramOp, QISKIT};
use crate::tensor::{DType, Dim, Tensor, TensorError, TensorType};

/// Which axes of each operand play which role in a contraction.
///
/// The lists are matched by position: `lhs_batch[i]` pairs with `rhs_batch[i]`, and `lhs_contract[i]`
/// pairs with `rhs_contract[i]`. Every axis that is not listed is a *free* axis.
///
/// | role     | in the result | effect                                                              |
/// |----------|---------------|---------------------------------------------------------------------|
/// | batch    | kept, once    | the pair is the same index. Sizes must be equal, or one of them `1` |
/// | contract | removed       | multiplied and summed over. Sizes must be equal                     |
/// | free     | kept          | passes through                                                      |
///
/// A batch pair is broadcast: a size of `1` stretches to the size of the other axis. A contracted pair
/// is not.
///
/// The result has the batch axes (in the order of `lhs_batch`), then the free axes of the left
/// operand, then the free axes of the right operand:
///
/// ```text
/// lhs (4, 6, 2)    lhs_batch [1]    lhs_contract [2]     free: axis 0 (size 4)
/// rhs (6, 2, 5)    rhs_batch [0]    rhs_contract [1]     free: axis 2 (size 5)
/// result (6, 4, 5)                  batch 6, then lhs free 4, then rhs free 5
/// ```
/// Broadcasting stretches the size of a paired batch axis, and nothing else. It does not add axes:
/// padding the operand that has fewer axes with leading axes of size `1`, as NumPy's `matmul` does for
/// operands of different rank, is not part of the broadcasting here. An axis that is not listed is a
/// free axis, and it comes after the batch axes. To pair the leading axes of operands of different
/// rank, add axes of size `1` to the shorter one before the op.
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
    /// lists do not fit them. Paired batch axes are broadcast, and paired contracted axes must be
    /// equal.
    pub(crate) fn output_shape(&self, lhs: &[Dim], rhs: &[Dim]) -> Result<Vec<Dim>, MathOpError> {
        // 1. Each pair of lists must have the same length.
        if self.lhs_contract.len() != self.rhs_contract.len() {
            return Err(MathOpError::AxisCountMismatch {
                kind: "contracting",
                lhs: self.lhs_contract.len(),
                rhs: self.rhs_contract.len(),
            });
        }
        if self.lhs_batch.len() != self.rhs_batch.len() {
            return Err(MathOpError::AxisCountMismatch {
                kind: "batch",
                lhs: self.lhs_batch.len(),
                rhs: self.rhs_batch.len(),
            });
        }

        // 2. Every listed axis must exist, and be listed only once for its operand.
        check_axes(0, &self.lhs_contract, &self.lhs_batch, lhs.len())?;
        check_axes(1, &self.rhs_contract, &self.rhs_batch, rhs.len())?;

        // 3. Contracted pairs must be equal. Batch pairs are broadcast, and start the result.
        check_contracted_pairs(&self.lhs_contract, &self.rhs_contract, lhs, rhs)?;
        let mut shape = broadcast_batch_pairs(&self.lhs_batch, &self.rhs_batch, lhs, rhs)?;

        // 4. Then come the free axes of the left operand, and those of the right operand.
        for axis in free_axes(lhs.len(), &self.lhs_contract, &self.lhs_batch) {
            shape.push(lhs[axis]);
        }
        for axis in free_axes(rhs.len(), &self.rhs_contract, &self.rhs_batch) {
            shape.push(rhs[axis]);
        }
        Ok(shape)
    }
}

/// Return the axes of an operand with `ndim` axes that are neither contracted nor batch axes.
fn free_axes(ndim: usize, contract: &[usize], batch: &[usize]) -> Vec<usize> {
    let mut free = Vec::new();
    for axis in 0..ndim {
        if !contract.contains(&axis) && !batch.contains(&axis) {
            free.push(axis);
        }
    }
    free
}

/// Check the axes listed for one operand: each must exist, and none may be listed twice.
fn check_axes(
    operand: usize,
    contract: &[usize],
    batch: &[usize],
    ndim: usize,
) -> Result<(), MathOpError> {
    let all_listed = [contract, batch].concat();
    let mut seen: Vec<usize> = Vec::new();
    for &axis in &all_listed {
        check_axis(axis, ndim)?;
        if seen.contains(&axis) {
            return Err(MathOpError::RepeatedAxis { operand, axis });
        }
        seen.push(axis);
    }
    Ok(())
}

/// Check that each contracted pair (`lhs_axes[i]` with `rhs_axes[i]`) is fixed and equal in size.
fn check_contracted_pairs(
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    lhs: &[Dim],
    rhs: &[Dim],
) -> Result<(), MathOpError> {
    for i in 0..lhs_axes.len() {
        let lhs_axis = lhs_axes[i];
        let rhs_axis = rhs_axes[i];
        match (lhs[lhs_axis], rhs[rhs_axis]) {
            (Dim::Fixed(m), Dim::Fixed(n)) => {
                if m != n {
                    return Err(MathOpError::PairedDimMismatch {
                        kind: "contracting",
                        lhs_axis,
                        rhs_axis,
                        lhs: Dim::Fixed(m),
                        rhs: Dim::Fixed(n),
                    });
                }
            }
            // A size that is only bounded cannot be compared with another one.
            (Dim::Bounded { .. }, _) => {
                return Err(TensorError::DynamicDim {
                    shape: lhs.to_vec(),
                }
                .into());
            }
            (_, Dim::Bounded { .. }) => {
                return Err(TensorError::DynamicDim {
                    shape: rhs.to_vec(),
                }
                .into());
            }
        }
    }
    Ok(())
}

/// Return the size of each batch pair (`lhs_axes[i]` with `rhs_axes[i]`) after broadcasting.
///
/// The two sizes must be equal, or one of them `1`. A bounded size can only meet a size of `1`,
/// because it cannot be compared with another one when the op is added.
/// Only the sizes of paired axes are broadcast. No axis is added: padding an operand with leading
/// axes of size `1``, so that more of its axes pair up, is not part of the broadcasting.
fn broadcast_batch_pairs(
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    lhs: &[Dim],
    rhs: &[Dim],
) -> Result<Vec<Dim>, MathOpError> {
    let mut sizes = Vec::new();
    for i in 0..lhs_axes.len() {
        let lhs_axis = lhs_axes[i];
        let rhs_axis = rhs_axes[i];
        let size = match (lhs[lhs_axis], rhs[rhs_axis]) {
            (Dim::Fixed(m), Dim::Fixed(n)) => {
                if m == n || n == 1 {
                    Dim::Fixed(m)
                } else if m == 1 {
                    Dim::Fixed(n)
                } else {
                    return Err(MathOpError::PairedDimMismatch {
                        kind: "batch",
                        lhs_axis,
                        rhs_axis,
                        lhs: Dim::Fixed(m),
                        rhs: Dim::Fixed(n),
                    });
                }
            }
            (Dim::Bounded { max }, Dim::Fixed(1)) => Dim::Bounded { max },
            (Dim::Fixed(1), Dim::Bounded { max }) => Dim::Bounded { max },
            (Dim::Bounded { .. }, _) => {
                return Err(TensorError::DynamicDim {
                    shape: lhs.to_vec(),
                }
                .into());
            }
            (_, Dim::Bounded { .. }) => {
                return Err(TensorError::DynamicDim {
                    shape: rhs.to_vec(),
                }
                .into());
            }
        };
        sizes.push(size);
    }
    Ok(sizes)
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
/// This follows the semantics of StableHLO and JAX's `dot_general`, except that a batch pair is
/// broadcast: a size of `1` stretches to the size of the other axis. No axes are added, so operands of
/// different rank are not padded. See [`DotDims`] for the axis lists. The operand dtypes are promoted to a common dtype,
/// which is the dtype of the result. Floating point, complex and integer dtypes are accepted.
/// Complex values are multiplied, not conjugated, and integer results wrap on overflow.

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
    use crate::{FunctionError, ProgramFunction};

    /// A tensor type with fixed sizes.
    fn ty(dtype: DType, sizes: &[usize]) -> TensorType {
        TensorType {
            dtype,
            shape: sizes.iter().map(|&n| Dim::Fixed(n)).collect(),
        }
    }

    /// `DotDims` from the four lists, in the order (lhs_contract, rhs_contract, lhs_batch, rhs_batch).
    fn dims(lc: &[usize], rc: &[usize], lb: &[usize], rb: &[usize]) -> DotDims {
        DotDims {
            lhs_contract: lc.to_vec(),
            rhs_contract: rc.to_vec(),
            lhs_batch: lb.to_vec(),
            rhs_batch: rb.to_vec(),
        }
    }

    /// The result type of the op on two `C128` tensors with the given fixed sizes.
    fn infer(d: DotDims, lhs: &[usize], rhs: &[usize]) -> Result<Vec<TensorType>, MathOpError> {
        DotGeneral(d).infer_output_types(&[ty(DType::C128, lhs), ty(DType::C128, rhs)])
    }

    /// Check the result shape for fixed operand sizes.
    fn assert_shape(d: DotDims, lhs: &[usize], rhs: &[usize], expected: &[usize]) {
        let result = infer(d.clone(), lhs, rhs).unwrap();
        assert_eq!(
            result,
            vec![ty(DType::C128, expected)],
            "{d:?} on {lhs:?} and {rhs:?}"
        );
    }

    /// The result type of the op on two `C128` tensors with arbitrary shapes (bounded sizes allowed).
    fn infer_dims(
        d: DotDims,
        lhs: Vec<Dim>,
        rhs: Vec<Dim>,
    ) -> Result<Vec<TensorType>, MathOpError> {
        let lhs_type = TensorType {
            dtype: DType::C128,
            shape: lhs,
        };
        let rhs_type = TensorType {
            dtype: DType::C128,
            shape: rhs,
        };
        DotGeneral(d).infer_output_types(&[lhs_type, rhs_type])
    }

    // ---------- the interface ----------

    #[test]
    fn test_metadata() {
        let op = DotGeneral(dims(&[1], &[0], &[], &[]));
        assert_eq!(op.full_name(), "qiskit.dot_general");
        assert_eq!(op.arity(), 2);
        assert!(!op.has_builtin_eval());
        assert!(matches!(op.eval(&[]), Err(MathOpError::NoBuiltinEval)));
    }

    // ---------- result shapes ----------

    #[test]
    fn test_result_shapes() {
        // A matrix product.
        assert_shape(dims(&[1], &[0], &[], &[]), &[2, 3], &[3, 4], &[2, 4]);
        // The pairwise product of many matrices: one batch axis.
        assert_shape(
            dims(&[2], &[1], &[0], &[0]),
            &[1000, 2, 2],
            &[1000, 2, 2],
            &[1000, 2, 2],
        );
        // One fixed matrix for many: no batch axis, so the left leading axis is a free axis.
        assert_shape(
            dims(&[2], &[0], &[], &[]),
            &[1000, 2, 2],
            &[2, 2],
            &[1000, 2, 2],
        );
        // Two batch axes, which both stretch: every left matrix with every right matrix.
        assert_shape(
            dims(&[3], &[2], &[0, 1], &[0, 1]),
            &[5, 1, 2, 2],
            &[1, 7, 2, 2],
            &[5, 7, 2, 2],
        );
        // The batch axis of the left operand is axis 1, so the result starts with its size.
        assert_shape(
            dims(&[2], &[1], &[1], &[0]),
            &[4, 6, 2],
            &[6, 2, 5],
            &[6, 4, 5],
        );
        // The dot product of two vectors has no axes left.
        assert_shape(dims(&[0], &[0], &[], &[]), &[3], &[3], &[]);
        // Nothing contracted: the outer product.
        assert_shape(dims(&[], &[], &[], &[]), &[3], &[4], &[3, 4]);
        // A batch of matrices times a batch of vectors.
        assert_shape(dims(&[2], &[1], &[0], &[0]), &[2, 3, 2], &[2, 2], &[2, 3]);
        // Two contracted axes.
        assert_shape(
            dims(&[1, 2], &[0, 1], &[], &[]),
            &[5, 2, 3],
            &[2, 3, 7],
            &[5, 7],
        );
    }

    #[test]
    fn test_axes_are_never_added_to_an_operand_with_fewer_axes() {
        // (5, 1, 2, 2) with (7, 2, 2): the extra leading axis of the left operand has no partner, so it
        // is a free axis, and the batch axis comes first. The result is not (5, 7, 2, 2), as it is for
        // the matmul of NumPy.
        assert_shape(
            dims(&[3], &[1], &[1], &[0]),
            &[5, 1, 2, 2],
            &[7, 2, 2],
            &[7, 5, 2, 2],
        );

        // To pair both leading axes, the right operand would need a batch axis for each of them. It is
        // not padded to get one.
        let result = infer(dims(&[3], &[1], &[0, 1], &[0]), &[5, 1, 2, 2], &[7, 2, 2]);
        assert!(matches!(
            result.unwrap_err(),
            MathOpError::AxisCountMismatch {
                kind: "batch",
                lhs: 2,
                rhs: 1
            }
        ));
    }

    #[test]
    fn test_axis_list_errors() {
        let (lhs, rhs) = (&[2, 3][..], &[3, 4][..]);
        let error = |d: DotDims| infer(d, lhs, rhs).unwrap_err();

        // The two lists of a pair have different lengths.
        assert!(matches!(
            error(dims(&[1], &[], &[], &[])),
            MathOpError::AxisCountMismatch {
                kind: "contracting",
                lhs: 1,
                rhs: 0
            }
        ));
        assert!(matches!(
            error(dims(&[], &[], &[0], &[])),
            MathOpError::AxisCountMismatch {
                kind: "batch",
                lhs: 1,
                rhs: 0
            }
        ));
        // An axis is listed twice for one operand: in two lists, or twice in one list.
        assert!(matches!(
            error(dims(&[0], &[0], &[0], &[1])),
            MathOpError::RepeatedAxis {
                operand: 0,
                axis: 0
            }
        ));
        assert!(matches!(
            error(dims(&[1, 1], &[0, 1], &[], &[])),
            MathOpError::RepeatedAxis {
                operand: 0,
                axis: 1
            }
        ));
        // An axis does not exist, in the left or in the right operand.
        assert!(matches!(
            error(dims(&[2], &[0], &[], &[])),
            MathOpError::InvalidAxis { axis: 2, ndim: 2 }
        ));
        assert!(matches!(
            error(dims(&[1], &[2], &[], &[])),
            MathOpError::InvalidAxis { axis: 2, ndim: 2 }
        ));
    }

    #[test]
    fn test_sizes_that_cannot_be_paired() {
        // Contracted axes must be equal. A size of 1 does not stretch there.
        assert!(matches!(
            infer(dims(&[1], &[0], &[], &[]), &[2, 3], &[4, 5]).unwrap_err(),
            MathOpError::PairedDimMismatch {
                kind: "contracting",
                ..
            }
        ));
        assert!(matches!(
            infer(dims(&[1], &[0], &[], &[]), &[2, 1], &[3, 5]).unwrap_err(),
            MathOpError::PairedDimMismatch {
                kind: "contracting",
                ..
            }
        ));
        // Batch axes must be equal or one of them 1.
        assert!(matches!(
            infer(dims(&[2], &[1], &[0], &[0]), &[3, 2, 2], &[4, 2, 2]).unwrap_err(),
            MathOpError::PairedDimMismatch { kind: "batch", .. }
        ));
    }

    #[test]
    fn test_bounded_axes() {
        // Which bounded batch pairs broadcast is checked against the elementwise rule in
        // `test_a_batch_pair_broadcasts_like_an_axis_of_an_elementwise_op`.
        let bounded = Dim::Bounded { max: 4000 };
        let two = Dim::Fixed(2);

        // A bounded free axis passes through: one fixed matrix for an unknown number of matrices.
        let result = infer_dims(
            dims(&[2], &[0], &[], &[]),
            vec![bounded, two, two],
            vec![two, two],
        );
        assert_eq!(result.unwrap()[0].shape, vec![bounded, two, two]);

        // A bounded batch axis cannot be compared with a size other than 1.
        let result = infer_dims(
            dims(&[2], &[1], &[0], &[0]),
            vec![bounded, two, two],
            vec![Dim::Fixed(5), two, two],
        );
        assert!(matches!(
            result.unwrap_err(),
            MathOpError::Tensor(TensorError::DynamicDim { .. })
        ));

        // A bounded contracted axis cannot be compared at all.
        let result = infer_dims(
            dims(&[1], &[0], &[], &[]),
            vec![two, bounded],
            vec![bounded, two],
        );
        assert!(matches!(
            result.unwrap_err(),
            MathOpError::Tensor(TensorError::DynamicDim { .. })
        ));
    }

    #[test]
    fn test_a_batch_pair_broadcasts_like_an_axis_of_an_elementwise_op() {
        use crate::tensor::broadcast_dims;
        let sizes = [
            Dim::Fixed(0),
            Dim::Fixed(1),
            Dim::Fixed(2),
            Dim::Fixed(3),
            Dim::Bounded { max: 4 },
            Dim::Bounded { max: 5 },
        ];
        // One batch axis and nothing else: a pair of sizes in, a size out.
        let op = dims(&[], &[], &[0], &[0]);
        for lhs in sizes {
            for rhs in sizes {
                let ours = op.output_shape(&[lhs], &[rhs]);
                let elementwise = broadcast_dims(&[lhs], &[rhs]);
                match (ours, elementwise) {
                    (Ok(ours), Ok(elementwise)) => {
                        assert_eq!(ours, elementwise, "{lhs} with {rhs}")
                    }
                    (Err(_), Err(_)) => {}
                    (ours, elementwise) => {
                        panic!("{lhs} with {rhs}: {ours:?} here, {elementwise:?} elementwise")
                    }
                }
            }
        }
    }

    // ---------- dtypes ----------

    #[test]
    fn test_dtypes_promote_and_the_result_keeps_the_promoted_dtype() {
        let op = DotGeneral(dims(&[1], &[0], &[], &[]));
        let cases = [
            (DType::F32, DType::C64, DType::C64),
            (DType::F64, DType::C64, DType::C128),
            (DType::I32, DType::F32, DType::F64),
            // Integers keep their promoted dtype: there is no widening.
            (DType::I16, DType::I16, DType::I16),
            // An integer pair that the promotion table sends to a float.
            (DType::U64, DType::I64, DType::F64),
        ];
        for (lhs, rhs, expected) in cases {
            let result = op.infer_output_types(&[ty(lhs, &[2, 2]), ty(rhs, &[2, 2])]);
            assert_eq!(
                result.unwrap(),
                vec![ty(expected, &[2, 2])],
                "{lhs} with {rhs}"
            );
        }
    }

    #[test]
    fn test_every_dtype_pair_is_accepted_or_rejected_cleanly() {
        use crate::tensor::PrngKind;
        use DType::*;
        let all = [
            Bit, U8, U16, U32, U64, I8, I16, I32, I64, F32, F64, C64, C128,
        ];
        let op = DotGeneral(dims(&[1], &[0], &[], &[]));
        for lhs in all {
            for rhs in all {
                let result = op.infer_output_types(&[ty(lhs, &[2, 2]), ty(rhs, &[2, 2])]);
                match result {
                    // Accepted: the result must be a dtype the op computes in.
                    Ok(out) => assert!(numeric(out[0].dtype), "{lhs} with {rhs}"),
                    // Rejected: only two `Bit` operands promote to `Bit`.
                    Err(MathOpError::UnsupportedPromotion { dtype: Bit, .. }) => {
                        assert_eq!((lhs, rhs), (Bit, Bit));
                    }
                    Err(other) => panic!("{lhs} with {rhs}: unexpected error {other}"),
                }
            }
        }
        // A PRNG key has no common dtype with anything, so it is rejected before the dtype filter.
        let key = Key(PrngKind::Philox4x32);
        for other in all.into_iter().chain([key]) {
            for (lhs, rhs) in [(key, other), (other, key)] {
                let result = op.infer_output_types(&[ty(lhs, &[2, 2]), ty(rhs, &[2, 2])]);
                assert!(
                    matches!(result.unwrap_err(), MathOpError::NoPromotion { .. }),
                    "{lhs} with {rhs}"
                );
            }
        }
    }

    #[test]
    fn test_a_function_checks_the_op_when_it_is_added() {
        let mut f = ProgramFunction::new();
        let a = f.add_parameter(ty(DType::C128, &[1000, 2, 2]));
        let b = f.add_parameter(ty(DType::C128, &[1000, 2, 2]));
        let c = f.add_parameter(ty(DType::C128, &[999, 2, 2]));

        // Matching batch sizes: accepted, with the inferred type.
        let pairwise = dims(&[2], &[1], &[0], &[0]);
        let out = f.add_op(DotGeneral(pairwise.clone()), &[a, b]).unwrap()[0];
        assert_eq!(f.type_of(out).unwrap(), &ty(DType::C128, &[1000, 2, 2]));

        // Batch sizes that cannot be paired: rejected, and nothing is added.
        let count_before = f.instruction_count();
        let result = f.add_op(DotGeneral(pairwise), &[a, c]);
        assert!(matches!(result, Err(FunctionError::TypeError { .. })));
        assert_eq!(f.instruction_count(), count_before);
    }
}
