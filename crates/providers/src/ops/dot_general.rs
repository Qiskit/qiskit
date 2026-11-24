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

use ndarray::{ArcArrayD, Array2, Array3, ArrayView2, Axis, IxDyn, LinalgScalar};
use num_complex::{Complex32, Complex64};

use super::error::{MathOpError, check_axis};
use super::inference::promoted_dtype;
use super::{ProgramOp, QISKIT};
use crate::tensor::{DType, Dim, Tensor, TensorError, TensorType, dim_shape, require_static};

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
/// A batch pair is broadcast: a size of `1` stretches to the size of the other axis. A contracted
/// pair is not. Broadcasting does not add axes: an operand with fewer axes is not padded with
/// leading axes of size `1`, as NumPy's `matmul` does. To pair the leading axes of operands of
/// different rank, add axes of size `1` to the shorter one before the op.
///
/// A paired axis must have a fixed size, except that a bounded batch axis may meet a size of `1`:
/// a size that is only known at run time cannot be compared when the op is added.
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

    /// Contract `lhs` with `rhs`, returning the result of shape `out_shape`.
    ///
    /// Each operand is read as a stack of matrices: its axes are put in the order batch, free,
    /// contract, its batch axes are stretched to the batch sizes of the result, and each group of axes
    /// is merged into one. That gives `(B, M, K)` for the left operand and `(B, N, K)` for the right
    /// one. Each matrix of the result is `lhs[t]` times the transpose of `rhs[t]`, computed by
    /// [`Element::product`].
    ///
    /// `out_shape` must be what [`DotDims::output_shape`] gives for the two operands, so its first axes
    /// are the batch axes.
    fn contract<T: Element>(
        &self,
        lhs: &ArcArrayD<T>,
        rhs: &ArcArrayD<T>,
        out_shape: &[usize],
    ) -> ArcArrayD<T> {
        let batch_sizes = &out_shape[..self.lhs_batch.len()];
        let b: usize = batch_sizes.iter().product();
        let l = StackLayout::new(
            lhs.shape(),
            &self.lhs_batch,
            &self.lhs_contract,
            batch_sizes,
        );
        let r = StackLayout::new(
            rhs.shape(),
            &self.rhs_batch,
            &self.rhs_contract,
            batch_sizes,
        );

        // Reordering and stretching give views. Making a view row-major copies only if it is not row-major
        // already, and merging the groups of axes then needs no copy.
        let lhs = lhs.view().permuted_axes(l.order);
        let lhs = lhs
            .broadcast(IxDyn(&l.stretched))
            .expect("the batch axes were checked to broadcast");
        let lhs = lhs.as_standard_layout();
        let lhs = lhs
            .view()
            .into_shape_with_order((b, l.rows, l.columns))
            .expect("the group sizes multiply to the element count");

        let rhs = rhs.view().permuted_axes(r.order);
        let rhs = rhs
            .broadcast(IxDyn(&r.stretched))
            .expect("the batch axes were checked to broadcast");
        let rhs = rhs.as_standard_layout();
        let rhs = rhs
            .view()
            .into_shape_with_order((b, r.rows, r.columns))
            .expect("the group sizes multiply to the element count");

        let mut out = Array3::<T>::zeros((b, l.rows, r.rows));
        for t in 0..b {
            let product = T::product(lhs.index_axis(Axis(0), t), rhs.index_axis(Axis(0), t));
            out.index_axis_mut(Axis(0), t).assign(&product);
        }
        out.into_shape_with_order(IxDyn(out_shape))
            .expect("the result has B * M * N elements")
            .into_shared()
    }
}

/// How [`DotDims::contract`] reads one operand as a stack of matrices.
struct StackLayout {
    /// The axes of the operand in the order batch, free, contract.
    order: Vec<usize>,
    /// The shape after reordering, with the batch axes at the batch sizes of the result.
    stretched: Vec<usize>,
    /// The number of rows of each matrix: the product of the sizes of the free axes.
    rows: usize,
    /// The number of columns of each matrix: the product of the sizes of the contracted axes.
    columns: usize,
}

impl StackLayout {
    /// The layout of an operand of shape `shape`, with the given batch and contracted axes, whose batch
    /// axes are stretched to `batch_sizes`.
    fn new(shape: &[usize], batch: &[usize], contract: &[usize], batch_sizes: &[usize]) -> Self {
        let free = free_axes(shape.len(), contract, batch);
        let mut stretched = batch_sizes.to_vec();
        for &axis in free.iter().chain(contract) {
            stretched.push(shape[axis]);
        }
        Self {
            order: [batch, &free, contract].concat(),
            stretched,
            rows: free.iter().map(|&axis| shape[axis]).product(),
            columns: contract.iter().map(|&axis| shape[axis]).product(),
        }
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
///
/// Only the sizes of paired axes are broadcast. No axis is added: padding an operand with leading
/// axes of size `1`, so that more of its axes pair up, is not part of the broadcasting.
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
///
/// This lists the dtypes rather than excluding `Bit`, so that a dtype added later is not accepted by
/// default.
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

/// An element type that the op computes in, with the product of two matrices that it uses.
trait Element: LinalgScalar {
    /// Return `lhs` times the transpose of `rhs`, where `lhs` is `(M, K)` and `rhs` is `(N, K)`.
    fn product(lhs: ArrayView2<'_, Self>, rhs: ArrayView2<'_, Self>) -> Array2<Self>;
}

/// Floating point and complex numbers use the matrix product of `ndarray`.
macro_rules! impl_element_with_dot {
    ($($t:ty),*) => {
        $(
            impl Element for $t {
                fn product(lhs: ArrayView2<'_, Self>, rhs: ArrayView2<'_, Self>) -> Array2<Self> {
                    lhs.dot(&rhs.t())
                }
            }
        )*
    };
}
impl_element_with_dot!(f32, f64, Complex32, Complex64);

/// Integers add up the products with wrapping arithmetic, so that a result that does not fit wraps around,
/// as in the elementwise integer ops. The matrix product of `ndarray` would panic on overflow in a debug
/// build.
macro_rules! impl_element_wrapping {
    ($($t:ty),*) => {
        $(
            impl Element for $t {
                fn product(lhs: ArrayView2<'_, Self>, rhs: ArrayView2<'_, Self>) -> Array2<Self> {
                    let (m, k) = lhs.dim();
                    let n = rhs.dim().0;
                    let mut out = Array2::<Self>::zeros((m, n));
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum: $t = 0;
                            for l in 0..k {
                                sum = sum.wrapping_add(lhs[[i, l]].wrapping_mul(rhs[[j, l]]));
                            }
                            out[[i, j]] = sum;
                        }
                    }
                    out
                }
            }
        )*
    };
}
impl_element_wrapping!(i8, i16, i32, i64, u8, u16, u32, u64);

/// Contract two tensors over chosen axes, keeping chosen batch axes.
///
/// This follows the semantics of StableHLO and JAX's `dot_general`, except that a batch pair is
/// broadcast: a size of `1` stretches to the size of the other axis. No axes are added, so operands of
/// different rank are not padded. See [`DotDims`] for the axis lists. The operand dtypes are promoted 
/// to a common dtype,  which is the dtype of the result. Floating point, complex and integer dtypes are 
/// accepted. Complex values are multiplied, not conjugated, and integer results wrap on overflow.
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
        true
    }
    fn infer_output_types(&self, inputs: &[TensorType]) -> Result<Vec<TensorType>, Self::Error> {
        crate::unpack_operands!(self, inputs, [x, y]);
        let dtype = promoted_dtype(x.dtype, y.dtype, numeric)?;
        let shape = self.0.output_shape(&x.shape, &y.shape)?;
        Ok(vec![TensorType { dtype, shape }])
    }
    fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
        crate::unpack_operands!(self, args, [x, y]);
        let dtype = promoted_dtype(x.dtype(), y.dtype(), numeric)?;

        // A tensor value has fixed sizes, so the result shape has fixed sizes too.
        let x_shape = dim_shape(x.shape().iter().copied());
        let y_shape = dim_shape(y.shape().iter().copied());
        let out_shape = require_static(&self.0.output_shape(&x_shape, &y_shape)?)?;

        // Bring both operands to the common dtype, then compute in it.
        let (x, y) = (x.clone().cast(dtype), y.clone().cast(dtype));
        let dims = &self.0;
        let result = match (&x, &y) {
            (Tensor::F32(a), Tensor::F32(b)) => Tensor::F32(dims.contract(a, b, &out_shape)),
            (Tensor::F64(a), Tensor::F64(b)) => Tensor::F64(dims.contract(a, b, &out_shape)),
            (Tensor::C64(a), Tensor::C64(b)) => Tensor::C64(dims.contract(a, b, &out_shape)),
            (Tensor::C128(a), Tensor::C128(b)) => Tensor::C128(dims.contract(a, b, &out_shape)),
            (Tensor::I8(a), Tensor::I8(b)) => Tensor::I8(dims.contract(a, b, &out_shape)),
            (Tensor::I16(a), Tensor::I16(b)) => Tensor::I16(dims.contract(a, b, &out_shape)),
            (Tensor::I32(a), Tensor::I32(b)) => Tensor::I32(dims.contract(a, b, &out_shape)),
            (Tensor::I64(a), Tensor::I64(b)) => Tensor::I64(dims.contract(a, b, &out_shape)),
            (Tensor::U8(a), Tensor::U8(b)) => Tensor::U8(dims.contract(a, b, &out_shape)),
            (Tensor::U16(a), Tensor::U16(b)) => Tensor::U16(dims.contract(a, b, &out_shape)),
            (Tensor::U32(a), Tensor::U32(b)) => Tensor::U32(dims.contract(a, b, &out_shape)),
            (Tensor::U64(a), Tensor::U64(b)) => Tensor::U64(dims.contract(a, b, &out_shape)),
            _ => unreachable!("both operands were cast to {dtype}"),
        };
        Ok(vec![result])
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{FunctionError, ProgramFunction};
    use ndarray::ArrayD;

    /// An array of the given shape, filled from `data` in row-major order.
    fn array<T: Clone>(sizes: &[usize], data: &[T]) -> ArcArrayD<T> {
        ArrayD::from_shape_vec(IxDyn(sizes), data.to_vec())
            .unwrap()
            .into_shared()
    }

    /// An `F64` tensor.
    fn f64s(sizes: &[usize], data: &[f64]) -> Tensor {
        Tensor::F64(array(sizes, data))
    }

    /// An `F64` tensor holding 1, 2, 3, ... in row-major order.
    fn counting(sizes: &[usize]) -> Tensor {
        let count: usize = sizes.iter().product();
        let data: Vec<f64> = (1..=count).map(|n| n as f64).collect();
        f64s(sizes, &data)
    }

    /// A tensor type with fixed sizes.
    fn ty(dtype: DType, sizes: &[usize]) -> TensorType {
        TensorType {
            dtype,
            shape: dim_shape(sizes.iter().copied()),
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
        assert!(op.has_builtin_eval());
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

    // ---------- values ----------

    #[test]
    fn test_matrix_product() {
        let op = DotGeneral(dims(&[1], &[0], &[], &[]));
        let a = f64s(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = f64s(&[2, 2], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(
            op.eval(&[a, b]).unwrap(),
            vec![f64s(&[2, 2], &[19.0, 22.0, 43.0, 50.0])]
        );

        // Not square: (2x3) times (3x1).
        let a = f64s(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = f64s(&[3, 1], &[1.0, 0.0, -1.0]);
        assert_eq!(
            op.eval(&[a, b]).unwrap(),
            vec![f64s(&[2, 1], &[-2.0, -2.0])]
        );
    }

    #[test]
    fn test_one_matrix_for_many_with_or_without_a_batch_axis() {
        // Three matrices times one matrix. The one matrix is given without a batch axis, and again with
        // a batch axis of size 1 that stretches to 3. Both give the same values. Values from NumPy.
        let m = [1.0, 2.0, 3.0, 4.0];
        let expected = f64s(
            &[3, 2, 2],
            &[
                7.0, 10.0, 15.0, 22.0, 23.0, 34.0, 31.0, 46.0, 39.0, 58.0, 47.0, 70.0,
            ],
        );
        let no_batch = DotGeneral(dims(&[2], &[0], &[], &[]));
        let result = no_batch.eval(&[counting(&[3, 2, 2]), f64s(&[2, 2], &m)]);
        assert_eq!(result.unwrap(), vec![expected.clone()]);
        let stretched = DotGeneral(dims(&[2], &[1], &[0], &[0]));
        let result = stretched.eval(&[counting(&[3, 2, 2]), f64s(&[1, 2, 2], &m)]);
        assert_eq!(result.unwrap(), vec![expected]);
    }

    #[test]
    fn test_batch_axes_that_stretch_on_both_operands() {
        // (5, 1, 2, 2) with (1, 7, 2, 2). Values from NumPy.
        let op = DotGeneral(dims(&[3], &[2], &[0, 1], &[0, 1]));
        let out = op
            .eval(&[counting(&[5, 1, 2, 2]), counting(&[1, 7, 2, 2])])
            .unwrap();
        assert_eq!(out[0].shape(), &[5, 7, 2, 2]);
        let Tensor::F64(c) = &out[0] else {
            panic!("expected F64")
        };
        assert_eq!(c.sum(), 42770.0);
        let at = |i: usize, j: usize| -> Vec<f64> {
            c.index_axis(Axis(0), i)
                .index_axis(Axis(0), j)
                .iter()
                .copied()
                .collect()
        };
        assert_eq!(at(0, 0), [7.0, 10.0, 15.0, 22.0]);
        assert_eq!(at(2, 3), [267.0, 286.0, 323.0, 346.0]);
        assert_eq!(at(4, 6), [911.0, 946.0, 1015.0, 1054.0]);
    }

    #[test]
    fn test_a_batch_axis_in_the_middle_that_stretches() {
        // lhs (4, 1, 2): free 4, batch 1, contract 2. rhs (3, 2, 5): batch 3, contract 2, free 5.
        // The batch pair stretches from 1 to 3. Values from NumPy.
        let op = DotGeneral(dims(&[2], &[1], &[1], &[0]));
        let out = op
            .eval(&[counting(&[4, 1, 2]), counting(&[3, 2, 5])])
            .unwrap();
        assert_eq!(out[0].shape(), &[3, 4, 5]);
        let Tensor::F64(c) = &out[0] else {
            panic!("expected F64")
        };
        assert_eq!(c.sum(), 8520.0);
        assert_eq!(c[IxDyn(&[0, 0, 4])], 25.0);
        assert_eq!(c[IxDyn(&[2, 3, 0])], 355.0);
    }

    #[test]
    fn test_batch_pairs_listed_in_a_different_order() {
        // lhs batch axes (0, 1) pair with rhs batch axes (1, 0). Values from NumPy.
        let op = DotGeneral(dims(&[3], &[2], &[0, 1], &[1, 0]));
        let lhs = counting(&[2, 3, 2, 2]);
        let out = op.eval(&[lhs.clone(), counting(&[3, 2, 2, 2])]).unwrap();
        assert_eq!(out[0].shape(), &[2, 3, 2, 2]);
        let Tensor::F64(c) = &out[0] else {
            panic!("expected F64")
        };
        assert_eq!(c.sum(), 9124.0);
        assert_eq!(c[IxDyn(&[0, 0, 0, 0])], 7.0);
        assert_eq!(c[IxDyn(&[1, 2, 0, 0])], 947.0);

        // The same with the right operand stretching along its axis 1.
        let out = op.eval(&[lhs, counting(&[3, 1, 2, 2])]).unwrap();
        assert_eq!(out[0].shape(), &[2, 3, 2, 2]);
        let Tensor::F64(c) = &out[0] else {
            panic!("expected F64")
        };
        assert_eq!(c.sum(), 4436.0);
        assert_eq!(c[IxDyn(&[1, 2, 0, 0])], 431.0);
    }

    #[test]
    fn test_two_batch_axes_and_a_contracted_last_axis() {
        // 'bhqd,bhkd->bhqk': the last axis of both operands is contracted. Values from NumPy.
        let op = DotGeneral(dims(&[3], &[3], &[0, 1], &[0, 1]));
        let q = f64s(&[2, 2, 3, 2], &(0..24).map(f64::from).collect::<Vec<_>>());
        let k = f64s(&[2, 2, 4, 2], &(0..32).map(f64::from).collect::<Vec<_>>());
        let out = op.eval(&[q, k]).unwrap();
        assert_eq!(out[0].shape(), &[2, 2, 3, 4]);
        let Tensor::F64(scores) = &out[0] else {
            panic!("expected F64")
        };
        assert_eq!(scores.sum(), 22896.0);
        assert_eq!(scores[IxDyn(&[1, 1, 2, 3])], 1373.0);
        assert_eq!(scores[IxDyn(&[0, 1, 0, 2])], 163.0);
    }

    #[test]
    fn test_vectors() {
        let v = f64s(&[3], &[1.0, 2.0, 3.0]);
        let w = f64s(&[3], &[4.0, 5.0, 6.0]);
        let dot = DotGeneral(dims(&[0], &[0], &[], &[]));
        assert_eq!(
            dot.eval(&[v.clone(), w.clone()]).unwrap(),
            vec![f64s(&[], &[32.0])]
        );
        let outer = DotGeneral(dims(&[], &[], &[], &[]));
        assert_eq!(
            outer.eval(&[v, w]).unwrap(),
            vec![f64s(
                &[3, 3],
                &[4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 12.0, 15.0, 18.0]
            )]
        );
    }

    #[test]
    fn test_axes_of_size_zero() {
        // An empty batch gives an empty result.
        let empty = f64s(&[0, 2, 2], &[]);
        let out = DotGeneral(dims(&[2], &[1], &[0], &[0]))
            .eval(&[empty.clone(), empty])
            .unwrap();
        assert_eq!(out[0].shape(), &[0, 2, 2]);

        // A sum over an empty axis is zero.
        let op = DotGeneral(dims(&[1], &[0], &[], &[]));
        let out = op.eval(&[f64s(&[2, 0], &[]), f64s(&[0, 3], &[])]).unwrap();
        assert_eq!(out, vec![f64s(&[2, 3], &[0.0; 6])]);
    }

    #[test]
    fn test_complex_values_are_not_conjugated() {
        let op = DotGeneral(dims(&[1], &[0], &[], &[]));
        let i = Tensor::C128(array(&[1, 1], &[Complex64::new(0.0, 1.0)]));
        let result = op.eval(&[i.clone(), i]).unwrap();
        assert_eq!(
            result,
            vec![Tensor::C128(array(&[1, 1], &[Complex64::new(-1.0, 0.0)]))]
        );
    }

    #[test]
    fn test_operands_of_different_dtypes_are_promoted() {
        let op = DotGeneral(dims(&[1], &[0], &[], &[]));
        let a = Tensor::F32(array(&[1, 1], &[2.0_f32]));
        let b = Tensor::C64(array(&[1, 1], &[Complex32::new(1.0, 1.0)]));
        let result = op.eval(&[a, b]).unwrap();
        assert_eq!(
            result,
            vec![Tensor::C64(array(&[1, 1], &[Complex32::new(2.0, 2.0)]))]
        );
    }

    #[test]
    fn test_integer_products() {
        let op = DotGeneral(dims(&[1], &[0], &[], &[]));

        // Without overflow.
        let a = Tensor::I16(array(&[2, 2], &[1, 2, 3, 4]));
        let b = Tensor::I16(array(&[2, 2], &[5, 6, 7, 8]));
        assert_eq!(
            op.eval(&[a, b]).unwrap(),
            vec![Tensor::I16(array(&[2, 2], &[19, 22, 43, 50]))]
        );

        // Mixed integer widths promote: I8 with U8 is computed in I16.
        let a = Tensor::I8(array(&[1, 2], &[-100, 100]));
        let b = Tensor::U8(array(&[2, 1], &[3, 2]));
        assert_eq!(
            op.eval(&[a, b]).unwrap(),
            vec![Tensor::I16(array(&[1, 1], &[-100]))]
        );

        // U8: 200 * 200 + 100 * 1 does not fit, and wraps. Values from NumPy.
        let u = Tensor::U8(array(&[2, 2], &[200, 100, 1, 1]));
        let result = op.eval(&[u.clone(), u]).unwrap();
        assert_eq!(
            result,
            vec![Tensor::U8(array(&[2, 2], &[164, 132, 201, 101]))]
        );

        // I32: 50000 * 50000 + 50000 wraps to a negative number.
        let i = Tensor::I32(array(&[2, 2], &[50000, 50000, 1, 1]));
        let result = op.eval(&[i.clone(), i]).unwrap();
        assert_eq!(
            result,
            vec![Tensor::I32(array(
                &[2, 2],
                &[-1_794_917_296, -1_794_917_296, 50001, 50001]
            ))]
        );
    }

    #[test]
    fn test_evaluation_agrees_with_inference() {
        let op = DotGeneral(dims(&[3], &[2], &[0, 1], &[0, 1]));
        let (a, b) = (counting(&[5, 1, 2, 3]), counting(&[1, 7, 3, 4]));
        let inferred = op
            .infer_output_types(&[a.tensor_type(), b.tensor_type()])
            .unwrap();
        let result = op.eval(&[a, b]).unwrap();
        assert!(result[0].matches(&inferred[0]));
    }

    #[test]
    fn test_a_function_checks_and_evaluates_the_op() {
        let mut f = ProgramFunction::new();
        let a = f.add_parameter(ty(DType::F64, &[5, 1, 2, 2]));
        let b = f.add_parameter(ty(DType::F64, &[1, 7, 2, 2]));
        let c = f.add_parameter(ty(DType::F64, &[4, 7, 2, 2]));
        let pairwise = dims(&[3], &[2], &[0, 1], &[0, 1]);

        // Broadcasting batch axes: accepted, with the inferred type.
        let out = f.add_op(DotGeneral(pairwise.clone()), &[a, b]).unwrap()[0];
        assert_eq!(f.type_of(out).unwrap(), &ty(DType::F64, &[5, 7, 2, 2]));

        // Batch sizes that cannot be paired: rejected, and nothing is added.
        let count_before = f.instruction_count();
        let result = f.add_op(DotGeneral(pairwise), &[a, c]);
        assert!(matches!(result, Err(FunctionError::TypeError { .. })));
        assert_eq!(f.instruction_count(), count_before);

        // The function evaluates.
        f.add_result(out).unwrap();
        assert!(f.has_builtin_eval());
        let result = f.eval(&[
            counting(&[5, 1, 2, 2]),
            counting(&[1, 7, 2, 2]),
            counting(&[4, 7, 2, 2]),
        ]);
        assert_eq!(result.unwrap()[0].shape(), &[5, 7, 2, 2]);
    }

    #[test]
    fn test_a_bounded_batch_axis_meets_a_size_of_one_when_evaluated() {
        let bounded = Dim::Bounded { max: 8 };
        let two = Dim::Fixed(2);
        let pairwise = dims(&[2], &[1], &[0], &[0]);

        // The number of matrices on the left is only bounded, and there is one matrix on the right,
        // with a batch axis of size 1. The op is accepted, and the result is bounded too.
        let mut f = ProgramFunction::new();
        let a = f.add_parameter(TensorType {
            dtype: DType::F64,
            shape: vec![bounded, two, two],
        });
        let b = f.add_parameter(ty(DType::F64, &[1, 2, 2]));
        let out = f.add_op(DotGeneral(pairwise.clone()), &[a, b]).unwrap()[0];
        assert_eq!(f.type_of(out).unwrap().shape, vec![bounded, two, two]);

        // The size of 1 stretches to the size that the left operand has when the function runs.
        f.add_result(out).unwrap();
        for n in [1, 3, 8] {
            let result = f
                .eval(&[counting(&[n, 2, 2]), counting(&[1, 2, 2])])
                .unwrap();
            assert_eq!(result[0].shape(), &[n, 2, 2], "n = {n}");
        }
        // More than the bound is refused by the function.
        assert!(
            f.eval(&[counting(&[9, 2, 2]), counting(&[1, 2, 2])])
                .is_err()
        );

        // A bounded axis against a fixed size other than 1 is refused when the op is added.
        let mut g = ProgramFunction::new();
        let a = g.add_parameter(TensorType {
            dtype: DType::F64,
            shape: vec![bounded, two, two],
        });
        let b = g.add_parameter(ty(DType::F64, &[5, 2, 2]));
        assert!(g.add_op(DotGeneral(pairwise), &[a, b]).is_err());
    }
}
