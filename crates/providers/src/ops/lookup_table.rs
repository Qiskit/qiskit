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

//! The op that reads one entry out of a table at a computed coordinate.

use ndarray::{ArrayD, IxDyn};
use thiserror::Error;

use super::{ProgramOp, QISKIT};
use crate::tensor::{
    DType, Dim, Tensor, TensorError, TensorType, broadcast_dims, broadcast_shape, require_static,
};

/// Read one entry out of a table at a coordinate the program computes.
///
/// The op owns the dtype and static shape an operand must have to be its table. The table is
/// operand `0` and operand `k + 1` addresses axis `k`. The addressed leading axes are a grid of
/// entries and the rest are one entry. A `[4, 4, 2, 2]` table with two indices is a four-by-four
/// grid of two-by-two entries.
///
/// The indices broadcast against one another into a batch. One entry is read per batch element so
/// a result is `batch ++ `[`Self::entry_shape`]. Nothing needs a
/// [`BroadcastTo`](super::BroadcastTo) in front.
///
/// Each index is reduced modulo the axis it addresses rather than range-checked. Wrapping keeps
/// every read in bounds and leaves a uniform index uniform. A power-of-two axis masks rather than
/// divides. An entry is contiguous because the addressed axes lead so a lookup is one `memcpy`.
///
/// A table may be of any dtype but a key one and its shape must be fully static. An index must be
/// of an integer dtype.
#[derive(Clone, Debug)]
pub struct LookupTable {
    /// The dtype an operand must have to be this op's table.
    dtype: DType,
    /// The static shape an operand must have to be this op's table.
    ///
    /// Held as sizes rather than a [`TensorType`] because a dimension is only usable as a size or
    /// a stride once it is known to be static.
    shape: Vec<usize>,
    /// How many leading axes a coordinate addresses. The op takes one index operand each.
    indices: usize,
    /// The table's row-major stride in elements along each addressed axis.
    strides: Vec<usize>,
    /// How many elements one entry holds.
    entry_len: usize,
}

impl LookupTable {
    /// Construct a lookup over `table` addressed by `indices` indices on its leading axes.
    ///
    /// The axes after them are the entry and are read whole. A `[4, 4, 2, 2]` table with two
    /// indices reads entries of shape `[2, 2]`.
    ///
    /// Zero indices is [`LookupTableError::NoIndices`] and more than the table's rank is
    /// [`LookupTableError::IndexedRank`]. An addressed axis of size zero is
    /// [`LookupTableError::EmptyAxis`]. An unaddressed one only makes the entry empty and is
    /// admitted.
    pub fn new(table: TensorType, indices: usize) -> Result<Self, LookupTableError> {
        if matches!(table.dtype, DType::Key(_)) {
            return Err(LookupTableError::KeyTable { dtype: table.dtype });
        }
        let shape = require_static(&table.shape)?;
        if indices == 0 {
            return Err(LookupTableError::NoIndices);
        }
        if indices > shape.len() {
            return Err(LookupTableError::IndexedRank {
                indices,
                rank: shape.len(),
            });
        }
        if let Some(axis) = (0..indices).find(|&axis| shape[axis] == 0) {
            return Err(LookupTableError::EmptyAxis { axis });
        }
        // The stride of the last addressed axis is one entry, and each earlier one is that times
        // every addressed axis in between.
        let entry_len: usize = shape[indices..].iter().product();
        let mut strides = vec![entry_len; indices];
        for axis in (0..indices - 1).rev() {
            strides[axis] = strides[axis + 1] * shape[axis + 1];
        }
        Ok(Self {
            dtype: table.dtype,
            shape,
            indices,
            strides,
            entry_len,
        })
    }

    /// Return the dtype an operand must have to be this op's table.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Return the static shape an operand must have to be this op's table.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return [`Self::dtype`] over [`Self::shape`] with every axis [`Dim::Fixed`].
    pub fn table_type(&self) -> TensorType {
        TensorType {
            dtype: self.dtype,
            shape: self.shape.iter().copied().map(Dim::Fixed).collect(),
        }
    }

    /// Return how many index operands this op takes.
    pub fn num_indices(&self) -> usize {
        self.indices
    }

    /// Return the grid of entries. These are the addressed leading axes.
    ///
    /// Index operand `k + 1` is reduced modulo `grid_shape()[k]`.
    pub fn grid_shape(&self) -> &[usize] {
        &self.shape[..self.indices]
    }

    /// Return how many entries the table holds, the product of [`Self::grid_shape`].
    pub fn entries(&self) -> usize {
        self.grid_shape().iter().product()
    }

    /// Return the shape of one entry. This is every unaddressed axis.
    pub fn entry_shape(&self) -> &[usize] {
        &self.shape[self.indices..]
    }

    /// Copy one entry of `table` per batch element of `indices` into a tensor of shape `shape`.
    fn gather(
        &self,
        table: &Tensor,
        indices: &[Tensor],
        batch_len: usize,
        shape: &[usize],
    ) -> Tensor {
        macro_rules! gather {
            ($variant:ident, $array:expr) => {{
                // A tensor is built in standard layout so this borrows rather than copies.
                let table = $array.as_standard_layout();
                let flat = table
                    .as_slice()
                    .expect("a standard layout array is contiguous");
                Tensor::$variant(
                    ArrayD::from_shape_vec(IxDyn(shape), self.collect(flat, indices, batch_len))
                        .expect("one entry is gathered per batch element")
                        .into_shared(),
                )
            }};
        }
        debug_assert_eq!(
            batch_len * self.entry_len,
            shape.iter().product::<usize>(),
            "the result holds one entry per batch element"
        );
        match table {
            Tensor::C128(a) => gather!(C128, a),
            Tensor::C64(a) => gather!(C64, a),
            Tensor::F64(a) => gather!(F64, a),
            Tensor::F32(a) => gather!(F32, a),
            Tensor::I64(a) => gather!(I64, a),
            Tensor::I32(a) => gather!(I32, a),
            Tensor::I16(a) => gather!(I16, a),
            Tensor::I8(a) => gather!(I8, a),
            Tensor::U64(a) => gather!(U64, a),
            Tensor::U32(a) => gather!(U32, a),
            Tensor::U16(a) => gather!(U16, a),
            Tensor::U8(a) => gather!(U8, a),
            Tensor::Bit(a) => gather!(Bit, a),
            Tensor::Key(_) => unreachable!("a key table is refused by LookupTable::new"),
        }
    }

    /// Collect one entry of `flat` per batch element of `indices` in row-major result order.
    ///
    /// Every index is reduced onto its own axis so each copy is in bounds and is a `memcpy`.
    fn collect<T: Copy>(&self, flat: &[T], indices: &[Tensor], batch_len: usize) -> Vec<T> {
        let mut data = Vec::with_capacity(batch_len * self.entry_len);
        let mut bases = vec![0_usize; TILE.min(batch_len)];
        for tile in (0..batch_len).step_by(TILE) {
            let bases = &mut bases[..TILE.min(batch_len - tile)];
            if let [index] = indices {
                // One index is the whole offset so it is written straight into the tile.
                write_offsets(bases, index, tile, self.strides[0], self.shape[0]);
            } else {
                bases.fill(0);
                for ((&stride, &size), index) in
                    self.strides.iter().zip(self.grid_shape()).zip(indices)
                {
                    add_offsets(bases, index, tile, stride, size);
                }
            }
            for &base in bases.iter() {
                data.extend_from_slice(&flat[base..base + self.entry_len]);
            }
        }
        data
    }

    /// Split operands into the table and the indices.
    ///
    /// # Panics
    ///
    /// Panics if the operand count is not this op's [`ProgramOp::arity`]. A
    /// [`ProgramFunction`](crate::ProgramFunction) rules that out as the op is added.
    fn split_operands<'a, T>(&self, operands: &'a [T]) -> (&'a T, &'a [T]) {
        operands
            .split_first()
            .filter(|(_, indices)| indices.len() == self.indices)
            .unwrap_or_else(|| {
                panic!(
                    "{} expects {} operand(s), got {}",
                    self.full_name(),
                    self.arity(),
                    operands.len()
                )
            })
    }
}

impl ProgramOp for LookupTable {
    type Error = LookupTableError;

    fn name(&self) -> &str {
        "lookup_table"
    }
    fn namespace(&self) -> &str {
        QISKIT
    }
    fn arity(&self) -> usize {
        1 + self.indices
    }
    fn has_builtin_eval(&self) -> bool {
        true
    }

    fn infer_output_types(&self, inputs: &[TensorType]) -> Result<Vec<TensorType>, Self::Error> {
        let (table, indices) = self.split_operands(inputs);
        let expected = self.table_type();
        if *table != expected {
            return Err(LookupTableError::WrongTable {
                expected,
                actual: table.clone(),
            });
        }
        let mut shape: Vec<Dim> = Vec::new();
        for (operand, index) in indices.iter().enumerate() {
            check_index_dtype(operand + 1, index.dtype)?;
            shape = broadcast_dims(&shape, &index.shape)?;
        }
        shape.extend(self.entry_shape().iter().copied().map(Dim::Fixed));
        Ok(vec![TensorType {
            dtype: self.dtype,
            shape,
        }])
    }

    fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
        let (table, indices) = self.split_operands(args);
        // Inference accepted the table as the op was added so evaluation reads it rather than
        // checking it again. Offsets come from the declared strides so a table of another shape
        // would be read at the wrong positions rather than refused by a bound.
        debug_assert_eq!(
            table.tensor_type(),
            self.table_type(),
            "the table operand is not the type inference accepted"
        );

        // The indices broadcast into the batch and each is laid out so the gather can read a tile
        // of its elements as a slice.
        let mut batch: Vec<usize> = Vec::new();
        for index in indices {
            batch = broadcast_shape(&batch, index.shape())?;
        }
        let indices = indices
            .iter()
            .map(|index| index.broadcast_to(&batch).map(standard_layout))
            .collect::<Result<Vec<_>, _>>()?;

        let batch_len = batch.iter().product();
        let mut shape = batch;
        shape.extend(self.entry_shape());
        Ok(vec![self.gather(table, &indices, batch_len, &shape)])
    }
}

/// Validate that an index operand's dtype can be read as an offset.
///
/// Inference is the only tier that calls this. Evaluation reads an index rather than checking it
/// again and [`offsets`] panics on a dtype that should never have reached it.
fn check_index_dtype(operand: usize, dtype: DType) -> Result<(), LookupTableError> {
    // The integer dtypes are named rather than the rest ruled out so a dtype added later is
    // refused here instead of reaching the panic in `offsets`.
    if !matches!(
        dtype,
        DType::Bit
            | DType::U8
            | DType::U16
            | DType::U32
            | DType::U64
            | DType::I8
            | DType::I16
            | DType::I32
            | DType::I64
    ) {
        return Err(LookupTableError::IndexDType { operand, dtype });
    }
    Ok(())
}

/// Return `index` in standard layout so a tile of its elements is a slice.
///
/// A tensor is normally in standard layout already and is then shared rather than copied.
fn standard_layout(index: Tensor) -> Tensor {
    macro_rules! standard {
        ($variant:ident, $array:expr) => {
            (!$array.is_standard_layout())
                .then(|| Tensor::$variant($array.as_standard_layout().into_owned().into_shared()))
        };
    }
    let copy = match &index {
        Tensor::C128(a) => standard!(C128, a),
        Tensor::C64(a) => standard!(C64, a),
        Tensor::F64(a) => standard!(F64, a),
        Tensor::F32(a) => standard!(F32, a),
        Tensor::I64(a) => standard!(I64, a),
        Tensor::I32(a) => standard!(I32, a),
        Tensor::I16(a) => standard!(I16, a),
        Tensor::I8(a) => standard!(I8, a),
        Tensor::U64(a) => standard!(U64, a),
        Tensor::U32(a) => standard!(U32, a),
        Tensor::U16(a) => standard!(U16, a),
        Tensor::U8(a) => standard!(U8, a),
        Tensor::Bit(a) => standard!(Bit, a),
        // A key holds no strided array to normalize and inference refuses a key index anyway.
        Tensor::Key(_) => None,
    };
    copy.unwrap_or(index)
}

/// How many offsets are computed before they are read back.
///
/// Small enough that a tile stays in cache so a large batch costs no more memory traffic than the
/// entries it copies.
const TILE: usize = 1024;

/// Write `index.rem_euclid(size) * stride` into every slot of `slots` reading `index` from `tile`.
///
/// One index addressing the table alone contributes the whole offset.
fn write_offsets(slots: &mut [usize], index: &Tensor, tile: usize, stride: usize, size: usize) {
    offsets::<false>(slots, index, tile, stride, size)
}

/// Add `index.rem_euclid(size) * stride` to every slot of `slots` reading `index` from `tile`.
///
/// One index of several contributes part of the offset so `slots` holds what the others gave.
fn add_offsets(slots: &mut [usize], index: &Tensor, tile: usize, stride: usize, size: usize) {
    offsets::<true>(slots, index, tile, stride, size)
}

/// Fold this tile of `index` into `slots`, adding to what is there if `ADD` and replacing it
/// otherwise.
///
/// `size` is the length of the axis `index` addresses so a reduced index is always a position on
/// it. `stride` is that axis's stride in table elements. `ADD` is a constant so the loop a caller
/// reaches holds no test of it.
///
/// # Panics
///
/// Panics if `index` is not of an integer dtype. [`check_index_dtype`] rules that out as the op is
/// added and this is the only guard left once it is evaluated.
fn offsets<const ADD: bool>(
    slots: &mut [usize],
    index: &Tensor,
    tile: usize,
    stride: usize,
    size: usize,
) {
    /// Fold this tile's indices into `slots` with `wrap`.
    macro_rules! fold {
        ($array:expr, $wrap:expr) => {{
            let flat = $array
                .as_slice()
                .expect("an index is laid out in standard layout before it is read");
            let wrap = $wrap;
            for (slot, &value) in slots.iter_mut().zip(&flat[tile..]) {
                let offset = wrap(value) * stride;
                *slot = if ADD { *slot + offset } else { offset };
            }
        }};
    }
    /// Reduce an unsigned index. It is never below the axis so `%` is already Euclidean.
    ///
    /// Both sides widen to `u64` so a size larger than the index dtype is not truncated into one
    /// that would fold away a position the axis has.
    macro_rules! unsigned {
        ($array:expr, $t:ty) => {{
            // A power-of-two size is the common case and masking is what a modulo costs there.
            if let Some(mask) = power_of_two_mask(size) {
                let mask = mask as u64;
                fold!($array, |value: $t| (u64::from(value) & mask) as usize)
            } else {
                let size = size as u64;
                fold!($array, |value: $t| (u64::from(value) % size) as usize)
            }
        }};
    }
    /// Reduce a signed index. Here `%` would keep the sign of a negative index.
    ///
    /// Masking in two's complement already gives the Euclidean remainder so the power-of-two case
    /// needs no sign handling.
    macro_rules! signed {
        ($array:expr, $t:ty) => {{
            if let Some(mask) = power_of_two_mask(size) {
                let mask = mask as i64;
                fold!($array, |value: $t| (i64::from(value) & mask) as usize)
            } else {
                let size = size as i64;
                fold!($array, |value: $t| i64::from(value).rem_euclid(size)
                    as usize)
            }
        }};
    }
    match index {
        Tensor::Bit(a) | Tensor::U8(a) => unsigned!(a, u8),
        Tensor::U16(a) => unsigned!(a, u16),
        Tensor::U32(a) => unsigned!(a, u32),
        Tensor::U64(a) => unsigned!(a, u64),
        Tensor::I8(a) => signed!(a, i8),
        Tensor::I16(a) => signed!(a, i16),
        Tensor::I32(a) => signed!(a, i32),
        Tensor::I64(a) => signed!(a, i64),
        Tensor::F32(_) | Tensor::F64(_) | Tensor::C64(_) | Tensor::C128(_) | Tensor::Key(_) => {
            panic!("type inference rejects a non-integer index so one never reaches here")
        }
    }
}

/// Return the mask standing in for a modulo by `divisor` when it is a power of two.
fn power_of_two_mask(divisor: usize) -> Option<usize> {
    divisor.is_power_of_two().then(|| divisor - 1)
}

/// Errors returned by [`LookupTable`].
#[derive(Debug, Error)]
pub enum LookupTableError {
    /// The lookup addresses no axis, so it reads nothing.
    #[error("a lookup with no index has no entry to address")]
    NoIndices,

    /// The lookup addresses more axes than the table has.
    #[error("{indices} indices address more axes than a table of rank {rank} has")]
    IndexedRank { indices: usize, rank: usize },

    /// An addressed axis is empty, so it holds no entry to read.
    #[error("table axis {axis} is empty so it holds no entry to address")]
    EmptyAxis { axis: usize },

    /// The table is of a key dtype. A key holds no array to copy an entry out of.
    #[error("a table of dtype {dtype} cannot be read; a key holds no entry to copy")]
    KeyTable { dtype: DType },

    /// The table operand is not of the type this op declares.
    ///
    /// Inference reports this. Evaluation assumes it instead since a program infers the types as
    /// the op is added.
    #[error("table operand: expected {expected}, got {actual}")]
    WrongTable {
        expected: TensorType,
        actual: TensorType,
    },

    /// An index operand is not of an integer dtype.
    #[error("operand {operand}: an index of dtype {dtype} cannot address a table; cast it first")]
    IndexDType { operand: usize, dtype: DType },

    /// A tensor operation failed or the indices do not broadcast or the shape is not static.
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::tensor::PrngKind;
    use ndarray::arr2;

    fn ty(dtype: DType, shape: &[usize]) -> TensorType {
        TensorType {
            dtype,
            shape: shape.iter().copied().map(Dim::Fixed).collect(),
        }
    }

    fn tensor(shape: &[usize], data: Vec<i64>) -> Tensor {
        Tensor::I64(
            ArrayD::from_shape_vec(IxDyn(shape), data)
                .unwrap()
                .into_shared(),
        )
    }

    fn u8_tensor(shape: &[usize], data: Vec<u8>) -> Tensor {
        Tensor::U8(
            ArrayD::from_shape_vec(IxDyn(shape), data)
                .unwrap()
                .into_shared(),
        )
    }

    fn offsets(shape: &[usize]) -> Tensor {
        let len: usize = shape.iter().product();
        tensor(shape, (0..len as i64).collect())
    }

    fn scalar(index: i64) -> Tensor {
        tensor(&[], vec![index])
    }

    fn elements(tensor: &Tensor) -> Vec<i64> {
        let Tensor::I64(array) = tensor else {
            panic!("expected an I64 tensor, got {}", tensor.dtype())
        };
        array.iter().copied().collect()
    }

    #[test]
    fn test_name_and_arity() {
        let op = LookupTable::new(ty(DType::F64, &[16, 4]), 1).unwrap();
        assert_eq!(op.full_name(), "qiskit.lookup_table");
        assert!(op.has_builtin_eval());
        assert_eq!(op.arity(), 2);
        assert_eq!(
            LookupTable::new(ty(DType::F64, &[4, 4, 2, 2]), 2)
                .unwrap()
                .arity(),
            3
        );
    }

    #[test]
    fn test_new_describes_the_grid_and_the_entries() {
        let op = LookupTable::new(ty(DType::F64, &[16, 2, 2]), 1).unwrap();
        assert_eq!(op.dtype(), DType::F64);
        assert_eq!(op.shape(), [16, 2, 2]);
        assert_eq!(op.table_type(), ty(DType::F64, &[16, 2, 2]));
        assert_eq!(op.num_indices(), 1);
        assert_eq!(op.grid_shape(), [16]);
        assert_eq!(op.entries(), 16);
        assert_eq!(op.entry_shape(), [2, 2]);

        let op = LookupTable::new(ty(DType::F64, &[16, 2, 2]), 2).unwrap();
        assert_eq!(op.num_indices(), 2);
        assert_eq!(op.grid_shape(), [16, 2]);
        assert_eq!(op.entries(), 32);
        assert_eq!(op.entry_shape(), [2]);

        let op = LookupTable::new(ty(DType::F64, &[16, 2, 2]), 3).unwrap();
        assert_eq!(op.entries(), 64);
        assert!(op.entry_shape().is_empty());
    }

    #[test]
    fn test_new_rejects_what_it_cannot_plan() {
        let dynamic = vec![Dim::Bounded { max: 16 }, Dim::Fixed(4)];
        assert!(matches!(
            LookupTable::new(
                TensorType {
                    dtype: DType::F64,
                    shape: dynamic.clone(),
                },
                1
            )
            .unwrap_err(),
            LookupTableError::Tensor(TensorError::DynamicDim { shape }) if shape == dynamic
        ));

        assert!(matches!(
            LookupTable::new(ty(DType::F64, &[4, 2]), 0).unwrap_err(),
            LookupTableError::NoIndices
        ));

        assert!(matches!(
            LookupTable::new(ty(DType::F64, &[4, 2]), 3).unwrap_err(),
            LookupTableError::IndexedRank {
                indices: 3,
                rank: 2
            }
        ));
        assert!(matches!(
            LookupTable::new(ty(DType::F64, &[]), 1).unwrap_err(),
            LookupTableError::IndexedRank {
                indices: 1,
                rank: 0
            }
        ));

        assert!(matches!(
            LookupTable::new(ty(DType::F64, &[4, 0]), 2).unwrap_err(),
            LookupTableError::EmptyAxis { axis: 1 }
        ));

        let key = DType::Key(PrngKind::Philox4x32);
        assert!(matches!(
            LookupTable::new(ty(key, &[4, 2]), 1).unwrap_err(),
            LookupTableError::KeyTable { dtype } if dtype == key
        ));

        let op = LookupTable::new(ty(DType::I64, &[4, 0]), 1).unwrap();
        assert_eq!(op.entries(), 4);
        assert_eq!(op.entry_shape(), [0]);
        let result = op
            .eval(&[tensor(&[4, 0], vec![]), tensor(&[3], vec![0, 1, 2])])
            .unwrap();
        assert_eq!(result[0].shape(), [3, 0]);
    }

    #[test]
    fn test_infer_output_types_is_the_batch_then_the_entry() {
        let op = LookupTable::new(ty(DType::F64, &[16, 4]), 1).unwrap();
        let table = ty(DType::F64, &[16, 4]);
        assert_eq!(
            op.infer_output_types(&[table.clone(), ty(DType::U8, &[32])])
                .unwrap(),
            vec![ty(DType::F64, &[32, 4])],
            "the result takes the table's dtype and the index's shape"
        );
        assert_eq!(
            op.infer_output_types(&[table.clone(), ty(DType::U8, &[])])
                .unwrap(),
            vec![ty(DType::F64, &[4])],
            "a scalar index leaves no batch axes"
        );

        let shots = Dim::Bounded { max: 4000 };
        assert_eq!(
            op.infer_output_types(&[
                table,
                TensorType {
                    dtype: DType::U8,
                    shape: vec![shots],
                },
            ])
            .unwrap(),
            vec![TensorType {
                dtype: DType::F64,
                shape: vec![shots, Dim::Fixed(4)],
            }]
        );
    }

    #[test]
    fn test_infer_output_types_broadcasts_the_indices_against_one_another() {
        let op = LookupTable::new(ty(DType::I64, &[4, 4, 2]), 2).unwrap();
        let table = ty(DType::I64, &[4, 4, 2]);
        assert_eq!(op.entry_shape(), [2]);

        assert_eq!(
            op.infer_output_types(&[table.clone(), ty(DType::I64, &[5]), ty(DType::I64, &[5])])
                .unwrap(),
            vec![ty(DType::I64, &[5, 2])]
        );
        assert_eq!(
            op.infer_output_types(&[
                table.clone(),
                ty(DType::I64, &[5, 1]),
                ty(DType::I64, &[1, 3]),
            ])
            .unwrap(),
            vec![ty(DType::I64, &[5, 3, 2])]
        );
        assert_eq!(
            op.infer_output_types(&[table.clone(), ty(DType::I64, &[]), ty(DType::I64, &[5])])
                .unwrap(),
            vec![ty(DType::I64, &[5, 2])]
        );
        assert_eq!(
            op.infer_output_types(&[table, ty(DType::I64, &[]), ty(DType::I64, &[])])
                .unwrap(),
            vec![ty(DType::I64, &[2])],
            "every index scalar reads one entry, so the entry is the whole result"
        );
    }

    #[test]
    fn test_infer_output_types_reports_indices_that_do_not_broadcast() {
        let op = LookupTable::new(ty(DType::I64, &[4, 4]), 2).unwrap();
        assert!(matches!(
            op.infer_output_types(&[
                ty(DType::I64, &[4, 4]),
                ty(DType::I64, &[3]),
                ty(DType::I64, &[5]),
            ])
            .unwrap_err(),
            LookupTableError::Tensor(TensorError::DimShapeMismatch { lhs, rhs })
                if lhs == [Dim::Fixed(3)] && rhs == [Dim::Fixed(5)]
        ));

        let shots = Dim::Bounded { max: 4000 };
        assert!(matches!(
            op.infer_output_types(&[
                ty(DType::I64, &[4, 4]),
                TensorType {
                    dtype: DType::I64,
                    shape: vec![shots],
                },
                ty(DType::I64, &[5]),
            ])
            .unwrap_err(),
            LookupTableError::Tensor(TensorError::DynamicDim { .. })
        ));
    }

    #[test]
    fn test_infer_output_types_rejects_a_table_of_another_type() {
        let op = LookupTable::new(ty(DType::I64, &[4, 2]), 1).unwrap();
        for table in [ty(DType::F64, &[4, 2]), ty(DType::I64, &[2, 2])] {
            assert!(matches!(
                op.infer_output_types(&[table.clone(), ty(DType::U8, &[3])])
                    .unwrap_err(),
                LookupTableError::WrongTable { expected, actual }
                    if expected == ty(DType::I64, &[4, 2]) && actual == table
            ));
        }
    }

    #[test]
    fn test_infer_output_types_rejects_an_index_that_is_not_an_integer() {
        let op = LookupTable::new(ty(DType::I64, &[4, 2]), 2).unwrap();
        let table = ty(DType::I64, &[4, 2]);
        for dtype in [
            DType::F32,
            DType::F64,
            DType::C64,
            DType::C128,
            DType::Key(PrngKind::Philox4x32),
        ] {
            assert!(matches!(
                op.infer_output_types(&[table.clone(), ty(dtype, &[3]), ty(DType::U8, &[3])])
                    .unwrap_err(),
                LookupTableError::IndexDType { operand: 1, dtype: reported } if reported == dtype
            ));
            assert!(
                matches!(
                    op.infer_output_types(&[table.clone(), ty(DType::U8, &[3]), ty(dtype, &[3])])
                        .unwrap_err(),
                    LookupTableError::IndexDType { operand: 2, dtype: reported }
                        if reported == dtype
                ),
                "the operand reported is the one that is wrong, not always the first"
            );
        }
        for dtype in [DType::Bit, DType::U8, DType::U64, DType::I8, DType::I64] {
            assert!(
                op.infer_output_types(&[table.clone(), ty(dtype, &[3]), ty(dtype, &[3])])
                    .is_ok(),
                "an index of dtype {dtype} addresses a table"
            );
        }
    }

    #[test]
    fn test_eval_reads_the_entry_one_index_names() {
        let op = LookupTable::new(ty(DType::I64, &[4, 2]), 1).unwrap();
        let result = op
            .eval(&[offsets(&[4, 2]), Tensor::from([2_i64, 0, 3])])
            .unwrap();
        assert_eq!(result[0].shape(), [3, 2]);
        assert_eq!(elements(&result[0]), [4, 5, 0, 1, 6, 7]);

        let result = op.eval(&[offsets(&[4, 2]), scalar(1)]).unwrap();
        assert_eq!(result[0].shape(), [2]);
        assert_eq!(elements(&result[0]), [2, 3]);

        let op = LookupTable::new(ty(DType::I64, &[4, 2, 2]), 1).unwrap();
        let result = op
            .eval(&[offsets(&[4, 2, 2]), Tensor::from([2_i64, 0])])
            .unwrap();
        assert_eq!(result[0].shape(), [2, 2, 2]);
        assert_eq!(elements(&result[0]), [8, 9, 10, 11, 0, 1, 2, 3]);
    }

    #[test]
    fn test_eval_reads_the_entry_a_coordinate_names() {
        let op = LookupTable::new(ty(DType::I64, &[4, 4, 2]), 2).unwrap();
        let result = op
            .eval(&[
                offsets(&[4, 4, 2]),
                Tensor::from([1_i64, 3]),
                Tensor::from([2_i64, 0]),
            ])
            .unwrap();
        assert_eq!(result[0].shape(), [2, 2]);
        assert_eq!(
            elements(&result[0]),
            [12, 13, 24, 25],
            "a coordinate names the entry at flat offset 8i + 2j"
        );

        let result = op
            .eval(&[offsets(&[4, 4, 2]), scalar(3), Tensor::from([2_i64, 0])])
            .unwrap();
        assert_eq!(result[0].shape(), [2, 2]);
        assert_eq!(
            elements(&result[0]),
            [28, 29, 24, 25],
            "a scalar index pairs with every element of the other"
        );

        let result = op
            .eval(&[offsets(&[4, 4, 2]), scalar(1), scalar(2)])
            .unwrap();
        assert_eq!(result[0].shape(), [2]);
        assert_eq!(
            elements(&result[0]),
            [12, 13],
            "every index scalar reads one entry"
        );
    }

    #[test]
    fn test_eval_broadcasts_the_indices_against_one_another() {
        let op = LookupTable::new(ty(DType::I64, &[4, 4, 2]), 2).unwrap();
        let first = [0_i64, 1, 3];
        let second = [0_i64, 1, 2, 3, 2];
        let result = op
            .eval(&[
                offsets(&[4, 4, 2]),
                tensor(&[1, 3], first.to_vec()),
                tensor(&[5, 1], second.to_vec()),
            ])
            .unwrap();

        assert_eq!(result[0].shape(), [5, 3, 2]);
        let expected: Vec<i64> = second
            .iter()
            .flat_map(|&j| {
                first
                    .iter()
                    .flat_map(move |&i| [8 * i + 2 * j, 8 * i + 2 * j + 1])
            })
            .collect();
        assert_eq!(elements(&result[0]), expected);
    }

    #[test]
    fn test_eval_wraps_an_index_onto_an_entry() {
        let op = LookupTable::new(ty(DType::I64, &[3, 2]), 1).unwrap();
        let result = op
            .eval(&[offsets(&[3, 2]), Tensor::from([-1_i64, 3, 7, 2])])
            .unwrap();
        assert_eq!(
            elements(&result[0]),
            [4, 5, 0, 1, 2, 3, 4, 5],
            "three entries is no power of two so the reduction divides and -1, 3, 7 and 2 reach \
             entries 2, 0, 1 and 2"
        );

        let op = LookupTable::new(ty(DType::I64, &[4, 2]), 1).unwrap();
        let result = op
            .eval(&[offsets(&[4, 2]), Tensor::from([-1_i64, 3, 7, 2])])
            .unwrap();
        assert_eq!(
            elements(&result[0]),
            [6, 7, 6, 7, 6, 7, 4, 5],
            "four entries is a power of two so masking runs and the same indices reach entries \
             3, 3, 3 and 2"
        );

        let table = Tensor::Bit(arr2(&[[0_u8, 1], [1, 0], [1, 1]]).into_dyn().into_shared());
        let op = LookupTable::new(ty(DType::Bit, &[3, 2]), 1).unwrap();
        let entries_one_and_zero = Tensor::Bit(arr2(&[[1_u8, 0], [0, 1]]).into_dyn().into_shared());
        for index in [
            Tensor::from([7_u8, 3]),
            Tensor::from([7_u16, 3]),
            Tensor::from([7_u32, 3]),
            Tensor::from([7_u64, 3]),
            Tensor::from([7_i8, 3]),
            Tensor::from([7_i16, 3]),
            Tensor::from([7_i32, 3]),
            Tensor::from([7_i64, 3]),
            Tensor::Bit(ndarray::arr1(&[1_u8, 0]).into_dyn().into_shared()),
        ] {
            let dtype = index.dtype();
            let result = op.eval(&[table.clone(), index]).unwrap();
            assert_eq!(
                result[0], entries_one_and_zero,
                "the indices reach entries 1 and 0 for an index of {dtype}"
            );
        }
    }

    #[test]
    fn test_eval_wraps_each_index_onto_its_own_axis() {
        let op = LookupTable::new(ty(DType::I64, &[3, 4, 2]), 2).unwrap();
        let result = op
            .eval(&[
                offsets(&[3, 4, 2]),
                Tensor::from([-1_i64, 3, 7]),
                Tensor::from([-1_i64, 5, 2]),
            ])
            .unwrap();
        assert_eq!(result[0].shape(), [3, 2]);
        assert_eq!(
            elements(&result[0]),
            [22, 23, 2, 3, 12, 13],
            "axis 0 of three divides and axis 1 of four masks so the indices reach 2, 0, 1 and \
             3, 1, 2 giving 8i + 2j"
        );
    }

    #[test]
    fn test_eval_reads_a_batch_that_spans_several_tiles() {
        let count = 2 * TILE + 37;
        let one = |n: i64| n % 6 - 1;
        let two = |n: i64| n % 7 - 2;

        let op = LookupTable::new(ty(DType::I64, &[4, 2]), 1).unwrap();
        let index: Vec<i64> = (0..count as i64).map(one).collect();
        let result = op
            .eval(&[offsets(&[4, 2]), tensor(&[count], index.clone())])
            .unwrap();
        assert_eq!(result[0].shape(), [count, 2]);
        let expected: Vec<i64> = index
            .iter()
            .flat_map(|&i| {
                let base = i.rem_euclid(4) * 2;
                [base, base + 1]
            })
            .collect();
        assert_eq!(elements(&result[0]), expected);

        let op = LookupTable::new(ty(DType::I64, &[4, 4, 2]), 2).unwrap();
        let first: Vec<i64> = (0..count as i64).map(one).collect();
        let second: Vec<i64> = (0..count as i64).map(two).collect();
        let result = op
            .eval(&[
                offsets(&[4, 4, 2]),
                tensor(&[count], first.clone()),
                tensor(&[count], second.clone()),
            ])
            .unwrap();
        assert_eq!(result[0].shape(), [count, 2]);
        let expected: Vec<i64> = first
            .iter()
            .zip(&second)
            .flat_map(|(&i, &j)| {
                let base = i.rem_euclid(4) * 8 + j.rem_euclid(4) * 2;
                [base, base + 1]
            })
            .collect();
        assert_eq!(elements(&result[0]), expected);
    }

    #[test]
    fn test_eval_reads_operands_that_are_not_in_standard_layout() {
        let table = Tensor::I64(
            arr2(&[[0_i64, 1], [2, 3], [4, 5], [6, 7]])
                .reversed_axes()
                .into_dyn()
                .into_shared(),
        );
        let index = Tensor::I64(
            arr2(&[[0_i64, 0], [1, 1]])
                .reversed_axes()
                .into_dyn()
                .into_shared(),
        );
        assert_eq!(table.shape(), [2, 4]);

        let op = LookupTable::new(ty(DType::I64, &[2, 4]), 1).unwrap();
        let result = op.eval(&[table.clone(), index.clone()]).unwrap();
        assert_eq!(result[0].shape(), [2, 2, 4]);
        assert_eq!(
            elements(&result[0]),
            [0, 2, 4, 6, 1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7],
            "the table reads as [[0, 2, 4, 6], [1, 3, 5, 7]] and the index as [[0, 1], [0, 1]]"
        );

        let op = LookupTable::new(ty(DType::I64, &[2, 4]), 2).unwrap();
        let other = Tensor::I64(
            arr2(&[[3_i64, 1], [2, 0]])
                .reversed_axes()
                .into_dyn()
                .into_shared(),
        );
        let result = op.eval(&[table, index, other]).unwrap();
        assert_eq!(result[0].shape(), [2, 2]);
        assert_eq!(
            elements(&result[0]),
            [6, 5, 2, 1],
            "the indices read as [[0, 1], [0, 1]] and [[3, 2], [1, 0]] so 4i + j gives offsets \
             3, 6, 1 and 4"
        );
    }

    #[test]
    #[should_panic(expected = "type inference rejects a non-integer index")]
    fn test_eval_panics_on_an_index_inference_would_have_rejected() {
        let op = LookupTable::new(ty(DType::I64, &[4, 2]), 1).unwrap();
        let _ = op.eval(&[offsets(&[4, 2]), Tensor::from([1.0_f64])]);
    }

    #[test]
    #[should_panic(expected = "qiskit.lookup_table expects 3 operand(s), got 2")]
    fn test_eval_panics_on_the_wrong_operand_count() {
        let op = LookupTable::new(ty(DType::I64, &[4, 4]), 2).unwrap();
        let _ = op.eval(&[offsets(&[4, 4]), Tensor::from([1_i64])]);
    }

    #[test]
    fn test_a_function_broadcasts_a_per_shot_and_a_per_qubit_index() {
        use crate::ops::Constant;
        use crate::program::ProgramFunction;

        let data: Vec<i64> = (0..4)
            .flat_map(|i| (0..4).flat_map(move |j| [i, j]))
            .collect();
        let mut function = ProgramFunction::new();
        let table = function
            .add_op(Constant::new(tensor(&[4, 4, 2], data)), &[])
            .unwrap()[0];
        let per_shot = function.add_parameter(ty(DType::U8, &[2, 1]));
        let per_qubit = function.add_parameter(ty(DType::U8, &[1, 3]));
        let entry = function
            .add_op(
                LookupTable::new(ty(DType::I64, &[4, 4, 2]), 2).unwrap(),
                &[table, per_shot, per_qubit],
            )
            .unwrap()[0];
        function.add_result(entry).unwrap();

        assert_eq!(function.type_of(entry), Some(&ty(DType::I64, &[2, 3, 2])));
        let result = function
            .eval(&[
                u8_tensor(&[2, 1], vec![1, 3]),
                u8_tensor(&[1, 3], vec![0, 2, 1]),
            ])
            .unwrap();
        assert_eq!(result[0].shape(), [2, 3, 2]);
        assert_eq!(
            elements(&result[0]),
            [1, 0, 1, 2, 1, 1, 3, 0, 3, 2, 3, 1],
            "shot 0 reads row 1 and shot 1 row 3, each against qubits 0, 2 and 1"
        );
    }
}
