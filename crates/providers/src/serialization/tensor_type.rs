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

#![allow(dead_code)] // Currently in progress.

use crate::serialization::formats::{DTypePack, DimPack, PrngKindPack, TensorTypePack};
use crate::tensor::{DType, Dim, PrngKind, TensorType};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum UnpackTensorTypeError {
    #[error("dimension {dimension} has size {size}, which cannot be represented on this platform")]
    DimensionOverflow { dimension: usize, size: u64 },
}

pub fn pack_tensor_type(tensor_type: &TensorType) -> TensorTypePack {
    let dtype = match tensor_type.dtype {
        DType::C128 => DTypePack::C128,
        DType::C64 => DTypePack::C64,
        DType::F64 => DTypePack::F64,
        DType::F32 => DTypePack::F32,
        DType::I64 => DTypePack::I64,
        DType::I32 => DTypePack::I32,
        DType::I16 => DTypePack::I16,
        DType::I8 => DTypePack::I8,
        DType::U64 => DTypePack::U64,
        DType::U32 => DTypePack::U32,
        DType::U16 => DTypePack::U16,
        DType::U8 => DTypePack::U8,
        DType::Bit => DTypePack::Bit,
        DType::Key(PrngKind::Philox4x32) => DTypePack::Key(PrngKindPack::Philox4x32),
    };
    let shape = tensor_type
        .shape
        .iter()
        .map(|dimension| match dimension {
            Dim::Fixed(size) => DimPack::Fixed(*size as u64),
            Dim::Bounded { max } => DimPack::Bounded(*max as u64),
        })
        .collect();
    TensorTypePack { dtype, shape }
}

pub fn unpack_tensor_type(
    tensor_type: TensorTypePack,
) -> Result<TensorType, UnpackTensorTypeError> {
    let dtype = match tensor_type.dtype {
        DTypePack::C128 => DType::C128,
        DTypePack::C64 => DType::C64,
        DTypePack::F64 => DType::F64,
        DTypePack::F32 => DType::F32,
        DTypePack::I64 => DType::I64,
        DTypePack::I32 => DType::I32,
        DTypePack::I16 => DType::I16,
        DTypePack::I8 => DType::I8,
        DTypePack::U64 => DType::U64,
        DTypePack::U32 => DType::U32,
        DTypePack::U16 => DType::U16,
        DTypePack::U8 => DType::U8,
        DTypePack::Bit => DType::Bit,
        DTypePack::Key(PrngKindPack::Philox4x32) => DType::Key(PrngKind::Philox4x32),
    };
    let shape = tensor_type
        .shape
        .into_iter()
        .enumerate()
        .map(|(dimension, packed)| {
            let (size, bounded) = match packed {
                DimPack::Fixed(size) => (size, false),
                DimPack::Bounded(max) => (max, true),
            };
            let size = usize::try_from(size)
                .map_err(|_| UnpackTensorTypeError::DimensionOverflow { dimension, size })?;
            Ok(if bounded {
                Dim::Bounded { max: size }
            } else {
                Dim::Fixed(size)
            })
        })
        .collect::<Result<_, _>>()?;
    Ok(TensorType { dtype, shape })
}

#[cfg(test)]
mod tests {
    use super::*;
    use binrw::{BinRead, BinWrite};
    use std::io::Cursor;

    const DTYPES: [DType; 14] = [
        DType::C128,
        DType::C64,
        DType::F64,
        DType::F32,
        DType::I64,
        DType::I32,
        DType::I16,
        DType::I8,
        DType::U64,
        DType::U32,
        DType::U16,
        DType::U8,
        DType::Bit,
        DType::Key(PrngKind::Philox4x32),
    ];

    #[test]
    fn round_trip_all_dtypes_and_dimension_kinds_through_binary_format() {
        for dtype in DTYPES {
            let tensor_type = TensorType {
                dtype,
                shape: vec![Dim::Fixed(0), Dim::Fixed(7), Dim::Bounded { max: 23 }],
            };
            let mut bytes = Cursor::new(Vec::new());
            pack_tensor_type(&tensor_type).write_be(&mut bytes).unwrap();
            bytes.set_position(0);
            let packed = TensorTypePack::read_be(&mut bytes).unwrap();
            assert_eq!(unpack_tensor_type(packed).unwrap(), tensor_type);
        }
    }

    #[test]
    fn round_trip_scalar_through_binary_format() {
        let tensor_type = TensorType {
            dtype: DType::F64,
            shape: Vec::new(),
        };
        let mut bytes = Cursor::new(Vec::new());
        pack_tensor_type(&tensor_type).write_be(&mut bytes).unwrap();
        bytes.set_position(0);
        let packed = TensorTypePack::read_be(&mut bytes).unwrap();
        assert_eq!(unpack_tensor_type(packed).unwrap(), tensor_type);
    }
}
