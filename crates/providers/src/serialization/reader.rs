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

use super::data_tree::{UnpackTreeError, unpack_tree};
use super::format::*;
use crate::ops::*;
use crate::tensor::{DType, Dim, KeyTensor, PrngKind, Tensor, TensorError, TensorType};
use crate::{FunctionError, FunctionId, ProgramError, ProgramFunction, QuantumProgram, Value};
use ndarray::{ArcArrayD, ShapeError};
use num_complex::{Complex32, Complex64};
use qiskit_qpy::{QISKIT_VERSION, QpyError, native_load_qpy};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum QuantumProgramReadError {
    #[error(transparent)]
    TensorShapeError(#[from] ShapeError),
    #[error(transparent)]
    QuantumProgramError(#[from] ProgramError),
    #[error(transparent)]
    QpyError(#[from] QpyError),
    #[error(transparent)]
    TensorError(#[from] TensorError),
    #[error(transparent)]
    FunctionError(#[from] FunctionError),
    #[error(transparent)]
    TreeError(#[from] UnpackTreeError),
    #[error("Unsupported version of serialization format {0}")]
    UnsupportedVersion(u8),
}

pub fn read_program_pack(prog: ProgramHeader) -> Result<QuantumProgram, QuantumProgramReadError> {
    if prog.serialization_version > 1 && prog.serialization_version != 0 {
        return Err(QuantumProgramReadError::UnsupportedVersion(
            prog.serialization_version,
        ));
    }
    if prog.qiskit_version > QISKIT_VERSION {
        // TODO: Return a warning
    }
    Ok(QuantumProgram::new(
        build_functions(prog.functions)?,
        unpack_tree(prog.input_structure)?,
        unpack_tree(prog.output_structure)?,
    )?)
}

fn build_functions(
    functions: Vec<ProgramFunctionPack>,
) -> Result<Vec<ProgramFunction>, QuantumProgramReadError> {
    functions
        .into_iter()
        .map(|func| {
            let mut out = ProgramFunction::new();
            for raw_inst in func.instructions {
                let operands: Vec<Value> = raw_inst
                    .operands
                    .iter()
                    .map(|x| Value::new(x.instruction, x.slot))
                    .collect();
                let output_types: Vec<TensorType> = raw_inst
                    .output_types
                    .iter()
                    .map(|x| TensorType {
                        dtype: x.dtype.into(),
                        shape: x.shape.iter().map(|dim| dim.into()).collect(),
                    })
                    .collect();
                match raw_inst.body {
                    InstructionBodyPack::Parameter => {
                        out.add_parameter(output_types[0].clone());
                    }
                    InstructionBodyPack::Call(id) => {
                        out.add_call_unchecked(
                            FunctionId::from_index(id as usize),
                            &output_types,
                            &operands,
                        )?;
                    }
                    InstructionBodyPack::Result => {
                        out.add_result(operands[0])?;
                    }
                    InstructionBodyPack::Op(op) => {
                        let op: BoxedProgramOp = match op {
                            ProgramOpPack::Add => Box::new(Add),
                            ProgramOpPack::Divide => Box::new(Divide),
                            ProgramOpPack::Multiply => Box::new(Multiply),
                            ProgramOpPack::Power => Box::new(Power),
                            ProgramOpPack::Remainder => Box::new(Remainder),
                            ProgramOpPack::Subtract => Box::new(Subtract),
                            ProgramOpPack::BitwiseAnd => Box::new(BitwiseAnd),
                            ProgramOpPack::BitwiseNot => Box::new(BitwiseNot),
                            ProgramOpPack::BitwiseOr => Box::new(BitwiseOr),
                            ProgramOpPack::BitwiseXor => Box::new(BitwiseXor),
                            ProgramOpPack::Parity(axis) => Box::new(Parity::new(axis as usize)),
                            ProgramOpPack::BroadcastTo { targets } => Box::new(BroadcastTo::new(
                                targets.iter().map(|x| x.into()).collect(),
                            )),
                            ProgramOpPack::Cast(dtype) => Box::new(Cast::new(dtype.into())),
                            ProgramOpPack::Constant(tensor) => {
                                let tensor_shape: Vec<usize> =
                                    tensor.shape.into_iter().map(|x| x as usize).collect();
                                let tensor_out = match tensor.data {
                                    TensorData::C128(vec) => {
                                        let temp: Vec<Complex64> = vec
                                            .into_iter()
                                            .map(|[re, im]| Complex64::new(re, im))
                                            .collect();
                                        Tensor::C128(ArcArrayD::from_shape_vec(tensor_shape, temp)?)
                                    }
                                    TensorData::C64(vec) => {
                                        let temp: Vec<Complex32> = vec
                                            .into_iter()
                                            .map(|[re, im]| Complex32::new(re, im))
                                            .collect();
                                        Tensor::C64(ArcArrayD::from_shape_vec(tensor_shape, temp)?)
                                    }
                                    TensorData::F64(vec) => {
                                        Tensor::F64(ArcArrayD::from_shape_vec(tensor_shape, vec)?)
                                    }
                                    TensorData::F32(vec) => {
                                        Tensor::F32(ArcArrayD::from_shape_vec(tensor_shape, vec)?)
                                    }
                                    TensorData::I64(vec) => {
                                        Tensor::I64(ArcArrayD::from_shape_vec(tensor_shape, vec)?)
                                    }
                                    TensorData::I32(vec) => {
                                        Tensor::I32(ArcArrayD::from_shape_vec(tensor_shape, vec)?)
                                    }
                                    TensorData::I16(vec) => {
                                        Tensor::I16(ArcArrayD::from_shape_vec(tensor_shape, vec)?)
                                    }
                                    TensorData::I8(vec) => {
                                        Tensor::I8(ArcArrayD::from_shape_vec(tensor_shape, vec)?)
                                    }
                                    TensorData::U64(vec) => {
                                        Tensor::U64(ArcArrayD::from_shape_vec(tensor_shape, vec)?)
                                    }
                                    TensorData::U32(vec) => {
                                        Tensor::U32(ArcArrayD::from_shape_vec(tensor_shape, vec)?)
                                    }
                                    TensorData::U16(vec) => {
                                        Tensor::U16(ArcArrayD::from_shape_vec(tensor_shape, vec)?)
                                    }
                                    TensorData::U8(vec) => {
                                        Tensor::U8(ArcArrayD::from_shape_vec(tensor_shape, vec)?)
                                    }
                                    TensorData::Bit(vec) => {
                                        Tensor::Bit(ArcArrayD::from_shape_vec(tensor_shape, vec)?)
                                    }
                                    TensorData::Philo4x32(vec) => Tensor::Key(KeyTensor::new(
                                        PrngKind::Philox4x32,
                                        ArcArrayD::from_shape_vec(tensor_shape, vec)?,
                                    )?),
                                };
                                Box::new(Constant::new(tensor_out))
                            }
                            ProgramOpPack::Mean(axis) => Box::new(Mean::new(axis as usize)),
                            ProgramOpPack::Std(axis, ddof) => {
                                Box::new(Std::new(axis as usize, ddof))
                            }
                            ProgramOpPack::Variance(axis, ddof) => {
                                Box::new(Variance::new(axis as usize, ddof))
                            }
                            ProgramOpPack::ShotLoop(data) => {
                                let circuits = native_load_qpy(&data.circuits.0)?;
                                Box::new(ShotLoop::new(circuits, data.shots as usize).unwrap())
                            }
                            ProgramOpPack::EulerAnglesZYZ(atol) => {
                                Box::new(EulerAnglesZYZ::new(atol))
                            }
                            ProgramOpPack::Entropy { shape } => Box::new(Entropy::new(
                                PrngKind::Philox4x32,
                                shape.into_iter().map(|x| x as usize).collect(),
                            )),
                            ProgramOpPack::Split { shape } => Box::new(Split::new(
                                shape.into_iter().map(|x| x as usize).collect(),
                            )),
                        };
                        out.add_boxed_op(op, &operands)?;
                    }
                };
            }
            Ok(out)
        })
        .collect()
}

impl From<DtypePack> for DType {
    fn from(dtype: DtypePack) -> DType {
        match dtype {
            DtypePack::C128 => DType::C128,
            DtypePack::C64 => DType::C64,
            DtypePack::F64 => DType::F64,
            DtypePack::F32 => DType::F32,
            DtypePack::I64 => DType::I64,
            DtypePack::I32 => DType::I32,
            DtypePack::I16 => DType::I16,
            DtypePack::I8 => DType::I8,
            DtypePack::U64 => DType::U64,
            DtypePack::U32 => DType::U32,
            DtypePack::U16 => DType::U16,
            DtypePack::U8 => DType::U8,
            DtypePack::Bit => DType::Bit,
            DtypePack::Philo4x32 => DType::Key(PrngKind::Philox4x32),
        }
    }
}

impl From<&DimPack> for Dim {
    fn from(dim: &DimPack) -> Dim {
        match dim {
            DimPack::Fixed(val) => Dim::Fixed(*val as usize),
            DimPack::Bounded(max) => Dim::Bounded { max: *max as usize },
        }
    }
}
