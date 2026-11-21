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

use std::io::Cursor;

use qiskit_qpy::{Bytes, QISKIT_VERSION, QpyError, native_dump_qpy};

use super::data_tree::pack_tree;
use super::format::*;
use crate::QuantumProgram;
use crate::ops::{
    BroadcastTo, Cast, Constant, Entropy, EulerAnglesZYZ, Mean, Parity, ShotLoop, Split, Std,
    Variance,
};
use crate::program::{InstructionView, Value};
use crate::tensor::{DType, Dim, PrngKind, Tensor, TensorType};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProgramWriteError {
    #[error("Invalid op name: {0}")]
    InvalidOpName(String),
    #[error(transparent)]
    QpyError(#[from] QpyError),
    #[error("Invalid operation shadowing the name {0}")]
    ShadowingBuiltinOp(String),
    #[error("Unsupported operation name {0}")]
    InvalidOperation(String),
}

fn tensor_to_pack(tensor: &Tensor) -> TensorPack {
    let shape: Vec<u64> = tensor.shape().iter().map(|x| *x as u64).collect();
    let num_elems: u64 = shape.iter().product();
    let (data, dtype) = match tensor {
        Tensor::C128(arr) => (
            TensorData::C128(arr.iter().map(|x| [x.re, x.im]).collect()),
            DtypePack::C128,
        ),
        Tensor::C64(arr) => (
            TensorData::C64(arr.iter().map(|x| [x.re, x.im]).collect()),
            DtypePack::C64,
        ),
        Tensor::F32(arr) => (
            TensorData::F32(arr.clone().into_flat().to_vec()),
            DtypePack::F32,
        ),
        Tensor::F64(arr) => (
            TensorData::F64(arr.clone().into_flat().to_vec()),
            DtypePack::F64,
        ),
        Tensor::I32(arr) => (
            TensorData::I32(arr.clone().into_flat().to_vec()),
            DtypePack::I32,
        ),
        Tensor::I64(arr) => (
            TensorData::I64(arr.clone().into_flat().to_vec()),
            DtypePack::I64,
        ),
        Tensor::I16(arr) => (
            TensorData::I16(arr.clone().into_flat().to_vec()),
            DtypePack::I16,
        ),
        Tensor::I8(arr) => (
            TensorData::I8(arr.clone().into_flat().to_vec()),
            DtypePack::I8,
        ),
        Tensor::U32(arr) => (
            TensorData::U32(arr.clone().into_flat().to_vec()),
            DtypePack::U32,
        ),
        Tensor::U64(arr) => (
            TensorData::U64(arr.clone().into_flat().to_vec()),
            DtypePack::U64,
        ),
        Tensor::U16(arr) => (
            TensorData::U16(arr.clone().into_flat().to_vec()),
            DtypePack::U16,
        ),
        Tensor::U8(arr) => (
            TensorData::U8(arr.clone().into_flat().to_vec()),
            DtypePack::U8,
        ),
        Tensor::Bit(arr) => (
            TensorData::U8(arr.clone().into_flat().to_vec()),
            DtypePack::Bit,
        ),
        Tensor::Key(tensor) => match tensor.kind() {
            PrngKind::Philox4x32 => (
                TensorData::Philo4x32(tensor.words().clone().into_flat().to_vec()),
                DtypePack::Philo4x32,
            ),
        },
    };
    TensorPack {
        shape,
        num_elems,
        data,
        dtype,
    }
}

/// Generate a ProgramHeaderPack that can be written out to a Write object
///
/// This function will generate a serializable ProgramHeader pack which is used
///
/// # Args
///
/// - `prog`: the quantum program to generate a serializable object for
/// - `qpy_version`: The QPY version to use for any circuits contained within shot loops. By
///   default this will use the latest qpy version available in Qiskit. This can be any version
///   of QPY supported by the native dump function (currently >= 13)
pub fn program_writer(
    prog: &QuantumProgram,
    qpy_verion: Option<u8>,
) -> Result<ProgramHeader, ProgramWriteError> {
    Ok(ProgramHeader {
        serialization_version: 1,
        qiskit_version: QISKIT_VERSION,
        input_structure: pack_tree(prog.input_structure()),
        output_structure: pack_tree(prog.output_structure()),
        functions: prog
            .functions()
            .iter()
            .map(
                |function| -> Result<ProgramFunctionPack, ProgramWriteError> {
                    Ok(ProgramFunctionPack {
                        instructions: function
                            .iter_instructions()
                            .map(|inst| {
                                let body = match inst.view() {
                                    InstructionView::Parameter => InstructionBodyPack::Parameter,
                                    InstructionView::Call(val) => {
                                        InstructionBodyPack::Call(val.index() as u32)
                                    }
                                    InstructionView::Result => InstructionBodyPack::Result,
                                    InstructionView::Op(op) => {
                                        InstructionBodyPack::Op(match op.name() {
                                            "add" => ProgramOpPack::Add,
                                            "divide" => ProgramOpPack::Divide,
                                            "power" => ProgramOpPack::Power,
                                            "multiply" => ProgramOpPack::Multiply,
                                            "remainder" => ProgramOpPack::Remainder,
                                            "subtract" => ProgramOpPack::Subtract,
                                            "bitwise_and" => ProgramOpPack::BitwiseAnd,
                                            "bitwise_not" => ProgramOpPack::BitwiseNot,
                                            "bitwise_or" => ProgramOpPack::BitwiseOr,
                                            "bitwise_xor" => ProgramOpPack::BitwiseXor,
                                            "parity" => {
                                                let op: &Parity = op.downcast_ref().ok_or(
                                                    ProgramWriteError::ShadowingBuiltinOp(
                                                        "parity".to_string(),
                                                    ),
                                                )?;
                                                ProgramOpPack::Parity(op.axis() as u64)
                                            }
                                            "broadcast_to" => {
                                                let op: &BroadcastTo = op.downcast_ref().ok_or(
                                                    ProgramWriteError::ShadowingBuiltinOp(
                                                        "broadcast_to".to_string(),
                                                    ),
                                                )?;
                                                ProgramOpPack::BroadcastTo {
                                                    targets: op
                                                        .target()
                                                        .iter()
                                                        .map(|x| x.into())
                                                        .collect(),
                                                }
                                            }
                                            "cast" => {
                                                let op: &Cast = op.downcast_ref().ok_or(
                                                    ProgramWriteError::ShadowingBuiltinOp(
                                                        "cast".to_string(),
                                                    ),
                                                )?;
                                                ProgramOpPack::Cast(op.target().into())
                                            }
                                            "constant" => {
                                                let op: &Constant = op.downcast_ref().ok_or(
                                                    ProgramWriteError::ShadowingBuiltinOp(
                                                        "constant".to_string(),
                                                    ),
                                                )?;
                                                ProgramOpPack::Constant(tensor_to_pack(op.value()))
                                            }
                                            "mean" => {
                                                let op: &Mean = op.downcast_ref().ok_or(
                                                    ProgramWriteError::ShadowingBuiltinOp(
                                                        "mean".to_string(),
                                                    ),
                                                )?;
                                                ProgramOpPack::Mean(op.axis() as u64)
                                            }
                                            "std" => {
                                                let op: &Std = op.downcast_ref().ok_or(
                                                    ProgramWriteError::ShadowingBuiltinOp(
                                                        "std".to_string(),
                                                    ),
                                                )?;
                                                ProgramOpPack::Std(op.axis() as u64, op.ddof())
                                            }
                                            "variance" => {
                                                let op: &Variance = op.downcast_ref().ok_or(
                                                    ProgramWriteError::ShadowingBuiltinOp(
                                                        "variance".to_string(),
                                                    ),
                                                )?;
                                                ProgramOpPack::Variance(op.axis() as u64, op.ddof())
                                            }
                                            "shot_loop" => {
                                                let op: &ShotLoop = op.downcast_ref().ok_or(
                                                    ProgramWriteError::ShadowingBuiltinOp(
                                                        "shot_loop".to_string(),
                                                    ),
                                                )?;
                                                let qpy_buf: Vec<u8> = Vec::new();
                                                let mut out_bytes = Cursor::new(qpy_buf);
                                                native_dump_qpy(
                                                    op.circuits().iter(),
                                                    &mut out_bytes,
                                                    qpy_verion,
                                                )?;
                                                let data = ShotLoopPack {
                                                    shots: op.shots() as u64,
                                                    circuits: Bytes(out_bytes.into_inner()),
                                                };
                                                ProgramOpPack::ShotLoop(data)
                                            }
                                            "euler_angles_zyz" => {
                                                let op: &EulerAnglesZYZ = op.downcast_ref().ok_or(
                                                    ProgramWriteError::ShadowingBuiltinOp(
                                                        "euler_angles_zyz".to_string(),
                                                    ),
                                                )?;
                                                ProgramOpPack::EulerAnglesZYZ(op.atol())
                                            }
                                            "entropy" => {
                                                let op: &Entropy = op.downcast_ref().ok_or(
                                                    ProgramWriteError::ShadowingBuiltinOp(
                                                        "entropy".to_string(),
                                                    ),
                                                )?;
                                                ProgramOpPack::Entropy {
                                                    shape: op
                                                        .shape()
                                                        .iter()
                                                        .map(|x| *x as u64)
                                                        .collect(),
                                                }
                                            }
                                            "split" => {
                                                let op: &Split = op.downcast_ref().ok_or(
                                                    ProgramWriteError::ShadowingBuiltinOp(
                                                        "split".to_string(),
                                                    ),
                                                )?;
                                                ProgramOpPack::Split {
                                                    shape: op
                                                        .shape()
                                                        .iter()
                                                        .map(|x| *x as u64)
                                                        .collect(),
                                                }
                                            }

                                            _ => {
                                                return Err(ProgramWriteError::InvalidOperation(
                                                    op.name().to_string(),
                                                ));
                                            }
                                        })
                                    }
                                };
                                Ok(ProgramInstructionPack {
                                    operands: inst.operands().iter().map(|x| x.into()).collect(),
                                    output_types: inst
                                        .output_types()
                                        .iter()
                                        .map(|out_type| out_type.into())
                                        .collect(),
                                    body,
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    })
                },
            )
            .collect::<Result<_, _>>()?,
    })
}

impl From<&Value> for ValuePack {
    fn from(value: &Value) -> ValuePack {
        ValuePack {
            instruction: value.instruction().index() as u32,
            slot: value.slot() as u16,
        }
    }
}

impl From<&TensorType> for TensorTypePack {
    fn from(tensor_type: &TensorType) -> TensorTypePack {
        TensorTypePack {
            dtype: tensor_type.dtype.into(),
            shape: tensor_type.shape.iter().map(|x: &Dim| x.into()).collect(),
        }
    }
}

impl From<&Dim> for DimPack {
    fn from(dim: &Dim) -> DimPack {
        match dim {
            Dim::Fixed(val) => DimPack::Fixed(*val as u64),
            Dim::Bounded { max } => DimPack::Bounded(*max as u64),
        }
    }
}

impl From<DType> for DtypePack {
    fn from(dtype: DType) -> DtypePack {
        match dtype {
            DType::C128 => DtypePack::C128,
            DType::C64 => DtypePack::C64,
            DType::F64 => DtypePack::F64,
            DType::F32 => DtypePack::F32,
            DType::I64 => DtypePack::I64,
            DType::I32 => DtypePack::I32,
            DType::I16 => DtypePack::I16,
            DType::I8 => DtypePack::I8,
            DType::U64 => DtypePack::U64,
            DType::U32 => DtypePack::U32,
            DType::U16 => DtypePack::U16,
            DType::U8 => DtypePack::U8,
            DType::Bit => DtypePack::Bit,
            DType::Key(PrngKind::Philox4x32) => DtypePack::Philo4x32,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::DataTree;
    use crate::ProgramFunction;
    use crate::ops::Add;
    use crate::serialization::read_program_pack;
    use binrw::{BinRead, BinWrite};
    use std::io::Seek;

    #[test]
    fn test_basic_program() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Fixed(1)],
        });
        let doubled = function.add_op(Add, &[x, x]).unwrap()[0];
        function.add_result(doubled).unwrap();
        let program =
            QuantumProgram::new(vec![function], DataTree::Leaf(()), DataTree::Leaf(())).unwrap();
        let buf: Vec<u8> = Vec::new();
        let mut buf = Cursor::new(buf);
        program_writer(&program, None)
            .unwrap()
            .write_be(&mut buf)
            .unwrap();
        buf.seek(std::io::SeekFrom::Start(0)).unwrap();
        let pack = ProgramHeader::read_be(&mut buf).unwrap();
        let result = read_program_pack(pack).unwrap();
        assert_eq!(
            result.eval(DataTree::Leaf(Tensor::from([3.0]))).unwrap(),
            program.eval(DataTree::Leaf(Tensor::from([3.0]))).unwrap()
        );
    }
}
