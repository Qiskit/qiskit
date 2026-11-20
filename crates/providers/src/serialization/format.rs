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

use std::io::Read;

use binrw::{BinRead, BinResult, BinWrite, binrw};
use qiskit_qpy::Bytes;

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct ProgramHeader {
    #[brw(magic = b"QKPROG")]
    pub(super) serialization_version: u8,
    pub(super) qiskit_version: (u8, u8, u8),
    pub(super) input_structure: DataTreePack,
    pub(super) output_structure: DataTreePack,
    #[bw(calc = functions.len() as u64)]
    pub(super) num_functions: u64,
    #[br(args((num_functions,)), parse_with=functions_decoder)]
    pub(super) functions: Vec<ProgramFunctionPack>,
}

#[binrw::parser(reader, endian)]
fn functions_decoder(args: (u64,)) -> BinResult<Vec<ProgramFunctionPack>> {
    let count = args.0 as usize;
    let mut out = Vec::new();
    out.try_reserve_exact(count).map_err(|e| {
        binrw::Error::Io(binrw::io::Error::new(
            binrw::io::ErrorKind::OutOfMemory,
            e.to_string(),
        ))
    })?;
    for _ in 0..count {
        out.push(<_>::read_options(reader, endian, ())?);
    }
    Ok(out)
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub(super) struct ProgramFunctionPack {
    #[bw(calc = instructions.len() as u32)]
    num_instructions: u32,
    #[br(count = num_instructions)]
    pub(super) instructions: Vec<ProgramInstructionPack>,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub(super) struct ProgramInstructionPack {
    pub(super) body: InstructionBodyPack,
    #[bw(calc = operands.len() as u32)]
    num_operands: u32,
    #[bw(calc = output_types.len() as u32)]
    num_output_types: u32,
    #[br(count = num_operands)]
    pub(super) operands: Vec<ValuePack>,
    #[br(count = num_output_types)]
    pub(super) output_types: Vec<TensorTypePack>,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub(super) struct ValuePack {
    pub(super) instruction: u32,
    pub(super) slot: u16,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub(super) struct TensorTypePack {
    pub(super) dtype: DtypePack,
    #[bw(calc = shape.len() as u32)]
    shape_len: u32,
    #[br(count = shape_len)]
    pub(super) shape: Vec<DimPack>,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub(super) enum DimPack {
    #[brw(magic = b'f')]
    Fixed(u64),
    #[brw(magic = b'b')]
    Bounded(u64),
}

#[binrw]
#[brw(big)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(super) enum DtypePack {
    #[brw(magic = b'C')]
    C128, // complex
    #[brw(magic = b'c')]
    C64,
    #[brw(magic = b'F')]
    F64, // real
    #[brw(magic = b'f')]
    F32,
    #[brw(magic = b'I')]
    I64, // signed integer
    #[brw(magic = b'i')]
    I32,
    #[brw(magic = b'l')]
    I16,
    #[brw(magic = b'b')]
    I8,
    #[brw(magic = b'U')]
    U64, // unsigned integer
    #[brw(magic = b'u')]
    U32,
    #[brw(magic = b'L')]
    U16,
    #[brw(magic = b'B')]
    U8,
    #[brw(magic = b'n')]
    Bit, // bool
    #[brw(magic = b'P')]
    Philo4x32,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub(super) enum InstructionBodyPack {
    #[brw(magic = b'p')]
    Parameter,
    #[brw(magic = b'o')]
    Op(ProgramOpPack),
    #[brw(magic = b'c')]
    Call(u32),
    #[brw(magic = b'r')]
    Result,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub(super) enum ProgramOpPack {
    #[brw(magic = b'+')]
    Add,
    #[brw(magic = b'/')]
    Divide,
    #[brw(magic = b'x')]
    Multiply,
    #[brw(magic = b'*')]
    Power,
    #[brw(magic = b'r')]
    Remainder,
    #[brw(magic = b's')]
    Subtract,
    #[brw(magic = b'&')]
    BitwiseAnd,
    #[brw(magic = b'!')]
    BitwiseNot,
    #[brw(magic = b'|')]
    BitwiseOr,
    #[brw(magic = b'^')]
    BitwiseXor,
    #[brw(magic = b'p')]
    Parity(u64),
    #[brw(magic = b'b')]
    BroadcastTo {
        #[bw(calc = targets.len() as u64)]
        num_targets: u64,
        #[br(count = num_targets)]
        targets: Vec<DimPack>,
    },
    #[brw(magic = b'c')]
    Cast(DtypePack),
    #[brw(magic = b'C')]
    Constant(TensorPack),
    #[brw(magic = b'm')]
    Mean(u64),
    #[brw(magic = b'S')]
    Std(u64, f64),
    #[brw(magic = b'v')]
    Variance(u64, f64),
    #[brw(magic = b'L')]
    ShotLoop(ShotLoopPack),
    #[brw(magic = b'E')]
    EulerAnglesZYZ(f64),
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub(super) struct ShotLoopPack {
    pub(super) shots: u64,
    #[bw(calc = circuits.len() as u64)]
    circuits_size: u64,
    #[br(count = circuits_size)]
    pub(super) circuits: Bytes,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub(super) struct TensorPack {
    pub(super) dtype: DtypePack,
    #[bw(calc = shape.len() as u32)]
    shape_len: u32,
    #[br(count = shape_len)]
    pub(super) shape: Vec<u64>,
    pub(super) num_elems: u64,
    #[br(args(dtype, num_elems as usize))]
    pub(super) data: TensorData,
}

#[binrw]
#[brw(big)]
#[br(import(dtype: DtypePack, len: usize))]
#[derive(Debug)]
pub(super) enum TensorData {
    #[br(pre_assert(dtype == DtypePack::C128))]
    C128(#[br(count = len)] Vec<[f64; 2]>),
    #[br(pre_assert(dtype == DtypePack::C64))]
    C64(#[br(count = len)] Vec<[f32; 2]>),
    #[br(pre_assert(dtype == DtypePack::F64))]
    F64(#[br(count = len)] Vec<f64>),
    #[br(pre_assert(dtype == DtypePack::F32))]
    F32(#[br(count = len)] Vec<f32>),
    #[br(pre_assert(dtype == DtypePack::I64))]
    I64(#[br(count = len)] Vec<i64>),
    #[br(pre_assert(dtype == DtypePack::I32))]
    I32(#[br(count = len)] Vec<i32>),
    #[br(pre_assert(dtype == DtypePack::I16))]
    I16(#[br(count = len)] Vec<i16>),
    #[br(pre_assert(dtype == DtypePack::I8))]
    I8(#[br(count = len)] Vec<i8>),
    #[br(pre_assert(dtype == DtypePack::U64))]
    U64(#[br(count = len)] Vec<u64>),
    #[br(pre_assert(dtype == DtypePack::U32))]
    U32(#[br(count = len)] Vec<u32>),
    #[br(pre_assert(dtype == DtypePack::U16))]
    U16(#[br(count = len)] Vec<u16>),
    #[br(pre_assert(dtype == DtypePack::U8))]
    U8(#[br(count = len)] Vec<u8>),
    #[br(pre_assert(dtype == DtypePack::Bit))]
    Bit(#[br(count = len)] Vec<u8>),
    #[br(pre_assert(dtype == DtypePack::Philo4x32))]
    Philo4x32(#[br(count = len)] Vec<u32>),
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct DataTreePack {
    #[bw(calc = nodes.len() as u64)]
    pub num_nodes: u64,
    #[br(count = num_nodes)]
    pub nodes: Vec<DataTreeNodePack>,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub enum DataTreeNodePack {
    #[brw(magic = 0u8)]
    Leaf,
    #[brw(magic = 1u8)]
    Branch(DataTreeBranchPack),
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct DataTreeBranchPack {
    pub num_children: u64,
    pub keys_pack: DataTreeBranchKeysPack,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct DataTreeBranchKeysPack {
    #[bw(calc = keys.len() as u32)]
    pub num_keys: u32,
    #[br(count = num_keys, )]
    pub keys: Vec<(StringPack, u64)>,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub(crate) struct StringPack {
    #[bw(calc = value.len() as u64)]
    size: u64,

    #[bw(write_with=key_encoder)]
    #[br(args((size,)), parse_with=key_decoder)]
    pub value: String,
}

#[binrw::parser(reader)]
fn key_decoder(args: (u64,)) -> BinResult<String> {
    let count = args.0 as usize;
    let mut out = Vec::new();
    out.try_reserve_exact(count).map_err(|e| {
        binrw::Error::Io(binrw::io::Error::new(
            binrw::io::ErrorKind::OutOfMemory,
            e.to_string(),
        ))
    })?;
    if reader.take(count as u64).read_to_end(&mut out)? == count {
        String::from_utf8(out).map_err(|e| binrw::Error::Custom {
            pos: reader.stream_position().unwrap_or(0),
            err: Box::new(e),
        })
    } else {
        Err(binrw::Error::Io(binrw::io::Error::new(
            binrw::io::ErrorKind::UnexpectedEof,
            "Insufficient bytes in QPY for specified size",
        )))
    }
}

#[binrw::writer(writer, endian)]
fn key_encoder(key: &String) -> BinResult<()> {
    key.as_bytes().write_options(writer, endian, ())
}
