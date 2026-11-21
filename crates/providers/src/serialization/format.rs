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

//! The Binary Serialization format for QuantumPrograms
//!
//! This module defines the binary serialization format details. All values in
//! the payload use network byte order (big endian) and the data is typically
//! represented in this doc using packed c structs. Except were explicitly noted
//! there is no padding between fields.
//!
//! The payload starts with the following 6 byte UTF8 string: `QKPROG`
//! which is immediately followed by following `PAYLOAD_HEADER` struct.
//!
//! ## `PAYLOAD_HEADER`
//! ```c
//! struct {
//!   uint8_t payload_version;
//!   uint8_t qiskit_major_version;
//!   uint8_t qiskit_minor_version;
//!   uint8_t qiskit_patch_version;
//! }
//!
//! This is then followed by two `DATA_TREE` structs representing first the input
//! structure data tree and then the output structure data tree. Following that
//! is a `uint64_t` representing the number of functions in the program. This is
//! followed by that value `PROGRAM_FUNCTION` representations.
//!
//! ## `DATA_TREE`
//!
//! ```c
//! struct {
//!   uint64_t num_nodes;
//! }
//!
//! which is immediately followed by an array of `num_nodes` node representations.
//! Each node representation starts with a discriminant of a single `uint8_t` that
//! is valued either 0 or 1, where 0 represents a leaf or 1 represents a branch. If
//! the value is 1 it is immediately followed by a `DATA_TREE_BRANCH` data.
//!
//! ### `DATA_TREE_BRANCH
//!
//! ```c
//! struct {
//!   uint64_t num_children;
//!   uin32_t num_keys;
//! }
//! ```
//! which is then immediately followed by `num_keys` key representations. Each key
//! representation starts with a `uint64_t` that representas the string length, that
//! is then followed by that number of UTF8 bytes for the string of the name. Finally
//! there is another `uint64_t` representing the index of the key in the branch.
//!
//! ## `PROGRAM_FUNCTION`
//!
//! Each `PROGRAM_FUNCTION` starts with a `uint32_t` representing howmany instructions
//! are in the function. This is followed by that number of the following `INSTRUCTION`
//! structs representing each instruction.
//!
//! ### `INSTRUCTION`
//! Each `INSTRUCTION` starts with a `INSTRUCTION_BODY` and then is immediately followed by the
//! following struct
//!
//! ```c
//! struct {
//!   uint32_t num_operands;
//!   uint32_t num_output_types;
//! }
//! ```
//!
//! which is then followed by an array `num_operands` `VALUE` structs and `num_output_type`
//! `TENSOR_TYPE` structs.
//!
//! ### `INSTRUCTION_BODY`
//!
//! The instruction body data starts with a char discriminant to determine the type of the
//! instruction
//!
//! | Value | Type |
//! |-------|------|
//! | p | Parameter |
//! | o | Operation |
//! | c | Call |
//! | r | Result |
//!
//! `p` and `r` are followed by no additional data. If the value is a `c` for a call this
//! includes a `uint32_t` for the index being called. If the value is an `o` the following
//! data immediately follows it.is another char discriminant indicating the operation and depnding
//! on the discriminant there might be additional data following it. You can refer to the following
//! table for the details
//!
//! | Value | Type | Subsequent data structure |
//! |-------|------|---------------------------|
//! | + | Add | N/A |
//! | / | Divide | N/A |
//! | x | Multiply | N/A |
//! | * | Power | N/A |
//! | r | Remainder | N/A |
//! | s | Subtract | N/A |
//! | & | Bitwise And | N/A |
//! | ! | Bitwise Not | N/A |
//! | | | Bitwise Or | N/A |
//! | ^ | Bitwise Xor | N/A |
//! | p | Parity | `uint64_t` representing the axis |
//! | b | Broadcast To | `uint64_t` representing the num of targets followed that number of a char `f` or `b` representing either a fixed or bound dimension and a `uint64_t` representing the dimension |
//! | c | Cast | char following the values in [DTYPE] |
//! | C | Constant | Data defined in TENSOR |
//! | m | Mean | `uint64_t` representing the axis |
//! | S | Std | `uint64_t` & `double` representing the axis and ddof |
//! | v | Variance | `uint64_t` & `double` representing the axis and ddof |
//! | L | Shot Loop | Data defined in SHOT_LOOP |
//! | E | Euler Angles ZYZ | `double` representing atol |
//! | e | Entropy | `uint64_t` and then that many `uint64_t` which represent the shape |
//! | P | Split |  `uint64_t` and then that many `uint64_t` which represent the shape |
//!
//! #### `SHOT_LOOP`
//!
//! ```c
//! struct {
//!    uint64_t shots;
//!    uint64_t circuits_bytes;
//! }
//! ```
//! which is immediately followed by `circuits_bytes` bytes of QPY data that encodes the circuits
//! in the shot loop.
//!
//! #### `TENSOR`
//!
//!```c
//! struct {
//!   char dtype;
//!   u32 shape_len;
//! }
//! ```
//! where dtype follows the values in DTYPE. This struct is immediately followed by `shape_len`
//! `uint64_t` values which represent the shape of the tensor. This is followed by a `uint64_t`
//! which is the number of elements (it must always be equal to the product of all the shape
//! values). This is then followed by `num_elems` values that match the dtype. The following
//! table contains the value data type for each dtype value.
//!
//! | Dtype | Data |
//! |-------|------|
//! | c128  | [double, double] (representing real and imaginary |
//! | c64 | [float, float] (representing real and imaginary |
//! | f64 | double |
//! | f32 | float |
//! | i64 | int64_t |
//! | i32 | int32_t |
//! | i16 | int16_t |
//! | i8 | int8_t |
//! | u64 | uint64_t |
//! | u32 | uint32_t |
//! | u16 | uint16_t |
//! | u8 | uint8_t |
//! | Bool | uint8_t |
//! | Philox 4x32 Key | uint32_t |
//!
//! #### `VALUE`
//!
//! ```c
//! struct {
//!   uint32_t instruction;
//!   uint16_t slot;
//! }
//!
//! ```
//!
//! where slot and instruction represent the index for each.
//!
//! #### `TENSOR_TYPE`
//!
//! ```c
//! struct {
//!   char dtype;
//!   uint32_t shape_len;
//! }
//! ```
//! This struct is followed immediately by an array of `shape_len` values which are a uint64_t
//! with a single byte discriminant prior to the value of either 'f' or 'b' representing either a
//! fixed or bound dimension value. The dtype field defines the dtype of the tensor type.
//!
//! #### DTYPE
//!
//! The following table maps the value to the dtype
//!
//! | Value | Type |
//! |-------|------|
//! | C | C128 |
//! | c | C64 |
//! | F |f64 |
//! | f | f32 |
//! | I | i64 |
//! | i | i32 |
//! | l | i16 |
//! | b | i8 |
//! | U | u64 |
//! | u | u32 |
//! | L | u16 |
//! | B | u8 |
//! | n | bool |
//! | P | Philox 4x32 rng key |
//!

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
    #[brw(magic = b"e\0")] // 0 here is a future use discriminant if we ever add additional prngs
    Entropy {
        #[bw(calc = shape.len() as u64)]
        shape_size: u64,
        #[br(count = shape_size)]
        shape: Vec<u64>,
    },
    #[brw(magic = b'P')]
    Split {
        #[bw(calc = shape.len() as u64)]
        shape_size: u64,
        #[br(count = shape_size)]
        shape: Vec<u64>,
    },
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
