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

#![allow(dead_code)] // Currently in progress
use binrw::binrw;

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct QuantumProgramFile {
    #[brw(magic = b"QISKIT_PROGRAM")]
    pub version: u8,
    pub qiskit_version: (u8, u8, u8),
    pub program: QuantumProgramPack,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct QuantumProgramPack {
    pub num_functions: u64,
    #[br(count = num_functions)]
    pub functions: Vec<ProgramFunctionPack>,
    pub input_structure: DataTreePack,
    pub output_structure: DataTreePack,
}

// nodes are stored in a flat vector
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
    #[br(count = num_keys)]
    pub keys: Vec<(StringU16Pack, u64)>,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct ProgramFunctionPack {
    #[bw(calc = instructions.len() as u64)]
    pub num_instructions: u64,
    #[br(count = num_instructions)]
    pub instructions: Vec<ProgramInstructionPack>,

    #[bw(calc = parameters.len() as u64)]
    pub num_parameters: u64,
    #[br(count = num_parameters)]
    pub parameters: Vec<u32>,

    #[bw(calc = results.len() as u64)]
    pub num_results: u64,
    #[br(count = num_results)]
    pub results: Vec<u32>,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct ProgramInstructionPack {
    pub body: InstructionBodyPack,

    #[bw(calc = operands.len() as u64)]
    pub num_operands: u64,
    #[br(count = num_operands)]
    pub operands: Vec<ProgramFunctionValuePack>,

    #[bw(calc = output_types.len() as u64)]
    pub num_output_types: u64,
    #[br(count = num_output_types)]
    pub output_types: Vec<TensorTypePack>,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct ProgramFunctionValuePack {
    pub instruction_id: u32,
    pub slot: u16,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct TensorTypePack {
    pub dtype: DTypePack,

    #[bw(calc = shape.len() as u64)]
    pub rank: u64,
    #[br(count = rank)]
    pub shape: Vec<DimPack>,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub enum DTypePack {
    #[brw(magic = 0u8)]
    C128,
    #[brw(magic = 1u8)]
    C64,
    #[brw(magic = 2u8)]
    F64,
    #[brw(magic = 3u8)]
    F32,
    #[brw(magic = 4u8)]
    I64,
    #[brw(magic = 5u8)]
    I32,
    #[brw(magic = 6u8)]
    I16,
    #[brw(magic = 7u8)]
    I8,
    #[brw(magic = 8u8)]
    U64,
    #[brw(magic = 9u8)]
    U32,
    #[brw(magic = 10u8)]
    U16,
    #[brw(magic = 11u8)]
    U8,
    #[brw(magic = 12u8)]
    Bit,
    #[brw(magic = 13u8)]
    Key(PrngKindPack),
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub enum PrngKindPack {
    #[brw(magic = 0u8)]
    Philox4x32,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub enum DimPack {
    #[brw(magic = 0u8)]
    Fixed(u64),
    #[brw(magic = 1u8)]
    Bounded(u64),
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct InstructionBodyPack {
    // placeholder
}

// Easy string storage when the string size is u16.
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub(crate) struct StringU16Pack {
    #[br(temp)]
    #[bw(try_calc = u16::try_from(value.len()))]
    size: u16,

    #[br(count = size, try_map = String::from_utf8)]
    #[bw(map = |value| value.as_bytes())]
    pub value: String,
}
