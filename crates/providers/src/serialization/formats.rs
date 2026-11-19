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
    // placeholder
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct InstructionBodyPack{
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
