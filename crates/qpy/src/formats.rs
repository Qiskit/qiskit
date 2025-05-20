// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use binrw::{binrw, binread, binwrite, BinRead, BinResult, BinWrite, Endian};
use std::io::{Read, Write, Seek};
use crate::bytes::Bytes;
use crate::value::{DumpedValue, ExpressionType};

/// The overall structure of the QPY file
#[derive(BinWrite, Debug)]
#[brw(big)]
pub struct QPYFormatV13 {
    pub header: CircuitHeaderV12Pack,
    pub standalone_vars: Vec<ExpressionVarDeclarationPack>,
    pub custom_instructions: CustomCircuitInstructionsPack,
    pub instructions: Vec<CircuitInstructionV2Pack>,
    pub calibrations: CalibrationsPack,
    pub layout: LayoutV2Pack,
}

#[binwrite]
#[binread]
#[derive(Debug)]
#[brw(big)]
pub struct CircuitHeaderV12Pack {
    #[bw(calc = circuit_name.as_bytes().len() as u16)]
    pub name_size: u16,
    #[bw(calc = global_phase_data.data_type)]
    pub global_phase_type: u8,
    #[bw(calc = global_phase_data.data.len() as u16)]
    pub global_phase_size: u16,
    pub num_qubits: u32,
    pub num_clbits: u32,
    #[bw(calc = metadata.len() as u64)]
    pub metadata_size: u64,
    #[bw(calc = registers.len() as u32)]
    pub num_registers: u32,
    pub num_instructions: u64,
    pub num_vars: u32,
    #[br(parse_with = read_string, args(name_size as usize))]
    #[bw(write_with = write_string)]
    pub circuit_name: String,
    #[br(parse_with = DumpedValue::read, args(global_phase_size as usize, global_phase_type))]
    #[bw(write_with = DumpedValue::write)]
    pub global_phase_data: DumpedValue,
    #[br(count = metadata_size)]
    pub metadata: Bytes,
    #[br(count = num_registers)] // TODO: this is wrong
    pub registers: Vec<RegisterV4Pack>,
}


// circuit instructions related
#[binwrite]
#[binread]
#[derive(Debug)]
#[brw(big)]
pub struct CircuitInstructionV2Pack {
    #[bw(calc = gate_class_name.len() as u16)]
    pub name_size: u16,
    #[bw(calc = label.len() as u16)]
    pub label_size: u16,
    #[bw(calc = params.len() as u16)]
    pub num_parameters: u16,
    pub num_qargs: u32,
    pub num_cargs: u32,
    #[bw(calc = condition.key)]
    pub conditional_key: u8,
    #[bw(calc = condition.register_size)]
    pub condition_register_size: u16,
    #[bw(calc = condition.value)]
    pub condition_value: i64,
    pub num_ctrl_qubits: u32,
    pub ctrl_state: u32,
    #[br(parse_with = read_string, args(name_size as usize))]
    #[bw(write_with = write_string)]
    pub gate_class_name: String,
    #[br(parse_with = read_string, args(label_size as usize))]
    #[bw(write_with = write_string)]
    pub label: String,
    #[br(parse_with = ConditionPack::read, args(condition_register_size, conditional_key, condition_value))]
    #[bw(write_with = ConditionPack::write)]
    pub condition: ConditionPack,
    #[br(count = num_qargs + num_cargs)]
    pub bit_data: Vec<CircuitInstructionArgPack>,
    #[br(count = num_parameters)]
    pub params: Vec<PackedParam>,
}

#[derive(BinWrite, BinRead)]
#[brw(big)]
#[derive(Debug)]
pub struct CircuitInstructionArgPack {
    pub bit_type: u8,
    pub index: u32,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct CustomCircuitInstructionsPack {
    #[bw(calc = custom_instructions.len() as u64)]
    pub custom_operations_length: u64,
    #[br(count = custom_operations_length)]
    pub custom_instructions: Vec<CustomCircuitInstructionDefPack>,
}

// This struct seems more suitable as enum with binrw magic numbers
// unfortunetly, since the gate_type field is second, binrw magic won't work here
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct CustomCircuitInstructionDefPack {
    #[bw(calc = name.len() as u16)]
    pub gate_name_size: u16,
    pub gate_type: u8,
    pub num_qubits: u32,
    pub num_clbits: u32,
    pub custom_definition: u8,
    #[bw(calc = data.len() as u64)]
    pub size: u64,
    pub num_ctrl_qubits: u32,
    pub ctrl_state: u32,
    #[bw(calc = base_gate_raw.len() as u64)]
    pub base_gate_size: u64,
    #[br(parse_with = read_string, args(gate_name_size as usize))]
    #[bw(write_with = write_string)]
    pub name: String,
    #[br(count = size)]
    pub data: Bytes,
    #[br(count = base_gate_size)]
    pub base_gate_raw: Bytes,
}

#[binread]
#[binwrite]
#[brw(big)]
#[derive(Debug)]
pub struct RegisterV4Pack {
    pub register_type: u8,
    pub standalone: u8,
    #[bw(calc = bit_indices.len() as u32)]
    pub size: u32,
    #[bw(calc = name.len() as u16)]
    pub name_size: u16,
    pub in_circuit: u8,
    #[br(parse_with = read_string, args(name_size as usize))]
    #[bw(write_with = write_string)]
    pub name: String,
    #[br(count = size)]
    pub bit_indices: Vec<i64>,
}

// Conditions

// The most natural thing is to encode conditions as enum with magic numbers differentiating between different kinds
// however, since in QPY the condition data is not stored consecutively, it's going to be a mass to implement, so we do something else
pub mod condition_types {
    pub const NONE: u8 = 0;
    pub const TWO_TUPLE: u8 = 1;
    pub const EXPRESSION: u8 = 2;
}

// register SHOULD be a string, but since we encode some registers starting with "\x00" they are rendered illegal
// we should probably change this in future versions to support magic numbers
#[derive(Debug)]
pub enum ConditionData {
    None,
    Register(Bytes), 
    Expression(GenericDataPack)
}
#[derive(Debug)]
pub struct ConditionPack {
    pub key: u8,
    pub register_size: u16,
    pub value: i64,
    pub data: ConditionData,
}

impl ConditionPack {
    pub fn write<W: Write + Seek>(
        value: &ConditionPack,
        writer: &mut W,
        endian: Endian,
        args: (),
    ) -> binrw::BinResult<()> {
        match &value.data {
            ConditionData::None => (),
            ConditionData::Register(register_data) => register_data.write_options(writer, endian, args)?,
            ConditionData::Expression(expression_data) => expression_data.write_options(writer, endian, args)?,
        }
        Ok(())
    }
    
    pub fn read<R: Read + Seek> (
        reader: &mut R,
        endian: Endian,
        (register_size, key, value): (u16, u8, i64),
    ) -> BinResult<ConditionPack> {
        
        let data = match key {
            condition_types::TWO_TUPLE => {
                let mut buf = vec![0u8; register_size as usize];
                reader.read_exact(&mut buf)?;
                ConditionData::Register(buf.into())
            },
            condition_types::EXPRESSION => ConditionData::Expression(GenericDataPack::read_options(reader, endian, ())?),
            condition_types::NONE | _ => ConditionData::None,
        };
        Ok(ConditionPack{
            key,
            register_size,
            value,
            data,
        })
    }
}

// impl fmt::Debug for ConditionPack {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         f
//         .debug_struct("ConditionPack")
//         .field("type", &self.condition_type)
//         .field("length", &self.condition_register_len)
//         .field("length", &self.condition_value)
//         .field("data", &self.condition_data)
//         .finish()
//     }
// }
// #[binread]
// #[binwrite]
// #[brw(big)]
// #[derive(Debug)]
// pub struct ConditionExpressionPack {
//     // in expression we don't use the condition register len or condition value fields, but still need to write then (with value 0) due to the QPY format
//     // TODO: this can be improved in a future version
//     #[br(temp)]
//     #[bw(calc = 0u16)]
//     pub _condition_register_len: u16,
//     #[br(temp)]
//     #[bw(calc = 0i64)]
//     pub _condition_value: i64,
//     pub condition_expression: GenericDataPack,
// }

// #[binread]
// #[binwrite]
// #[brw(big)]
// #[derive(Debug)]
// pub struct ConditionRegisterPack {
//     pub condition_type: u8,
//     #[bw(calc = condition_register.len() as u16)]
//     pub condition_register_len: u16,
//     pub condition_value: i64,
//     #[br(count = condition_register_len)]
//     pub condition_register: Bytes,
// }

// pub mod condition_types {
//     pub const NONE: u8 = 0;
//     pub const TWO_TUPLE: u8 = 1;
//     pub const EXPRESSION: u8 = 2;
// }

// #[binrw]
// #[brw(big)]
// #[derive(Debug)]
// enum ConditionPack {
//     // sadly, we literally need to use magic numbers and not condition_types
//     #[br(magic = 0u8)]
//     None(u16, i64),
//     #[br(magic = 1u8)]
//     Register(ConditionRegisterPack),
//     #[br(magic = 2u8)]
//     Expression(ConditionExpressionPack),
// }
// pub struct ConditionPack {
//     pub condition_type: u8,
//     #[bw(calc = condition_register.len() as u16)]
//     pub condition_register_len: u16,
//     pub condition_value: i64,
//     #[br(count = condition_register_len)]
//     pub condition_register: Bytes,
// }

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct LayoutV2Pack {
    pub exists: u8,
    pub initial_layout_size: i32,
    pub input_mapping_size: i32,
    pub final_layout_size: i32,
    #[bw(calc = extra_registers.len() as u32)]
    pub extra_registers_length: u32,
    pub input_qubit_count: i32,
    #[br(count = extra_registers_length)]
    pub extra_registers: Vec<RegisterV4Pack>,
    #[br(count = initial_layout_size.max(0))]
    pub initial_layout_items: Vec<InitialLayoutItemV2Pack>,
    #[br(count = input_mapping_size.max(0))]
    pub input_mapping_items: Vec<u32>,
    #[br(count = final_layout_size.max(0))]
    pub final_layout_items: Vec<u32>,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct InitialLayoutItemV2Pack {
    pub index_value: i32,
    pub register_name_length: i32, // in this special case, reg_name_length can be -1 indicating "no name"
    #[br(parse_with = read_string, args(register_name_length.max(0) as usize))]
    #[bw(write_with = write_string)]
    pub register_name: String,
}

#[binread]
#[binwrite]
#[brw(big)]
#[derive(Debug)]
pub struct GenericDataPack {
    pub type_key: u8,
    #[bw(calc = data.len() as u64)]
    pub data_len: u64,
    #[br(count = data_len)]
    pub data: Bytes,
}

#[derive(BinWrite, Debug)]
#[brw(big)]
pub struct GenericDataSequencePack {
    pub num_elements: u64,
    pub elements: Vec<GenericDataPack>,
}

// parameter related
#[binrw]
#[derive(Debug)]
#[brw(big)]
pub struct PackedParam {
    pub type_key: u8,
    #[bw(calc = data.len() as u64)]
    pub data_len: u64,
    #[br(count = data_len)]
    pub data: Bytes,
}

#[binrw]
#[brw(big)]
pub struct ParameterPack {
    #[bw(calc = name.as_bytes().len() as u16)]
    pub name_length: u16,
    pub uuid: [u8; 16],
    #[br(parse_with = read_string, args(name_length as usize))]
    #[bw(write_with = write_string)]
    pub name: String,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct ParameterVectorPack {
    #[bw(calc = name.as_bytes().len() as u16)]
    pub name_size: u16,
    pub vector_size: u64,
    pub uuid: [u8; 16],
    pub index: u64,
    #[br(parse_with = read_string, args(name_size as usize))]
    #[bw(write_with = write_string)]
    pub name: String,
}

#[derive(BinWrite, BinRead, Debug)]
#[brw(big)]
pub struct ParameterExpressionElementPack {
    pub op_code: u8,
    pub lhs_type: u8,
    pub lhs: [u8; 16],
    pub rhs_type: u8,
    pub rhs: [u8; 16],
}

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
pub struct ParameterExpressionPack {
    pub symbol_tables_length: u64,
    pub expression_data_length: u64,
    pub expression_data: Bytes,
    pub symbol_table_data: Bytes,
    pub extra_symbol_table_data: Bytes,
}

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
pub struct ParameterExpressionSymbolPack {
    pub symbol_key: u8,
    pub value_key: u8,
    pub value_data_len: u64,
    pub symbol_data: Bytes,
    pub value_data: Bytes,
}

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
pub struct ExtraSymbolsTablePack {
    pub keys: Vec<ParameterExpressionSymbolPack>,
    pub values: Vec<ParameterExpressionSymbolPack>,
}

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
pub struct PauliEvolutionDefPack {
    pub operator_size: u64,
    pub standalone_op: u8,
    pub time_type: u8,
    pub time_size: u64,
    pub synth_method_size: u64,
    pub pauli_data: Vec<Bytes>,
    pub time_data: Bytes,
    pub synth_data: Bytes,
}

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
pub struct SparsePauliOpListElemPack {
    pub size: u64,
    pub data: Bytes,
}

// general data types

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
pub struct MappingPack {
    pub num_elements: u64,
    pub items: Vec<MappingItem>,
}

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
pub struct MappingItem {
    pub item_header: MappingItemHeader,
    pub key_bytes: Bytes,
    pub item_bytes: Bytes,
}

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
pub struct MappingItemHeader {
    pub key_size: u16,
    pub item_type: u8,
    pub size: u16,
}

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
pub struct RangePack {
    pub start: i64,
    pub stop: i64,
    pub step: i64,
}

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
pub struct ModifierPack {
    pub modifier_type: u8,
    pub num_ctrl_qubits: u32,
    pub ctrl_state: u32,
    pub power: f64,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct ExpressionVarDeclarationPack {
    pub uuid_bytes: [u8; 16],
    pub usage: u8,
    #[bw(calc = name.as_bytes().len() as u16)]
    pub name_size: u16,
    pub exp_type: ExpressionType,
    #[br(parse_with = read_string, args(name_size as usize))]
    #[bw(write_with = write_string)]
    pub name: String,
}

// calibrations are obsolete and won't be used in Qiskit 2.0 and beyond, but we still need to consume that part of the qpy file
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct CalibrationsPack {
    pub num_cals: u16,
    // TODO: incomplete
}
// implementations of custom read/write for the more complex data types

// impl ReadEndian for QPYFormatV13 {
//     const ENDIAN: EndianKind = EndianKind::Endian(())
// }
impl BinRead for QPYFormatV13 {
    type Args<'a> = ();
    
    fn read_options<R: Read + Seek>(
        reader: &mut R,
        endian: Endian,
        _: Self::Args<'_>,
    ) -> BinResult<Self> {
        let header = CircuitHeaderV12Pack::read_options(reader, endian, ())?;
        let mut standalone_vars = Vec::with_capacity(header.num_vars as usize);
        for _ in 0..header.num_vars {
            standalone_vars.push(ExpressionVarDeclarationPack::read_options(reader, endian, ())?);
        }
        let custom_instructions = CustomCircuitInstructionsPack::read_options(reader, endian, ())?;
        let mut instructions = Vec::with_capacity(header.num_vars as usize);
        for _ in 0..header.num_instructions {
            instructions.push(CircuitInstructionV2Pack::read_options(reader, endian, ())?);
        }
        let calibrations = CalibrationsPack::read_options(reader, endian, ())?;
        let layout = LayoutV2Pack::read_options(reader, endian, ())?;
        Ok(Self { header, standalone_vars, custom_instructions, instructions, calibrations, layout })
    }
}

// impl BinWrite for QPYFormatV13 {
//     type Args<'a> = ();

//     fn write_options<W: Write + Seek>(
//             &self,
//             writer: &mut W,
//             endian: Endian,
//             args: Self::Args<'_>,
//         ) -> BinResult<()> {
//             self.header.write_options(writer, endian, ())?;
//             for var in self.standalone_vars {
//                 var.write_options(writer, endian, ())?;
//             }
//             self.custom_instructions.write_options(writer, endian, ());
//             for instruction in self.instructions {
//                 instruction.write_options(writer, endian, ());
//             }
//             self.calibrations.write_options(writer, endian, ());
//             self.layout.write_options(writer, endian, ());
//             Ok(())
//     }
// }

fn write_string<W: Write>(
    value: &String,
    writer: &mut W,
    _endian: Endian,
    _args: (),
) -> binrw::BinResult<()> {
    Ok(writer.write_all(value.as_bytes())?)
}

fn read_string<R: Read + Seek> (
    reader: &mut R,
    _endian: Endian,
    (len, ): (usize,),
) -> BinResult<String> {
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf).unwrap())
}