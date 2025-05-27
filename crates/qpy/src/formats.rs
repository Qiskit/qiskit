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

use crate::bytes::Bytes;
use crate::value::{DumpedValue, ExpressionType};
use binrw::{binread, binrw, binwrite, BinRead, BinResult, BinWrite, Endian};
use std::io::{Read, Seek, Write};

/// The overall structure of the QPY file

// TODO: Still need to split by versioning and have a struct for the whole file, not a single circuit
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

#[binrw]
#[brw(big)]
#[derive(Debug)]
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
    #[br(count = num_registers)]
    pub registers: Vec<RegisterV4Pack>,
}

// circuit instructions related
#[binrw]
#[brw(big)]
#[derive(Debug)]
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
    #[br(count = num_parameters as usize)]
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
    Expression(GenericDataPack),
}

//most of the data here is "virtual" in the sense that is is not stored as-is
//we use custom reader/writer to enable spreading the data to its relevant places in the qpy instruction
//in newer versions of qpy it may be better to store all the data consecutively
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
            ConditionData::Register(register_data) => {
                register_data.write_options(writer, endian, args)?
            }
            ConditionData::Expression(expression_data) => {
                expression_data.write_options(writer, endian, args)?
            }
        }
        Ok(())
    }

    pub fn read<R: Read + Seek>(
        reader: &mut R,
        endian: Endian,
        (register_size, key, value): (u16, u8, i64),
    ) -> BinResult<ConditionPack> {
        let data = match key {
            condition_types::TWO_TUPLE => {
                let mut buf = vec![0u8; register_size as usize];
                reader.read_exact(&mut buf)?;
                ConditionData::Register(buf.into())
            }
            condition_types::EXPRESSION => {
                ConditionData::Expression(GenericDataPack::read_options(reader, endian, ())?)
            }
            _ => ConditionData::None,
        };
        Ok(ConditionPack {
            key,
            register_size,
            value,
            data,
        })
    }
}

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
#[derive(Debug)]
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
pub enum ParameterExpressionElementPack {
    #[brw(magic = 0u8)]
    Add(ParameterExpressionStandardOpPack),
    #[brw(magic = 1u8)]
    Sub(ParameterExpressionStandardOpPack),
    #[brw(magic = 2u8)]
    Mul(ParameterExpressionStandardOpPack),
    #[brw(magic = 3u8)]
    Div(ParameterExpressionStandardOpPack),
    #[brw(magic = 4u8)]
    Pow(ParameterExpressionStandardOpPack),
    #[brw(magic = 5u8)]
    Sin(ParameterExpressionStandardOpPack),
    #[brw(magic = 6u8)]
    Cos(ParameterExpressionStandardOpPack),
    #[brw(magic = 7u8)]
    Tan(ParameterExpressionStandardOpPack),
    #[brw(magic = 8u8)]
    Asin(ParameterExpressionStandardOpPack),
    #[brw(magic = 9u8)]
    Acos(ParameterExpressionStandardOpPack),
    #[brw(magic = 10u8)]
    Exp(ParameterExpressionStandardOpPack),
    #[brw(magic = 11u8)]
    Log(ParameterExpressionStandardOpPack),
    #[brw(magic = 12u8)]
    Sign(ParameterExpressionStandardOpPack),
    #[brw(magic = 13u8)]
    Grad(ParameterExpressionStandardOpPack),
    #[brw(magic = 14u8)]
    Conj(ParameterExpressionStandardOpPack),
    #[brw(magic = 15u8)]
    Substitute(ParameterExpressionSubsOpPack),
    #[brw(magic = 16u8)]
    Abs(ParameterExpressionStandardOpPack),
    #[brw(magic = 17u8)]
    Atan(ParameterExpressionStandardOpPack),
    #[brw(magic = 18u8)]
    Rsub(ParameterExpressionStandardOpPack),
    #[brw(magic = 19u8)]
    Rdiv(ParameterExpressionStandardOpPack),
    #[brw(magic = 20u8)]
    Rpow(ParameterExpressionStandardOpPack),
    #[brw(magic = 255u8)]
    Expression(ParameterExpressionStandardOpPack),
}

#[derive(BinWrite, BinRead, Debug)]
#[brw(big)]
pub struct ParameterExpressionStandardOpPack {
    pub lhs_type: u8,
    pub lhs: [u8; 16],
    pub rhs_type: u8,
    pub rhs: [u8; 16],
}
#[binrw]
#[derive(Debug)]
#[brw(big)]
pub struct ParameterExpressionSubsOpPack {
    #[bw(calc = "u".as_bytes()[0])]
    pub _lhs_type: u8,
    #[bw(calc = {
        let mut size = [0u8; 16];
        let len_bytes = (mapping_data.len() as u64).to_be_bytes();
        size[..8].copy_from_slice(&len_bytes);
        size
    })]
    pub mapping_data_size: [u8; 16],
    #[bw(calc = "n".as_bytes()[0])]
    pub _rhs_type: u8,
    #[bw(calc = [0u8; 16])]
    pub _rhs: [u8; 16],

    #[br(count = u64::from_be_bytes(mapping_data_size[..8].try_into().unwrap()))]
    pub mapping_data: Bytes,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct ParameterExpressionPack {
    #[bw(calc = symbol_table_data.len() as u64)]
    pub symbol_tables_length: u64,
    #[bw(calc = expression_data.len() as u64)]
    pub expression_data_length: u64,
    #[br(count = expression_data_length)]
    pub expression_data: Bytes,
    #[br(count = symbol_tables_length)]
    pub symbol_table_data: Vec<ParameterExpressionSymbolPack>,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub enum ParameterExpressionSymbolPack {
    #[brw(magic = b'p')]
    Parameter(ParameterExpressionParameterSymbolPack),
    #[brw(magic = b'v')]
    ParameterVector(ParameterExpressionParameterVectorSymbolPack),
    #[brw(magic = b'e')]
    ParameterExpression(ParameterExpressionParameterExpressionSymbolPack),
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct ParameterExpressionParameterSymbolPack {
    pub value_key: u8,
    #[bw(calc = value_data.len() as u64)]
    pub value_data_len: u64,
    pub symbol_data: ParameterPack,
    #[br(count = value_data_len)]
    pub value_data: Bytes,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct ParameterExpressionParameterVectorSymbolPack {
    pub value_key: u8,
    #[bw(calc = value_data.len() as u64)]
    pub value_data_len: u64,
    pub symbol_data: ParameterVectorPack,
    #[br(count = value_data_len)]
    pub value_data: Bytes,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct ParameterExpressionParameterExpressionSymbolPack {
    pub value_key: u8,
    #[bw(calc = value_data.len() as u64)]
    pub value_data_len: u64,
    pub symbol_data: ParameterExpressionPack,
    #[br(count = value_data_len)]
    pub value_data: Bytes,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct PauliEvolutionDefPack {
    #[bw(calc = pauli_data.len() as u64)]
    pub operator_size: u64,
    pub standalone_op: u8,
    pub time_type: u8,
    #[bw(calc = time_data.len() as u64)]
    pub time_size: u64,
    #[bw(calc = synth_data.len() as u64)]
    pub synth_method_size: u64,
    #[br(count = operator_size)]
    pub pauli_data: Vec<SparsePauliOpListElemPack>,
    #[br(count = time_size)]
    pub time_data: Bytes,
    #[br(count = synth_method_size)]
    pub synth_data: Bytes,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct SparsePauliOpListElemPack {
    #[bw(calc = data.len() as u64)]
    pub size: u64,
    #[br(count = size)]
    pub data: Bytes,
}

// general data types

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct MappingPack {
    #[bw(calc = items.len() as u64)]
    pub num_elements: u64,
    #[br(count = num_elements)]
    pub items: Vec<MappingItem>,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct MappingItem {
    #[bw(calc = key_bytes.len() as u16)]
    pub key_size: u16,
    pub item_type: u8,
    #[bw(calc = item_bytes.len() as u16)]
    pub size: u16,
    #[br(count = usize::from(key_size))]
    pub key_bytes: Bytes,
    #[br(count = usize::from(size))]
    pub item_bytes: Bytes,
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
            standalone_vars.push(ExpressionVarDeclarationPack::read_options(
                reader,
                endian,
                (),
            )?);
        }
        let custom_instructions = CustomCircuitInstructionsPack::read_options(reader, endian, ())?;
        let mut instructions = Vec::with_capacity(header.num_vars as usize);
        for _ in 0..header.num_instructions {
            instructions.push(CircuitInstructionV2Pack::read_options(reader, endian, ())?);
        }
        let calibrations = CalibrationsPack::read_options(reader, endian, ())?;
        let layout = LayoutV2Pack::read_options(reader, endian, ())?;
        Ok(Self {
            header,
            standalone_vars,
            custom_instructions,
            instructions,
            calibrations,
            layout,
        })
    }
}

fn write_string<W: Write>(
    value: &String,
    writer: &mut W,
    _endian: Endian,
    _args: (),
) -> binrw::BinResult<()> {
    Ok(writer.write_all(value.as_bytes())?)
}

fn read_string<R: Read + Seek>(
    reader: &mut R,
    _endian: Endian,
    (len,): (usize,),
) -> BinResult<String> {
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf).unwrap())
}
