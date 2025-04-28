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

use binrw::BinWrite;

pub type Bytes = Vec<u8>;

/// The overall structure of the QPY file
#[derive(BinWrite)]
#[brw(big)]
pub struct QPYFormatV13 {
    pub header: HeaderData,
    pub custom_instructions: CustomCircuitInstructionsPack,
    pub instructions: Vec<CircuitInstructionV2Pack>,
    pub calibration_header: Bytes,
    pub layout: LayoutV2Pack,
}

// header related
#[derive(BinWrite)]
#[brw(big)]
pub struct HeaderData {
    pub header: CircuitHeaderV12Pack,
    pub circuit_name: Bytes,
    pub global_phase_data: Bytes,
    pub metadata: Bytes,
    pub qregs: Bytes,
    pub cregs: Bytes,
}

#[derive(BinWrite)]
#[brw(big)]
pub struct CircuitHeaderV12Pack {
    pub name_size: u16,
    pub global_phase_type: u8,
    pub global_phase_size: u16,
    pub num_qubits: u32,
    pub num_clbits: u32,
    pub metadata_size: u64,
    pub num_registers: u32,
    pub num_instructions: u64,
    pub num_vars: u32,
}

// circuit instructions related
#[derive(BinWrite)]
#[brw(big)]
pub struct CircuitInstructionV2Pack {
    pub name_size: u16,
    pub label_size: u16,
    pub num_parameters: u16,
    pub num_qargs: u32,
    pub num_cargs: u32,
    pub conditional_key: u8,
    pub condition_register_size: u16,
    pub condition_value: i64,
    pub num_ctrl_qubits: u32,
    pub ctrl_state: u32,
    pub gate_class_name: Bytes,
    pub label_raw: Bytes,
    pub condition_raw: Bytes,
    pub bit_data: Vec<CircuitInstructionArgPack>,
    pub params: Vec<PackedParam>,
}

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
pub struct CircuitInstructionArgPack {
    pub bit_type: u8,
    pub index: u32,
}

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
pub struct CustomCircuitInstructionsPack {
    pub custom_operations_length: u64,
    pub custom_instructions: Vec<CustomCircuitInstructionDefPack>,
}

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
pub struct CustomCircuitInstructionDefPack {
    pub gate_name_size: u16,
    pub gate_type: u8,
    pub num_qubits: u32,
    pub num_clbits: u32,
    pub custom_definition: u8,
    pub size: u64,
    pub num_ctrl_qubits: u32,
    pub ctrl_state: u32,
    pub base_gate_size: u64,
    pub name_raw: Bytes,
    pub data: Bytes,
    pub base_gate_raw: Bytes
}
// "CUSTOM_CIRCUIT_INST_DEF",
// [
//     "gate_name_size",
//     "type",
//     "num_qubits",
//     "num_clbits",
//     "custom_definition",
//     "size",
//     "num_ctrl_qubits",
//     "ctrl_state",
//     "base_gate_size",
// ],
// )
// ustom_operation_raw = struct.pack(
//     formats.CUSTOM_CIRCUIT_INST_DEF_V2_PACK,
//     len(name_raw),
//     type_key,
//     num_qubits,
//     num_clbits,
//     has_definition,
//     size,
//     num_ctrl_qubits,
//     ctrl_state,
//     len(base_gate_raw),
// )

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
pub struct RegisterV4Pack {
    pub register_type: u8,
    pub standalone: u8,
    pub size: u32,
    pub name_size: u16,
    pub in_circuit: u8,
    pub name: Bytes,
    pub bit_indices: Vec<i64>,
}

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
pub struct LayoutV2Pack {
    pub exists: u8,
    pub initial_layout_size: i32,
    pub input_mapping_size: i32,
    pub final_layout_size: i32,
    pub extra_registers_length: u32,
    pub input_qubit_count: i32,
    // TODO: incomplete; should have the layout data here
}

// parameter related
#[derive(BinWrite)]
#[brw(big)]
pub struct PackedParam {
    pub type_key: u8,
    pub data_len: u64,
    pub data: Bytes,
}

#[derive(BinWrite)]
#[brw(big)]
pub struct ParameterPack {
    pub name_length: u16,
    pub uuid: [u8; 16],
    pub name: Bytes,
}

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
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
    pub item_bytes: Bytes
}

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
pub struct MappingItemHeader {
    pub key_size: u16,
    pub item_type: u8,
    pub size: u16,
}