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
use crate::params::{SerializableParam};

/// The overall structure of the QPY file
#[derive(BinWrite)]
#[brw(big)]
pub struct QPYFormatV13 {
    pub header: HeaderData,
    pub custom_instructions: CustomCircuitInstructionsPack,
    pub instructions: Vec<CircuitInstructionV2Pack>,
    pub calibration_header: Vec<u8>,
    pub layout: LayoutV2Pack
}

#[derive(BinWrite)]
#[brw(big)]
pub struct HeaderData {
    pub header: CircuitHeaderV12Pack,
    pub circuit_name: Vec<u8>,
    pub global_phase_data: Vec<u8>,
    pub metadata: Vec<u8>,
    pub qregs: Vec<u8>,
    pub cregs: Vec<u8>
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

#[derive(BinWrite)]
#[brw(big)]
pub struct CircuitInstructionV2Pack {
    pub gate_class_name_length: u16,
    pub label_raw_length: u16,
    pub instruction_params_length: u16,
    pub num_qubits: u32,
    pub num_clbits: u32,
    pub condition_type_value: u8,
    pub condition_register_length: u16,
    pub condition_value: i64,
    pub num_ctrl_qubits: u32,
    pub ctrl_state: u32,
    pub gate_class_name: Vec<u8>,
    pub label_raw: Vec<u8>,
    pub condition_raw: Vec<u8>,
    pub bit_data: Vec<CircuitInstructionArgPack>,
    pub params: Vec<SerializableParam>
}

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
pub struct CircuitInstructionArgPack {
    pub bit_type: u8,
    pub bit_value: u32,
}
#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
pub struct CustomCircuitInstructionsPack {
    pub custom_operations_length: u64,
}

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
pub struct RegisterV4Pack {
    pub reg_type: u8,
    pub standalone: u8,
    pub reg_size: u32,
    pub reg_name_length: u16,
    pub is_in_circuit: u8,
    pub name: Vec<u8>,
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