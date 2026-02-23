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

// Methods and data structures related to backwards compatibility with old QPY versions
// which are irrelevant for the current version

use crate::bytes::Bytes;
use crate::circuit_reader::instruction_values_to_params;
use crate::formats;
use crate::value::GenericValue;
use crate::value::{QPYReadData, ValueType};
use binrw::{binread, binrw, binwrite};
use qiskit_circuit::operations::Condition;
use qiskit_circuit::packed_instruction::PackedOperation;

use qiskit_circuit::Block;
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::interner::Interned;
use qiskit_circuit::operations::{ControlFlow, ControlFlowInstruction};
use qiskit_circuit::{Clbit, Qubit};

use pyo3::prelude::*;

// Calibrations are obsolete and won't be used in Qiskit 2.0 and beyond, but we still need to consume that part of the qpy file
#[binrw]
#[brw(big)]
#[derive(Debug)]
#[brw(import(version: u32))]
pub struct CalibrationsPack {
    #[bw(calc = calibrations.len() as u16)]
    pub num_cals: u16,
    #[br(count = num_cals, args { inner: (version,) })]
    pub calibrations: Vec<CalibrationDefPack>,
}

// A single calibration definition
// Contains the gate name, qubits, parameters, and schedule block
#[binrw]
#[brw(big)]
#[derive(Debug)]
#[brw(import(version: u32))]
pub struct CalibrationDefPack {
    #[bw(calc = name.len() as u16)]
    pub name_size: u16,
    #[bw(calc = qubits.len() as u16)]
    pub num_qubits: u16,
    #[bw(calc = params.len() as u16)]
    pub num_params: u16,
    pub cal_type: ValueType,
    #[br(count = name_size as usize, try_map = String::from_utf8)]
    #[bw(map = |s| s.as_bytes())]
    pub name: String,
    #[br(count = num_qubits)]
    pub qubits: Vec<i64>,
    #[br(count = num_params)]
    pub params: Vec<formats::GenericDataPack>,
    #[br(args(version,))]
    pub schedule: ScheduleBlockPack,
}

// Schedule block header and data
// This is legacy pulse data that we need to consume but not use
#[binrw]
#[brw(big)]
#[derive(Debug)]
#[brw(import(version: u32))]
pub struct ScheduleBlockPack {
    #[bw(calc = name.len() as u16)]
    pub name_size: u16,
    #[bw(calc = metadata.len() as u64)]
    pub metadata_size: u64,
    #[bw(calc = elements.len() as u16)]
    pub num_elements: u16,
    #[br(count = name_size as usize, try_map = String::from_utf8)]
    #[bw(map = |s| s.as_bytes())]
    pub name: String,
    #[br(count = metadata_size)]
    pub metadata: Bytes,
    pub alignment_context: AlignmentContextPack,
    #[br(count = num_elements, args { inner: (version,) })]
    pub elements: Vec<ScheduleBlockElementPack>,
    #[br(if(version >= 7))]
    pub references: Option<ScheduleReferencePack>,
}

// The same an generic data pack, but without strict type keys, since calibrations had many more now obsolete
#[binread]
#[binwrite]
#[brw(big)]
#[derive(Debug)]
pub struct GenericCalibrationDataPack {
    pub type_key: u8,
    #[bw(calc = data.len() as u64)]
    pub data_len: u64,
    #[br(count = data_len)]
    pub data: Bytes,
}

// A simple contained for a sequence of generic data items, containing its length.
#[binrw]
#[derive(Debug)]
#[brw(big)]
pub struct GenericCalibratinoDataSequencePack {
    #[bw(calc = elements.len() as u64)]
    pub num_elements: u64,
    #[br(count = num_elements)]
    pub elements: Vec<GenericCalibrationDataPack>,
}

// Alignment context for schedule blocks
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct AlignmentContextPack {
    pub type_key: u8,
    pub sequence: GenericCalibratinoDataSequencePack,
}

// A single element in a schedule block
// If element_type is 's' (SCHEDULE_BLOCK), it contains a nested ScheduleBlockPack
// Otherwise, it contains operands sequence and a name value
#[binrw]
#[brw(big)]
#[derive(Debug)]
#[brw(import(version: u32))]
pub struct ScheduleBlockElementPack {
    pub element_type: u8,
    #[br(if(element_type == b's'), args(version,))]
    pub nested_schedule: Option<ScheduleBlockPack>,
    #[br(if(element_type != b's'))]
    pub operands: Option<GenericCalibratinoDataSequencePack>,
    #[br(if(element_type != b's'))]
    pub name: Option<GenericCalibrationDataPack>,
}

// References in schedule blocks (version 7+)
// This is a mapping structure: num_elements followed by MAP_ITEM entries
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct ScheduleReferencePack {
    #[bw(calc = items.len() as u64)]
    pub num_elements: u64,
    #[br(count = num_elements)]
    pub items: Vec<ScheduleReferenceMapItem>,
}

// A single map item in the schedule reference mapping
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct ScheduleReferenceMapItem {
    #[bw(calc = key.len() as u16)]
    pub key_size: u16,
    pub item_type: u8,
    #[bw(calc = data.len() as u16)]
    pub size: u16,
    #[br(count = key_size)]
    pub key: Bytes,
    #[br(count = size)]
    pub data: Bytes,
}

// handling for non control flow gates with conditionals

pub fn wrap_conditional_gate(
    instruction: &formats::CircuitInstructionV2Pack,
    op: PackedOperation,
    cond: Condition,
    qubits: Interned<[Qubit]>,
    clbits: Interned<[Clbit]>,
    params: Option<Box<Parameters<Block>>>,
    qpy_data: &mut QPYReadData,
) -> PyResult<(PackedOperation, Option<Box<Parameters<Block>>>)> {
    use qiskit_circuit::circuit_data::CircuitData;
    use qiskit_circuit::operations::Param;
    use smallvec::SmallVec;

    // Create an IfElseOp wrapping this instruction
    let control_flow = ControlFlow::IfElse { condition: cond };
    let control_flow_instruction = ControlFlowInstruction {
        control_flow,
        num_qubits: instruction.num_qargs,
        num_clbits: instruction.num_cargs,
    };
    let if_else_op = PackedOperation::from_control_flow(Box::new(control_flow_instruction));

    // Get the actual qubit and clbit indices from the interned values
    let qubit_indices = qpy_data.circuit_data.get_qargs(qubits);
    let clbit_indices = qpy_data.circuit_data.get_cargs(clbits);

    // Convert params from Parameters<Block> to SmallVec<[Param; 3]>
    let param_vec: SmallVec<[Param; 3]> = if let Some(params_box) = params {
        match params_box.as_ref() {
            qiskit_circuit::instruction::Parameters::Params(p) => p.clone(),
            qiskit_circuit::instruction::Parameters::Blocks(_) => {
                // If we have blocks, we can't easily convert them to params
                // This shouldn't happen for simple gates with conditions
                SmallVec::new()
            }
        }
    } else {
        SmallVec::new()
    };

    // Create the body circuit using from_packed_operations
    // This avoids manually creating PackedInstruction and pushing it
    let body_data = CircuitData::from_packed_operations(
        instruction.num_qargs,
        instruction.num_cargs,
        std::iter::once(Ok((
            op,
            param_vec,
            qubit_indices.to_vec(),
            clbit_indices.to_vec(),
        ))),
        Param::Float(0.0),
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e)))?;

    // Create GenericValue::CircuitData from the CircuitData
    // The instruction_values_to_params function will handle converting this to a Block
    let if_params = vec![GenericValue::CircuitData(body_data)];
    let if_params_converted = instruction_values_to_params(if_params, qpy_data)?;

    Ok((if_else_op, if_params_converted))
}
