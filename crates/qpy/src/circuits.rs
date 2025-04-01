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
use std::io::Cursor;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::intern;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::operations::{Operation, OperationRef};
use binrw::BinWrite;
use crate::params::{SerializableParam, param_to_serializable};

#[derive(BinWrite)]
#[brw(big)]
struct CircuitInstructionV2Pack {
    gate_class_name_length: u16,
    label_raw_length: u16,
    instruction_params_length: u16,
    num_qubits: u32,
    num_clbits: u32,
    condition_type_value: u8,
    condition_register_length: u16,
    condition_value: i64,
    num_ctrl_qubits: u32,
    ctrl_state: u32,
    gate_class_name: Vec<u8>,
    label_raw: Vec<u8>,
    bit_data: Vec<CircuitInstructionArgPack>,
    params: Vec<SerializableParam>
}

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
struct CircuitInstructionArgPack {
    bit_type: u8,
    bit_value: u32,
}

fn get_packed_bit_list(inst: &PackedInstruction, circuit_data: &CircuitData) -> Vec<CircuitInstructionArgPack>{
    let mut result: Vec<CircuitInstructionArgPack> = Vec::new();
    for qubit in circuit_data.get_qargs(inst.qubits).iter(){
        result.push(CircuitInstructionArgPack { bit_type: ('q' as u8), bit_value: (qubit.index() as u32) });
    }
    for clbit in circuit_data.get_cargs(inst.clbits).iter(){
        result.push(CircuitInstructionArgPack { bit_type: ('c' as u8), bit_value: (clbit.index() as u32) });
    }
    result
}

fn gate_class_name(py: Python, inst: &PackedInstruction) -> PyResult<Vec<u8>> {
    let name = match inst.op.view() {
        OperationRef::StandardGate(gate) => {
            gate.get_gate_class(py)?
            .bind(py)
            .getattr(intern!(py, "__name__"))?
            .extract::<String>()
        }
        _ => Ok(String::from("unknown"))
    }?;
    Ok(name.into_bytes())
}

fn gate_label(inst: &PackedInstruction) -> Vec<u8> {
    match inst.label() {
        Some(label) => label.as_bytes().to_vec(),
        None => Vec::new()
    }
}

fn get_num_ctrl_qubits(py: Python, inst: &PackedInstruction) -> PyResult<u32> {
    match inst.op.view() {
        OperationRef::StandardGate(gate) => Ok(gate.num_ctrl_qubits()),
        OperationRef::Gate(py_gate) => {
            py_gate.gate
            .getattr(py, "num_ctrl_qubits")?
            .extract::<u32>(py)
        }
        OperationRef::Instruction(py_inst) => {
            py_inst.instruction
            .getattr(py, "num_ctrl_qubits")?
            .extract::<u32>(py)
        }
        _ => Ok(0)
    }
}

fn get_ctrl_state(py: Python, inst: &PackedInstruction, num_ctrl_qubits: u32) -> PyResult<u32> {
    match inst.op.view() {
        OperationRef::Gate(py_gate) => {
            py_gate.gate
            .getattr(py, "ctrl_state")?
            .extract::<u32>(py)
        }
        OperationRef::Instruction(py_inst) => {
            py_inst.instruction
            .getattr(py, "ctrl_state")?
            .extract::<u32>(py)
        }
        _ => Ok(2u32.pow(num_ctrl_qubits) - 1)
    }
}

fn get_instruction_params(instruction: &PackedInstruction) -> Vec<SerializableParam> {
    instruction
    .params_view()
    .iter()
    .map(|x| param_to_serializable(&x))
    .collect()
}

fn instruction_raw(py: Python, instruction: &PackedInstruction, circuit_data: &CircuitData) -> CircuitInstructionV2Pack{
    let class_name = gate_class_name(py, instruction).unwrap();
    let label_raw = gate_label(instruction);
    let num_ctrl_qubits = get_num_ctrl_qubits(py,instruction).unwrap_or(0);
    let ctrl_state = get_ctrl_state(py, instruction, num_ctrl_qubits).unwrap_or(0);
    let instruction_params: Vec<SerializableParam> = get_instruction_params(instruction);
    let bit_data = get_packed_bit_list(instruction, circuit_data);
    CircuitInstructionV2Pack{
        gate_class_name_length: class_name.len() as u16,
        label_raw_length: label_raw.len() as u16,
        instruction_params_length: instruction_params.len() as u16,
        num_qubits: instruction.op.num_qubits(),
        num_clbits: instruction.op.num_clbits(),
        condition_type_value: 0,
        condition_register_length: 0,
        condition_value: 0,
        num_ctrl_qubits: num_ctrl_qubits,
        ctrl_state: ctrl_state,
        gate_class_name: class_name,
        label_raw: label_raw,
        bit_data: bit_data,
        params: instruction_params,
    }
    
}
#[pyfunction]
#[pyo3(signature = (file_obj, circuit_data))]
pub fn py_write_instructions(
    py: Python,
    file_obj: &Bound<PyAny>,
    circuit_data: CircuitData,
) -> PyResult<usize> {
    for instruction in circuit_data.data(){
        let raw_instruction = instruction_raw(py, instruction, &circuit_data);
        let mut buffer = Cursor::new(Vec::new());
        raw_instruction.write(&mut buffer).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("BinRW write failed: {e}"))
        })?;
        let bytes = buffer.into_inner();
        file_obj.call_method1("write", (pyo3::types::PyBytes::new(py, &bytes),))?;
    }
    Ok(55)
}