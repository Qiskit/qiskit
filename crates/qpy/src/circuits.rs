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

// We use the following terminology:
// 1. "Pack": To create a struct (from formats.rs) from the original data
// 2. "Serialize": To create binary data (Bytes) from the original data
// 3. "Write": To write to a file obj the serialization of the original data
// Ideally, serialization is done by packing in a binrw-enhanced struct and using the
// `write` method into a `Cursor` buffer, but there might be exceptions.

use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyIterator, PyList, PyTuple};
use qiskit_circuit::bit::{PyClassicalRegister, PyClbit, ShareableClbit};
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::imports::{BARRIER, DELAY, MEASURE, RESET};
use qiskit_circuit::operations::{Operation, OperationRef, StandardInstruction};
use qiskit_circuit::packed_instruction::PackedInstruction;
use std::io::Cursor;

use crate::formats::{
    Bytes, CircuitHeaderV12Pack, CircuitInstructionArgPack, CircuitInstructionV2Pack,
    CustomCircuitInstructionsPack, HeaderData, LayoutV2Pack, QPYFormatV13, RegisterV4Pack,
    PackedParam
};
use crate::params::pack_param;
use crate::value::dumps_value;
use binrw::BinWrite;

// For debugging purposes
fn hex_string(bytes: &Bytes) -> String {
    bytes
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
}

fn get_packed_bit_list(
    inst: &PackedInstruction,
    circuit_data: &CircuitData,
) -> Vec<CircuitInstructionArgPack> {
    let mut result: Vec<CircuitInstructionArgPack> = Vec::new();
    for qubit in circuit_data.get_qargs(inst.qubits).iter() {
        result.push(CircuitInstructionArgPack {
            bit_type: ('q' as u8),
            index: (qubit.index() as u32),
        });
    }
    for clbit in circuit_data.get_cargs(inst.clbits).iter() {
        result.push(CircuitInstructionArgPack {
            bit_type: ('c' as u8),
            index: (clbit.index() as u32),
        });
    }
    result
}

fn gate_class_name(py: Python, inst: &PackedInstruction) -> PyResult<Bytes> {
    let name = match inst.op.view() {
        // getting __name__ for standard gates and instructions should
        // eventually be replaced with a Rust-side mapping
        OperationRef::StandardGate(gate) => gate
            .get_gate_class(py)?
            .bind(py)
            .getattr(intern!(py, "__name__"))?
            .extract::<String>(),
        OperationRef::StandardInstruction(inst) => match inst {
            StandardInstruction::Measure => {
                MEASURE.get_bound(py).getattr(intern!(py, "__name__"))?
            }
            StandardInstruction::Delay(_) => {
                DELAY.get_bound(py).getattr(intern!(py, "__name__"))?
            }
            StandardInstruction::Barrier(_) => {
                BARRIER.get_bound(py).getattr(intern!(py, "__name__"))?
            }
            StandardInstruction::Reset => RESET.get_bound(py).getattr(intern!(py, "__name__"))?,
        }
        .extract::<String>(),
        OperationRef::Gate(pygate) => pygate
            .gate
            .bind(py)
            .getattr(intern!(py, "__class__"))?
            .getattr(intern!(py, "__name__"))?
            .extract::<String>(),
        OperationRef::Instruction(pyinst) => pyinst
            .instruction
            .bind(py)
            .getattr(intern!(py, "__class__"))?
            .getattr(intern!(py, "__name__"))?
            .extract::<String>(),
        _ => Ok(String::from("unknown")),
    }?;
    Ok(name.into_bytes())
}

fn gate_label(inst: &PackedInstruction) -> Bytes {
    match inst.label() {
        Some(label) => label.as_bytes().to_vec(),
        None => Vec::new(),
    }
}

fn get_num_ctrl_qubits(py: Python, inst: &PackedInstruction) -> PyResult<u32> {
    match inst.op.view() {
        OperationRef::StandardGate(gate) => Ok(gate.num_ctrl_qubits()),
        OperationRef::Gate(py_gate) => py_gate
            .gate
            .getattr(py, "num_ctrl_qubits")?
            .extract::<u32>(py),
        OperationRef::Instruction(py_inst) => py_inst
            .instruction
            .getattr(py, "num_ctrl_qubits")?
            .extract::<u32>(py),
        _ => Ok(0),
    }
}

fn get_ctrl_state(py: Python, inst: &PackedInstruction, num_ctrl_qubits: u32) -> PyResult<u32> {
    match inst.op.view() {
        OperationRef::Gate(py_gate) => py_gate.gate.getattr(py, "ctrl_state")?.extract::<u32>(py),
        OperationRef::Instruction(py_inst) => py_inst
            .instruction
            .getattr(py, "ctrl_state")?
            .extract::<u32>(py),
        _ => Ok(2u32.pow(num_ctrl_qubits) - 1),
    }
}

fn get_instruction_params(py: Python, instruction: &PackedInstruction) -> Vec<PackedParam> {
    instruction
        .params_view()
        .iter()
        .map(|x| pack_param(py, &x))
        .collect()
}

fn dump_register(
    py: Python,
    register: Bound<PyAny>,
    circuit_data: &CircuitData,
) -> PyResult<Bytes> {
    if register.is_instance_of::<PyClassicalRegister>() {
        Ok(register.getattr("name")?.extract::<String>()?.into_bytes())
    } else if register.is_instance_of::<PyClbit>() {
        let key = &register.extract::<ShareableClbit>()?;
        let name = circuit_data
            .get_clbit_indices(py)
            .bind(py)
            .get_item(key)?
            .ok_or(PyErr::new::<PyValueError, _>("Clbit not found"))?
            .getattr("index")?
            .str()?;
        let mut bytes: Bytes = Vec::with_capacity(name.len()? + 1);
        bytes.push(0u8);
        bytes.extend_from_slice(name.extract::<String>()?.as_bytes());
        Ok(bytes)
    } else {
        Ok(Vec::new())
    }
}

pub mod condition_types {
    pub const NONE: u8 = 0;
    pub const TWO_TUPLE: u8 = 1;
    pub const EXPRESSION: u8 = 2;
}

fn get_condition_data_from_inst(
    py: Python,
    inst: &Py<PyAny>,
    circuit_data: &CircuitData,
) -> PyResult<(u8, u16, i64, Bytes)> {
    let condition = inst.bind(py).getattr("_condition")?;
    if condition.is_instance_of::<PyTuple>() {
        let condition_type = condition_types::TWO_TUPLE;
        let condition_value = condition
            .downcast::<PyTuple>()?
            .get_item(1)?
            .extract::<i64>()?;
        let condition_register = dump_register(
            py,
            condition.downcast::<PyTuple>()?.get_item(0)?,
            circuit_data,
        )?;
        return Ok((
            condition_type,
            condition_register.len() as u16,
            condition_value,
            condition_register,
        ));
    } else {
        // TODO: handle expressions
        return Ok((0, 0, 0, Vec::new()));
    }
}

fn get_condition_data(
    py: Python,
    inst: &PackedInstruction,
    circuit_data: &CircuitData,
) -> (u8, u16, i64, Bytes) {
    //getattr(instruction.operation, "_condition"
    let default_return_value = (condition_types::NONE, 0, 0, Vec::new());
    match inst.op.view() {
        OperationRef::Instruction(py_inst) => {
            get_condition_data_from_inst(py, &py_inst.instruction, circuit_data)
                .unwrap_or(default_return_value)
        }
        _ => default_return_value,
    }
}

fn pack_instruction(
    py: Python,
    instruction: &PackedInstruction,
    circuit_data: &CircuitData,
) -> CircuitInstructionV2Pack {
    let class_name = gate_class_name(py, instruction).unwrap();
    let label_raw = gate_label(instruction);
    let num_ctrl_qubits = get_num_ctrl_qubits(py, instruction).unwrap_or(0);
    let ctrl_state = get_ctrl_state(py, instruction, num_ctrl_qubits).unwrap_or(0);
    let instruction_params: Vec<PackedParam> = get_instruction_params(py, instruction);
    let bit_data = get_packed_bit_list(instruction, circuit_data);
    let (condition_type_value, condition_register_length, condition_value, condition_raw) =
        get_condition_data(py, instruction, circuit_data);
    CircuitInstructionV2Pack {
        name_size: class_name.len() as u16,
        label_size: label_raw.len() as u16,
        num_parameters: instruction_params.len() as u16,
        num_qargs: instruction.op.num_qubits(),
        num_cargs: instruction.op.num_clbits(),
        conditional_key: condition_type_value,
        condition_register_size: condition_register_length,
        condition_value: condition_value,
        num_ctrl_qubits: num_ctrl_qubits,
        ctrl_state: ctrl_state,
        gate_class_name: class_name,
        label_raw: label_raw,
        condition_raw: condition_raw,
        bit_data: bit_data,
        params: instruction_params,
    }
}

pub fn pack_instructions(
    py: Python,
    circuit_data: CircuitData,
) -> PyResult<Vec<CircuitInstructionV2Pack>> {
    Ok(circuit_data
        .data()
        .iter()
        .map(|instruction| pack_instruction(py, instruction, &circuit_data))
        .collect())
}

fn serialize_metadata(
    py: Python,
    metadata: &Bound<PyAny>,
    metadata_serializer: &Bound<PyAny>,
) -> PyResult<Bytes> {
    let json = py.import("json")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("separators", PyTuple::new(py, [",", ":"])?)?;
    kwargs.set_item("cls", metadata_serializer)?;
    Ok(json
        .call_method("dumps", (metadata,), Some(&kwargs))?
        .extract::<String>()?
        .into_bytes())
}

fn pack_register(
    register: &Bound<PyAny>,
    bitmap: &Bound<PyDict>,
    is_in_circuit: bool,
) -> PyResult<RegisterV4Pack> {
    let reg_name = register.getattr("name")?.extract::<String>()?.into_bytes();
    let reg_type = register
        .getattr("prefix")?
        .extract::<String>()?
        .into_bytes()[0];
    let mut standalone = true;
    let mut bit_indices: Vec<i64> = Vec::new();
    for (index, bit) in PyIterator::from_object(register)?.enumerate() {
        let bit_val = bit?;
        if !(register
            .rich_compare(bit_val.getattr("_register")?, pyo3::basic::CompareOp::Eq)?
            .is_truthy()?)
        {
            standalone = false;
        }
        if bit_val.getattr("_index")?.extract::<usize>()? != index {
            standalone = false;
        }

        if let Some(index) = bitmap.get_item(bit_val)? {
            bit_indices.push(index.extract::<i64>()?);
        } else {
            bit_indices.push(-1);
        }
    }
    let packed_reg = RegisterV4Pack {
        register_type: reg_type,
        standalone: standalone as u8,
        size: register.getattr("size")?.extract::<u32>()?,
        name_size: reg_name.len() as u16,
        in_circuit: is_in_circuit as u8,
        name: reg_name,
        bit_indices: bit_indices,
    };
    Ok(packed_reg)
}

fn serialize_register(
    py: Python,
    register: &Bound<PyAny>,
    bitmap: &Bound<PyDict>,
    is_in_circuit: bool,
) -> PyResult<Bytes> {
    let mut buffer = Cursor::new(Vec::new());
    let packed_register = pack_register(register, bitmap, is_in_circuit)?;
    packed_register.write(&mut buffer).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("BinRW write failed: {e}"))
    })?;
    Ok(buffer.into_inner())
}

fn serialize_registers(
    py: Python,
    in_circ_regs: &Bound<PyAny>,
    bits: &Bound<PyAny>,
) -> PyResult<(u32, Bytes)> {
    let bitmap = PyDict::new(py);
    let out_circ_regs = PyList::new(py, Vec::<PyObject>::new())?;

    bits.downcast::<PyList>()?
        .iter()
        .enumerate()
        .try_for_each(|(index, bit)| -> PyResult<()> {
            bitmap.set_item(&bit, index)?;
            if let Ok(register) = bit.getattr("_register") {
                if !(in_circ_regs.contains(&register).unwrap_or(false))
                    && !(out_circ_regs.contains(&register).unwrap_or(true))
                {
                    out_circ_regs.append(register)?;
                }
            }
            Ok(())
        })?;
    let mut result: Bytes = Vec::new();
    in_circ_regs
        .downcast::<PyList>()?
        .iter()
        .try_for_each(|register| -> PyResult<()> {
            result.extend(serialize_register(py, &register, &bitmap, true)?);
            Ok(())
        })?;

    out_circ_regs
        .iter()
        .try_for_each(|register| -> PyResult<()> {
            result.extend(serialize_register(py, &register, &bitmap, false)?);
            Ok(())
        })?;

    let length = in_circ_regs.call_method0("__len__")?.extract::<u32>()?
        + out_circ_regs.call_method0("__len__")?.extract::<u32>()?;
    Ok((length, result))
}

fn pack_circuit_header(
    py: Python,
    circuit: &Bound<PyAny>,
    metadata_serializer: &Bound<PyAny>,
) -> PyResult<HeaderData> {
    let circuit_name = circuit.getattr("name")?.extract::<String>()?.into_bytes();
    let metadata_raw = serialize_metadata(py, &circuit.getattr("metadata")?, metadata_serializer)?;
    let (global_phase_type, global_phase_data) =
        dumps_value(py, &circuit.getattr("global_phase")?)?;
    let (num_qregs, qregs_raw) =
        serialize_registers(py, &circuit.getattr("qregs")?, &circuit.getattr("qubits")?)?;
    let (num_cregs, cregs_raw) =
        serialize_registers(py, &circuit.getattr("cregs")?, &circuit.getattr("clbits")?)?;
    let header = CircuitHeaderV12Pack {
        name_size: circuit_name.len() as u16,
        global_phase_type: global_phase_type,
        global_phase_size: global_phase_data.len() as u16,
        num_qubits: circuit.getattr("num_qubits")?.extract::<u32>()?,
        num_clbits: circuit.getattr("num_clbits")?.extract::<u32>()?,
        metadata_size: metadata_raw.len() as u64,
        num_registers: num_qregs + num_cregs,
        num_instructions: circuit.call_method0("__len__")?.extract::<u64>()?,
        num_vars: circuit.getattr("num_identifiers")?.extract::<u32>()?,
    };

    Ok(HeaderData {
        header: header,
        circuit_name: circuit_name,
        global_phase_data: global_phase_data,
        metadata: metadata_raw,
        qregs: qregs_raw,
        cregs: cregs_raw,
    })
}

fn pack_layout(circuit: &Bound<PyAny>) -> PyResult<LayoutV2Pack> {
    if circuit.getattr("layout")?.is_none() {
        Ok(LayoutV2Pack {
            exists: 0,
            initial_layout_size: -1,
            input_mapping_size: -1,
            final_layout_size: -1,
            extra_registers_length: 0,
            input_qubit_count: 0,
        })
    } else {
        // TODO: placeholder; should handle custom layout here
        Ok(LayoutV2Pack {
            exists: 0,
            initial_layout_size: -1,
            input_mapping_size: -1,
            final_layout_size: -1,
            extra_registers_length: 0,
            input_qubit_count: 0,
        })
    }
}

fn pack_circuit(
    py: Python,
    circuit: &Bound<PyAny>,
    metadata_serializer: &Bound<PyAny>,
    _use_symengine: bool,
    _version: u32,
) -> PyResult<QPYFormatV13> {
    let circuit_data = circuit.getattr("_data")?.extract::<CircuitData>()?;
    let header_raw = pack_circuit_header(py, circuit, metadata_serializer)?;
    // TODO: still need to implement write_standalone_vars
    // Pulse has been removed in Qiskit 2.0. As long as we keep QPY at version 13,
    // we need to write an empty calibrations header since read_circuit expects it
    let calibration_header_value: u16 = 0;
    Ok(QPYFormatV13 {
        header: header_raw,
        custom_instructions: CustomCircuitInstructionsPack {
            custom_operations_length: (0),
        },
        instructions: pack_instructions(py, circuit_data)?,
        calibration_header: calibration_header_value.to_be_bytes().to_vec(),
        layout: pack_layout(circuit)?,
    })
}

fn serialize_circuit(
    py: Python,
    circuit: &Bound<PyAny>,
    metadata_serializer: &Bound<PyAny>,
    use_symengine: bool,
    version: u32,
) -> PyResult<Bytes> {
    let mut buffer = Cursor::new(Vec::new());
    let packed_circuit = pack_circuit(py, circuit, metadata_serializer, use_symengine, version)?;
    packed_circuit.write(&mut buffer).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("BinRW write failed: {e}"))
    })?;
    Ok(buffer.into_inner())
}

#[pyfunction]
#[pyo3(signature = (file_obj, circuit, metadata_serializer, use_symengine, version))]
pub fn py_write_circuit(
    py: Python,
    file_obj: &Bound<PyAny>,
    circuit: &Bound<PyAny>,
    metadata_serializer: &Bound<PyAny>,
    use_symengine: bool,
    version: u32,
) -> PyResult<usize> {
    let serialized_circuit =
        serialize_circuit(py, circuit, metadata_serializer, use_symengine, version)?;
    file_obj.call_method1(
        "write",
        (pyo3::types::PyBytes::new(py, &serialized_circuit),),
    )?;
    Ok(serialized_circuit.len())
}
