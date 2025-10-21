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

use hashbrown::{HashMap, HashSet};
use indexmap::IndexSet;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3::types::{PyAny, PyBytes, PyDict, PyList, PyString, PyTuple, PyType};
use pyo3::IntoPyObjectExt;

use qiskit_circuit::bit::{
    ClassicalRegister, PyClbit, PyQubit, QuantumRegister, Register, ShareableClbit,
    ShareableQubit,
};
use qiskit_circuit::circuit_data::{CircuitData, CircuitStretchType, CircuitVarType};
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::classical;
use qiskit_circuit::imports;
use qiskit_circuit::operations::{Operation, OperationRef, Param, StandardInstruction};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::{Clbit, Qubit};
use qiskit_circuit::converters::QuantumCircuitData;

use smallvec::SmallVec;

use crate::annotations::AnnotationHandler;
use crate::bytes::Bytes;
use crate::consts::standard_gate_from_gate_class_name;
use crate::formats;
use crate::params::unpack_param;
use crate::value::{
    circuit_instruction_types, deserialize, deserialize_with_args,
    expression_var_declaration, pack_standalone_var, pack_stretch, 
    serialize, tags, unpack_generic_data, DumpedValue, ExpressionType, QPYReadData, QPYWriteData, VarOrStretch
};
use crate::py_methods::{serialize_metadata, pack_py_registers, pack_py_instruction, pack_py_custom_instruction, get_condition_data_from_inst, get_python_gate_class};

use crate::UnsupportedFeatureForVersion;



// This is a helper struct, designed to pass data within methods
// It is not meant to be serialized, so it's not in formats.rs
#[derive(Debug)]
struct CustomCircuitInstructionData {
    gate_type: u8,
    num_qubits: u32,
    num_clbits: u32,
    definition_circuit: Option<Py<PyAny>>,
    base_gate_raw: Bytes,
}

// Standard char representation of register types: 'q' qreg, 'c' for creg
pub mod register_types {
    pub const QREG: u8 = b'q';
    pub const CREG: u8 = b'c';
}

pub mod bit_types {
    pub const QUBIT: u8 = b'q';
    pub const CLBIT: u8 = b'c';
}

pub fn get_packed_bit_list(
    inst: &PackedInstruction,
    circuit_data: &CircuitData,
) -> Vec<formats::CircuitInstructionArgPack> {
    let mut result: Vec<formats::CircuitInstructionArgPack> = Vec::new();
    for qubit in circuit_data.get_qargs(inst.qubits).iter() {
        result.push(formats::CircuitInstructionArgPack {
            bit_type: bit_types::QUBIT,
            index: (qubit.index() as u32),
        });
    }
    for clbit in circuit_data.get_cargs(inst.clbits).iter() {
        result.push(formats::CircuitInstructionArgPack {
            bit_type: bit_types::CLBIT,
            index: (clbit.index() as u32),
        });
    }
    result
}



pub fn load_register(
    py: Python,
    data_bytes: Bytes,
    circuit_data: &CircuitData,
) -> PyResult<Py<PyAny>> {
    // If register name prefixed with null character it's a clbit index for single bit condition.
    if data_bytes.is_empty() {
        return Err(PyValueError::new_err(
            "Failed to load register - name missing",
        ));
    }
    if data_bytes[0] == 0u8 {
        let index: Clbit = Clbit(std::str::from_utf8(&data_bytes[1..])?.parse()?);
        match circuit_data.clbits().get(index) {
            Some(shareable_clbit) => {
                Ok(shareable_clbit.into_pyobject(py)?.as_any().clone().unbind())
            }
            None => Err(PyValueError::new_err(format!(
                "Could not find clbit {:?}",
                index
            ))),
        }
    } else {
        let name = std::str::from_utf8(&data_bytes)?;
        let mut register = None;
        for creg in circuit_data.cregs() {
            if creg.name() == name {
                register = Some(creg);
            }
        }
        match register {
            Some(register) => Ok(register.into_py_any(py)?),
            None => Err(PyValueError::new_err(format!(
                "Could not find classical register {:?}",
                name
            ))),
        }
    }
}



pub fn get_condition_data(
    op: &PackedOperation,
    circuit_data: &CircuitData,
    qpy_data: &QPYWriteData,
) -> PyResult<formats::ConditionPack> {
    match op.view() {
        OperationRef::Instruction(py_inst) => {
            get_condition_data_from_inst(&py_inst.instruction, circuit_data, qpy_data)
        }
        // we assume only PyInstructions have condition data at this stage
        _ => Ok(formats::ConditionPack {
            key: formats::condition_types::NONE,
            register_size: 0u16,
            value: 0i64,
            data: formats::ConditionData::None,
        }),
    }
}



fn deserialize_standard_instruction(
    instruction: &formats::CircuitInstructionV2Pack,
) -> Option<StandardInstruction> {
    match instruction.gate_class_name.as_str() {
        "Barrier" => Some(StandardInstruction::Barrier(instruction.num_qargs)),
        "Measure" => Some(StandardInstruction::Measure),
        "Reset" => Some(StandardInstruction::Reset),
        // TODO it seems currently delays are not treated correctly, and the unit is not stored
        // even in Python QPY. We need to fix the writer to correctly store the delay unit as second parameter
        "Delay" => {
            let unit = if !instruction.params.is_empty() {
                if instruction.params.len() >= 2 {
                    // both duration and unit params; extract the unit param
                    // TODO: for now returning default value; this should be fixed as part of the fix mentioned above
                    qiskit_circuit::operations::DelayUnit::DT
                } else {
                    // only duration; check whether it's an expression
                    if [
                        tags::EXPRESSION,
                        tags::PARAMETER_EXPRESSION,
                        tags::PARAMETER,
                        tags::PARAMETER_VECTOR,
                    ]
                    .contains(&instruction.params[0].type_key)
                    {
                        qiskit_circuit::operations::DelayUnit::EXPR
                    } else {
                        qiskit_circuit::operations::DelayUnit::DT
                    }
                }
            } else {
                // default case
                qiskit_circuit::operations::DelayUnit::DT
            };
            Some(StandardInstruction::Delay(unit))
        }
        _ => None,
    }
}

fn unpack_custom_instruction(
    py: Python,
    instruction: &formats::CircuitInstructionV2Pack,
    py_params: &Vec<Bound<PyAny>>,
    label: Option<&String>,
    custom_instruction: &CustomCircuitInstructionData,
    qpy_data: &mut QPYReadData,
    custom_instructions_map: &HashMap<String, CustomCircuitInstructionData>,
) -> PyResult<PackedOperation> {
    // TODO: should have "if version >= 11" check here once we introduce versioning to rust
    let mut gate_class_name = match instruction.gate_class_name.rfind('_') {
        Some(pos) => &instruction.gate_class_name[..pos],
        None => &instruction.gate_class_name,
    };
    let inst_obj = match custom_instruction.gate_type {
        circuit_instruction_types::GATE => {
            let gate_object = imports::GATE.get_bound(py).call1((
                &gate_class_name,
                custom_instruction.num_qubits,
                py_params,
            ))?;
            if let Some(definition) = &custom_instruction.definition_circuit {
                gate_object.setattr("definition", definition)?;
            }
            if let Some(label_string) = label {
                gate_object.setattr("label", label_string.as_str())?;
            }
            gate_object.unbind()
        }
        circuit_instruction_types::INSTRUCTION => {
            let instruction_object = imports::INSTRUCTION.get_bound(py).call1((
                &gate_class_name,
                custom_instruction.num_qubits,
                custom_instruction.num_clbits,
                py_params,
            ))?;
            if let Some(definition) = &custom_instruction.definition_circuit {
                instruction_object.setattr("definition", definition)?;
            }
            if let Some(label_string) = label {
                instruction_object.setattr("label", label_string.as_str())?;
            }
            instruction_object.unbind()
        }
        circuit_instruction_types::PAULI_EVOL_GATE => {
            if let Some(definition) = &custom_instruction.definition_circuit {
                let inst = definition.clone();
                if let Some(label_string) = label {
                    inst.setattr(py, "label", label_string.as_str())?;
                }
                inst
            } else {
                return Err(PyValueError::new_err(
                    "Pauli Evolution Gate missing definition",
                ));
            }
        }
        circuit_instruction_types::CONTROLLED_GATE => {
            let packed_base_gate = deserialize_with_args::<
                formats::CircuitInstructionV2Pack,
                (bool,),
            >(&custom_instruction.base_gate_raw, (false,))?
            .0;
            let base_gate =
                unpack_instruction(py, &packed_base_gate, custom_instructions_map, qpy_data)?;
            // If open controls, we need to discard the control suffix when setting the name.
            if instruction.ctrl_state < (1u32 << instruction.num_ctrl_qubits) - 1 {
                gate_class_name = match gate_class_name.rfind('_') {
                    Some(pos) => &gate_class_name[..pos],
                    None => gate_class_name,
                };
            }
            let kwargs = PyDict::new(py);
            kwargs.set_item(intern!(py, "num_ctrl_qubits"), instruction.num_ctrl_qubits)?;
            kwargs.set_item(intern!(py, "ctrl_state"), instruction.ctrl_state)?;
            kwargs.set_item(intern!(py, "base_gate"), base_gate.unpack_py_op(py)?)?;

            let controlled_gate_object = imports::CONTROLLED_GATE.get_bound(py).call(
                (&gate_class_name, custom_instruction.num_qubits, py_params),
                Some(&kwargs),
            )?;
            if let Some(definition) = &custom_instruction.definition_circuit {
                controlled_gate_object.setattr("definition", definition)?;
            }
            controlled_gate_object.unbind()
        }
        circuit_instruction_types::ANNOTATED_OPERATION => {
            let packed_base_gate = deserialize_with_args::<
                formats::CircuitInstructionV2Pack,
                (bool,),
            >(&custom_instruction.base_gate_raw, (false,))?
            .0;
            let base_gate =
                unpack_instruction(py, &packed_base_gate, custom_instructions_map, qpy_data)?;
            let kwargs = PyDict::new(py);
            kwargs.set_item(intern!(py, "base_op"), base_gate.unpack_py_op(py)?)?;
            kwargs.set_item(intern!(py, "modifiers"), py_params)?;
            imports::ANNOTATED_OPERATION
                .get_bound(py)
                .call((), Some(&kwargs))?
                .unbind()
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "Custom gate type {:?} not handled",
                custom_instruction.gate_type
            )))
        }
    };
    Ok(inst_obj.extract::<OperationFromPython>(py)?.operation)
}

fn unpack_condition(
    py: Python,
    condition: &formats::ConditionPack,
    qpy_data: &mut QPYReadData,
) -> PyResult<Option<Py<PyAny>>> {
    match &condition.data {
        formats::ConditionData::Expression(exp_data_pack) => {
            Ok(Some(unpack_generic_data(py, exp_data_pack, qpy_data)?))
        }
        formats::ConditionData::Register(register_data) => {
            let register = load_register(py, register_data.clone(), qpy_data.circuit_data)?;
            let condition_value = condition.value.into_py_any(py)?;
            let tuple = PyTuple::new(py, &[register, condition_value])?;
            Ok(Some(tuple.into_any().unbind()))
        }
        formats::ConditionData::None => Ok(None),
    }
}

fn unpack_instruction(
    py: Python,
    instruction: &formats::CircuitInstructionV2Pack,
    custom_instructions: &HashMap<String, CustomCircuitInstructionData>,
    qpy_data: &mut QPYReadData,
) -> PyResult<PackedInstruction> {
    let name = instruction.gate_class_name.clone();
    let label = (!instruction.label.is_empty()).then(|| Box::new(instruction.label.clone()));
    let condition = unpack_condition(py, &instruction.condition, qpy_data)?;
    let mut qubit_indices = Vec::new();
    let mut clbit_indices = Vec::new();
    for arg in &instruction.bit_data {
        match arg.bit_type {
            bit_types::QUBIT => qubit_indices.push(Qubit(arg.index)),
            bit_types::CLBIT => clbit_indices.push(Clbit(arg.index)),
            _ => return Err(PyValueError::new_err("Unrecognized bit type")),
        };
    }
    let qubits = qpy_data.circuit_data.add_qargs(&qubit_indices);
    let clbits = qpy_data.circuit_data.add_cargs(&clbit_indices);
    let mut inst_params: Vec<Param> = instruction
        .params
        .iter()
        .map(|packed_param| unpack_param(py, packed_param, qpy_data))
        .collect::<PyResult<_>>()?;
    let mut py_params: Vec<Bound<'_, PyAny>> = inst_params
        .iter()
        .map(|param: &Param| param.into_pyobject(py))
        .collect::<PyResult<_>>()?;
    // TODO: currently we check has_nonstandard_control only for standard gates. should we check for more?
    let has_nonstandard_control = instruction.num_ctrl_qubits > 0
        && (instruction.ctrl_state != (1 << instruction.num_ctrl_qubits) - 1);
    let standard_gate = (!has_nonstandard_control)
        .then(|| standard_gate_from_gate_class_name(name.as_str()))
        .flatten();
    let op = if let Some(custom_instruction) = custom_instructions.get(&name) {
        unpack_custom_instruction(
            py,
            instruction,
            &py_params,
            label.as_deref(),
            custom_instruction,
            qpy_data,
            custom_instructions,
        )?
    } else if let Some(gate) = standard_gate {
        PackedOperation::from_standard_gate(gate)
    } else if let Some(std_instruction) = deserialize_standard_instruction(instruction) {
        PackedOperation::from_standard_instruction(std_instruction)
    } else {
        let gate_class = get_python_gate_class(py, &instruction.gate_class_name)?;
        let mut gate_object = match name.as_str() {
            "IfElseOp" | "WhileLoopOp" => {
                let py_condition = match condition {
                    Some(py_obj) => py_obj,
                    None => py.None(),
                };
                let mut args = vec![py_condition];
                for param in py_params {
                    args.push(param.unbind());
                }
                gate_class.call1(PyTuple::new(py, args)?)?
            }
            "BoxOp" => {
                if py_params.len() < 2 {
                    return Err(PyValueError::new_err(format!(
                        "BoxOp instruction has only {:?} params; should have at least 2",
                        py_params.len()
                    )));
                }
                let unit = py_params.pop().unwrap();
                inst_params.pop();
                let duration = py_params.pop().unwrap();
                inst_params.pop();
                let annotations = match &instruction.annotations {
                    Some(annotation_pack) => annotation_pack
                        .annotations
                        .iter()
                        .map(|annotation| {
                            qpy_data.annotation_handler.load(
                                py,
                                annotation.namespace_index,
                                annotation.payload.clone(),
                            )
                        })
                        .collect::<PyResult<_>>()?,
                    None => Vec::new(),
                }
                .into_pyarray(py)
                .into_any();
                let kwargs = [
                    ("unit", unit),
                    ("duration", duration),
                    ("annotations", annotations),
                ]
                .into_py_dict(py)?;
                let args = PyTuple::new(py, &py_params)?;
                gate_class.call(args, Some(&kwargs))?
            }
            "BreakLoopOp" | "ContinueLoopOp" => {
                let args = (qubit_indices.len(), clbit_indices.len());
                gate_class.call1(args)?
            }
            "Initialize" | "StatePreparation" => {
                if py_params[0].is_instance_of::<PyString>() {
                    // the params are the labels of the initial state
                    let label = py_params
                        .iter()
                        .map(|param| param.extract())
                        .collect::<PyResult<Vec<String>>>()?
                        .join("");
                    gate_class.call1((label,))?
                } else if py_params.len() == 1 {
                    // the params is the integer indicating which qubits to initialize
                    let real_param: f64 = py_params[0].getattr("real")?.extract()?;
                    let qubits_to_initialize = real_param as u32;
                    gate_class.call1((qubits_to_initialize, instruction.num_qargs))?
                } else {
                    // the params represent a list of complex amplitudes
                    gate_class.call1((py_params,))?
                }
            }
            "QFTGate" => {
                let mut args: Vec<Py<PyAny>> = vec![instruction.num_qargs.into_py_any(py)?];
                for param in py_params {
                    args.push(param.unbind());
                }
                gate_class.call1(PyTuple::new(py, args)?)?
            }
            "UCRXGate" | "UCRYGate" | "UCRZGate" | "DiagonalGate" => {
                gate_class.call1((py_params,))?
            }
            "MCPhaseGate" | "MCU1Gate" | "MCXGrayCode" | "MCXGate" | "MCXRecursive"
            | "MCXVChain" => {
                let mut args: Vec<Py<PyAny>> = Vec::new();
                for param in py_params {
                    args.push(param.unbind());
                }
                args.push(instruction.num_ctrl_qubits.into_py_any(py)?);
                gate_class.call1(PyTuple::new(py, args)?)?
                // if condition:
                // body = QuantumCircuit(qargs, cargs)
                // body.append(gate, qargs, cargs)
                // gate = IfElseOp(condition, body)
            }
            _ => {
                let args = PyTuple::new(py, &py_params)?;
                gate_class.call1(args)?
                // if condition:
                // body = QuantumCircuit(qargs, cargs)
                // body.append(gate, qargs, cargs)
                // gate = IfElseOp(condition, body)
            }
        };
        if let Some(label_text) = &label {
            if !gate_object.hasattr("label")? || gate_object.getattr("label")?.is_none() {
                gate_object.setattr("label", label_text.as_str())?;
            }
        }
        if gate_class
            .downcast_into::<PyType>()?
            .is_subclass(imports::CONTROLLED_GATE.get_bound(py))?
            && (gate_object.getattr("num_ctrl_qubits")?.extract::<u32>()?
                != instruction.num_ctrl_qubits
                || gate_object.getattr("ctrl_state")?.extract::<u32>()? != instruction.ctrl_state)
        {
            gate_object = gate_object.call_method0("to_mutable")?;
            gate_object.setattr("num_ctrl_qubits", instruction.num_ctrl_qubits)?;
            gate_object.setattr("ctrl_state", instruction.ctrl_state)?;
        }

        let op_parts = gate_object.extract::<OperationFromPython>()?;
        op_parts.operation
    };
    let params =
        (!inst_params.is_empty()).then(|| Box::new(SmallVec::<[Param; 3]>::from_vec(inst_params)));
    Ok(PackedInstruction {
        op,
        qubits,
        clbits,
        params,
        label,
        #[cfg(feature = "cache_pygates")]
        py_op: std::sync::OnceLock::new(),
    })
}

pub fn pack_instructions(
    py: Python,
    circuit_data: &CircuitData,
    qpy_data: &mut QPYWriteData,
) -> PyResult<(
    Vec<formats::CircuitInstructionV2Pack>,
    HashMap<String, PackedOperation>,
)> {
    let mut custom_operations = HashMap::new();
    let mut custom_new_operations = Vec::new();
    Ok((
        circuit_data
            .data()
            .iter()
            .map(|instruction| {
                pack_py_instruction(
                    py,
                    instruction,
                    circuit_data,
                    &mut custom_operations,
                    &mut custom_new_operations,
                    qpy_data,
                )
            })
            .collect::<PyResult<_>>()?,
        custom_operations,
    ))
}



fn deserialize_metadata(
    py: Python,
    metadata_bytes: &Bytes,
    metadata_deserializer: &Bound<PyAny>,
) -> PyResult<PyObject> {
    let json = py.import("json")?;
    let kwargs: Bound<'_, PyDict> = PyDict::new(py);
    kwargs.set_item("cls", metadata_deserializer)?;
    let metadata_string = PyString::new(py, metadata_bytes.try_into()?);
    Ok(json
        .call_method("loads", (metadata_string,), Some(&kwargs))?
        .unbind())
}


// packs the quantum registers in the circuit. we pack:
// 1) registers appearing explicitly in the circuit register list (identified using in_circ_lookup)
// 2) registers appearing implicitly for bits (in python: as their "_register" data field)
fn pack_quantum_registers(circuit_data: &CircuitData) -> Vec<formats::RegisterV4Pack> {
    let in_circ_lookup: HashSet<QuantumRegister> = circuit_data.qregs().iter().cloned().collect();
    let mut registers_to_pack: IndexSet<QuantumRegister> = circuit_data.qregs().iter().cloned().collect();
    registers_to_pack.extend(
        circuit_data
        .qubits()
        .objects()
        .iter()
        .filter_map(|qubit| circuit_data.qubit_indices().get(qubit)) // only qubits with register data
        .flat_map(|qreg_locations| qreg_locations.registers().iter().map(|(qreg, _)| qreg.clone()))
    );
    registers_to_pack
    .iter()
    .map(|qreg| {
        let bit_indices = qreg
        .bits()
        .map(|qubit| circuit_data.qubit_index(qubit).map(|index| index as i64).unwrap_or(-1))
        .collect();
        formats::RegisterV4Pack {
            register_type: 'q' as u8,
            standalone: qreg.is_owning() as u8,
            in_circuit: in_circ_lookup.contains(qreg) as u8,
            name: qreg.name().to_string(),
            bit_indices
        }
    })
    .collect()
}

fn pack_classical_registers(circuit_data: &CircuitData) -> Vec<formats::RegisterV4Pack> {
    let in_circ_lookup: HashSet<ClassicalRegister> = circuit_data.cregs().iter().cloned().collect();
    let mut registers_to_pack: IndexSet<ClassicalRegister> = circuit_data.cregs().iter().cloned().collect();
    registers_to_pack.extend(
        circuit_data
        .clbits()
        .objects()
        .iter()
        .filter_map(|clbit| circuit_data.clbit_indices().get(clbit)) // only qubits with register data
        .flat_map(|creg_locations| creg_locations.registers().iter().map(|(creg, _)| creg.clone()))
    );
    registers_to_pack
    .iter()
    .map(|creg| {
        let bit_indices = creg
        .bits()
        .map(|clbit| circuit_data.clbit_index(clbit).map(|index| index as i64).unwrap_or(-1))
        .collect();
        formats::RegisterV4Pack {
            register_type: 'c' as u8,
            standalone: creg.is_owning() as u8,
            in_circuit: in_circ_lookup.contains(creg) as u8,
            name: creg.name().to_string(),
            bit_indices
        }
    })
    .collect()
}

fn pack_circuit_header(
    circuit: &QuantumCircuitData,
    metadata_serializer: &Bound<PyAny>,
    qpy_data: &QPYWriteData,
) -> PyResult<formats::CircuitHeaderV12Pack> {
    // let py = circuit.py();
    // let circuit_name = circuit.getattr(intern!(py, "name"))?.extract::<String>()?;
    let metadata = serialize_metadata(
        circuit.metadata,
        metadata_serializer,
    )?;
    let global_phase_data = DumpedValue::from_param(circuit.data.global_phase(), qpy_data)?;
        // DumpedValue::from(&circuit.getattr(intern!(py, "global_phase"))?, qpy_data)?;
    // let qregs = pack_registers(
    //     &circuit.getattr(intern!(py, "qregs"))?,
    //     circuit
    //         .getattr(intern!(py, "qubits"))?
    //         .downcast::<PyList>()?,
    // )?;
    let qregs = pack_quantum_registers(&circuit.data);
    let cregs = pack_classical_registers(&circuit.data);
    // let cregs = pack_registers(
    //     &circuit.getattr(intern!(py, "cregs"))?,
    //     circuit
    //         .getattr(intern!(py, "clbits"))?
    //         .downcast::<PyList>()?,
    // )?;
    let mut registers = qregs;
    registers.extend(cregs);
    let header = formats::CircuitHeaderV12Pack {
        num_qubits: circuit.data.num_qubits() as u32,
        num_clbits: circuit.data.num_clbits() as u32,
        num_instructions: circuit.data.__len__() as u64,
        num_vars: circuit.data.num_identifiers() as u32,
        circuit_name: circuit.name.unwrap_or_else(|| String::new()),
        global_phase_data,
        metadata,
        registers,
    };

    Ok(header)
}

pub fn pack_layout(circuit: &QuantumCircuitData) -> PyResult<formats::LayoutV2Pack> {
    match circuit.custom_layout {
        None => Ok(formats::LayoutV2Pack {
            exists: 0,
            initial_layout_size: -1,
            input_mapping_size: -1,
            final_layout_size: -1,
            input_qubit_count: 0,
            extra_registers: Vec::new(),
            initial_layout_items: Vec::new(),
            input_mapping_items: Vec::new(),
            final_layout_items: Vec::new(),
        }),
        Some(custom_layout) => pack_custom_layout(&custom_layout, &circuit.data)
    }
}

fn unpack_layout<'py>(
    py: Python<'py>,
    layout: &formats::LayoutV2Pack,
    circuit_data: &CircuitData,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match layout.exists {
        0 => Ok(None),
        _ => Ok(Some(unpack_custom_layout(py, layout, circuit_data)?)),
    }
}

fn pack_custom_layout(layout: &Bound<PyAny>, circuit: &CircuitData) -> PyResult<formats::LayoutV2Pack> {
    let py = layout.py();
    let mut initial_layout_size = -1; // initial_size
    let input_qubit_mapping = PyDict::new(py);
    let initial_layout_array = PyList::empty(py);
    let extra_registers = PyDict::new(py);

    let initial_layout = layout.getattr("initial_layout")?;
    if !initial_layout.is_none() {
        initial_layout_size = initial_layout.call_method0("__len__")?.extract::<i32>()?;
        let layout_mapping = initial_layout.call_method0("get_physical_bits")?;
        for i in 0..circuit.num_qubits() {
            let qubit = layout_mapping.get_item(i)?;
            input_qubit_mapping.set_item(&qubit, i)?;
            let register = qubit.getattr("_register")?;
            let index = qubit.getattr("_index")?;
            if !register.is_none() || !index.is_none() {
                if !circuit.qregs().contains(&register.extract()?) {
                    let extra_register_list = match extra_registers.get_item(&register)? {
                        Some(list) => list,
                        None => {
                            let new_list = PyList::empty(py);
                            extra_registers.set_item(&register, &new_list)?;
                            new_list.into_any()
                        }
                    };
                    extra_register_list.downcast::<PyList>()?.append(qubit)?;
                }
                initial_layout_array.append((index, register))?;
            } else {
                initial_layout_array.append((py.None(), py.None()))?;
            }
        }
    }

    let mut input_mapping_size: i32 = -1; //input_qubit_size
    let mut input_qubit_mapping_array = PyList::new(py, Vec::<Bound<PyAny>>::new())?;
    let layout_input_qubit_mapping = layout.getattr("input_qubit_mapping")?;
    if !layout_input_qubit_mapping.is_none() {
        input_mapping_size = layout_input_qubit_mapping
            .call_method0("__len__")?
            .extract()?;
        input_qubit_mapping_array = PyList::new(
            py,
            std::iter::repeat(py.None())
                .take(input_mapping_size as usize)
                .collect::<Vec<_>>(),
        )?;
        let layout_mapping = initial_layout.call_method0("get_virtual_bits")?;
        for (qubit, index) in layout_input_qubit_mapping.downcast::<PyDict>()? {
            let register = qubit.getattr("_register")?;
            if !register.is_none()
                && !qubit.getattr("_index")?.is_none()
                && !circuit.qregs().contains(&register.extract()?)
            {
                let extra_register_list = match extra_registers.get_item(&register)? {
                    Some(list) => list,
                    None => {
                        let new_list = PyList::empty(py);
                        extra_registers.set_item(&register, &new_list)?;
                        new_list.into_any()
                    }
                };
                extra_register_list.downcast::<PyList>()?.append(&qubit)?;
            }
            input_qubit_mapping_array
                .set_item(index.extract()?, layout_mapping.get_item(&qubit)?)?;
        }
    }

    let mut final_layout_size: i32 = -1;
    let final_layout_array = PyList::empty(py);
    let final_layout = layout.getattr("final_layout")?;
    if !final_layout.is_none() {
        final_layout_size = final_layout.call_method0("__len__")?.extract()?;
        let final_layout_physical = final_layout.call_method0("get_physical_bits")?;
        for i in 0..circuit.num_qubits() {
            // this part is alternative to calling `find_bit` for the python version of the quantum circuit
            let virtual_bit = final_layout_physical.downcast::<PyDict>()?.get_item(i)?.unwrap(); // TODO: handle unwrap failure
            if virtual_bit.is_instance_of::<PyClbit>() {
                match circuit.get_clbit_indices(py).bind(py).get_item(virtual_bit)? {
                    None => (), // TODO: error?
                    Some(bit_data) => final_layout_array.append(bit_data.getattr("index")?)?
                }
            }
            if virtual_bit.is_instance_of::<PyQubit>() {
                match circuit.get_qubit_indices(py).bind(py).get_item(virtual_bit)? {
                    None => (), // TODO: error?
                    Some(bit_data) => final_layout_array.append(bit_data.getattr("index")?)?
                }
            }
        }
    }

    let input_qubit_count: i32 = if layout.getattr("_input_qubit_count")?.is_none() {
        -1
    } else {
        layout.getattr("_input_qubit_count")?.extract()?
    };

    let mut bits = Vec::new();
    for register_bit_list in extra_registers.values() {
        for x in register_bit_list.downcast::<PyList>()? {
            bits.push(x);
        }
    }
    let extra_registers = pack_py_registers(&extra_registers.keys(), &PyList::new(py, bits)?)?;
    let mut initial_layout_items = Vec::with_capacity(initial_layout_size.max(0) as usize);
    for item in initial_layout_array {
        let tuple = item.downcast::<PyTuple>()?;
        let index = tuple.get_item(0)?;
        let register = tuple.get_item(1)?;
        let reg_name_bytes = if !register.is_none() {
            Some(register.getattr("name")?.extract::<String>()?)
        } else {
            None
        };
        let index_value = if index.is_none() {
            -1
        } else {
            index.extract::<i32>()?
        };
        let (register_name, register_name_length) = reg_name_bytes
            .as_ref()
            .map(|name| (name.clone(), name.len() as i32))
            .unwrap_or((String::new(), -1));
        initial_layout_items.push(formats::InitialLayoutItemV2Pack {
            index_value,
            register_name_length,
            register_name,
        });
    }

    let mut input_mapping_items = Vec::with_capacity(input_mapping_size.max(0) as usize);
    for i in &input_qubit_mapping_array {
        input_mapping_items.push(i.extract::<u32>()?);
        // buffer.write_all(&i.extract::<u32>()?.to_be_bytes())?;
    }

    let mut final_layout_items = Vec::with_capacity(final_layout_size.max(0) as usize);
    for i in &final_layout_array {
        final_layout_items.push(i.extract::<u32>()?);
        // buffer.write_all(&i.extract::<u32>()?.to_be_bytes())?;
    }

    Ok(formats::LayoutV2Pack {
        exists: true as u8,
        initial_layout_size,
        input_mapping_size,
        final_layout_size,
        input_qubit_count,
        extra_registers,
        initial_layout_items,
        input_mapping_items,
        final_layout_items,
    })
}

fn unpack_custom_layout<'py>(
    py: Python<'py>,
    layout: &formats::LayoutV2Pack,
    circuit_data: &CircuitData,
) -> PyResult<Bound<'py, PyAny>> {
    let mut initial_layout = py.None();
    let mut input_qubit_mapping = py.None();
    let mut final_layout = py.None();

    let mut extra_register_map = HashMap::new();
    let mut existing_register_map = HashMap::new();
    for packed_register in &layout.extra_registers {
        if packed_register.register_type == bit_types::QUBIT {
            let register = QuantumRegister::new_owning(
                packed_register.name.clone(),
                packed_register.bit_indices.len() as u32,
            );
            extra_register_map.insert(packed_register.name.as_str(), register);
        }
    }
    // add the registers from the circuit, to streamline the search phase
    for qreg in circuit_data.qregs() {
        existing_register_map.insert(qreg.name(), qreg);
    }
    let initial_layout_virtual_bits = PyList::new(py, Vec::<PyObject>::new())?;
    for virtual_bit in &layout.initial_layout_items {
        let qubit = if let Some(register) =
            extra_register_map.get(virtual_bit.register_name.as_str())
        {
            if let Some(qubit) = register.get(virtual_bit.index_value as usize) {
                qubit
            } else {
                ShareableQubit::new_anonymous()
            }
        } else if let Some(register) = existing_register_map.get(virtual_bit.register_name.as_str())
        {
            if let Some(qubit) = register.get(virtual_bit.index_value as usize) {
                qubit
            } else {
                ShareableQubit::new_anonymous()
            }
        } else {
            ShareableQubit::new_anonymous()
        };
        initial_layout_virtual_bits.append(qubit)?;
    }
    if initial_layout_virtual_bits.len() > 0 {
        initial_layout = imports::LAYOUT
            .get_bound(py)
            .call_method1("from_qubit_list", (initial_layout_virtual_bits,))?
            .unbind();
    }

    if layout.input_mapping_size > 0 {
        let input_qubit_mapping_data = PyDict::new(py);
        let physical_bits_object = initial_layout.call_method0(py, "get_physical_bits")?;
        let physical_bits = physical_bits_object.downcast_bound::<PyDict>(py)?;
        for (index, bit) in layout.input_mapping_items.iter().enumerate() {
            let physical_bit =
                physical_bits
                    .get_item(bit)?
                    .ok_or(PyValueError::new_err(format!(
                        "Could not get physical bit for bit {:?}",
                        bit
                    )))?;
            input_qubit_mapping_data.set_item(physical_bit, index)?;
        }
        input_qubit_mapping = input_qubit_mapping_data.into_py_any(py)?;
    }

    if layout.final_layout_size > 0 {
        let final_layout_dict = PyDict::new(py);
        let py_qubits = circuit_data.py_qubits(py);
        let qubits = py_qubits.bind(py);
        for (index, bit) in layout.final_layout_items.iter().enumerate() {
            let qubit = qubits.get_item(*bit as usize)?;
            final_layout_dict.set_item(qubit, index)?;
        }
        final_layout = imports::LAYOUT
            .get_bound(py)
            .call1((final_layout_dict,))?
            .unbind();
    }
    let transpiled_layout = imports::TRANSPILER_LAYOUT.get_bound(py).call1((
        initial_layout,
        input_qubit_mapping,
        final_layout,
    ))?;
    // TODO: this is for version >= 10
    if layout.input_qubit_count >= 0 {
        transpiled_layout.setattr("_input_qubit_count", layout.input_qubit_count)?;
        transpiled_layout.setattr("_output_qubit_list", circuit_data.py_qubits(py))?;
    }
    Ok(transpiled_layout)
}

pub fn pack_custom_instructions(
    py: Python,
    custom_instructions_hash: &mut HashMap<String, PackedOperation>,
    circuit_data: &mut CircuitData,
    qpy_data: &mut QPYWriteData,
) -> PyResult<formats::CustomCircuitInstructionsPack> {
    let mut custom_instructions: Vec<formats::CustomCircuitInstructionDefPack> = Vec::new();
    let mut instructions_to_pack: Vec<String> = custom_instructions_hash.keys().cloned().collect();
    while let Some(name) = instructions_to_pack.pop() {
        custom_instructions.push(pack_py_custom_instruction(
            py,
            &name,
            custom_instructions_hash,
            &mut instructions_to_pack,
            circuit_data,
            qpy_data,
        )?);
    }
    Ok(formats::CustomCircuitInstructionsPack {
        custom_instructions,
    })
}

fn pack_standalone_vars(
    circuit_data: &CircuitData,
    version: u32,
    standalone_var_indices: &mut HashMap<VarOrStretch, usize>,
) -> PyResult<Vec<formats::ExpressionVarDeclarationPack>> {
    let mut result = Vec::new();
    let mut index: usize = 0;
    // input vars
    for var in circuit_data.get_vars(CircuitVarType::Input) {
        result.push(pack_standalone_var(
            &var,
            expression_var_declaration::INPUT,
            version,
        )?);
        standalone_var_indices.insert(VarOrStretch::Var(var.clone()), index);
        index += 1;
    }
    
    // captured vars
    for var in circuit_data.get_vars(CircuitVarType::Capture) {
        result.push(pack_standalone_var(
            &var,
            expression_var_declaration::CAPTURE,
            version,
        )?);
        standalone_var_indices.insert(VarOrStretch::Var(var.clone()), index);
        index += 1;
    }

    // declared vars
    for var in circuit_data.get_vars(CircuitVarType::Declare) {
        result.push(pack_standalone_var(
            &var,
            expression_var_declaration::LOCAL,
            version,
        )?);
        standalone_var_indices.insert(VarOrStretch::Var(var.clone()), index);
        index += 1;
    }
    if version < 14 && (circuit_data.num_captured_stretches() > 0 || circuit_data.num_declared_stretches() > 0) {
        return Err(UnsupportedFeatureForVersion::new_err((
            "circuits containing stretch variables",
            14,
            version,
        )));
    }
    for stretch in circuit_data.get_stretches(CircuitStretchType::Capture)
    {
        result.push(pack_stretch(
            stretch,
            expression_var_declaration::STRETCH_CAPTURE,
        ));
        standalone_var_indices.insert(VarOrStretch::Stretch(stretch.clone()), index);
        index += 1;
    }
    for stretch in circuit_data.get_stretches(CircuitStretchType::Declare)
    {
        result.push(pack_stretch(
            stretch,
            expression_var_declaration::STRETCH_LOCAL,
        ));
        standalone_var_indices.insert(VarOrStretch::Stretch(stretch.clone()), index);
        index += 1;
    }
    Ok(result)
}

pub fn pack_circuit(
    mut circuit: QuantumCircuitData,
    metadata_serializer: &Bound<PyAny>,
    use_symengine: bool,
    version: u32,
    annotation_factories: &Bound<PyDict>,
) -> PyResult<formats::QPYFormatV15> {
    let py = metadata_serializer.py();
//    circuit.getattr("data")?; // in case _data is lazily generated in python
//    let mut circuit_data = circuit.getattr("_data")?.extract::<CircuitData>()?;
    let clbit_indices = circuit.data.get_clbit_indices(py).clone();
    let mut standalone_var_indices: HashMap<VarOrStretch, usize> = HashMap::new();
    let standalone_vars = pack_standalone_vars(&circuit.data, version, &mut standalone_var_indices)?;
    let annotation_handler = AnnotationHandler::new(annotation_factories);
    let mut qpy_data = QPYWriteData {
        version,
        _use_symengine: use_symengine,
        clbit_indices,
        standalone_var_indices,
        py_standalone_var_indices: PyDict::new(py).unbind(),
        annotation_handler,
    };
    let header = pack_circuit_header(&circuit, metadata_serializer, &qpy_data)?;
    // Pulse has been removed in Qiskit 2.0. As long as we keep QPY at version 13,
    // we need to write an empty calibrations header since read_circuit expects it
    let calibrations = formats::CalibrationsPack { num_cals: 0 };
    let (instructions, mut custom_instructions_hash) =
        pack_instructions(py, &circuit.data, &mut qpy_data)?;
    let custom_instructions = pack_custom_instructions(
        py,
        &mut custom_instructions_hash,
        &mut circuit.data,
        &mut qpy_data,
    )?;
    let layout = pack_layout(&circuit)?;
    let state_headers: Vec<formats::AnnotationStateHeaderPack> = qpy_data
        .annotation_handler
        .dump_serializers()?
        .into_iter()
        .map(|(namespace, state)| formats::AnnotationStateHeaderPack { namespace, state })
        .collect();

    let annotation_headers = formats::AnnotationHeaderStaticPack { state_headers };
    Ok(formats::QPYFormatV15 {
        header,
        standalone_vars,
        annotation_headers,
        custom_instructions,
        instructions,
        calibrations,
        layout,
    })
}

fn deserialize_pauli_evolution_gate(
    py: Python,
    data: &Bytes,
    qpy_data: &mut QPYReadData,
) -> PyResult<Py<PyAny>> {
    let json = py.import("json")?;
    let evo_synth_library = py.import("qiskit.synthesis.evolution")?;
    let (packed_data, _) = deserialize::<formats::PauliEvolutionDefPack>(data)?;
    // operators as stored as a numpy dump that can be loaded into Python's SparsePauliOp.from_list
    let operators: Vec<Bound<PyAny>> = packed_data
        .pauli_data
        .iter()
        .map(|elem| {
            let data = elem.data.clone();
            let data_type = tags::NUMPY_OBJ;
            let op_raw_data = DumpedValue { data_type, data }.to_python(py, qpy_data)?;
            imports::SPARSE_PAULI_OP
                .get_bound(py)
                .call_method1("from_list", (op_raw_data,))
        })
        .collect::<PyResult<_>>()?;
    let py_operators = if packed_data.standalone_op != 0 {
        operators[0].clone()
    } else {
        PyList::new(py, operators)?.into_any()
    };

    let time = DumpedValue {
        data_type: packed_data.time_type,
        data: packed_data.time_data,
    }
    .to_python(py, qpy_data)?;
    let synth_data = json.call_method1("loads", (packed_data.synth_data,))?;
    let synth_data = synth_data.downcast::<PyDict>()?;
    let synthesis_class_name = synth_data.get_item("class")?.ok_or_else(|| {
        PyValueError::new_err("Could not find synthesis class name for Pauli Evolution Gate")
    })?;
    let synthesis_class_settings = synth_data.get_item("settings")?.ok_or_else(|| {
        PyValueError::new_err("Could not find synthesis class settings for Pauli Evolution Gate")
    })?;
    let synthesis_class =
        evo_synth_library.getattr(synthesis_class_name.downcast::<PyString>()?)?;
    let synthesis =
        synthesis_class.call((), Some(synthesis_class_settings.downcast::<PyDict>()?))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "time"), time)?;
    kwargs.set_item(intern!(py, "synthesis"), synthesis)?;
    Ok(imports::PAULI_EVOLUTION_GATE
        .get_bound(py)
        .call((py_operators,), Some(&kwargs))?
        .unbind())
}

fn read_custom_instructions(
    py: Python,
    packed_circuit: &formats::QPYFormatV15,
    qpy_data: &mut QPYReadData,
) -> PyResult<HashMap<String, CustomCircuitInstructionData>> {
    let mut result = HashMap::new();
    for operation in &packed_circuit.custom_instructions.custom_instructions {
        let definition = if operation.custom_definition != 0 {
            if operation.name.starts_with("###PauliEvolutionGate_") {
                Some(deserialize_pauli_evolution_gate(
                    py,
                    &operation.data,
                    qpy_data,
                )?)
            } else {
                Some(
                    deserialize_circuit(
                        py,
                        &operation.data,
                        qpy_data.version,
                        py.None().bind(py),
                        qpy_data.use_symengine,
                        qpy_data.annotation_handler.annotation_factories,
                    )?
                    .0
                    .unbind(),
                )
            }
        } else {
            None
        };
        let custom_instruction_data = CustomCircuitInstructionData {
            gate_type: operation.gate_type,
            num_qubits: operation.num_qubits,
            num_clbits: operation.num_clbits,
            definition_circuit: definition,
            base_gate_raw: operation.base_gate_raw.clone(),
        };
        result.insert(operation.name.clone(), custom_instruction_data);
    }
    Ok(result)
}
fn add_standalone_vars(
    py: Python,
    packed_circuit: &formats::QPYFormatV15,
    qpy_data: &mut QPYReadData,
) -> PyResult<()> {
    for packed_var in &packed_circuit.standalone_vars {
        let ty = match packed_var.exp_type {
            ExpressionType::Bool => classical::types::Type::Bool,
            ExpressionType::Duration => classical::types::Type::Duration,
            ExpressionType::Float => classical::types::Type::Float,
            ExpressionType::Uint(val) => classical::types::Type::Uint(val.try_into().unwrap()), // TODO: why rust uses u16 and not u32?
        };
        let uuid = u128::from_be_bytes(packed_var.uuid_bytes);
        let name = packed_var.name.clone();
        match packed_var.usage {
            expression_var_declaration::LOCAL => {
                let var = classical::expr::Var::Standalone { uuid, name, ty };
                qpy_data.standalone_vars.bind(py).append(var.clone())?;
                qpy_data
                    .circuit_data
                    .add_var(var, CircuitVarType::Declare)?;
            }
            expression_var_declaration::INPUT => {
                let var = classical::expr::Var::Standalone { uuid, name, ty };
                qpy_data.standalone_vars.bind(py).append(var.clone())?;
                qpy_data.circuit_data.add_var(var, CircuitVarType::Input)?;
            }
            expression_var_declaration::CAPTURE => {
                let var = classical::expr::Var::Standalone { uuid, name, ty };
                qpy_data.standalone_vars.bind(py).append(var.clone())?;
                qpy_data
                    .circuit_data
                    .add_var(var, CircuitVarType::Capture)?;
            }
            expression_var_declaration::STRETCH_LOCAL => {
                let var = classical::expr::Stretch { uuid, name };
                qpy_data.standalone_vars.bind(py).append(var.clone())?;
                qpy_data
                    .circuit_data
                    .add_stretch(var, CircuitStretchType::Declare)?;
            }
            expression_var_declaration::STRETCH_CAPTURE => {
                let var = classical::expr::Stretch { uuid, name };
                qpy_data.standalone_vars.bind(py).append(var.clone())?;
                qpy_data
                    .circuit_data
                    .add_stretch(var, CircuitStretchType::Capture)?;
            }
            _ => (),
        }
    }
    Ok(())
}

fn add_registers_and_bits(
    packed_circuit: &formats::QPYFormatV15,
    qpy_data: &mut QPYReadData,
) -> PyResult<()> {
    let num_qubits = packed_circuit.header.num_qubits as usize;
    let num_clbits = packed_circuit.header.num_clbits as usize;
    let mut qubits: Vec<Option<ShareableQubit>> = vec![None; num_qubits];
    let mut clbits: Vec<Option<ShareableClbit>> = vec![None; num_clbits];
    let mut qregs = Vec::new();
    let mut cregs = Vec::new();

    // first, create all owning registers and collect their bits
    let mut non_standalone_registers = Vec::new();
    for packed_register in &packed_circuit.header.registers {
        if packed_register.standalone == 0 {
            non_standalone_registers.push(packed_register);
        } else {
            match packed_register.register_type {
                register_types::QREG => {
                    let qreg = QuantumRegister::new_owning(
                        &packed_register.name,
                        packed_register.bit_indices.len() as u32,
                    );
                    for (qubit, &index) in qreg.bits().zip(packed_register.bit_indices.iter()) {
                        if index >= 0 {
                            // index can be -1, indicating this bit is not in the circuit
                            qubits[index as usize] = Some(qubit);
                        }
                    }
                    if packed_register.in_circuit != 0 {
                        qregs.push(qreg);
                    }
                }
                register_types::CREG => {
                    let creg = ClassicalRegister::new_owning(
                        &packed_register.name,
                        packed_register.bit_indices.len() as u32,
                    );
                    for (clbit, &index) in creg.bits().zip(packed_register.bit_indices.iter()) {
                        if index >= 0 {
                            // index can be -1, indicating this bit is not in the circuit
                            clbits[index as usize] = Some(clbit);
                        }
                    }
                    Python::with_gil(|py| -> PyResult<()> {
                        qpy_data
                            .cregs
                            .bind(py)
                            .set_item(&packed_register.name, creg.clone())?;
                        Ok(())
                    })?;
                    if packed_register.in_circuit != 0 {
                        cregs.push(creg);
                    }
                }
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Unrecognized register type for {:?}",
                        packed_register.name
                    )))
                }
            }
        }
    }
    // First pass is done. Now add Anonymous bits that are not part of any register
    let final_qubit_list: Vec<ShareableQubit> = qubits
        .iter()
        .map(|element| match element {
            Some(qubit) => qubit.clone(),
            None => ShareableQubit::new_anonymous(),
        })
        .collect();
    let final_clbit_list: Vec<ShareableClbit> = clbits
        .iter()
        .map(|element| match element {
            Some(clbit) => clbit.clone(),
            None => ShareableClbit::new_anonymous(),
        })
        .collect();

    // We collected owning registers to qregs, cregs and added all remaining bits and can now deal with the non-standalone registers
    for packed_register in non_standalone_registers {
        match packed_register.register_type {
            register_types::QREG => {
                let bits: Vec<ShareableQubit> = packed_register
                    .bit_indices
                    .iter()
                    .filter_map(|&index| {
                        if index >= 0 {
                            Some(final_qubit_list[index as usize].clone())
                        } else {
                            None
                        }
                    })
                    .collect();
                let qreg = QuantumRegister::new_alias(Some(packed_register.name.clone()), bits);
                qregs.push(qreg);
            }
            register_types::CREG => {
                let bits: Vec<ShareableClbit> = packed_register
                    .bit_indices
                    .iter()
                    .filter_map(|&index| {
                        if index >= 0 {
                            Some(final_clbit_list[index as usize].clone())
                        } else {
                            None
                        }
                    })
                    .collect();
                let creg = ClassicalRegister::new_alias(Some(packed_register.name.clone()), bits);
                cregs.push(creg);
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unrecognized register type for {:?}",
                    packed_register.name
                )))
            }
        }
    }

    // now add the bits to the ciruit, and then add the registers
    for qubit in final_qubit_list {
        qpy_data.circuit_data.add_qubit(qubit, true)?;
    }
    for clbit in final_clbit_list {
        qpy_data.circuit_data.add_clbit(clbit, true)?;
    }

    for qreg in qregs {
        qpy_data.circuit_data.add_qreg(qreg, true)?;
    }

    for creg in cregs {
        qpy_data.circuit_data.add_creg(creg, true)?;
    }

    Python::with_gil(|py| -> PyResult<()> {
        qpy_data.clbit_indices = qpy_data.circuit_data.get_clbit_indices(py).clone();
        Ok(())
    })?;
    Ok(())
}

pub fn deserialize_circuit<'py>(
    py: Python<'py>,
    serialized_circuit: &[u8],
    version: u32,
    metadata_deserializer: &Bound<PyAny>,
    use_symengine: bool,
    annotation_factories: &Bound<PyDict>,
) -> PyResult<(Bound<'py, PyAny>, usize)> {
    let (packed_circuit, pos) = deserialize::<formats::QPYFormatV15>(serialized_circuit)?;
    let instruction_capacity = packed_circuit.instructions.len();
    // create an empty circuit; we'll fill data as we go along
    let mut circuit_data =
        CircuitData::with_capacity(0, 0, instruction_capacity, Param::Float(0.0))?;

    let annotation_handler = AnnotationHandler::new(annotation_factories);
    let mut qpy_data = QPYReadData {
        circuit_data: &mut circuit_data,
        version,
        use_symengine,
        clbit_indices: PyDict::new(py).unbind(),
        cregs: PyDict::new(py).unbind(),
        standalone_vars: PyList::empty(py).unbind(),
        vectors: HashMap::new(),
        annotation_handler,
    };

    let annotation_deserializers_data: Vec<(String, Bytes)> = packed_circuit
        .annotation_headers
        .state_headers
        .iter()
        .map(|data| (data.namespace.clone(), data.state.clone()))
        .collect();
    qpy_data
        .annotation_handler
        .load_deserializers(annotation_deserializers_data)?;
    let global_phase = packed_circuit
        .header
        .global_phase_data
        .to_param(py, &mut qpy_data)?;
    qpy_data.circuit_data.set_global_phase(global_phase)?;

    add_standalone_vars(py, &packed_circuit, &mut qpy_data)?;
    add_registers_and_bits(&packed_circuit, &mut qpy_data)?;

    let custom_instructions = read_custom_instructions(py, &packed_circuit, &mut qpy_data)?;
    for instruction in &packed_circuit.instructions {
        let inst = unpack_instruction(py, instruction, &custom_instructions, &mut qpy_data)?;
        qpy_data.circuit_data.push(inst)?;
    }

    for (vector, initialized_params) in qpy_data.vectors.values() {
        let vector_length = vector
            .bind(py)
            .call_method0("__len__")?
            .extract::<usize>()?;
        let missing_indices: Vec<u64> = (0..vector_length as u64)
            .filter(|x| !initialized_params.contains(x))
            .collect();
        if initialized_params.len() != vector_length {
            let msg = format!(
                "The ParameterVector: '{:}' is not fully identical to its \
                pre-serialization state. Elements {:} \
                in the ParameterVector will be not equal to the pre-serialized ParameterVector \
                as they weren't used in the circuit: {:}",
                vector.getattr(py, "name")?.extract::<String>(py)?,
                missing_indices
                    .iter()
                    .map(|index| index.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                packed_circuit.header.circuit_name
            );
            imports::WARNINGS_WARN.get_bound(py).call1((
                msg,
                imports::BUILTIN_USER_WARNING.get_bound(py),
                1,
            ))?;
        }
    }

    let unpacked_layout = unpack_layout(py, &packed_circuit.layout, &circuit_data)?;
    let metadata =
        deserialize_metadata(py, &packed_circuit.header.metadata, metadata_deserializer)?;
    let circuit = imports::QUANTUM_CIRCUIT
        .get_bound(py)
        .call_method1(intern!(py, "_from_circuit_data"), (circuit_data,))?;
    // add registers

    circuit.setattr("metadata", metadata)?;
    circuit.setattr("name", &packed_circuit.header.circuit_name)?;
    if let Some(layout) = unpacked_layout {
        circuit.setattr("_layout", layout)?;
    }
    Ok((circuit, pos))
}

#[pyfunction]
#[pyo3(signature = (file_obj, circuit, metadata_serializer, use_symengine, version, annotation_factories))]
pub fn py_write_circuit(
    py: Python,
    file_obj: &Bound<PyAny>,
    circuit: &Bound<PyAny>,
    metadata_serializer: &Bound<PyAny>,
    use_symengine: bool,
    version: u32,
    annotation_factories: &Bound<PyDict>,
) -> PyResult<usize> {
    let serialized_circuit = serialize(&pack_circuit(
        circuit.extract()?,
        metadata_serializer,
        use_symengine,
        version,
        annotation_factories,
    )?)?;
    file_obj.call_method1(
        "write",
        (pyo3::types::PyBytes::new(py, &serialized_circuit),),
    )?;
    Ok(serialized_circuit.len())
}

#[pyfunction]
#[pyo3(signature = (file_obj, version, metadata_deserializer, use_symengine, annotation_factories))]
pub fn py_read_circuit<'py>(
    py: Python<'py>,
    file_obj: &Bound<PyAny>,
    version: u32,
    metadata_deserializer: &Bound<PyAny>,
    use_symengine: bool,
    annotation_factories: &Bound<PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let pos = file_obj.call_method0("tell")?.extract::<usize>()?;
    let bytes = file_obj.call_method0("read")?;
    let serialized_circuit: &[u8] = bytes.downcast::<PyBytes>()?.as_bytes();
    let (deserialized_ciruit, bytes_read) = deserialize_circuit(
        py,
        serialized_circuit,
        version,
        metadata_deserializer,
        use_symengine,
        annotation_factories,
    )?;
    file_obj.call_method1("seek", (pos + bytes_read,))?;
    Ok(deserialized_ciruit)
}
