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

// Circuit reader module: converts a qpy file into a  QuantumCircuit

// We use the following terminology:
// 1. "Pack": To create a struct (from formats.rs) from the original data
// 2. "Serialize": To create binary data (Bytes) from the original data
// 3. "Write": To write to a file obj the serialization of the original data
// Ideally, serialization is done by packing in a binrw-enhanced struct and using the
// `write` method into a `Cursor` buffer, but there might be exceptions.

use hashbrown::HashMap;
use numpy::IntoPyArray;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3::types::{PyAny, PyBytes, PyDict, PyList, PyString, PyTuple, PyType};

use qiskit_circuit::bit::{
    ClassicalRegister, QuantumRegister, Register, ShareableClbit, ShareableQubit,
};
use qiskit_circuit::circuit_data::{CircuitData, CircuitStretchType, CircuitVarType};
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::classical;
use qiskit_circuit::imports;
use qiskit_circuit::operations::{Param, StandardInstruction};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::{Clbit, Qubit};

use smallvec::SmallVec;

use crate::annotations::AnnotationHandler;
use crate::bytes::Bytes;
use crate::consts::standard_gate_from_gate_class_name;
use crate::formats;
use crate::formats::QPYFormatV15;
use crate::params::{unpack_param, unpack_param_from_data};
use crate::py_methods::{get_python_gate_class, py_load_register, py_unpack_generic_data};
use crate::value::{
    DumpedPyValue, ExpressionType, QPYReadData, bit_types, circuit_instruction_types, deserialize,
    deserialize_with_args, expression_var_declaration, register_types, tags,
};

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
            )));
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
            Ok(Some(py_unpack_generic_data(py, exp_data_pack, qpy_data)?))
        }
        formats::ConditionData::Register(register_data) => {
            let register = py_load_register(py, register_data.clone(), qpy_data.circuit_data)?;
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
    // for some reason, the QPY convention for storing ints/floats params is in little endian and not big endian as usual
    let mut inst_params: Vec<Param> = instruction
        .params
        .iter()
        .map(|packed_param| unpack_param(packed_param, qpy_data, binrw::Endian::Little))
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
            .cast_into::<PyType>()?
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

fn deserialize_metadata(
    py: Python,
    metadata_bytes: &Bytes,
    metadata_deserializer: &Bound<PyAny>,
) -> PyResult<Py<PyAny>> {
    let json = py.import("json")?;
    let kwargs: Bound<'_, PyDict> = PyDict::new(py);
    kwargs.set_item("cls", metadata_deserializer)?;
    let metadata_string = PyString::new(py, metadata_bytes.try_into()?);
    Ok(json
        .call_method("loads", (metadata_string,), Some(&kwargs))?
        .unbind())
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
    let initial_layout_virtual_bits = PyList::new(py, Vec::<Py<PyAny>>::new())?;
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
        let physical_bits = physical_bits_object.cast_bound::<PyDict>(py)?;
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
            let op_raw_data = DumpedPyValue { data_type, data }.to_python(py, qpy_data)?;
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

    let time = DumpedPyValue {
        data_type: packed_data.time_type,
        data: packed_data.time_data,
    }
    .to_python(py, qpy_data)?;
    let synth_data = json.call_method1("loads", (packed_data.synth_data,))?;
    let synth_data = synth_data.cast::<PyDict>()?;
    let synthesis_class_name = synth_data.get_item("class")?.ok_or_else(|| {
        PyValueError::new_err("Could not find synthesis class name for Pauli Evolution Gate")
    })?;
    let synthesis_class_settings = synth_data.get_item("settings")?.ok_or_else(|| {
        PyValueError::new_err("Could not find synthesis class settings for Pauli Evolution Gate")
    })?;
    let synthesis_class = evo_synth_library.getattr(synthesis_class_name.cast::<PyString>()?)?;
    let synthesis = synthesis_class.call((), Some(synthesis_class_settings.cast::<PyDict>()?))?;
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
                    unpack_circuit(
                        py,
                        &deserialize::<QPYFormatV15>(&operation.data)?.0,
                        qpy_data.version,
                        py.None().bind(py),
                        qpy_data.use_symengine,
                        qpy_data.annotation_handler.annotation_factories,
                    )?
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
    packed_circuit: &formats::QPYFormatV15,
    qpy_data: &mut QPYReadData,
) -> PyResult<()> {
    let mut index: u16 = 0;
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
                let var = qpy_data.circuit_data.add_var(
                    classical::expr::Var::Standalone { uuid, name, ty },
                    CircuitVarType::Declare,
                )?;
                qpy_data.standalone_vars.insert(index, var);
                index += 1;
            }
            expression_var_declaration::INPUT => {
                let var = qpy_data.circuit_data.add_var(
                    classical::expr::Var::Standalone { uuid, name, ty },
                    CircuitVarType::Input,
                )?;
                qpy_data.standalone_vars.insert(index, var);
                index += 1;
            }
            expression_var_declaration::CAPTURE => {
                let var = qpy_data.circuit_data.add_var(
                    classical::expr::Var::Standalone { uuid, name, ty },
                    CircuitVarType::Capture,
                )?;
                qpy_data.standalone_vars.insert(index, var);
                index += 1;
            }
            expression_var_declaration::STRETCH_LOCAL => {
                let stretch = qpy_data.circuit_data.add_stretch(
                    classical::expr::Stretch { uuid, name },
                    CircuitStretchType::Declare,
                )?;
                qpy_data.standalone_stretches.insert(index, stretch);
                index += 1;
            }
            expression_var_declaration::STRETCH_CAPTURE => {
                let stretch = qpy_data.circuit_data.add_stretch(
                    classical::expr::Stretch { uuid, name },
                    CircuitStretchType::Capture,
                )?;
                qpy_data.standalone_stretches.insert(index, stretch);
                index += 1;
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
                    Python::attach(|py| -> PyResult<()> {
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
                    )));
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
                )));
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

    Ok(())
}

pub fn unpack_circuit<'py>(
    py: Python<'py>,
    packed_circuit: &QPYFormatV15,
    version: u32,
    metadata_deserializer: &Bound<PyAny>,
    use_symengine: bool,
    annotation_factories: &Bound<PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let instruction_capacity = packed_circuit.instructions.len();
    // create an empty circuit; we'll fill data as we go along
    let mut circuit_data =
        CircuitData::with_capacity(0, 0, instruction_capacity, Param::Float(0.0))?;

    let annotation_handler = AnnotationHandler::new(annotation_factories);
    let mut qpy_data = QPYReadData {
        circuit_data: &mut circuit_data,
        version,
        use_symengine,
        cregs: PyDict::new(py).unbind(),
        standalone_vars: HashMap::new(),
        standalone_stretches: HashMap::new(),
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
    let global_phase = unpack_param_from_data(
        packed_circuit.header.global_phase_data.clone(),
        packed_circuit.header.global_phase_type,
        &mut qpy_data,
        binrw::Endian::Big,
    )?;
    qpy_data.circuit_data.set_global_phase(global_phase)?;

    add_standalone_vars(&packed_circuit, &mut qpy_data)?;
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
            .filter(|x| !initialized_params.contains(&(*x as u32)))
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
    Ok(circuit)
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
    let serialized_circuit: &[u8] = bytes.cast::<PyBytes>()?.as_bytes();
    let (packed_circuit, bytes_read) = deserialize::<formats::QPYFormatV15>(serialized_circuit)?;
    let unpacked_ciruit = unpack_circuit(
        py,
        &packed_circuit,
        version,
        metadata_deserializer,
        use_symengine,
        annotation_factories,
    )?;
    file_obj.call_method1("seek", (pos + bytes_read,))?;
    Ok(unpacked_ciruit)
}
