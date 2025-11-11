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

// Circuit writer module: converts a QuantumCircuit into a qpy file

// We use the following terminology:
// 1. "Pack": To create a struct (from formats.rs) from the original data
// 2. "Serialize": To create binary data (Bytes) from the original data
// 3. "Write": To write to a file obj the serialization of the original data
// Ideally, serialization is done by packing in a binrw-enhanced struct and using the
// `write` method into a `Cursor` buffer, but there might be exceptions.

use hashbrown::{HashMap, HashSet};
use indexmap::IndexSet;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyTuple};

use qiskit_circuit::bit::{ClassicalRegister, PyClbit, PyQubit, QuantumRegister, Register};
use qiskit_circuit::circuit_data::{CircuitData, CircuitStretchType, CircuitVarType};
use qiskit_circuit::converters::QuantumCircuitData;
use qiskit_circuit::operations::{Operation, OperationRef};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};

use crate::annotations::AnnotationHandler;
use crate::formats;
use crate::py_methods::{
    gate_class_name, get_condition_data_from_inst, get_instruction_annotations,
    get_instruction_params, pack_custom_instruction, pack_py_registers, recognize_custom_operation,
    serialize_metadata,
};
use crate::value::{
    DumpedPyValue, QPYWriteData, bit_types, expression_var_declaration, pack_standalone_var,
    pack_stretch, register_types, serialize,
};

use crate::UnsupportedFeatureForVersion;

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

pub fn pack_instructions(
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
                pack_instruction(
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

pub fn pack_instruction(
    instruction: &PackedInstruction,
    circuit_data: &CircuitData,
    custom_operations: &mut HashMap<String, PackedOperation>,
    new_custom_operations: &mut Vec<String>,
    qpy_data: &mut QPYWriteData,
) -> PyResult<formats::CircuitInstructionV2Pack> {
    let mut gate_class_name = gate_class_name(&instruction.op)?;
    if let Some(new_name) = recognize_custom_operation(&instruction.op, &gate_class_name)? {
        gate_class_name = new_name;
        new_custom_operations.push(gate_class_name.clone());
        custom_operations.insert(gate_class_name.clone(), instruction.op.clone());
    }
    let label = match instruction.label() {
        Some(label) => String::from(label),
        None => String::from(""),
    };
    let num_ctrl_qubits = match &instruction.op.view() {
        OperationRef::StandardGate(gate) => gate.num_ctrl_qubits(),
        OperationRef::Gate(py_gate) => py_gate.num_ctrl_qubits(),
        OperationRef::Instruction(py_inst) => py_inst.num_ctrl_qubits(),
        _ => 0,
    };
    let ctrl_state = match &instruction.op.view() {
        OperationRef::Gate(py_gate) => py_gate.ctrl_state(),
        OperationRef::Instruction(py_inst) => py_inst.ctrl_state(),
        _ => (1 << num_ctrl_qubits) - 1,
    };
    let params: Vec<formats::PackedParam> = get_instruction_params(instruction, qpy_data)?;
    let bit_data = get_packed_bit_list(instruction, circuit_data);
    let condition = get_condition_data(&instruction.op, circuit_data, qpy_data)?;
    let annotations = get_instruction_annotations(instruction, qpy_data)?;
    let mut extras_key = condition.key;
    if annotations.is_some() {
        extras_key |= formats::extras_key_parts::ANNOTATIONS;
    }
    Ok(formats::CircuitInstructionV2Pack {
        num_qargs: instruction.op.num_qubits(),
        num_cargs: instruction.op.num_clbits(),
        extras_key,
        num_ctrl_qubits,
        ctrl_state,
        gate_class_name,
        label,
        condition,
        bit_data,
        params,
        annotations,
    })
}

// packs the quantum registers in the circuit. we pack:
// 1) registers appearing explicitly in the circuit register list (identified using in_circ_lookup)
// 2) registers appearing implicitly for bits (in python: as their "_register" data field)
fn pack_quantum_registers(circuit_data: &CircuitData) -> Vec<formats::RegisterV4Pack> {
    let in_circ_lookup: HashSet<QuantumRegister> = circuit_data.qregs().iter().cloned().collect();
    let mut registers_to_pack: IndexSet<QuantumRegister> =
        circuit_data.qregs().iter().cloned().collect();
    registers_to_pack.extend(
        circuit_data
            .qubits()
            .objects()
            .iter()
            .filter_map(|qubit| circuit_data.qubit_indices().get(qubit)) // only qubits with register data
            .flat_map(|qreg_locations| {
                qreg_locations
                    .registers()
                    .iter()
                    .map(|(qreg, _)| qreg.clone())
            }),
    );
    registers_to_pack
        .iter()
        .map(|qreg| {
            let bit_indices = qreg
                .bits()
                .map(|qubit| {
                    circuit_data
                        .qubit_index(qubit)
                        .map(|index| index as i64)
                        .unwrap_or(-1)
                })
                .collect();
            formats::RegisterV4Pack {
                register_type: register_types::QREG,
                standalone: qreg.is_owning() as u8,
                in_circuit: in_circ_lookup.contains(qreg) as u8,
                name: qreg.name().to_string(),
                bit_indices,
            }
        })
        .collect()
}

fn pack_classical_registers(circuit_data: &CircuitData) -> Vec<formats::RegisterV4Pack> {
    let in_circ_lookup: HashSet<ClassicalRegister> = circuit_data.cregs().iter().cloned().collect();
    let mut registers_to_pack: IndexSet<ClassicalRegister> =
        circuit_data.cregs().iter().cloned().collect();
    registers_to_pack.extend(
        circuit_data
            .clbits()
            .objects()
            .iter()
            .filter_map(|clbit| circuit_data.clbit_indices().get(clbit)) // only qubits with register data
            .flat_map(|creg_locations| {
                creg_locations
                    .registers()
                    .iter()
                    .map(|(creg, _)| creg.clone())
            }),
    );
    registers_to_pack
        .iter()
        .map(|creg| {
            let bit_indices = creg
                .bits()
                .map(|clbit| {
                    circuit_data
                        .clbit_index(clbit)
                        .map(|index| index as i64)
                        .unwrap_or(-1)
                })
                .collect();
            formats::RegisterV4Pack {
                register_type: register_types::CREG,
                standalone: creg.is_owning() as u8,
                in_circuit: in_circ_lookup.contains(creg) as u8,
                name: creg.name().to_string(),
                bit_indices,
            }
        })
        .collect()
}

fn pack_circuit_header(
    circuit: &QuantumCircuitData,
    metadata_serializer: &Bound<PyAny>,
    qpy_data: &QPYWriteData,
) -> PyResult<formats::CircuitHeaderV12Pack> {
    let metadata = serialize_metadata(&circuit.metadata, metadata_serializer)?;
    let global_phase_data = DumpedPyValue::from_param(circuit.data.global_phase(), qpy_data)?;
    let qregs = pack_quantum_registers(&circuit.data);
    let cregs = pack_classical_registers(&circuit.data);
    let mut registers = qregs;
    registers.extend(cregs);
    let header = formats::CircuitHeaderV12Pack {
        num_qubits: circuit.data.num_qubits() as u32,
        num_clbits: circuit.data.num_clbits() as u32,
        num_instructions: circuit.data.__len__() as u64,
        num_vars: circuit.data.num_identifiers() as u32,
        circuit_name: circuit.name.clone().unwrap_or_default(),
        global_phase_data: global_phase_data.data,
        global_phase_type: global_phase_data.data_type,
        metadata,
        registers,
    };

    Ok(header)
}

pub fn pack_layout(circuit: &QuantumCircuitData) -> PyResult<formats::LayoutV2Pack> {
    let default_layout = formats::LayoutV2Pack {
        exists: 0,
        initial_layout_size: -1,
        input_mapping_size: -1,
        final_layout_size: -1,
        input_qubit_count: 0,
        extra_registers: Vec::new(),
        initial_layout_items: Vec::new(),
        input_mapping_items: Vec::new(),
        final_layout_items: Vec::new(),
    };
    match &circuit.custom_layout {
        None => Ok(default_layout),
        Some(custom_layout) => {
            if custom_layout.is_none() {
                Ok(default_layout)
            } else {
                pack_custom_layout(custom_layout, &circuit.data)
            }
        }
    }
}

fn pack_custom_layout(
    layout: &Bound<PyAny>,
    circuit: &CircuitData,
) -> PyResult<formats::LayoutV2Pack> {
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
                    extra_register_list.cast::<PyList>()?.append(qubit)?;
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
        for (qubit, index) in layout_input_qubit_mapping.cast::<PyDict>()? {
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
                extra_register_list.cast::<PyList>()?.append(&qubit)?;
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
            let virtual_bit = final_layout_physical
                .cast::<PyDict>()?
                .get_item(i)?
                .unwrap(); // TODO: handle unwrap failure
            if virtual_bit.is_instance_of::<PyClbit>() {
                match circuit
                    .get_clbit_indices(py)
                    .bind(py)
                    .get_item(virtual_bit)?
                {
                    None => (), // TODO: error?
                    Some(bit_data) => final_layout_array.append(bit_data.getattr("index")?)?,
                }
            } else if virtual_bit.is_instance_of::<PyQubit>() {
                match circuit
                    .get_qubit_indices(py)
                    .bind(py)
                    .get_item(virtual_bit)?
                {
                    None => (), // TODO: error?
                    Some(bit_data) => final_layout_array.append(bit_data.getattr("index")?)?,
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
        for x in register_bit_list.cast::<PyList>()? {
            bits.push(x);
        }
    }
    let extra_registers = pack_py_registers(&extra_registers.keys(), &PyList::new(py, bits)?)?;
    let mut initial_layout_items = Vec::with_capacity(initial_layout_size.max(0) as usize);
    for item in initial_layout_array {
        let tuple = item.cast::<PyTuple>()?;
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
    }

    let mut final_layout_items = Vec::with_capacity(final_layout_size.max(0) as usize);
    for i in &final_layout_array {
        final_layout_items.push(i.extract::<u32>()?);
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

pub fn pack_custom_instructions(
    custom_instructions_hash: &mut HashMap<String, PackedOperation>,
    circuit_data: &mut CircuitData,
    qpy_data: &mut QPYWriteData,
) -> PyResult<formats::CustomCircuitInstructionsPack> {
    let mut custom_instructions: Vec<formats::CustomCircuitInstructionDefPack> = Vec::new();
    let mut instructions_to_pack: Vec<String> = custom_instructions_hash.keys().cloned().collect();
    while let Some(name) = instructions_to_pack.pop() {
        custom_instructions.push(pack_custom_instruction(
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

pub fn pack_standalone_vars(
    circuit_data: &CircuitData,
    version: u32,
    standalone_var_indices: &mut HashMap<u128, u16>,
) -> PyResult<Vec<formats::ExpressionVarDeclarationPack>> {
    let mut result = Vec::new();
    let mut index: u16 = 0;
    let mut uuid: u128 = 0;
    // input vars
    for var in circuit_data.get_vars(CircuitVarType::Input) {
        let var_pack =
            pack_standalone_var(var, expression_var_declaration::INPUT, version, &mut uuid)?;
        result.push(var_pack);
        standalone_var_indices.insert(uuid, index);
        index += 1;
    }

    // captured vars
    for var in circuit_data.get_vars(CircuitVarType::Capture) {
        result.push(pack_standalone_var(
            var,
            expression_var_declaration::CAPTURE,
            version,
            &mut uuid,
        )?);
        standalone_var_indices.insert(uuid, index);
        index += 1;
    }

    // declared vars
    for var in circuit_data.get_vars(CircuitVarType::Declare) {
        result.push(pack_standalone_var(
            var,
            expression_var_declaration::LOCAL,
            version,
            &mut uuid,
        )?);
        standalone_var_indices.insert(uuid, index);
        index += 1;
    }
    if version < 14
        && (circuit_data.num_captured_stretches() > 0 || circuit_data.num_declared_stretches() > 0)
    {
        return Err(UnsupportedFeatureForVersion::new_err((
            "circuits containing stretch variables",
            14,
            version,
        )));
    }
    for stretch in circuit_data.get_stretches(CircuitStretchType::Capture) {
        result.push(pack_stretch(
            stretch,
            expression_var_declaration::STRETCH_CAPTURE,
        ));
        standalone_var_indices.insert(stretch.uuid, index);
        index += 1;
    }
    for stretch in circuit_data.get_stretches(CircuitStretchType::Declare) {
        result.push(pack_stretch(
            stretch,
            expression_var_declaration::STRETCH_LOCAL,
        ));
        standalone_var_indices.insert(stretch.uuid, index);
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
    // let clbit_indices = circuit.data.get_clbit_indices(py).clone();
    let mut standalone_var_indices: HashMap<u128, u16> = HashMap::new();
    let standalone_vars =
        pack_standalone_vars(&circuit.data, version, &mut standalone_var_indices)?;
    let annotation_handler = AnnotationHandler::new(annotation_factories);
    let mut qpy_data = QPYWriteData {
        version,
        _use_symengine: use_symengine,
        clbits: &circuit.data.clbits().clone(), // we need to clone since circuit_data might change when serializing custom instructions, explicitly creating the inner instructions
        standalone_var_indices,
        annotation_handler,
    };
    let header = pack_circuit_header(&circuit, metadata_serializer, &qpy_data)?;
    // Pulse has been removed in Qiskit 2.0. As long as we keep QPY at version 13,
    // we need to write an empty calibrations header since read_circuit expects it
    let calibrations = formats::CalibrationsPack { num_cals: 0 };
    let (instructions, mut custom_instructions_hash) =
        pack_instructions(&circuit.data, &mut qpy_data)?;
    let custom_instructions = pack_custom_instructions(
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
    let packed_circuit = pack_circuit(
        circuit.extract()?,
        metadata_serializer,
        use_symengine,
        version,
        annotation_factories,
    )?;
    let serialized_circuit = serialize(&packed_circuit);
    file_obj.call_method1(
        "write",
        (pyo3::types::PyBytes::new(py, &serialized_circuit),),
    )?;
    Ok(serialized_circuit.len())
}
