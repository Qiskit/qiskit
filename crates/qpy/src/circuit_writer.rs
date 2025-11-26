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

use binrw::Endian;
use hashbrown::{HashMap, HashSet};
use indexmap::IndexSet;
use numpy::ToPyArray;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use qiskit_circuit::bit::{
    ClassicalRegister, PyClbit, PyQubit, QuantumRegister, Register, ShareableClbit, ShareableQubit,
};
use qiskit_circuit::circuit_data::{CircuitData, CircuitStretchType, CircuitVarType};
use qiskit_circuit::circuit_instruction::CircuitInstruction;
use qiskit_circuit::converters::QuantumCircuitData;
use qiskit_circuit::imports;
use qiskit_circuit::operations::{ArrayType, Operation, OperationRef};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};

use crate::annotations::AnnotationHandler;
use crate::bytes::Bytes;
use crate::formats::{self, InitialLayoutItemV2Pack};
use crate::params::pack_param_obj;
use crate::py_methods::{
    gate_class_name, getattr_or_none, py_get_condition_data_from_inst,
    py_get_instruction_annotations, py_pack_param, py_pack_pauli_evolution_gate,
    recognize_custom_operation, serialize_metadata,
};
use crate::value::{
    GenericValue, QPYWriteData, bit_types, circuit_instruction_types, expression_var_declaration,
    get_circuit_type_key, pack_generic_value, pack_standalone_var, pack_stretch, register_types,
    serialize,
};

use crate::UnsupportedFeatureForVersion;

/// packing the qubits and clbits of a specific instruction into CircuitInstructionArgPack
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
    qpy_data: &QPYWriteData,
) -> PyResult<formats::ConditionPack> {
    match op.view() {
        OperationRef::Instruction(py_inst) => {
            py_get_condition_data_from_inst(&py_inst.instruction, qpy_data)
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

/// pack all the instructions in the circuit, returning both the packed instructions
/// and the dictionary of custom operations generated in the process
pub fn pack_instructions(
    qpy_data: &mut QPYWriteData,
) -> PyResult<(
    Vec<formats::CircuitInstructionV2Pack>,
    HashMap<String, PackedOperation>,
)> {
    let mut custom_operations = HashMap::new();
    let mut custom_new_operations = Vec::new();
    let instructions = qpy_data.circuit_data.data().to_vec();
    Ok((
        instructions
            .iter()
            .map(|instruction| {
                pack_instruction(
                    instruction,
                    &mut custom_operations,
                    &mut custom_new_operations,
                    qpy_data,
                )
            })
            .collect::<PyResult<_>>()?,
        custom_operations,
    ))
}

/// packs one specific instruction into CircuitInstructionV2Pack, creating a new custom operation if needed
pub fn pack_instruction(
    instruction: &PackedInstruction,
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
    let label = instruction.label().unwrap_or("").to_string();
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
    // this relies heavily on python-space as instruction params are usually added ad-hoc to arbitrary field in the python instruction
    let params: Vec<formats::GenericDataPack> = get_instruction_params(instruction, qpy_data)?;
    let bit_data = get_packed_bit_list(instruction, qpy_data.circuit_data);
    let condition = get_condition_data(&instruction.op, qpy_data)?;
    let annotations = py_get_instruction_annotations(instruction, qpy_data)?;
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

pub fn get_instruction_params(
    instruction: &PackedInstruction,
    qpy_data: &QPYWriteData,
) -> PyResult<Vec<formats::GenericDataPack>> {
    // The instruction params we store are about being able to reconstruct the objects; they don't
    // necessarily need to match one-to-one to the `params` field.
    Python::attach(|py| {
        if let OperationRef::Instruction(inst) = instruction.op.view() {
            if inst
                .instruction
                .bind(py)
                .is_instance(imports::CONTROL_FLOW_SWITCH_CASE_OP.get_bound(py))?
            {
                let op = inst.instruction.bind(py);
                let target = op.getattr("target")?;
                let cases = op.call_method0("cases_specifier")?;
                let cases_tuple = imports::BUILTIN_TUPLE.get_bound(py).call1((cases,))?;
                return Ok(vec![
                    py_pack_param(&target, qpy_data, Endian::Little)?,
                    py_pack_param(&cases_tuple, qpy_data, Endian::Little)?,
                ]);
            }
            if inst
                .instruction
                .bind(py)
                .is_instance(imports::CONTROL_FLOW_BOX_OP.get_bound(py))?
            {
                let op = inst.instruction.bind(py);
                let first_block = op
                    .getattr("blocks")?
                    .try_iter()?
                    .next()
                    .transpose()?
                    .ok_or_else(|| PyValueError::new_err("No blocks in box control flow op"))?;
                let duration = op.getattr("duration")?;
                let unit = op.getattr("unit")?;
                return Ok(vec![
                    py_pack_param(&first_block, qpy_data, Endian::Little)?,
                    py_pack_param(&duration, qpy_data, Endian::Little)?,
                    py_pack_param(&unit, qpy_data, Endian::Little)?,
                ]);
            }
        }

        if let OperationRef::Operation(op) = instruction.op.view() {
            if op
                .operation
                .bind(py)
                .is_instance(imports::CLIFFORD.get_bound(py))?
            {
                let op = op.operation.bind(py);
                let tableau = op.getattr("tableau")?;
                return Ok(vec![py_pack_param(&tableau, qpy_data, Endian::Little)?]);
            }
            if op
                .operation
                .bind(py)
                .is_instance(imports::ANNOTATED_OPERATION.get_bound(py))?
            {
                let op = op.operation.bind(py);
                let modifiers = op.getattr("modifiers")?;
                return modifiers
                    .try_iter()?
                    .map(|modifier| py_pack_param(&modifier?, qpy_data, Endian::Little))
                    .collect::<PyResult<_>>();
            }
            if op
                .operation
                .bind(py)
                .is_instance(imports::PAULI_PRODUCT_MEASUREMENT.get_bound(py))?
            {
                let op = op.operation.bind(py);
                let pauli_data = op.call_method0("_to_pauli_data")?;
                return pauli_data
                    .try_iter()?
                    .map(|pauli| py_pack_param(&pauli?, qpy_data, Endian::Little))
                    .collect::<PyResult<_>>();
            }
        }

        if let OperationRef::Unitary(unitary) = instruction.op.view() {
            // unitary gates are special since they are uniquely determined by a matrix, which is not
            // a "parameter", strictly speaking, but is treated as such when serializing

            // until we change the QPY version or verify we get the exact same result,
            // we translate the matrix to numpy and then serialize it like python does
            let out_array = match &unitary.array {
                ArrayType::NDArray(arr) => arr.to_pyarray(py),
                ArrayType::OneQ(arr) => arr.to_pyarray(py),
                ArrayType::TwoQ(arr) => arr.to_pyarray(py),
            };
            return Ok(vec![py_pack_param(&out_array, qpy_data, Endian::Little)?]);
        }
        if let OperationRef::PauliProductMeasurement(pauli_product_measurement) =
            instruction.op.view()
        {
            let z_values = GenericValue::Tuple(
                pauli_product_measurement
                    .z
                    .iter()
                    .cloned()
                    .map(GenericValue::Bool)
                    .collect(),
            );
            let x_values = GenericValue::Tuple(
                pauli_product_measurement
                    .x
                    .iter()
                    .cloned()
                    .map(GenericValue::Bool)
                    .collect(),
            );
            let neg_value = GenericValue::Bool(pauli_product_measurement.neg);
            return Ok(vec![
                pack_generic_value(&z_values, qpy_data)?,
                pack_generic_value(&x_values, qpy_data)?,
                pack_generic_value(&neg_value, qpy_data)?,
            ]);
        }
        instruction
            .params_view()
            .iter()
            .map(|x| pack_param_obj(x, qpy_data, Endian::Little))
            .collect::<PyResult<_>>()
    })
}

// packs the quantum registers in the circuit. we pack:
// 1) registers appearing explicitly in the circuit register list (identified using in_circ_lookup)
// 2) registers appearing implicitly for bits (in python: as their "_register" data field)
fn pack_quantum_registers(circuit_data: &CircuitData) -> Vec<formats::RegisterV4Pack> {
    let in_circ_lookup: HashSet<QuantumRegister> = circuit_data.qregs().iter().cloned().collect();
    let mut registers_to_pack: IndexSet<QuantumRegister> =
        circuit_data.qregs().iter().cloned().collect();

    // add all owning registers for qubits in the circuits (even registers not included in the circuit itself)
    registers_to_pack.extend(
        circuit_data
            .qubits()
            .objects()
            .iter()
            .filter_map(|qubit| qubit.owning_register()),
    );
    // add all registers showing up as qubit indices for some qubit in the circuit
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
        .map(|qreg| pack_quantum_register(qreg, circuit_data, in_circ_lookup.contains(qreg)))
        .collect()
}

fn pack_quantum_register(
    qreg: &QuantumRegister,
    circuit_data: &CircuitData,
    in_circuit: bool,
) -> formats::RegisterV4Pack {
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
        in_circuit: in_circuit as u8,
        name: qreg.name().to_string(),
        bit_indices,
    }
}

fn pack_classical_registers(circuit_data: &CircuitData) -> Vec<formats::RegisterV4Pack> {
    let in_circ_lookup: HashSet<ClassicalRegister> = circuit_data.cregs().iter().cloned().collect();
    let mut registers_to_pack: IndexSet<ClassicalRegister> =
        circuit_data.cregs().iter().cloned().collect();
    // add all owning registers for clbits in the circuits (even registers not included in the circuit itself)
    registers_to_pack.extend(
        circuit_data
            .clbits()
            .objects()
            .iter()
            .filter_map(|clbit| clbit.owning_register()),
    );
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
    circuit_name: Option<String>,
    circuit_metadata: Option<Bound<PyAny>>,
    // circuit: &QuantumCircuitData,
    metadata_serializer: Option<&Bound<PyAny>>,
    qpy_data: &QPYWriteData,
) -> PyResult<formats::CircuitHeaderV12Pack> {
    let metadata = serialize_metadata(&circuit_metadata, metadata_serializer)?;
    let global_phase_data = pack_param_obj(
        qpy_data.circuit_data.global_phase(),
        qpy_data,
        binrw::Endian::Big,
    )?;
    let qregs = pack_quantum_registers(qpy_data.circuit_data);
    let cregs = pack_classical_registers(qpy_data.circuit_data);
    let mut registers = qregs;
    registers.extend(cregs);
    let header = formats::CircuitHeaderV12Pack {
        num_qubits: qpy_data.circuit_data.num_qubits() as u32,
        num_clbits: qpy_data.circuit_data.num_clbits() as u32,
        num_instructions: qpy_data.circuit_data.__len__() as u64,
        num_vars: qpy_data.circuit_data.num_identifiers() as u32,
        circuit_name: circuit_name.unwrap_or_default(),
        global_phase_data: global_phase_data.data,
        global_phase_type: global_phase_data.type_key,
        metadata,
        registers,
    };

    Ok(header)
}

pub fn pack_layout(
    custom_layout: Option<Bound<PyAny>>,
    qpy_data: &QPYWriteData,
) -> PyResult<formats::LayoutV2Pack> {
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
    match custom_layout {
        None => Ok(default_layout),
        Some(custom_layout) => {
            if custom_layout.is_none() {
                Ok(default_layout)
            } else {
                pack_custom_layout(&custom_layout, qpy_data)
            }
        }
    }
}

fn pack_custom_layout(
    layout: &Bound<PyAny>,
    qpy_data: &QPYWriteData,
) -> PyResult<formats::LayoutV2Pack> {
    let mut initial_layout_size = -1; // initial_size
    let mut input_qubit_mapping: HashMap<ShareableQubit, usize> = HashMap::new();
    let mut initial_layout_array: Vec<(Option<u32>, Option<QuantumRegister>)> = Vec::new();
    let mut extra_registers: HashSet<QuantumRegister> = HashSet::new();
    let mut extra_registers_qubits: HashSet<ShareableQubit> = HashSet::new();

    let initial_layout = layout.getattr("initial_layout")?;
    if !initial_layout.is_none() {
        initial_layout_size = initial_layout.call_method0("__len__")?.extract::<i32>()?;
        let layout_mapping = initial_layout.call_method0("get_physical_bits")?;
        for i in 0..qpy_data.circuit_data.num_qubits() {
            let qubit = layout_mapping.get_item(i)?.extract::<ShareableQubit>()?;
            input_qubit_mapping.insert(qubit.clone(), i);
            let register = qubit.owning_register();
            let index = qubit.owning_register_index();
            if let Some(reg) = register.clone() {
                extra_registers.insert(reg);
                extra_registers_qubits.insert(qubit);
            } else if index.is_some() {
                extra_registers_qubits.insert(qubit);
            }
            initial_layout_array.push((index, register));
        }
    }

    let mut input_mapping_size: i32 = -1; //input_qubit_size
    let layout_input_qubit_mapping = layout.getattr("input_qubit_mapping")?;
    let input_mapping_items = if !layout_input_qubit_mapping.is_none() {
        input_mapping_size = layout_input_qubit_mapping
            .call_method0("__len__")?
            .extract()?;
        let mut input_qubit_mapping_array: Vec<u32> = vec![0; input_mapping_size as usize];
        let layout_mapping = initial_layout.call_method0("get_virtual_bits")?;
        for (qubit, index) in layout_input_qubit_mapping.cast::<PyDict>()? {
            let qubit = qubit.extract::<ShareableQubit>()?;
            let register = qubit.owning_register();
            if let Some(reg) = register {
                if qubit.owning_register_index().is_some()
                    && !qpy_data.circuit_data.qregs().contains(&reg)
                {
                    extra_registers.insert(reg);
                }
            };
            let i: usize = index.extract()?;
            input_qubit_mapping_array[i] = layout_mapping.get_item(&qubit)?.extract::<u32>()?;
        }
        input_qubit_mapping_array
    } else {
        Vec::new()
    };

    let mut final_layout_size: i32 = -1; // this is the default value if final_layout is not present
    // let final_layout_array = PyList::empty(py);
    let final_layout = layout.getattr("final_layout")?;
    let final_layout_items: Vec<u32> = if !final_layout.is_none() {
        final_layout_size = final_layout.call_method0("__len__")?.extract::<i32>()?;
        let mut final_layout_items: Vec<u32> = Vec::with_capacity(final_layout_size as usize);
        let final_layout_physical = final_layout.call_method0("get_physical_bits")?;
        for i in 0..qpy_data.circuit_data.num_qubits() {
            let virtual_bit = final_layout_physical.get_item(i)?;
            if virtual_bit.is_instance_of::<PyClbit>() {
                let virtual_clbit = virtual_bit.extract::<ShareableClbit>()?;
                let index = qpy_data
                    .circuit_data
                    .clbit_index(virtual_clbit)
                    .ok_or_else(|| PyValueError::new_err("Clbit missing an index"))?;
                final_layout_items.push(index);
            } else if virtual_bit.is_instance_of::<PyQubit>() {
                let virtual_qubit = virtual_bit.extract::<ShareableQubit>()?;
                let index = qpy_data
                    .circuit_data
                    .qubit_index(virtual_qubit)
                    .ok_or_else(|| PyValueError::new_err("Clbit missing an index"))?;
                final_layout_items.push(index);
            }
        }
        final_layout_items
    } else {
        Vec::new()
    };

    let input_qubit_count: i32 = if layout.getattr("_input_qubit_count")?.is_none() {
        -1
    } else {
        layout.getattr("_input_qubit_count")?.extract()?
    };

    let extra_registers =
        pack_extra_registers(&extra_registers, &extra_registers_qubits, qpy_data)?;

    let initial_layout_items: Vec<InitialLayoutItemV2Pack> = initial_layout_array
        .iter()
        .map(|(index, qreg)| {
            let index_value = match index {
                None => -1,
                Some(val) => *val as i32,
            };
            let (register_name, register_name_length) = match qreg {
                None => (String::new(), -1),
                Some(qreg_val) => (qreg_val.name().to_string(), qreg_val.name().len() as i32),
            };
            formats::InitialLayoutItemV2Pack {
                index_value,
                register_name_length,
                register_name,
            }
        })
        .collect();

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
    qpy_data: &mut QPYWriteData,
) -> PyResult<formats::CustomCircuitInstructionsPack> {
    let mut custom_instructions: Vec<formats::CustomCircuitInstructionDefPack> = Vec::new();
    let mut instructions_to_pack: Vec<String> = custom_instructions_hash.keys().cloned().collect();
    while let Some(name) = instructions_to_pack.pop() {
        custom_instructions.push(pack_custom_instruction(
            &name,
            custom_instructions_hash,
            &mut instructions_to_pack,
            qpy_data,
        )?);
    }
    Ok(formats::CustomCircuitInstructionsPack {
        custom_instructions,
    })
}

pub fn pack_extra_registers(
    in_circ_regs: &HashSet<QuantumRegister>,
    qubits: &HashSet<ShareableQubit>,
    qpy_data: &QPYWriteData,
) -> PyResult<Vec<formats::RegisterV4Pack>> {
    let mut out_circ_regs: HashSet<QuantumRegister> = HashSet::new();
    for qubit in qubits.iter() {
        if let Some(qreg) = qubit.owning_register() {
            if !in_circ_regs.contains(&qreg) {
                out_circ_regs.insert(qreg);
            }
        }
    }
    let mut result = Vec::new();
    for qreg in in_circ_regs.iter() {
        result.push(pack_quantum_register(qreg, qpy_data.circuit_data, true));
    }
    for qreg in out_circ_regs.iter() {
        result.push(pack_quantum_register(qreg, qpy_data.circuit_data, false));
    }
    Ok(result)
}

pub fn pack_custom_instruction(
    name: &String,
    custom_instructions_hash: &mut HashMap<String, PackedOperation>,
    new_instructions_list: &mut Vec<String>,
    qpy_data: &mut QPYWriteData,
) -> PyResult<formats::CustomCircuitInstructionDefPack> {
    Python::attach(|py| {
        let operation = custom_instructions_hash.get(name).ok_or_else(|| {
            PyValueError::new_err(format!("Could not find operation data for {}", name))
        })?;
        let gate_type = get_circuit_type_key(operation)?;
        let mut has_definition = false;
        let mut data: Bytes = Bytes::new();
        let mut num_ctrl_qubits = 0;
        let mut ctrl_state = 0;
        let mut base_gate: Bound<PyAny> = py.None().bind(py).clone();
        let mut base_gate_raw: Bytes = Bytes::new();

        if gate_type == circuit_instruction_types::PAULI_EVOL_GATE {
            if let OperationRef::Gate(gate) = operation.view() {
                has_definition = true;
                data = serialize(&py_pack_pauli_evolution_gate(gate.gate.bind(py), qpy_data)?);
            }
        } else if gate_type == circuit_instruction_types::CONTROLLED_GATE {
            // For ControlledGate, we have to access and store the private `_definition` rather than the
            // public one, because the public one is mutated to include additional logic if the control
            // state is open, and the definition setter (during a subsequent read) uses the "fully
            // excited" control definition only.
            if let OperationRef::Gate(pygate) = operation.view() {
                has_definition = true;
                // Build internal definition to support overloaded subclasses by
                // calling definition getter on object
                let gate = pygate.gate.bind(py);
                gate.getattr("definition")?; // this creates the _definition field
                data = serialize(&pack_circuit(
                    &mut gate.getattr("_definition")?.extract()?,
                    Some(py.None().bind(py)),
                    false,
                    qpy_data.version,
                    qpy_data.annotation_handler.annotation_factories,
                )?);
                num_ctrl_qubits = gate.getattr("num_ctrl_qubits")?.extract::<u32>()?;
                ctrl_state = gate.getattr("ctrl_state")?.extract::<u32>()?;
                base_gate = gate.getattr("base_gate")?.clone();
            }
        } else if gate_type == circuit_instruction_types::ANNOTATED_OPERATION {
            if let OperationRef::Operation(operation) = operation.view() {
                has_definition = false; // just making sure
                base_gate = operation.operation.bind(py).getattr("base_op")?.clone();
            }
        } else {
            match operation.view() {
                // all-around catch for "operation" field; should be easier once we switch from python to rust
                OperationRef::Gate(pygate) => {
                    let gate = pygate.gate.bind(py);
                    match getattr_or_none(gate, "definition") {
                        None => (),
                        Some(definition) => {
                            has_definition = true;
                            data = serialize(&pack_circuit(
                                &mut definition.extract()?,
                                Some(py.None().bind(py)),
                                false,
                                qpy_data.version,
                                qpy_data.annotation_handler.annotation_factories,
                            )?);
                        }
                    }
                }
                OperationRef::Instruction(pyinst) => {
                    let inst = pyinst.instruction.bind(py);
                    match getattr_or_none(inst, "definition") {
                        None => (),
                        Some(definition) => {
                            has_definition = true;
                            data = serialize(&pack_circuit(
                                &mut definition.extract()?,
                                Some(py.None().bind(py)),
                                false,
                                qpy_data.version,
                                qpy_data.annotation_handler.annotation_factories,
                            )?);
                        }
                    }
                }
                OperationRef::Operation(pyoperation) => {
                    let operation = pyoperation.operation.bind(py);
                    match getattr_or_none(operation, "definition") {
                        None => (),
                        Some(definition) => {
                            has_definition = true;
                            data = serialize(&pack_circuit(
                                &mut definition.extract()?,
                                Some(py.None().bind(py)),
                                false,
                                qpy_data.version,
                                qpy_data.annotation_handler.annotation_factories,
                            )?);
                        }
                    }
                }
                _ => (),
            }
        }
        let num_qubits = operation.num_qubits();
        let num_clbits = operation.num_clbits();
        if !base_gate.is_none() {
            let instruction = qpy_data
                .circuit_data
                .pack(py, &CircuitInstruction::py_new(&base_gate, None, None)?)?;
            base_gate_raw = serialize(&pack_instruction(
                &instruction,
                custom_instructions_hash,
                new_instructions_list,
                qpy_data,
            )?);
        }
        Ok(formats::CustomCircuitInstructionDefPack {
            gate_type,
            num_qubits,
            num_clbits,
            custom_definition: has_definition as u8,
            num_ctrl_qubits,
            ctrl_state,
            name: name.to_string(),
            data,
            base_gate_raw,
        })
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
    circuit: &mut QuantumCircuitData,
    metadata_serializer: Option<&Bound<PyAny>>,
    use_symengine: bool,
    version: u32,
    annotation_factories: &Bound<PyDict>,
) -> PyResult<formats::QPYFormatV15> {
    let mut standalone_var_indices: HashMap<u128, u16> = HashMap::new();
    let standalone_vars =
        pack_standalone_vars(&circuit.data, version, &mut standalone_var_indices)?;
    let annotation_handler = AnnotationHandler::new(annotation_factories);
    let clbits = circuit.data.clbits().clone();
    let mut qpy_data = QPYWriteData {
        circuit_data: &mut circuit.data,
        version,
        _use_symengine: use_symengine,
        clbits: &clbits, // we need to clone since circuit_data might change when serializing custom instructions, explicitly creating the inner instructions
        standalone_var_indices,
        annotation_handler,
    };
    let header = pack_circuit_header(
        circuit.name.clone(),
        circuit.metadata.clone(),
        metadata_serializer,
        &qpy_data,
    )?;
    // Pulse has been removed in Qiskit 2.0. As long as we keep QPY at version 13,
    // we need to write an empty calibrations header since read_circuit expects it
    let calibrations = formats::CalibrationsPack { num_cals: 0 };
    let (instructions, mut custom_instructions_hash) = pack_instructions(&mut qpy_data)?;
    let custom_instructions =
        pack_custom_instructions(&mut custom_instructions_hash, &mut qpy_data)?;
    let layout = pack_layout(circuit.custom_layout.clone(), &qpy_data)?;
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
#[pyo3(name = "write_circuit")]
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
        &mut circuit.extract()?,
        Some(metadata_serializer),
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
