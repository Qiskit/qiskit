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

use hashbrown::HashMap;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyIterator, PyList, PyTuple};
use qiskit_circuit::bit::{PyClassicalRegister, PyClbit, ShareableClbit};
use qiskit_circuit::circuit;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::imports::{BARRIER, DELAY, MEASURE, RESET, PAULI_EVOLUTION_GATE, BLUEPRINT_CIRCUIT, CONTROL_FLOW_SWITCH_CASE_OP, CONTROL_FLOW_BOX_OP, BUILTIN_TUPLE, EXPR, CLIFFORD, ANNOTATED_OPERATION};
use qiskit_circuit::operations::Param;
use qiskit_circuit::operations::{Operation, OperationRef, StandardInstruction, ArrayType};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::circuit_instruction::CircuitInstruction;
use std::io::Cursor;
use std::io::Write;
use numpy::ToPyArray;
use uuid::Uuid;

use crate::formats::{
    Bytes, CircuitHeaderV12Pack, CircuitInstructionArgPack, CircuitInstructionV2Pack,
    CustomCircuitInstructionsPack, HeaderData, LayoutV2Pack, QPYFormatV13, RegisterV4Pack,
    CustomCircuitInstructionDefPack, PauliEvolutionDefPack, PackedParam, SparsePauliOpListElemPack,
    ExpressionVarDeclarationPack
};
use crate::params::pack_param;
use crate::value::{serialize, pack_generic_data, pack_standalone_var, expression_var_declaration};
use crate::value::{QPYData, dumps_value, get_circuit_type_key, circuit_instruction_types};
use crate::UnsupportedFeatureForVersion;
use binrw::BinWrite;

const UNITARY_GATE_CLASS_NAME: &str = "UnitaryGate";
type CustomOperationsMap = HashMap<String, PackedOperation>;
type CustomOperationsList = Vec<String>;

// For debugging purposes
fn hex_string(bytes: &Bytes) -> String {
    bytes
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
}

fn getattr_or_none<'py>(py_object: &'py Bound<'py, PyAny>, name: &str) -> PyResult<Option<Bound<'py, PyAny>>>{
    if py_object.hasattr(name)? {
        let attr = py_object.getattr(name)?;
        if attr.is_none() {
            Ok(None)
        } else {
            Ok(Some(attr))
        }
    } else {
        Ok(None)
    }
}
fn is_python_gate(py: Python, op:&PackedOperation, python_gate: &Bound<PyAny>) -> PyResult<bool> {
    match op.view() {
        OperationRef::Gate(pygate) => {
            if pygate.gate.bind(py).is_instance(python_gate)? {
                Ok(true)
            } else {
                Ok(false)
            }
        }
        _ => Ok(false)
    }
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

fn gate_class_name(py: Python, op: &PackedOperation) -> PyResult<String> {
    let name = match op.view() {
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
        OperationRef::Unitary(_) => Ok(UNITARY_GATE_CLASS_NAME.to_string()),
        OperationRef::Operation(py_op) => py_op
            .operation
            .bind(py)
            .getattr(intern!(py, "__class__"))?
            .getattr(intern!(py, "__name__"))?
            .extract::<String>(),
    }?;
    Ok(name)
}

fn gate_label(inst: &PackedInstruction) -> Bytes {
    match inst.label() {
        Some(label) => label.as_bytes().to_vec(),
        None => Vec::new(),
    }
}

fn get_num_ctrl_qubits(py: Python, op: &PackedOperation) -> PyResult<u32> {
    match op.view() {
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

fn get_ctrl_state(py: Python, op: &PackedOperation, num_ctrl_qubits: u32) -> PyResult<u32> {
    match op.view() {
        OperationRef::Gate(py_gate) => py_gate.gate.getattr(py, "ctrl_state")?.extract::<u32>(py),
        OperationRef::Instruction(py_inst) => py_inst
            .instruction
            .getattr(py, "ctrl_state")?
            .extract::<u32>(py),
        _ => Ok(2u32.pow(num_ctrl_qubits) - 1),
    }
}

fn get_instruction_params(py: Python, instruction: &PackedInstruction, circuit_data: &CircuitData, qpy_data: &QPYData) -> PyResult<Vec<PackedParam>> {
    // The instruction params we store are about being able to reconstruct the objects; they don't
    // necessarily need to match one-to-one to the `params` field.
    // TODO: handle these cases
    if let OperationRef::Instruction(inst) = instruction.op.view() {
        if inst.instruction.bind(py).is_instance(CONTROL_FLOW_SWITCH_CASE_OP.get_bound(py))?{
            let op = inst.instruction.bind(py);
            let target = op.getattr("target")?;
            let cases = op.call_method0("cases_specifier")?;
            let cases_tuple = BUILTIN_TUPLE.get_bound(py).call1((cases,))?;
            return Ok(vec![
                pack_param(py, &target.extract::<Param>()?, qpy_data)?,
                pack_param(py, &cases_tuple.extract::<Param>()?, qpy_data)?
            ]);
        }
        if inst.instruction.bind(py).is_instance(CONTROL_FLOW_BOX_OP.get_bound(py))?{
            let op = inst.instruction.bind(py);
            let first_block = op.getattr("blocks")?
            .try_iter()?
            .next()
            .transpose()?
            .ok_or_else(|| PyErr::new::<PyValueError, _>("No blocks in box control flow op"))?;
            let duration = op.getattr("duration")?;
            let unit = op.getattr("unit")?;
            return Ok(vec![
                pack_param(py, &first_block.extract::<Param>()?, qpy_data)?,
                pack_param(py, &duration.extract::<Param>()?, qpy_data)?,
                pack_param(py, &unit.extract::<Param>()?, qpy_data)?
            ]);
        }
    }

    if let OperationRef::Operation(op) = instruction.op.view() {
        if op.operation.bind(py).is_instance(CLIFFORD.get_bound(py))?{
            let op = op.operation.bind(py);
            let tableau = op.getattr("tableau")?;
            return Ok(vec![
                pack_param(py, &tableau.extract::<Param>()?, qpy_data)?,
            ]);
        }
        if op.operation.bind(py).is_instance(ANNOTATED_OPERATION.get_bound(py))?{
            let op = op.operation.bind(py);
            let modifiers = op.getattr("modifiers")?;
            return Ok(modifiers
                .try_iter()?
                .map(|modifier| pack_param(py, &modifier?.extract::<Param>()?, qpy_data))
                .collect::<PyResult<_>>()?
            );
        }
    }

    // elif isinstance(instruction.operation, AnnotatedOperation):
    //instruction_params = instruction.operation.modifiers
    if let OperationRef::Unitary(unitary) = instruction.op.view() {
        // unitary gates are special since they are uniquely determined by a matrix, which is not
        // a "parameter", strictly speaking, but is treated as such when serializing
        //return unitary.array;

        // until we change the QPY version or verify we get the exact same result,
        // we translate the matrix to numpy and then serialize it like python does
        let out_array = match &unitary.array {
            ArrayType::NDArray(arr) => arr.to_pyarray(py),
            ArrayType::OneQ(arr) => arr.to_pyarray(py),
            ArrayType::TwoQ(arr) => arr.to_pyarray(py),
        };
        return Ok(vec![pack_param(py, &out_array.extract::<Param>()?, qpy_data)?]);
    }
    Ok(instruction
        .params_view()
        .iter()
        .map(|x| pack_param(py, &x, qpy_data))
        .collect::<PyResult<_>>()?
    )
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
    qpy_data: &QPYData,
) -> PyResult<(u8, u16, i64, Bytes)> {
    match getattr_or_none(inst.bind(py), "_condition")? {
        None => Ok((0, 0, 0, Vec::new())),
        Some(condition) => {
            if condition.is_instance(EXPR.get_bound(py))? {
                let condition_raw = serialize(&pack_generic_data(py,&condition, qpy_data)?)?;
                Ok((condition_types::EXPRESSION, 0, 0, condition_raw))
            }
            else if condition.is_instance_of::<PyTuple>() {
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
                Ok((
                    condition_type,
                    condition_register.len() as u16,
                    condition_value,
                    condition_register,
                ))
            } else {
                Err(PyErr::new::<PyValueError, _>("Expression handling not implemented for get_condition_data_from_inst"))
            }
        }
    }
}

fn get_condition_data(
    py: Python,
    op: &PackedOperation,
    circuit_data: &CircuitData,
    qpy_data: &QPYData,
) -> PyResult<(u8, u16, i64, Bytes)> {
    let default_return_value = (condition_types::NONE, 0, 0, Vec::new());
    match op.view() {
        OperationRef::Instruction(py_inst) => {
            get_condition_data_from_inst(py, &py_inst.instruction, circuit_data, qpy_data)
        }
        _ => Ok(default_return_value), // we assume only PyInstructions have condition data at this stage
    }
}

fn recognize_custom_operation(py:Python, op: &PackedOperation, name: &String) -> PyResult<Option<String>>{
    let library = py.import("qiskit.circuit.library")?;
    let circuit_mod = py.import("qiskit.circuit")?;
    let controlflow = py.import("qiskit.circuit.controlflow")?;
    
    if (!library.hasattr(name)?
        && !circuit_mod.hasattr(name)?
        && !controlflow.hasattr(name)?
        && name != "Clifford")
        || name == "Gate"
        || name == "Instruction"
        || is_python_gate(py, op, BLUEPRINT_CIRCUIT.get_bound(py))? {
            // Assign a uuid to each instance of a custom operation
            let new_name = if !["ucrx_dg", "ucry_dg", "ucrz_dg"].contains(&op.name()) {
                format!("{}_{}", &op.name(), Uuid::new_v4().as_simple().to_string())
            } else {
                // ucr*_dg gates can have different numbers of parameters,
                // the uuid is appended to avoid storing a single definition
                // in circuits with multiple ucr*_dg gates. For legacy reasons
                // the uuid is stored in a different format as this was done
                // prior to QPY 11.
                format!("{}_{}", &op.name(), Uuid::new_v4())
            };
            return Ok(Some(new_name));
        }

    if ["ControlledGate", "AnnotatedOperation"].contains(&name.as_str()) {
        return Ok(Some(format!("{}_{}", op.name(), Uuid::new_v4())));
    }

    if is_python_gate(py, op, PAULI_EVOLUTION_GATE.get_bound(py))? {
        return Ok(Some(format!("###PauliEvolutionGate_{}", Uuid::new_v4())));
    }

    Ok(None)
}

fn pack_instruction(
    py: Python,
    instruction: &PackedInstruction,
    circuit_data: &CircuitData,
    custom_operations: &mut CustomOperationsMap,
    new_custom_operations: &mut CustomOperationsList,
    qpy_data: &QPYData,    
) -> PyResult<CircuitInstructionV2Pack> {
    let mut class_name = gate_class_name(py, &instruction.op)?;
    println!("pack_instruction called for {:?} {:?}", &instruction.op, &class_name);
    match recognize_custom_operation(py, &instruction.op, &class_name)? {
        Some(new_name) => {
            class_name = new_name;
            new_custom_operations.push(class_name.clone());
            custom_operations.insert(class_name.clone(), instruction.op.clone());
        },
        None => (),
    }
    let label_raw = gate_label(instruction);
    let num_ctrl_qubits = get_num_ctrl_qubits(py, &instruction.op).unwrap_or(0);
    let ctrl_state = get_ctrl_state(py, &instruction.op, num_ctrl_qubits).unwrap_or(0);
    let params: Vec<PackedParam> = get_instruction_params(py, instruction, circuit_data, qpy_data)?;
    let bit_data = get_packed_bit_list(instruction, circuit_data);
    let (conditional_key, condition_register_size, condition_value, condition_raw) =
        get_condition_data(py, &instruction.op, circuit_data, qpy_data)?;
    println!("params = {:?}", hex_string(&serialize(&params)?));
    println!("pack_instruction DONE");
    Ok(CircuitInstructionV2Pack {
        name_size: class_name.len() as u16,
        label_size: label_raw.len() as u16,
        num_parameters: params.len() as u16,
        num_qargs: instruction.op.num_qubits(),
        num_cargs: instruction.op.num_clbits(),
        conditional_key,
        condition_register_size,
        condition_value,
        num_ctrl_qubits,
        ctrl_state,
        gate_class_name: class_name.into_bytes(),
        label_raw,
        condition_raw,
        bit_data,
        params,
    })
}

fn serialize_instruction(
    py: Python,
    circuit_instruction: &PackedInstruction,
    circuit_data: &CircuitData,
    custom_operations: &mut CustomOperationsMap,
    new_custom_operations: &mut CustomOperationsList,
    qpy_data: &QPYData,
) -> PyResult<Bytes> {
    let mut buffer = Cursor::new(Vec::new());
    let packed_instruction = pack_instruction(py, circuit_instruction, &circuit_data, custom_operations, new_custom_operations, qpy_data)?;
    packed_instruction.write(&mut buffer).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("BinRW write failed: {e}"))
    })?;
    Ok(buffer.into_inner())
}

pub fn pack_instructions(
    py: Python,
    circuit_data: &CircuitData,
    qpy_data: &QPYData,
) -> PyResult<(Vec<CircuitInstructionV2Pack>, CustomOperationsMap)> {
    let mut custom_operations: CustomOperationsMap = HashMap::new();
    let mut custom_new_operations: CustomOperationsList = Vec::new();
    println!("Running pack instructions on {:?}", circuit_data.data());
    Ok((circuit_data
        .data()
        .iter()
        .map(|instruction| Ok(pack_instruction(py, instruction, circuit_data, &mut custom_operations, &mut custom_new_operations, qpy_data)?))
        .collect::<PyResult<_>>()?,
        custom_operations))
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
        match getattr_or_none(&bit_val, "_index")? {
            None => (),
            Some(value) => if value.extract::<usize>()? != index {
                standalone = false
            }
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
    bits: &Bound<PyList>,
) -> PyResult<(u32, Bytes)> {
    let bitmap = PyDict::new(py);
    let out_circ_regs = PyList::new(py, Vec::<PyObject>::new())?;

    bits
    .iter()
    .enumerate()
    .try_for_each(|(index, bit)| -> PyResult<()> {
        bitmap.set_item(&bit, index)?;
        match getattr_or_none(&bit, "_register")? {
            None => Ok(()),
            Some(register) => {
                if !(in_circ_regs.contains(&register).unwrap_or(false))
                && !(out_circ_regs.contains(&register).unwrap_or(true))
                {
                out_circ_regs.append(register)?;
                    }
            Ok(())
            }
        }
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
    qpy_data: &QPYData,
) -> PyResult<HeaderData> {
    let circuit_name = circuit.getattr("name")?.extract::<String>()?.into_bytes();
    let metadata_raw = serialize_metadata(py, &circuit.getattr("metadata")?, metadata_serializer)?;
    let (global_phase_type, global_phase_data) =
        dumps_value(py, &circuit.getattr("global_phase")?, qpy_data)?;
    let (num_qregs, qregs_raw) =
        serialize_registers(py, &circuit.getattr("qregs")?, circuit.getattr("qubits")?.downcast::<PyList>()?)?;
    let (num_cregs, cregs_raw) =
        serialize_registers(py, &circuit.getattr("cregs")?, circuit.getattr("clbits")?.downcast::<PyList>()?)?;
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
            extra_registers_data: Bytes::new(),
            array_data: Bytes::new(),
        })
    } else {
        pack_custom_layout(circuit)
    }
}

fn pack_custom_layout(circuit: &Bound<PyAny>) -> PyResult<LayoutV2Pack>{
    let layout = circuit.getattr("layout")?;
    let mut initial_layout_size = -1; // initial_size
    let py = circuit.py();
    let input_qubit_mapping = PyDict::new(py);
    let initial_layout_array = PyList::empty(py);
    let extra_registers = PyDict::new(py);

    let initial_layout = layout.getattr("initial_layout")?;
    let num_qubits: usize = circuit.getattr("num_qubits")?.extract()?;
    if !initial_layout.is_none() {
        initial_layout_size = initial_layout.call_method0("__len__")?.extract::<i32>()?;
        let layout_mapping = initial_layout.call_method0("get_physical_bits")?;
        for i in 0..num_qubits {
            let qubit = layout_mapping.get_item(i)?;
            input_qubit_mapping.set_item(&qubit, i)?;
            let register = qubit.getattr("_register")?;
            let index = qubit.getattr("_index")?;
            if !register.is_none() || !index.is_none() {
                if !circuit.getattr("qregs")?.contains(&register)? {
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
        input_mapping_size  = layout_input_qubit_mapping.call_method0("__len__")?.extract()?;
        input_qubit_mapping_array = PyList::new(py, std::iter::repeat(py.None()).take(input_mapping_size as usize).collect::<Vec<_>>())?;
        let layout_mapping = initial_layout.call_method0("get_virtual_bits")?;
        for (qubit, index) in layout_input_qubit_mapping.downcast::<PyDict>()? {
            let register = qubit.getattr("_register")?;
            if !register.is_none() && !qubit.getattr("_index")?.is_none() {
                if !circuit.getattr("qregs")?.contains(&register)? {
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
                // this was originally like that in the python code, but I suspect maybe we should use the `_index` field here instead
                
            }
            input_qubit_mapping_array.set_item(index.extract()?, layout_mapping.get_item(&qubit)?)?; 
        };   
    }
   
    let mut final_layout_size = -1;
    let final_layout_array = PyList::empty(py);
    let final_layout = layout.getattr("final_layout")?;
    if !final_layout.is_none(){
        final_layout_size = final_layout.call_method0("__len__")?.extract()?;
        let final_layout_physical = final_layout.call_method0("get_physical_bits")?;
        for i in 0..num_qubits {
            let virtual_bit = final_layout_physical.downcast::<PyList>()?.get_item(i)?;
            final_layout_array.append(circuit
                .call_method1("find_bit", (virtual_bit,))?
                .getattr("index")?
            )?;
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
    let (_, extra_registers_data) = serialize_registers(py, &extra_registers.keys(), &PyList::new(py, bits)?)?;
    let mut buffer = Cursor::new(Vec::new());
    for item in initial_layout_array {
        let tuple = item.downcast::<PyTuple>()?;
        let index = tuple.get_item(0)?;
        let register = tuple.get_item(1)?;
        let reg_name_bytes = if !register.is_none() {Some(register.getattr("name")?.extract::<String>()?)} else {None};
        let index_value = if index.is_none() {-1} else {index.extract::<i32>()?};
        let reg_name_length = reg_name_bytes.as_ref().map(|name| name.len() as i32).unwrap_or(-1);
        buffer.write_all(&index_value.to_be_bytes())?;
        buffer.write_all(&reg_name_length.to_be_bytes())?;
        match reg_name_bytes {Some(name) => buffer.write_all(name.as_bytes()), None => Ok(())}?;
    }

    for i in &input_qubit_mapping_array {
        buffer.write_all(&i.extract::<u32>()?.to_be_bytes())?;
    }

    for i in &final_layout_array {
        buffer.write_all(&i.extract::<u32>()?.to_be_bytes())?;
    }

    let array_data = buffer.into_inner(); // TODO this can be probably done using a struct, not dumping items to a buffer

    Ok(LayoutV2Pack {
        exists: true as u8,
        initial_layout_size,
        input_mapping_size,
        final_layout_size,
        extra_registers_length: extra_registers.len() as u32,
        input_qubit_count,
        extra_registers_data,
        array_data,
    })  
}

fn serialize_sparse_pauli_op(py: Python, operator: &Bound<PyAny>, qpy_data: &QPYData) -> PyResult<Bytes> {
    let op_as_np_list = operator.call_method1("to_list", (true,))?;
    let (_, data) = dumps_value(py, &op_as_np_list, qpy_data)?;
    let pauli_pack = SparsePauliOpListElemPack {
        size: data.len() as u64,
        data,
    };
    let mut buffer = Cursor::new(Vec::new());
    pauli_pack.write(&mut buffer).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("BinRW write failed: {e}"))
    })?;
    Ok(buffer.into_inner())
}

fn serialize_pauli_evolution_gate(py: Python, evolution_gate: &Bound<PyAny>, qpy_data: &QPYData) -> PyResult<Bytes>{
    println!("RUST: Serializing PAULI {:?}", evolution_gate);
    let operators = evolution_gate.getattr("operator")?;
    let mut standalone = false;
    let operator_list: Bound<PyList> = if !operators.is_instance_of::<PyList>() {
        standalone = true;
        PyList::new(py, [operators])?
    } else {
        operators.downcast()?.clone()
    };
    let operator_size = operator_list.call_method0("__len__")?.extract::<u64>()?;
    
    let pauli_data: Vec<Bytes> = operator_list
    .iter()
    .map(|operator| Ok(serialize_sparse_pauli_op(py, &operator, qpy_data)?))
    .collect::<PyResult<_>>()?;

    let (time_type, time_data) = dumps_value(py, &evolution_gate.getattr("time")?, qpy_data)?;
    println!("time: {:?}, time type: {:?}, time_data: {:?}", &evolution_gate.getattr("time")?, time_type, hex_string(&time_data));
    let synth_class = evolution_gate.getattr("synthesis")?.get_type().getattr("__name__")?;
    let settings_dict = evolution_gate.getattr("synthesis")?.getattr("settings")?;
    let json = py.import("json")?;
    let args = PyDict::new(py);
    args.set_item("class", synth_class)?;
    args.set_item("settings", settings_dict)?;
    let synth_data = json.call_method1("dumps", (args,))?.extract::<String>()?.as_bytes().to_vec();
    
    let standalone_op = standalone as u8;
    let packed_pauli_data = PauliEvolutionDefPack {
        operator_size,
        standalone_op,
        time_type,
        time_size: time_data.len() as u64,
        synth_method_size: synth_data.len() as u64,
        pauli_data,
        time_data,
        synth_data
    };
    let mut buffer = Cursor::new(Vec::new());
    packed_pauli_data.write(&mut buffer).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("BinRW write failed: {e}"))
    })?;
    Ok(buffer.into_inner())
    
}
fn pack_custom_instruction(
    py: Python,
    name: &String,
    custom_instructions_hash: &mut CustomOperationsMap,
    new_instructions_list: &mut Vec<String>,
    circuit_data: &mut CircuitData,
    qpy_data: &QPYData,
) -> PyResult<CustomCircuitInstructionDefPack> {
    print!("Rust: serializing custom gate: {:?}", name);
    let operation = custom_instructions_hash.get(name).ok_or_else(|| PyErr::new::<PyValueError, _>(format!("Could not find operation data for {}", name)))?.clone();
    let gate_type = get_circuit_type_key(py, &operation)?;
    let mut has_definition = false;
    let mut data: Bytes = Bytes::new();
    let mut num_ctrl_qubits = 0;
    let mut ctrl_state = 0;
    let mut base_gate: Bound<PyAny> = py.None().bind(py).clone();
    let mut base_gate_raw: Bytes = Bytes::new();
    
    if gate_type == circuit_instruction_types::PAULI_EVOL_GATE {
        println!("Pauli evol case");
        println!("operation.view() {:?}",operation.view());
        if let OperationRef::Gate(gate) = operation.view() {
            println!("Pauli evol GATE case");
            has_definition = true;
            data = serialize_pauli_evolution_gate(py,&gate.gate.bind(py), qpy_data)?;
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
            data = serialize_circuit(py, &gate.getattr("_definition")?, py.None().bind(py), false, qpy_data.version)?;
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
        match operation.view(){ // all-around catch for "operation" field; should be easier once we switch from python to rust            
            OperationRef::Gate(pygate) => {
                let gate = pygate.gate.bind(py);
                gate.getattr("definition")?; // this creates the _definition field
                println!("Gate case");
                match getattr_or_none(gate, "definition")? {
                    None => (),
                    Some(definition) => {
                        has_definition = true;
                        data = serialize_circuit(py, &definition, py.None().bind(py), false, qpy_data.version)?;
                    }
                }
            }
            OperationRef::Instruction(pyinst) => {
                let inst = pyinst.instruction.bind(py);
                inst.getattr("definition")?; // this creates the _definition field
                println!("Inst case");
                match getattr_or_none(inst, "definition")? {
                    None => (),
                    Some(definition) => {
                        has_definition = true;
                        data = serialize_circuit(py, &definition, py.None().bind(py), false, qpy_data.version)?;
                    }
                }
            }
            OperationRef::Operation(pyoperation) => {
                let operation = pyoperation.operation.bind(py);
                operation.getattr("definition")?; // this creates the _definition field
                println!("Op case");
                match getattr_or_none(operation, "definition")? {
                    None => (),
                    Some(definition) => {
                        has_definition = true;
                        data = serialize_circuit(py, &definition, py.None().bind(py), false, qpy_data.version)?;
                    }
                }
            }
            _ => ()
        }
    }

    if !base_gate.is_none(){
        let instruction = circuit_data.pack(py, &CircuitInstruction::py_new(&base_gate, None, None)?)?;
        base_gate_raw = serialize_instruction(py, &instruction, circuit_data, custom_instructions_hash, new_instructions_list, qpy_data)?;
    }
    let name_raw = name.as_bytes().to_vec();
    Ok(CustomCircuitInstructionDefPack{
        gate_name_size: name_raw.len() as u16,
        gate_type,
        num_qubits: operation.num_qubits(),
        num_clbits: operation.num_clbits(),
        custom_definition: has_definition as u8,
        size: data.len() as u64,
        num_ctrl_qubits,
        ctrl_state,
        base_gate_size: base_gate_raw.len() as u64,
        name_raw,
        data,
        base_gate_raw
    })
}

fn pack_custom_instructions(py: Python, custom_instructions_hash: &mut CustomOperationsMap, circuit_data: &mut CircuitData, qpy_data: &QPYData) -> PyResult<CustomCircuitInstructionsPack> {
    let mut custom_instructions: Vec<CustomCircuitInstructionDefPack> = Vec::new();
    let mut instructions_to_pack: Vec<String> = custom_instructions_hash.keys().cloned().collect();
    while let Some(name) = instructions_to_pack.pop() {
        custom_instructions.push(pack_custom_instruction(py, &name, custom_instructions_hash, &mut instructions_to_pack, circuit_data, qpy_data)?);
    }
    Ok(CustomCircuitInstructionsPack {
        custom_operations_length: custom_instructions.len() as u64,
        custom_instructions
    })
}

fn pack_standalone_vars(circuit: &Bound<PyAny>, version: u32, standalone_var_indices: &Bound<PyDict>) -> PyResult<Vec<ExpressionVarDeclarationPack>>{
    let mut result = Vec::new();
    let mut index = 0;
    for item in circuit.call_method0("iter_input_vars")?.try_iter()? {
        let var = item?;
        result.push(pack_standalone_var(&var, expression_var_declaration::INPUT, version)?);
        standalone_var_indices.set_item(&var, index)?;
        index += 1;
    }
    for item in circuit.call_method0("iter_captured_vars")?.try_iter()? {
        let var = item?;
        result.push(pack_standalone_var(&var, expression_var_declaration::CAPTURE, version)?);
        standalone_var_indices.set_item(&var, index)?;
        index += 1;
    }
    for item in circuit.call_method0("iter_declared_vars")?.try_iter()? {
        let var = item?;
        result.push(pack_standalone_var(&var, expression_var_declaration::LOCAL, version)?);
        standalone_var_indices.set_item(&var, index)?;
        index += 1;
    }
    if version < 14 {
        match getattr_or_none(circuit, "num_stretches")? {
            None => (),
            Some(value) => {
                if value.extract::<usize>()? > 0 {
                    return Err(UnsupportedFeatureForVersion::new_err(("circuits containing stretch variables", 14, version)));
                }
                ()
            }
        }
    }
    for item in circuit.call_method0("iter_captured_stretches")?.try_iter()? {
        let var = item?;
        result.push(pack_standalone_var(&var, expression_var_declaration::STRETCH_CAPTURE, version)?);
        standalone_var_indices.set_item(&var, index)?;
        index += 1;
    }
    for item in circuit.call_method0("iter_declared_stretches")?.try_iter()? {
        let var = item?;
        result.push(pack_standalone_var(&var, expression_var_declaration::STRETCH_LOCAL, version)?);
        standalone_var_indices.set_item(&var, index)?;
        index += 1;
    }
    Ok(result)
}

fn pack_circuit(
    py: Python,
    circuit: &Bound<PyAny>,
    metadata_serializer: &Bound<PyAny>,
    use_symengine: bool,
    version: u32,
) -> PyResult<QPYFormatV13> {
    println!("pack_circuit called for circuit {:?}", circuit);
    circuit.getattr("data")?; // in case _data is lazily generated in python
    let mut circuit_data = circuit.getattr("_data")?.extract::<CircuitData>()?;
    let clbit_indices = circuit_data.get_clbit_indices(py).clone();
    let standalone_var_indices = PyDict::new(py);
    println!("Calling pack_standalone_vars");
    let standalone_vars = pack_standalone_vars(circuit, version, &standalone_var_indices)?;
    println!("Calling pack_standalone_vars DONE");
    let qpy_data = QPYData {
        version,
        use_symengine,
        clbit_indices,
        standalone_var_indices: standalone_var_indices.unbind()
    };
    let header_raw = pack_circuit_header(py, circuit, metadata_serializer, &qpy_data)?;
    // TODO: still need to implement write_standalone_vars
    // Pulse has been removed in Qiskit 2.0. As long as we keep QPY at version 13,
    // we need to write an empty calibrations header since read_circuit expects it
    let calibration_header_value: u16 = 0;
    let (instructions, mut custom_instructions_hash) = pack_instructions(py, &circuit_data, &qpy_data)?;
    let custom_instructions = pack_custom_instructions(py, &mut custom_instructions_hash, &mut circuit_data, &qpy_data)?;
    
    Ok(QPYFormatV13 {
        header: header_raw,
        standalone_vars,
        custom_instructions,
        instructions,
        calibration_header: calibration_header_value.to_be_bytes().to_vec(),
        layout: pack_layout(circuit)?,
    })
}

pub fn serialize_circuit(
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

