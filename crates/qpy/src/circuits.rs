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
use numpy::ToPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyString;
use pyo3::types::{PyAny, PyDict, PyIterator, PyList, PyTuple, PyBytes};
use qiskit_circuit::bit::Register;
use qiskit_circuit::bit::ShareableQubit;
use qiskit_circuit::bit::{QuantumRegister, ClassicalRegister};
use qiskit_circuit::bit::{PyClassicalRegister, PyClbit, ShareableClbit};
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::circuit_instruction::CircuitInstruction;
use qiskit_circuit::imports;

use qiskit_circuit::classical::expr::Expr;
use qiskit_circuit::operations::Param;
use qiskit_circuit::operations::PyGate;
use qiskit_circuit::operations::{ArrayType, Operation, OperationRef, StandardInstruction};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{Clbit, Qubit};
use smallvec::SmallVec;

use uuid::Uuid;

use crate::formats;
use crate::formats::RegisterV4Pack;
use crate::params::pack_param;
use crate::params::unpack_param;
use crate::value::{circuit_instruction_types, dumps_value, get_circuit_type_key, QPYData, DumpedValue};
use crate::value::{expression_var_declaration, pack_generic_data, pack_standalone_var, serialize, deserialize};
use crate::bytes::Bytes;
use crate::UnsupportedFeatureForVersion;
use crate::consts::standard_gate_from_gate_class_name;

const UNITARY_GATE_CLASS_NAME: &str = "UnitaryGate";
type CustomOperationsMap = HashMap<String, PackedOperation>;
type CustomOperationsList = Vec<String>;

pub mod register_types {
    pub const QREG: u8 = b'q';
    pub const CREG: u8 = b'c';
}

// For debugging purposes
fn hex_string(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
}

fn getattr_or_none<'py>(
    py_object: &'py Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
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
fn is_python_gate(py: Python, op: &PackedOperation, python_gate: &Bound<PyAny>) -> PyResult<bool> {
    match op.view() {
        OperationRef::Gate(pygate) => {
            if pygate.gate.bind(py).is_instance(python_gate)? {
                Ok(true)
            } else {
                Ok(false)
            }
        }
        _ => Ok(false),
    }
}

pub mod bit_types {
    pub const QUBIT: u8 = b'q';
    pub const CLBIT: u8 = b'c';

}

fn get_packed_bit_list(
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
            StandardInstruction::Measure => imports::MEASURE
                .get_bound(py)
                .getattr(intern!(py, "__name__"))?,
            StandardInstruction::Delay(_) => imports::DELAY
                .get_bound(py)
                .getattr(intern!(py, "__name__"))?,
            StandardInstruction::Barrier(_) => imports::BARRIER
                .get_bound(py)
                .getattr(intern!(py, "__name__"))?,
            StandardInstruction::Reset => imports::RESET
                .get_bound(py)
                .getattr(intern!(py, "__name__"))?,
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

fn get_instruction_params(
    py: Python,
    instruction: &PackedInstruction,
    qpy_data: &QPYData,
) -> PyResult<Vec<formats::PackedParam>> {
    // The instruction params we store are about being able to reconstruct the objects; they don't
    // necessarily need to match one-to-one to the `params` field.
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
                pack_param(py, &target.extract::<Param>()?, qpy_data)?,
                pack_param(py, &cases_tuple.extract::<Param>()?, qpy_data)?,
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
                .ok_or_else(|| PyErr::new::<PyValueError, _>("No blocks in box control flow op"))?;
            let duration = op.getattr("duration")?;
            let unit = op.getattr("unit")?;
            return Ok(vec![
                pack_param(py, &first_block.extract::<Param>()?, qpy_data)?,
                pack_param(py, &duration.extract::<Param>()?, qpy_data)?,
                pack_param(py, &unit.extract::<Param>()?, qpy_data)?,
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
            return Ok(vec![pack_param(
                py,
                &tableau.extract::<Param>()?,
                qpy_data,
            )?]);
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
                .map(|modifier| pack_param(py, &modifier?.extract::<Param>()?, qpy_data))
                .collect::<PyResult<_>>();
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
        return Ok(vec![pack_param(
            py,
            &out_array.extract::<Param>()?,
            qpy_data,
        )?]);
    }
    instruction
        .params_view()
        .iter()
        .map(|x| pack_param(py, x, qpy_data))
        .collect::<PyResult<_>>()
}

fn dump_register(
    py: Python,
    register: Bound<PyAny>,
    circuit_data: &CircuitData,
) -> PyResult<Bytes> {
    if register.is_instance_of::<PyClassicalRegister>() {
        Ok(register.getattr("name")?.extract::<String>()?.into())
    } else if register.is_instance_of::<PyClbit>() {
        let key = &register.extract::<ShareableClbit>()?;
        let name = circuit_data
            .get_clbit_indices(py)
            .bind(py)
            .get_item(key)?
            .ok_or(PyErr::new::<PyValueError, _>("Clbit not found"))?
            .getattr("index")?
            .str()?;
        let mut bytes: Bytes = Bytes(Vec::with_capacity(name.len()? + 1));
        bytes.push(0u8);
        bytes.extend_from_slice(name.extract::<String>()?.as_bytes());
        Ok(bytes)
    } else {
        Ok(Bytes::new())
    }
}

fn get_condition_data_from_inst(
    py: Python,
    inst: &Py<PyAny>,
    circuit_data: &CircuitData,
    qpy_data: &QPYData,
) -> PyResult<formats::ConditionPack> {
    match getattr_or_none(inst.bind(py), "_condition")? {
        None => Ok(formats::ConditionPack{ key: formats::condition_types::NONE, register_size: 0u16, value:0i64, data: formats::ConditionData::None}),
        Some(condition) => {
            if condition.extract::<Expr>().is_ok() {
                let expression = pack_generic_data(&condition, qpy_data)?;
                Ok(formats::ConditionPack{ key: formats::condition_types::EXPRESSION,
                    register_size: 0u16,
                    value:0i64,
                    data: formats::ConditionData::Expression(expression)})
            } else if condition.is_instance_of::<PyTuple>() {
                let key = formats::condition_types::TWO_TUPLE;
                let value = condition
                    .downcast::<PyTuple>()?
                    .get_item(1)?
                    .extract::<i64>()?;
                let register = dump_register(
                    py,
                    condition.downcast::<PyTuple>()?.get_item(0)?,
                    circuit_data,
                )?;
                Ok(formats::ConditionPack {
                    key,
                    register_size: register.len() as u16,
                    value,
                    data: formats::ConditionData::Register(register),
            })
            } else {
                Err(PyErr::new::<PyValueError, _>(
                    "Expression handling not implemented for get_condition_data_from_inst",
                ))
            }
        }
    }
}

fn get_condition_data(
    py: Python,
    op: &PackedOperation,
    circuit_data: &CircuitData,
    qpy_data: &QPYData,
) -> PyResult<formats::ConditionPack> {
    match op.view() {
        OperationRef::Instruction(py_inst) => {
            get_condition_data_from_inst(py, &py_inst.instruction, circuit_data, qpy_data)
        }
        // we assume only PyInstructions have condition data at this stage
        _ => Ok(formats::ConditionPack{ key: formats::condition_types::NONE, register_size: 0u16, value:0i64, data: formats::ConditionData::None}),
    }
}

fn recognize_custom_operation(
    py: Python,
    op: &PackedOperation,
    name: &String,
) -> PyResult<Option<String>> {
    let library = py.import("qiskit.circuit.library")?;
    let circuit_mod = py.import("qiskit.circuit")?;
    let controlflow = py.import("qiskit.circuit.controlflow")?;

    if (!library.hasattr(name)?
        && !circuit_mod.hasattr(name)?
        && !controlflow.hasattr(name)?
        && name != "Clifford")
        || name == "Gate"
        || name == "Instruction"
        || is_python_gate(py, op, imports::BLUEPRINT_CIRCUIT.get_bound(py))?
    {
        // Assign a uuid to each instance of a custom operation
        let new_name = if !["ucrx_dg", "ucry_dg", "ucrz_dg"].contains(&op.name()) {
            format!("{}_{}", &op.name(), Uuid::new_v4().as_simple())
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

    if ["ControlledGate", "AnnotatedOperation"].contains(&name.as_str())
        || is_python_gate(py, op, imports::MCMT_GATE.get_bound(py))?
    {
        return Ok(Some(format!("{}_{}", op.name(), Uuid::new_v4())));
    }

    if is_python_gate(py, op, imports::PAULI_EVOLUTION_GATE.get_bound(py))? {
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
) -> PyResult<formats::CircuitInstructionV2Pack> {
    let mut gate_class_name = gate_class_name(py, &instruction.op)?;
    if let Some(new_name) = recognize_custom_operation(py, &instruction.op, &gate_class_name)? {
        gate_class_name = new_name;
        new_custom_operations.push(gate_class_name.clone());
        custom_operations.insert(gate_class_name.clone(), instruction.op.clone());
    }
    let label = match instruction.label() {
        Some(label) => String::from(label),
        None => String::from(""),
    };
    let num_ctrl_qubits = get_num_ctrl_qubits(py, &instruction.op).unwrap_or(0);
    let ctrl_state = get_ctrl_state(py, &instruction.op, num_ctrl_qubits).unwrap_or(0);
    let params: Vec<formats::PackedParam> = get_instruction_params(py, instruction, qpy_data)?;
    let bit_data = get_packed_bit_list(instruction, circuit_data);
    let condition =  get_condition_data(py, &instruction.op, circuit_data, qpy_data)?;
    Ok(formats::CircuitInstructionV2Pack {
        num_qargs: instruction.op.num_qubits(),
        num_cargs: instruction.op.num_clbits(),
        num_ctrl_qubits,
        ctrl_state,
        gate_class_name,
        label,
        condition,
        bit_data,
        params,
    })
}

fn get_python_gate_class<'a>(py: Python<'a>, gate_class_name: &String) -> PyResult<Bound<'a, PyAny>> {
    let library = py.import("qiskit.circuit.library")?;
    let control_flow = py.import("qiskit.circuit.controlflow")?;
    if library.hasattr(&gate_class_name)? {
        library.getattr(gate_class_name)
    }
    else if control_flow.hasattr(&gate_class_name)? {
        control_flow.getattr(gate_class_name)
    }
    else {
        Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Gate class not found: {:?}", gate_class_name)))
    }
}

fn deserialize_standard_instruction(instruction: &formats::CircuitInstructionV2Pack) -> Option<StandardInstruction> {
    match instruction.gate_class_name.as_str() {
        "Barrier" => Some(StandardInstruction::Barrier(instruction.num_qargs)),
        "Measure" => Some(StandardInstruction::Measure),
        "Reset" => Some(StandardInstruction::Reset),
        // TODO: where is the delay unit stored? Can't find how it's treated in the python qpy. Using the default dt for now
        "Delay" => Some(StandardInstruction::Delay(qiskit_circuit::operations::DelayUnit::DT)), 
        _ => None,
    }
}
fn unpack_instruction(py: Python, instruction: &formats::CircuitInstructionV2Pack, qpy_data: &QPYData) -> PyResult<Instruction> {
    println!("unpack_instruction {:?}", instruction);
    let name = &instruction.gate_class_name;
    let params: Vec<Param> = instruction.params.iter()
    .map(|packed_param| unpack_param(py, packed_param, &qpy_data))
    .collect::<PyResult<_>>()?;
    println!("params: {:?}", params);
    let op = if let Some(gate) = standard_gate_from_gate_class_name(name.as_str()) {
        println!("Standard gate: {:?}", name);
        PackedOperation::from_standard_gate(gate)
    } else if let Some(std_instruction) = deserialize_standard_instruction(instruction) {
        println!("Standard instruction: {:?}", name);
        PackedOperation::from_standard_instruction(std_instruction)
    } else {
        println!("Non standard gate: {:?}", name);
        println!("params: {:?}", instruction.params);
        let gate_class = get_python_gate_class(py, &instruction.gate_class_name)?;
        let py_params: Vec<Bound<'_, PyAny>> = params.iter().map(|param: &Param| param.into_pyobject(py))
        .collect::<PyResult<_>>()?;
        let gate_object = match name.as_str() {
            "IfElseOp" | "WhileLoopOp" => {
                // TODO: should load condition and do gate_class(condition, *params, label=label)
                let args = PyTuple::new(py, &py_params)?;
                gate_class.call1(args)?
            }
            _ => {
                let args = PyTuple::new(py, &py_params)?;
                gate_class.call1(args)?
            }
        };
        
        // let gate_object = gate_class.call0()?;
        let pygate = PyGate {
            qubits: instruction.num_qargs,
            clbits: instruction.num_cargs, 
            params: instruction.params.len() as u32,
            op_name: name.clone(),
            gate: gate_object.unbind()
        };
        PackedOperation::from_gate(Box::new(pygate))
    };
    let mut qubits = Vec::new();
    let mut clbits = Vec::new();
    for arg in &instruction.bit_data {
        match arg.bit_type {
            bit_types::QUBIT => qubits.push(Qubit(arg.index)),
            bit_types::CLBIT => clbits.push(Clbit(arg.index)),
            _ =>  return Err(PyErr::new::<PyValueError, _>("Unrecognized bit type",)),
        };
    }
    Ok((op, SmallVec::from(params), qubits, clbits))
}


pub fn pack_instructions(
    py: Python,
    circuit_data: &CircuitData,
    qpy_data: &QPYData,
) -> PyResult<(Vec<formats::CircuitInstructionV2Pack>, CustomOperationsMap)> {
    let mut custom_operations: CustomOperationsMap = HashMap::new();
    let mut custom_new_operations: CustomOperationsList = Vec::new();
    Ok((
        circuit_data
            .data()
            .iter()
            .map(|instruction| {
                pack_instruction(
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

fn serialize_metadata(
    metadata: &Bound<PyAny>,
    metadata_serializer: &Bound<PyAny>,
) -> PyResult<Bytes> {
    let py = metadata.py();
    let json = py.import("json")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("separators", PyTuple::new(py, [",", ":"])?)?;
    kwargs.set_item("cls", metadata_serializer)?;
    Ok(json
        .call_method("dumps", (metadata,), Some(&kwargs))?
        .extract::<String>()?
        .into())
}


fn deserialize_metadata(
    py: Python,
    metadata_bytes: Bytes,
    metadata_deserializer: &Bound<PyAny>,
) -> PyResult<PyObject> {
    let json = py.import("json")?;
    let kwargs: Bound<'_, PyDict> = PyDict::new(py);
    kwargs.set_item("cls", metadata_deserializer)?;
    let metadata_string = PyString::new(py, (&metadata_bytes).try_into()?);
    Ok(json.call_method("loads", (metadata_string,), Some(&kwargs))?.unbind())
}



fn pack_register(
    register: &Bound<PyAny>,
    bitmap: &Bound<PyDict>,
    is_in_circuit: bool,
) -> PyResult<formats::RegisterV4Pack> {
    let reg_name = register.getattr("name")?.extract::<String>()?.into();
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
            Some(value) => {
                if value.extract::<usize>()? != index {
                    standalone = false
                }
            }
        }

        if let Some(index) = bitmap.get_item(bit_val)? {
            bit_indices.push(index.extract::<i64>()?);
        } else {
            bit_indices.push(-1);
        }
    }
    let packed_reg = formats::RegisterV4Pack {
        register_type: reg_type,
        standalone: standalone as u8,
        in_circuit: is_in_circuit as u8,
        name: reg_name,
        bit_indices,
    };
    Ok(packed_reg)
}

fn pack_registers(
    in_circ_regs: &Bound<PyAny>,
    bits: &Bound<PyList>,
) -> PyResult<Vec<RegisterV4Pack>> {
    let py = in_circ_regs.py();
    let bitmap = PyDict::new(py);
    let out_circ_regs = PyList::new(py, Vec::<PyObject>::new())?;

    bits.iter()
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
    let mut result = Vec::new();
    in_circ_regs
        .downcast::<PyList>()?
        .iter()
        .try_for_each(|register| -> PyResult<()> {
            result.push(pack_register(&register, &bitmap, true)?);
            Ok(())
        })?;

    out_circ_regs
        .iter()
        .try_for_each(|register| -> PyResult<()> {
            result.push(pack_register(&register, &bitmap, false)?);
            Ok(())
        })?;
    Ok(result)
}

fn pack_circuit_header(
    circuit: &Bound<PyAny>,
    metadata_serializer: &Bound<PyAny>,
    qpy_data: &QPYData,
) -> PyResult<formats::CircuitHeaderV12Pack> {
    let py = circuit.py();
    let circuit_name = circuit
        .getattr(intern!(py, "name"))?
        .extract::<String>()?;
    let metadata = serialize_metadata(
        &circuit.getattr(intern!(py, "metadata"))?,
        metadata_serializer,
    )?;
    let global_phase_data = DumpedValue::from(&circuit.getattr(intern!(py, "global_phase"))?, qpy_data)?;
    let qregs = pack_registers(
        &circuit.getattr(intern!(py, "qregs"))?,
        circuit
            .getattr(intern!(py, "qubits"))?
            .downcast::<PyList>()?,
    )?;
    let cregs= pack_registers(
        &circuit.getattr(intern!(py, "cregs"))?,
        circuit
            .getattr(intern!(py, "clbits"))?
            .downcast::<PyList>()?,
    )?;
    let mut registers = qregs;
    registers.extend(cregs);
    let header = formats::CircuitHeaderV12Pack {
        num_qubits: circuit
            .getattr(intern!(py, "num_qubits"))?
            .extract::<u32>()?,
        num_clbits: circuit
            .getattr(intern!(py, "num_clbits"))?
            .extract::<u32>()?,
        num_instructions: circuit
            .call_method0(intern!(py, "__len__"))?
            .extract::<u64>()?,
        num_vars: circuit
            .getattr(intern!(py, "num_identifiers"))?
            .extract::<u32>()?,
        circuit_name,
        global_phase_data,
        metadata,
        registers,
    };

    Ok(header)
}

fn pack_layout(circuit: &Bound<PyAny>) -> PyResult<formats::LayoutV2Pack> {
    if circuit.getattr(intern!(circuit.py(), "layout"))?.is_none() {
        Ok(formats::LayoutV2Pack {
            exists: 0,
            initial_layout_size: -1,
            input_mapping_size: -1,
            final_layout_size: -1,
            input_qubit_count: 0,
            extra_registers: Vec::new(),
            initial_layout_items: Vec::new(),
            input_mapping_items: Vec::new(),
            final_layout_items: Vec::new(),
        })
    } else {
        pack_custom_layout(circuit)
    }
}

fn unpack_layout<'py>(py: Python<'py>, layout: formats::LayoutV2Pack, circuit_data: &CircuitData) -> PyResult<Option<Bound<'py, PyAny>>> {
    match layout.exists {
        0 => Ok(None),
        _ => Ok(Some(unpack_custom_layout(py, layout, circuit_data)?)),
    }
}

fn pack_custom_layout(circuit: &Bound<PyAny>) -> PyResult<formats::LayoutV2Pack> {
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
                && !circuit.getattr("qregs")?.contains(&register)?
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
        for i in 0..num_qubits {
            let virtual_bit = final_layout_physical.downcast::<PyDict>()?.get_item(i)?;
            final_layout_array.append(
                circuit
                    .call_method1("find_bit", (virtual_bit,))?
                    .getattr("index")?,
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
    let extra_registers = pack_registers(&extra_registers.keys(), &PyList::new(py, bits)?)?;
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
        final_layout_size: final_layout_size as i32,
        input_qubit_count,
        extra_registers,
        initial_layout_items,
        input_mapping_items,
        final_layout_items,
        
    })
}

fn unpack_custom_layout<'py>(py: Python<'py>, layout: formats::LayoutV2Pack, circuit_data: &CircuitData) -> PyResult<Bound<'py, PyAny>> {
    let layout_libray = py.import("qiskit.transpiler.layout")?;
    let transpiler_layout_class = layout_libray.getattr("TranspileLayout")?;
    let layout_class = layout_libray.getattr("Layout")?;

    let mut initial_layout = py.None();
    let mut input_qubit_mapping = py.None();
    let final_layout = py.None();   

    let mut extra_register_map: HashMap<String, QuantumRegister> = HashMap::new();
    let mut existing_register_map: HashMap<String, QuantumRegister> = HashMap::new();
    for packed_register in layout.extra_registers {
        if packed_register.register_type == bit_types::QUBIT {
            println!("extra register {:?}", &packed_register);
            let register = QuantumRegister::new_owning(packed_register.name.clone(), packed_register.bit_indices.len() as u32);
            extra_register_map.insert(packed_register.name.clone(), register);
        }
    }
    // add the registers from the circuit, to streamline the search phase
    for qreg in circuit_data.qregs() {
        println!("existing register {:?}", &qreg);
        existing_register_map.insert(qreg.name().to_string(), qreg.clone()); // TODO: can we avoid cloning?
    }
    let initial_layout_virtual_bits = PyList::new(py, Vec::<PyObject>::new())?;
    for virtual_bit in layout.initial_layout_items {
        let qubit = 
            if let Some(register) = extra_register_map.get(&virtual_bit.register_name) {
                if let Some(qubit) = register.get(virtual_bit.index_value as usize) {
                    qubit
                } else {
                    ShareableQubit::new_anonymous()
                }
            } else if let Some(register) = existing_register_map.get(&virtual_bit.register_name) {
                if let Some(qubit) = register.get(virtual_bit.index_value as usize) {
                    qubit
                } else {
                    ShareableQubit::new_anonymous()
                }
            } else {
                ShareableQubit::new_anonymous()
            };
        println!("For virtual bit {:?} got qubit {:?}", virtual_bit, qubit);
        initial_layout_virtual_bits.append(qubit)?;
    }
    if initial_layout_virtual_bits.len() > 0 {
        initial_layout = layout_class.call_method1("from_qubit_list", (initial_layout_virtual_bits,))?.unbind();
    }

    if layout.input_mapping_size > 0 {
        let input_qubit_mapping_data = PyDict::new(py);
        let physical_bits_object = initial_layout.call_method0(py, "get_physical_bits")?;
        let physical_bits = physical_bits_object.downcast_bound::<PyDict>(py)?;
        for (index, bit) in layout.input_mapping_items.iter().enumerate() {
            let physical_bit = physical_bits
            .get_item(bit)?
            .ok_or(PyErr::new::<PyValueError, _>(format!("Could not get physical bit for bit {:?}", bit)))?;
            input_qubit_mapping_data.set_item(physical_bit, index)?;
        }
        input_qubit_mapping = input_qubit_mapping_data.unbind().as_any().clone();
    }
    
    if layout.final_layout_size > 0 {
        return Err(PyErr::new::<PyValueError, _>("Final layout handling is not implemented"));
        // let final_layout_dict = PyDict::new(py);
        // for (index, bit) in layout.final_layout_items.iter().enumerate() {
            // TODO: not sure what to do here yet
            // layout_dict = {circuit.qubits[bit]: index for index, bit in enumerate(final_layout_array)}

            // let qubit = circuit_data.qubits().get(index)
            // .get_item(bit)?
            // .ok_or(PyErr::new::<PyValueError, _>(format!("Could not get physical bit for bit {:?}", bit)))?;
            
        // final_layout_dict.set_item(physical_bit, index)?;
        // }
    }
    let transpiled_layout = transpiler_layout_class.call1((initial_layout, input_qubit_mapping, final_layout))?;
    // TODO: this is for version >= 10
    if layout.input_qubit_count >= 0 {
        transpiled_layout.setattr("_input_qubit_count", layout.input_qubit_count)?;
        transpiled_layout.setattr("_output_qubit_list", circuit_data.py_qubits(py))?;
    }
    Ok(transpiled_layout)
}

fn pack_sparse_pauli_op(operator: &Bound<PyAny>, qpy_data: &QPYData) -> PyResult<formats::SparsePauliOpListElemPack> {
    let op_as_np_list = operator.call_method1("to_list", (true,))?;
    let (_, data) = dumps_value(&op_as_np_list, qpy_data)?;
    Ok(formats::SparsePauliOpListElemPack {data})
}

fn pack_pauli_evolution_gate(
    evolution_gate: &Bound<PyAny>,
    qpy_data: &QPYData,
) -> PyResult<formats::PauliEvolutionDefPack> {
    let py = evolution_gate.py();
    let operators = evolution_gate.getattr("operator")?;
    let mut standalone = false;
    let operator_list: Bound<PyList> = if !operators.is_instance_of::<PyList>() {
        standalone = true;
        PyList::new(py, [operators])?
    } else {
        operators.downcast()?.clone()
    };
    let pauli_data = operator_list
        .iter()
        .map(|operator| pack_sparse_pauli_op(&operator, qpy_data))
        .collect::<PyResult<_>>()?;

    let (time_type, time_data) = dumps_value(&evolution_gate.getattr("time")?, qpy_data)?;
    let synth_class = evolution_gate
        .getattr("synthesis")?
        .get_type()
        .getattr("__name__")?;
    let settings_dict = evolution_gate.getattr("synthesis")?.getattr("settings")?;
    let json = py.import("json")?;
    let args = PyDict::new(py);
    args.set_item("class", synth_class)?;
    args.set_item("settings", settings_dict)?;
    let synth_data: Bytes = json
        .call_method1("dumps", (args,))?
        .extract::<String>()?
        .into();

    let standalone_op = standalone as u8;
    Ok(formats::PauliEvolutionDefPack {
        standalone_op,
        time_type,
        pauli_data,
        time_data,
        synth_data,
    })
}

fn pack_custom_instruction(
    py: Python,
    name: &String,
    custom_instructions_hash: &mut CustomOperationsMap,
    new_instructions_list: &mut Vec<String>,
    circuit_data: &mut CircuitData,
    qpy_data: &QPYData,
) -> PyResult<formats::CustomCircuitInstructionDefPack> {
    let operation = custom_instructions_hash
        .get(name)
        .ok_or_else(|| {
            PyErr::new::<PyValueError, _>(format!("Could not find operation data for {}", name))
        })?
        .clone();
    let gate_type = get_circuit_type_key(py, &operation)?;
    let mut has_definition = false;
    let mut data: Bytes = Bytes::new();
    let mut num_ctrl_qubits = 0;
    let mut ctrl_state = 0;
    let mut base_gate: Bound<PyAny> = py.None().bind(py).clone();
    let mut base_gate_raw: Bytes = Bytes::new();

    if gate_type == circuit_instruction_types::PAULI_EVOL_GATE {
        if let OperationRef::Gate(gate) = operation.view() {
            has_definition = true;
            data = serialize(&pack_pauli_evolution_gate(gate.gate.bind(py), qpy_data)?)?;
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
                py,
                &gate.getattr("_definition")?,
                py.None().bind(py),
                false,
                qpy_data.version,
            )?)?;
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
                match getattr_or_none(gate, "definition")? {
                    None => (),
                    Some(definition) => {
                        has_definition = true;
                        data = serialize(&pack_circuit(
                            py,
                            &definition,
                            py.None().bind(py),
                            false,
                            qpy_data.version,
                        )?)?;
                    }
                }
            }
            OperationRef::Instruction(pyinst) => {
                let inst = pyinst.instruction.bind(py);
                match getattr_or_none(inst, "definition")? {
                    None => (),
                    Some(definition) => {
                        has_definition = true;
                        data = serialize(&pack_circuit(
                            py,
                            &definition,
                            py.None().bind(py),
                            false,
                            qpy_data.version,
                        )?)?;
                    }
                }
            }
            OperationRef::Operation(pyoperation) => {
                let operation = pyoperation.operation.bind(py);
                match getattr_or_none(operation, "definition")? {
                    None => (),
                    Some(definition) => {
                        has_definition = true;
                        data = serialize(&pack_circuit(
                            py,
                            &definition,
                            py.None().bind(py),
                            false,
                            qpy_data.version,
                        )?)?;
                    }
                }
            }
            _ => (),
        }
    }

    if !base_gate.is_none() {
        let instruction =
            circuit_data.pack(py, &CircuitInstruction::py_new(&base_gate, None, None)?)?;
        base_gate_raw = serialize(&pack_instruction(
            py,
            &instruction,
            circuit_data,
            custom_instructions_hash,
            new_instructions_list,
            qpy_data,
        )?)?;
    }
    Ok(formats::CustomCircuitInstructionDefPack {
        gate_type,
        num_qubits: operation.num_qubits(),
        num_clbits: operation.num_clbits(),
        custom_definition: has_definition as u8,
        num_ctrl_qubits,
        ctrl_state,
        name: name.to_string(),
        data,
        base_gate_raw,
    })
}

fn pack_custom_instructions(
    py: Python,
    custom_instructions_hash: &mut CustomOperationsMap,
    circuit_data: &mut CircuitData,
    qpy_data: &QPYData,
) -> PyResult<formats::CustomCircuitInstructionsPack> {
    let mut custom_instructions: Vec<formats::CustomCircuitInstructionDefPack> = Vec::new();
    let mut instructions_to_pack: Vec<String> = custom_instructions_hash.keys().cloned().collect();
    while let Some(name) = instructions_to_pack.pop() {
        custom_instructions.push(pack_custom_instruction(
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
    circuit: &Bound<PyAny>,
    version: u32,
    standalone_var_indices: &Bound<PyDict>,
) -> PyResult<Vec<formats::ExpressionVarDeclarationPack>> {
    let mut result = Vec::new();
    let mut index = 0;
    for item in circuit.call_method0("iter_input_vars")?.try_iter()? {
        let var = item?;
        result.push(pack_standalone_var(
            &var,
            expression_var_declaration::INPUT,
            version,
        )?);
        standalone_var_indices.set_item(&var, index)?;
        index += 1;
    }
    for item in circuit.call_method0("iter_captured_vars")?.try_iter()? {
        let var = item?;
        result.push(pack_standalone_var(
            &var,
            expression_var_declaration::CAPTURE,
            version,
        )?);
        standalone_var_indices.set_item(&var, index)?;
        index += 1;
    }
    for item in circuit.call_method0("iter_declared_vars")?.try_iter()? {
        let var = item?;
        result.push(pack_standalone_var(
            &var,
            expression_var_declaration::LOCAL,
            version,
        )?);
        standalone_var_indices.set_item(&var, index)?;
        index += 1;
    }
    if version < 14 {
        match getattr_or_none(circuit, "num_stretches")? {
            None => (),
            Some(value) => {
                if value.extract::<usize>()? > 0 {
                    return Err(UnsupportedFeatureForVersion::new_err((
                        "circuits containing stretch variables",
                        14,
                        version,
                    )));
                }
            }
        }
    }
    for item in circuit
        .call_method0("iter_captured_stretches")?
        .try_iter()?
    {
        let var = item?;
        result.push(pack_standalone_var(
            &var,
            expression_var_declaration::STRETCH_CAPTURE,
            version,
        )?);
        standalone_var_indices.set_item(&var, index)?;
        index += 1;
    }
    for item in circuit
        .call_method0("iter_declared_stretches")?
        .try_iter()?
    {
        let var = item?;
        result.push(pack_standalone_var(
            &var,
            expression_var_declaration::STRETCH_LOCAL,
            version,
        )?);
        standalone_var_indices.set_item(&var, index)?;
        index += 1;
    }
    Ok(result)
}

pub fn pack_circuit(
    py: Python,
    circuit: &Bound<PyAny>,
    metadata_serializer: &Bound<PyAny>,
    use_symengine: bool,
    version: u32,
) -> PyResult<formats::QPYFormatV13> {
    circuit.getattr("data")?; // in case _data is lazily generated in python
    let mut circuit_data = circuit.getattr("_data")?.extract::<CircuitData>()?;
    let clbit_indices = circuit_data.get_clbit_indices(py).clone();
    let standalone_var_indices = PyDict::new(py);
    let standalone_vars = pack_standalone_vars(circuit, version, &standalone_var_indices)?;
    let qpy_data = QPYData {
        version,
        _use_symengine: use_symengine,
        clbit_indices,
        standalone_var_indices: standalone_var_indices.unbind(),
    };
    let header = pack_circuit_header(circuit, metadata_serializer, &qpy_data)?;
    // Pulse has been removed in Qiskit 2.0. As long as we keep QPY at version 13,
    // we need to write an empty calibrations header since read_circuit expects it
    let calibrations = formats::CalibrationsPack {num_cals: 0};
    let (instructions, mut custom_instructions_hash) =
        pack_instructions(py, &circuit_data, &qpy_data)?;
    let custom_instructions = pack_custom_instructions(
        py,
        &mut custom_instructions_hash,
        &mut circuit_data,
        &qpy_data,
    )?;
    let layout = pack_layout(circuit)?;
    Ok(formats::QPYFormatV13 {
        header,
        standalone_vars,
        custom_instructions,
        instructions,
        calibrations,
        layout,
    })
}

type Instruction = (
    PackedOperation,
    SmallVec<[Param; 3]>,
    Vec<Qubit>,
    Vec<Clbit>,
);

pub fn deserialize_circuit<'py>(
    py: Python<'py>,
    serialized_circuit: &[u8],
    version: u32,
    metadata_deserializer: &Bound<PyAny>,
    use_symengine: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let qpy_data = QPYData {
        version,
        _use_symengine: use_symengine,
        clbit_indices: PyDict::new(py).unbind(),
        standalone_var_indices: PyDict::new(py).unbind(),
    };
    // println!("Deserializing circuit {:?}", hex_string(serialized_circuit));
    let (packed_circuit, _) = deserialize::<formats::QPYFormatV13>(serialized_circuit)?;
    println!("Got packed_circuit {:?}", packed_circuit);
    let num_qubits = packed_circuit.header.num_qubits;
    let num_clbits = packed_circuit.header.num_clbits;
    let global_phase = packed_circuit.header.global_phase_data.to_param(py, &qpy_data)?;
    println!("Got num qubits: {:?}", num_qubits);
    println!("Got num clbits: {:?}", num_clbits);
    println!("Got global_phase: {:?}", global_phase);

    let mut instructions: Vec<Instruction> = Vec::new();
    for instruction in packed_circuit.instructions {
        let inst = unpack_instruction(py, &instruction, &qpy_data)?;
        println!("Got unpacked instruction {:?}", inst);
        instructions.push(inst);
    }
    let mut circuit_data = CircuitData::from_packed_operations(py, num_qubits, num_clbits, instructions.into_iter().map(Ok), global_phase)?;
    // since from_packed_operations does not generate register data, we use replace_bits to add them retroactively
    let mut qubits = Vec::new();
    let mut clbits = Vec::new();
    let mut qregs = Vec::new();
    let mut cregs = Vec::new();
    for packed_register in packed_circuit.header.registers {
        match packed_register.register_type {
            register_types::QREG => {
                let qreg = QuantumRegister::new_owning(packed_register.name, packed_register.bit_indices.len() as u32);
                for qubit in qreg.bits() {
                    qubits.push(qubit);
                }
                qregs.push(qreg);
                //circuit_data.add_qreg(qreg, false)?;
            },
            register_types::CREG => {
                let creg = ClassicalRegister::new_owning(packed_register.name, packed_register.bit_indices.len() as u32);
                for clbit in creg.bits() {
                    clbits.push(clbit);
                }
                cregs.push(creg);
                //circuit_data.add_creg(creg, false)?;
            }
            _ => return Err(PyErr::new::<PyValueError, _>(format!(
                "Unrecognized register type for {:?}", packed_register.name)
            ))
        }
    }
    circuit_data.replace_bits(Some(qubits), Some(clbits), Some(qregs), Some(cregs))?;

    //let qreg = QuantumRegister::new_owning("q".to_string(), num_qubits);
    let unpacked_layout = unpack_layout(py, packed_circuit.layout, &circuit_data)?;
    let metadata = deserialize_metadata(py,packed_circuit.header.metadata, metadata_deserializer)?;
    let circuit = imports::QUANTUM_CIRCUIT
    .get_bound(py)
    .call_method1(intern!(py, "_from_circuit_data"), (circuit_data,))?;
    // add registers
    
    circuit.setattr("metadata", metadata)?;
    circuit.setattr("name", packed_circuit.header.circuit_name)?;
    match unpacked_layout {
        Some(layout) => {circuit.setattr("_layout", layout)?;},
        None => (),
    }
    Ok(circuit)
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
        serialize(&pack_circuit(py, circuit, metadata_serializer, use_symengine, version)?)?;
    file_obj.call_method1(
        "write",
        (pyo3::types::PyBytes::new(py, &serialized_circuit),),
    )?;
    Ok(serialized_circuit.len())
}

#[pyfunction]
#[pyo3(signature = (file_obj, version, metadata_deserializer, use_symengine))]
pub fn py_read_circuit<'py>(
    py: Python<'py>,
    file_obj: &Bound<PyAny>,
    version: u32,
    metadata_deserializer: &Bound<PyAny>,
    use_symengine: bool,    
) -> PyResult<Bound<'py, PyAny>> {
    // TODO: this currently reads *everything* so storing multiple files will fail
    let bytes = file_obj.call_method0("read")?;
    let serialized_circuit: &[u8] = bytes.downcast::<PyBytes>()?.as_bytes();
    println!("Got serialized data: {:?}", hex_string(serialized_circuit));
    deserialize_circuit(py, serialized_circuit, version, metadata_deserializer, use_symengine)    
}

