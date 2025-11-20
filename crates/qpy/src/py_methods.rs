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

// Methods for QPY serialization working directly with Python-based data
use binrw::Endian;
use hashbrown::HashMap;
use numpy::{Complex64, ToPyArray};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyIOError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{
    PyAny, PyBytes, PyComplex, PyDict, PyFloat, PyInt, PyIterator, PyList, PyString, PyTuple,
};
use qiskit_circuit::classical::expr::Expr;
use std::io::Cursor;

use qiskit_circuit::Clbit;
use qiskit_circuit::bit::{PyClassicalRegister, PyClbit, ShareableClbit};
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::circuit_instruction::CircuitInstruction;
use qiskit_circuit::classical;
use qiskit_circuit::imports;
use qiskit_circuit::operations::{ArrayType, Operation, OperationRef, StandardInstruction};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::parameter::parameter_expression::{
    PyParameter, PyParameterExpression, PyParameterVectorElement,
};

use uuid::Uuid;

use crate::bytes::Bytes;
use crate::formats;
use crate::params::pack_param_obj;
use crate::value::{
    GenericValue, QPYWriteData, circuit_instruction_types, deserialize, get_circuit_type_key,
    modifier_types, pack_generic_value, serialize, serialize_generic_value, tags, type_name,
};
use binrw::BinWrite;

use crate::circuit_writer::{pack_circuit, pack_instruction};

const UNITARY_GATE_CLASS_NAME: &str = "UnitaryGate";

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

/// custom gates have unique UUID attached to their name
/// this method recognizes whether we have such a gate and returns a unique name for it
/// since custom gates are implemented in python, this is a heavy python-space function
pub fn recognize_custom_operation(op: &PackedOperation, name: &String) -> PyResult<Option<String>> {
    Python::attach(|py| {
        let library = py.import("qiskit.circuit.library")?;
        let circuit_mod = py.import("qiskit.circuit")?;
        let controlflow = py.import("qiskit.circuit.controlflow")?;

        if (!library.hasattr(name)?
            && !circuit_mod.hasattr(name)?
            && !controlflow.hasattr(name)?
            && (name != "Clifford" && name != "PauliProductMeasurement"))
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
    })
}

/// when trying to instantiate nonstandard gates, we turn to the relevant python clas
/// this function obtains the class based on the gate class name
pub fn get_python_gate_class<'a>(
    py: Python<'a>,
    gate_class_name: &String,
) -> PyResult<Bound<'a, PyAny>> {
    let library = py.import("qiskit.circuit.library")?;
    let circuit_mod = py.import("qiskit.circuit")?;
    let control_flow = py.import("qiskit.circuit.controlflow")?;
    if library.hasattr(gate_class_name)? {
        library.getattr(gate_class_name)
    } else if circuit_mod.hasattr(gate_class_name)? {
        circuit_mod.getattr(gate_class_name)
    } else if control_flow.hasattr(gate_class_name)? {
        control_flow.getattr(gate_class_name)
    } else if gate_class_name == "Clifford" {
        Ok(imports::CLIFFORD.get_bound(py).clone())
    } else if gate_class_name == "pauli_product_measurement" {
        Ok(imports::PAULI_PRODUCT_MEASUREMENT.get_bound(py).clone())
    } else {
        Err(PyIOError::new_err(format!(
            "Gate class not found: {:?}",
            gate_class_name
        )))
    }
}

pub fn py_bytes_to_uuid(py: Python, bytes: [u8; 16]) -> PyResult<Py<PyAny>> {
    let uuid_module = py.import("uuid")?;
    let py_bytes = PyBytes::new(py, &bytes);
    let kwargs = PyDict::new(py);
    kwargs.set_item("bytes", py_bytes)?;
    Ok(uuid_module.call_method("UUID", (), Some(&kwargs))?.unbind())
}

// pub fn py_pack_generic_data(
//     py_data: &Bound<PyAny>,
//     qpy_data: &QPYWriteData,
// ) -> PyResult<formats::GenericDataPack> {
//     let (type_key, data) = py_dumps_value(py_data.clone().unbind(), qpy_data)?;
//     Ok(formats::GenericDataPack { type_key, data })
// }

// serializes python metadata to JSON using a python JSON serializer
pub fn serialize_metadata(
    metadata_opt: &Option<Bound<PyAny>>,
    metadata_serializer: Option<&Bound<PyAny>>,
) -> PyResult<Bytes> {
    match metadata_opt {
        None => Ok(Bytes::new()),
        Some(metadata) => {
            let py = metadata.py();
            let none = py.None();
            let py_serializer = metadata_serializer.unwrap_or(none.bind(py));
            let json = py.import("json")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("separators", PyTuple::new(py, [",", ":"])?)?;
            kwargs.set_item("cls", py_serializer)?;
            Ok(json
                .call_method("dumps", (metadata,), Some(&kwargs))?
                .extract::<String>()?
                .into())
        }
    }
}

// helper method to extract attribute from a py_object
pub fn getattr_or_none<'py>(
    py_object: &'py Bound<'py, PyAny>,
    name: &str,
) -> Option<Bound<'py, PyAny>> {
    match py_object.getattr(name) {
        Ok(val) => {
            if val.is_none() {
                None
            } else {
                Some(val)
            }
        }
        Err(_) => None,
    }
}

pub fn py_serialize_numpy_object(py_object: &Py<PyAny>) -> PyResult<Bytes> {
    Python::attach(|py| {
        let np = py.import("numpy")?;
        let io = py.import("io")?;
        let buffer = io.call_method0("BytesIO")?;
        np.call_method1("save", (&buffer, py_object))?;
        buffer.call_method0("getvalue")?.extract::<Bytes>()
    })
}

pub fn py_deserialize_numpy_object(data: &Bytes) -> PyResult<Py<PyAny>> {
    Python::attach(|py| {
        let np = py.import("numpy")?;
        let io = py.import("io")?;
        let buffer = io.call_method0("BytesIO")?;
        buffer.call_method1("write", (data.clone(),))?;
        buffer.call_method1("seek", (0,))?;
        Ok(np.call_method1("load", (buffer,))?.unbind())
    })
}

pub fn py_pack_registers(
    in_circ_regs: &Bound<PyAny>,
    bits: &Bound<PyList>,
) -> PyResult<Vec<formats::RegisterV4Pack>> {
    let py = in_circ_regs.py();
    let bitmap = PyDict::new(py);
    let out_circ_regs = PyList::new(py, Vec::<Py<PyAny>>::new())?;

    bits.iter()
        .enumerate()
        .try_for_each(|(index, bit)| -> PyResult<()> {
            bitmap.set_item(&bit, index)?;
            match getattr_or_none(&bit, "_register") {
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
        .cast::<PyList>()?
        .iter()
        .try_for_each(|register| -> PyResult<()> {
            result.push(py_pack_register(&register, &bitmap, true)?);
            Ok(())
        })?;

    out_circ_regs
        .iter()
        .try_for_each(|register| -> PyResult<()> {
            result.push(py_pack_register(&register, &bitmap, false)?);
            Ok(())
        })?;
    Ok(result)
}

fn py_pack_register(
    register: &Bound<PyAny>,
    bitmap: &Bound<PyDict>,
    is_in_circuit: bool,
) -> PyResult<formats::RegisterV4Pack> {
    let reg_name = register.getattr("name")?.extract::<String>()?;
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
        match getattr_or_none(&bit_val, "_index") {
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

fn pack_sparse_pauli_op(
    operator: &Bound<PyAny>,
    qpy_data: &QPYWriteData,
) -> PyResult<formats::SparsePauliOpListElemPack> {
    let op_as_np_list = operator.call_method1("to_list", (true,))?;
    let value = py_convert_to_generic_value(&op_as_np_list)?;
    let (_, data) = serialize_generic_value(&value, qpy_data)?;
    Ok(formats::SparsePauliOpListElemPack { data })
}

fn pack_pauli_evolution_gate(
    evolution_gate: &Bound<PyAny>,
    qpy_data: &QPYWriteData,
) -> PyResult<formats::PauliEvolutionDefPack> {
    let py = evolution_gate.py();
    let operators = evolution_gate.getattr("operator")?;
    let mut standalone = false;
    let operator_list: Bound<PyList> = if !operators.is_instance_of::<PyList>() {
        standalone = true;
        PyList::new(py, [operators])?
    } else {
        operators.cast()?.clone()
    };
    let pauli_data = operator_list
        .iter()
        .map(|operator| pack_sparse_pauli_op(&operator, qpy_data))
        .collect::<PyResult<_>>()?;

    let time_value = py_convert_to_generic_value(&evolution_gate.getattr("time")?)?;
    let (time_type, time_data) = serialize_generic_value(&time_value, qpy_data)?;
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

pub fn gate_class_name(op: &PackedOperation) -> PyResult<String> {
    Python::attach(|py| {
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
            OperationRef::PauliProductMeasurement(_) => imports::PAULI_PRODUCT_MEASUREMENT
                .get_bound(py)
                .getattr(intern!(py, "__name__"))?
                .extract::<String>(),
        }?;
        Ok(name)
    })
}

pub fn py_get_type_key(py_object: &Bound<PyAny>) -> PyResult<u8> {
    let py: Python<'_> = py_object.py();
    if py_object
        .is_instance(imports::PARAMETER_VECTOR_ELEMENT.get_bound(py))
        .unwrap()
    {
        return Ok(tags::PARAMETER_VECTOR);
    } else if py_object
        .is_instance(imports::PARAMETER.get_bound(py))
        .unwrap()
    {
        return Ok(tags::PARAMETER);
    } else if py_object.is_instance(imports::PARAMETER_EXPRESSION.get_bound(py))? {
        return Ok(tags::PARAMETER_EXPRESSION);
    } else if py_object.is_instance(imports::QUANTUM_CIRCUIT.get_bound(py))? {
        return Ok(tags::CIRCUIT);
    } else if py_object.is_instance(imports::CLBIT.get_bound(py))?
        || py_object.is_instance(imports::CLASSICAL_REGISTER.get_bound(py))?
    {
        return Ok(tags::REGISTER);
    } else if py_object.extract::<classical::expr::Expr>().is_ok() {
        return Ok(tags::EXPRESSION);
    } else if py_object.is_instance(imports::BUILTIN_RANGE.get_bound(py))? {
        return Ok(tags::RANGE);
    } else if py_object.is_instance(imports::NUMPY_ARRAY.get_bound(py))? {
        return Ok(tags::NUMPY_OBJ);
    } else if py_object.is_instance(imports::MODIFIER.get_bound(py))? {
        return Ok(tags::MODIFIER);
    } else if py_object.is_instance_of::<PyInt>() {
        return Ok(tags::INTEGER);
    } else if py_object.is_instance_of::<PyFloat>() {
        return Ok(tags::FLOAT);
    } else if py_object.is_instance_of::<PyComplex>() {
        return Ok(tags::COMPLEX);
    } else if py_object.is_instance_of::<PyString>() {
        return Ok(tags::STRING);
    } else if py_object.is_instance_of::<PyTuple>() {
        return Ok(tags::TUPLE);
    } else if py_object.is(imports::CASE_DEFAULT.get_bound(py)) {
        return Ok(tags::CASE_DEFAULT);
    } else if py_object.is_none() {
        return Ok(tags::NULL);
    }

    Err(PyTypeError::new_err(format!(
        "Unidentified type_key for: {}",
        py_object
    )))
}

pub fn py_convert_to_generic_value(py_object: &Bound<PyAny>) -> PyResult<GenericValue> {
    let type_key: u8 = py_get_type_key(py_object)?;
    match type_key {
        tags::BOOL => Ok(GenericValue::Bool(py_object.extract::<bool>()?)),
        tags::INTEGER => Ok(GenericValue::Int64(py_object.extract::<i64>()?)),
        tags::FLOAT => Ok(GenericValue::Float64(py_object.extract::<f64>()?)),
        tags::COMPLEX => Ok(GenericValue::Complex64(py_object.extract::<Complex64>()?)),
        tags::STRING => Ok(GenericValue::String(py_object.extract::<String>()?)),
        tags::EXPRESSION => Ok(GenericValue::Expression(py_object.extract::<Expr>()?)),
        tags::CASE_DEFAULT => Ok(GenericValue::CaseDefault),
        tags::NULL => Ok(GenericValue::Null),
        tags::PARAMETER => Ok(GenericValue::ParameterExpressionSymbol(
            py_object.extract::<PyParameter>()?.symbol().clone(),
        )),
        tags::PARAMETER_VECTOR => Ok(GenericValue::ParameterExpressionVectorSymbol(
            py_object
                .extract::<PyParameterVectorElement>()?
                .symbol()
                .clone(),
        )),
        tags::PARAMETER_EXPRESSION => Ok(GenericValue::ParameterExpression(
            py_object.extract::<PyParameterExpression>()?.inner,
        )),
        tags::CIRCUIT => Ok(GenericValue::Circuit(py_object.clone().unbind())),
        tags::TUPLE => {
            let elements: Vec<GenericValue> = py_object
                .try_iter()?
                .map(|data_item| {
                    // let data_item = possible_data_item?;
                    py_convert_to_generic_value(&data_item?)
                })
                .collect::<PyResult<_>>()?;
            Ok(GenericValue::Tuple(elements))
        }
        // the python-managed data types
        tags::RANGE => Ok(GenericValue::Range(py_object.clone().unbind())),
        tags::NUMPY_OBJ => Ok(GenericValue::NumpyObject(py_object.clone().unbind())),
        tags::MODIFIER => Ok(GenericValue::Modifier(py_object.clone().unbind())),
        tags::REGISTER => Ok(GenericValue::Register(py_object.clone().unbind())),
        _ => Err(PyTypeError::new_err(format!(
            "py_convert_to_generic_value: Unhandled type_key: {} ({})",
            type_key,
            type_name(type_key)
        ))),
    }
}

pub fn py_convert_from_generic_value(value: &GenericValue) -> PyResult<Py<PyAny>> {
    Python::attach(|py| match value {
        GenericValue::Bool(value) => value.into_py_any(py),
        GenericValue::Int64(value) => value.into_py_any(py),
        GenericValue::Float64(value) => value.into_py_any(py),
        GenericValue::Complex64(value) => value.into_py_any(py),
        GenericValue::String(value) => value.into_py_any(py),
        GenericValue::Expression(exp) => exp.clone().into_py_any(py),
        GenericValue::CaseDefault => Ok(imports::CASE_DEFAULT.get(py).clone()),
        GenericValue::Null => Ok(py.None()),
        GenericValue::ParameterExpressionSymbol(symbol) => symbol.clone().into_py_any(py),
        GenericValue::ParameterExpressionVectorSymbol(symbol) => symbol.clone().into_py_any(py),
        GenericValue::ParameterExpression(exp) => exp.clone().into_py_any(py),
        GenericValue::Circuit(py_object) => Ok(py_object.clone()),
        GenericValue::Register(py_object) => Ok(py_object.clone()),
        GenericValue::Modifier(py_object) => Ok(py_object.clone()),
        GenericValue::Range(py_object) => Ok(py_object.clone()),
        GenericValue::NumpyObject(py_object) => Ok(py_object.clone()),
        GenericValue::Tuple(values) => {
            let elements: Vec<Py<PyAny>> = values
                .iter()
                .map(py_convert_from_generic_value)
                .collect::<PyResult<_>>()?;
            PyTuple::new(py, &elements)?.into_py_any(py)
        }
    })
}

pub fn py_serialize_range(py_object: &Py<PyAny>) -> PyResult<Bytes> {
    Python::attach(|py| {
        let py_object = py_object.bind(py);
        let start = py_object.getattr("start")?.extract::<i64>()?;
        let stop = py_object.getattr("stop")?.extract::<i64>()?;
        let step = py_object.getattr("step")?.extract::<i64>()?;
        let range_pack = formats::RangePack { start, stop, step };
        let mut buffer = Cursor::new(Vec::new());
        range_pack.write(&mut buffer).unwrap();
        Ok(buffer.into())
    })
}

pub fn py_deserialize_range(raw_range: &Bytes) -> PyResult<Py<PyAny>> {
    Python::attach(|py| {
        let range_pack = deserialize::<formats::RangePack>(raw_range)?.0;
        Ok(imports::BUILTIN_RANGE
            .get_bound(py)
            .call1((range_pack.start, range_pack.stop, range_pack.step))?
            .unbind())
    })
}

// This functions packs an instruction parameter, which can be an arbitrary piece of data
// Not to be confused with Parameter, which is an atom of ParameterExpression
pub fn py_pack_param(
    py_object: &Bound<PyAny>,
    qpy_data: &QPYWriteData,
    endian: Endian,
) -> PyResult<formats::GenericDataPack> {
    let value = py_convert_to_generic_value(py_object)?;
    let (type_key, data) = match endian {
        Endian::Big => serialize_generic_value(&value, qpy_data)?,
        Endian::Little => serialize_generic_value(&value.as_le(), qpy_data)?,
    };
    Ok(formats::GenericDataPack { type_key, data })
}

// When a register is stored as an instruction param, it is serialized compactly
// For a classical register its name is saved as a string; for a clbit
// its index in the full clbit list is converted into a string, with 0x00 appended at the start
// to differentiate from the register case
pub fn py_serialize_register_param(
    register: &Py<PyAny>,
    qpy_data: &QPYWriteData,
) -> PyResult<Bytes> {
    Python::attach(|py| {
        let register = register.bind(py);
        if register.is_instance_of::<PyClassicalRegister>() {
            Ok(register.getattr("name")?.extract::<String>()?.into())
        } else if register.is_instance_of::<PyClbit>() {
            let key = &register.extract::<ShareableClbit>()?;
            let name = qpy_data
                .circuit_data
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
    })
}

pub fn py_deserialize_register_param(
    data_bytes: &Bytes,
    circuit_data: &CircuitData,
) -> PyResult<Py<PyAny>> {
    Python::attach(|py| {
        // If register name prefixed with null character it's a clbit index for single bit condition.
        if data_bytes.is_empty() {
            return Err(PyValueError::new_err(
                "Failed to load register - name missing",
            ));
        }
        if data_bytes[0] == 0u8 {
            let index = Clbit(std::str::from_utf8(&data_bytes[1..])?.parse()?);
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
            let name = std::str::from_utf8(data_bytes)?;
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

pub fn get_instruction_annotations(
    instruction: &PackedInstruction,
    qpy_data: &mut QPYWriteData,
) -> PyResult<Option<formats::InstructionsAnnotationPack>> {
    Python::attach(|py| {
        if let OperationRef::Instruction(inst) = instruction.op.view() {
            let op = inst.instruction.bind(py);
            if op.is_instance(imports::CONTROL_FLOW_BOX_OP.get_bound(py))? {
                let annotations_iter = PyIterator::from_object(&op.getattr("annotations")?)?;
                let annotations: Vec<formats::InstructionAnnotationPack> = annotations_iter
                    .map(|annotation| {
                        let (namespace_index, payload) =
                            qpy_data.annotation_handler.serialize(&annotation?)?;
                        Ok(formats::InstructionAnnotationPack {
                            namespace_index,
                            payload,
                        })
                    })
                    .collect::<PyResult<_>>()?;
                if !annotations.is_empty() {
                    return Ok(Some(formats::InstructionsAnnotationPack { annotations }));
                }
            }
        }
        Ok(None)
    })
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
                data = serialize(&pack_pauli_evolution_gate(gate.gate.bind(py), qpy_data)?);
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

pub fn py_pack_modifier(modifier: &Py<PyAny>) -> PyResult<formats::ModifierPack> {
    Python::attach(|py| {
        let modifier = modifier.bind(py);
        let module = py.import("qiskit.circuit.annotated_operation")?;
        if modifier.is_instance(&module.getattr("InverseModifier")?)? {
            Ok(formats::ModifierPack {
                modifier_type: modifier_types::INVERSE,
                num_ctrl_qubits: 0,
                ctrl_state: 0,
                power: 0.0,
            })
        } else if modifier.is_instance(&module.getattr("ControlModifier")?)? {
            Ok(formats::ModifierPack {
                modifier_type: modifier_types::CONTROL,
                num_ctrl_qubits: modifier.getattr("num_ctrl_qubits")?.extract::<u32>()?,
                ctrl_state: modifier.getattr("ctrl_state")?.extract::<u32>()?,
                power: 0.0,
            })
        } else if modifier.is_instance(&module.getattr("PowerModifier")?)? {
            Ok(formats::ModifierPack {
                modifier_type: modifier_types::POWER,
                num_ctrl_qubits: 0,
                ctrl_state: 0,
                power: modifier.getattr("power")?.extract::<f64>()?,
            })
        } else {
            Err(PyTypeError::new_err("Unsupported modifier."))
        }
    })
}

pub fn py_unpack_modifier(packed_modifier: &formats::ModifierPack) -> PyResult<Py<PyAny>> {
    Python::attach(|py| match packed_modifier.modifier_type {
        modifier_types::INVERSE => Ok(imports::INVERSE_MODIFIER.get_bound(py).call0()?.unbind()),
        modifier_types::CONTROL => {
            let kwargs = PyDict::new(py);
            kwargs.set_item(
                intern!(py, "num_ctrl_qubits"),
                packed_modifier.num_ctrl_qubits,
            )?;
            kwargs.set_item(intern!(py, "ctrl_state"), packed_modifier.ctrl_state)?;
            Ok(imports::CONTROL_MODIFIER
                .get_bound(py)
                .call((), Some(&kwargs))?
                .unbind())
        }
        modifier_types::POWER => {
            let kwargs = PyDict::new(py);
            kwargs.set_item(intern!(py, "power"), packed_modifier.power)?;
            Ok(imports::POWER_MODIFIER
                .get_bound(py)
                .call((), Some(&kwargs))?
                .unbind())
        }
        _ => Err(PyTypeError::new_err("Unsupported modifier.")),
    })
}

// condition is stored in an ad-hoc manner in the instruction's "_condition" field, and we need python to access it
pub fn get_condition_data_from_inst(
    inst: &Py<PyAny>,
    qpy_data: &QPYWriteData,
) -> PyResult<formats::ConditionPack> {
    Python::attach(|py| match getattr_or_none(inst.bind(py), "_condition") {
        None => Ok(formats::ConditionPack {
            key: formats::condition_types::NONE,
            register_size: 0u16,
            value: 0i64,
            data: formats::ConditionData::None,
        }),
        Some(condition) => {
            if condition.extract::<classical::expr::Expr>().is_ok() {
                let value = GenericValue::Expression(condition.extract::<classical::expr::Expr>()?);
                let expression_pack = pack_generic_value(&value, qpy_data)?;
                Ok(formats::ConditionPack {
                    key: formats::condition_types::EXPRESSION,
                    register_size: 0u16,
                    value: 0i64,
                    data: formats::ConditionData::Expression(expression_pack),
                })
            } else if condition.is_instance_of::<PyTuple>() {
                let key = formats::condition_types::TWO_TUPLE;
                let value = condition.cast::<PyTuple>()?.get_item(1)?.extract::<i64>()?;
                let register = py_serialize_register_param(
                    &condition.cast::<PyTuple>()?.get_item(0)?.unbind(),
                    qpy_data,
                )?;
                Ok(formats::ConditionPack {
                    key,
                    register_size: register.len() as u16,
                    value,
                    data: formats::ConditionData::Register(register),
                })
            } else {
                Err(PyValueError::new_err(
                    "Expression handling not implemented for get_condition_data_from_inst",
                ))
            }
        }
    })
}
