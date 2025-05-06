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

use binrw::meta::WriteEndian;
use pyo3::{prelude::*, IntoPyObjectExt};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyAny, PyComplex, PyDict, PyFloat, PyInt, PyString, PyTuple};
use binrw::BinWrite;
use qiskit_circuit::imports::{PARAMETER, PARAMETER_EXPRESSION, PARAMETER_VECTOR_ELEMENT, QUANTUM_CIRCUIT, NUMPY_ARRAY, CONTROLLED_GATE, PAULI_EVOLUTION_GATE, ANNOTATED_OPERATION, BUILTIN_RANGE, CLBIT, CLASSICAL_REGISTER, CASE_DEFAULT, EXPR};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::operations::OperationRef;
use qiskit_circuit::Clbit;

use crate::params::{serialize_parameter, serialize_parameter_expression, serialize_parameter_vector};
use crate::circuits::serialize_circuit;
use crate::formats::{Bytes, RangePack, GenericDataPack, GenericDataSequencePack};
use std::fmt::Debug;
use std::io::{Cursor, Write};

const QPY_VERSION: u32 = 14;

pub struct QPYData {
    pub version: u32,
    pub use_symengine: bool,
    pub clbit_indices: Py<PyDict>

}
// For debugging purposes
fn hex_string(bytes: &Bytes) -> String {
    bytes
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
}

pub mod tags {
    pub const INTEGER: u8 = b'i';
    pub const FLOAT: u8 = b'f';
    pub const COMPLEX: u8 = b'c';
    pub const CASE_DEFAULT: u8 = b'd';
    pub const REGISTER: u8 = b'R';
    pub const RANGE: u8 = b'r';
    pub const TUPLE: u8 = b't';
    pub const NUMPY_OBJ: u8 = b'n';
    pub const PARAMETER: u8 = b'p';
    pub const PARAMETER_VECTOR: u8 = b'v';
    pub const PARAMETER_EXPRESSION: u8 = b'e';
    pub const STRING: u8 = b's';
    pub const NULL: u8 = b'z';
    pub const EXPRESSION: u8 = b'x';
    pub const MODIFIER: u8 = b'm';
    pub const CIRCUIT: u8 = b'q';
}

pub fn serialize<T>(value: T) -> PyResult<Bytes>
where 
    T: BinWrite + WriteEndian + Debug,
    for<'a> <T as BinWrite>::Args<'a>: Default
{
    let mut buffer = Cursor::new(Vec::new());
    value.write(&mut buffer).unwrap();
    println!("Serializing {:?}", value);
    println!("Got {:?}", hex_string(buffer.get_ref()));
    Ok(buffer.into_inner())
}

pub fn get_type_key(py: Python, py_object: &Bound<PyAny>) -> PyResult<u8> {
    println!("get_type_key called for {:?}", py_object);
    if py_object.is_instance(PARAMETER_VECTOR_ELEMENT.get_bound(py)).unwrap() {
        return Ok(tags::PARAMETER_VECTOR);
    } else if py_object.is_instance(PARAMETER.get_bound(py)).unwrap() {
        return Ok(tags::PARAMETER);
    } else if py_object.is_instance(PARAMETER_EXPRESSION.get_bound(py))? {
        return Ok(tags::PARAMETER_EXPRESSION);
    } else if py_object.is_instance(QUANTUM_CIRCUIT.get_bound(py))? {
        return Ok(tags::CIRCUIT);
    } else if py_object.is_instance(CLBIT.get_bound(py))? || py_object.is_instance(CLASSICAL_REGISTER.get_bound(py))? {
        return Ok(tags::REGISTER);
    } else if py_object.is_instance(EXPR.get_bound(py))? || py_object.is_instance(CLASSICAL_REGISTER.get_bound(py))? {
        return Ok(tags::EXPRESSION);
    } else if py_object.is_instance(BUILTIN_RANGE.get_bound(py))? {
        return Ok(tags::RANGE);
    } else if py_object.is_instance(NUMPY_ARRAY.get_bound(py))? {
        return Ok(tags::NUMPY_OBJ);
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
    } else if py_object.is(CASE_DEFAULT.get_bound(py)) {
        return Ok(tags::CASE_DEFAULT);        
    } else if py_object.is_none() {
        return Ok(tags::NULL);
    }
    
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Unidentified type_key for: {}", py_object)))
}

pub fn dumps_value(py: Python, py_object: &Bound<PyAny>, qpy_data: &QPYData) -> PyResult<(u8, Bytes)> {
    println!("Dumping value: {:?}", py_object);
    let type_key: u8 = get_type_key(py, py_object)?;
    println!("Dumping value with type key {:?}", type_key);
    let value: Bytes = match type_key {
        tags::INTEGER => py_object.extract::<i64>()?.to_be_bytes().to_vec(),
        tags::FLOAT => py_object.extract::<f64>()?.to_be_bytes().to_vec(),
        tags::COMPLEX => {
            let complex_num = py_object.downcast::<PyComplex>()?;
            let mut bytes = Vec::with_capacity(16);
            bytes.extend_from_slice(&complex_num.real().to_be_bytes());
            bytes.extend_from_slice(&complex_num.imag().to_be_bytes());
            bytes
        }
        tags::RANGE => serialize_range(py_object)?,
        tags::TUPLE => serialize(pack_generic_sequence(py, py_object, qpy_data)?)?,
        tags::PARAMETER => serialize_parameter(py, py_object)?,
        tags::PARAMETER_VECTOR => serialize_parameter_vector(py, py_object)?,
        tags::PARAMETER_EXPRESSION => serialize_parameter_expression(py, py_object, qpy_data)?,
        tags::NUMPY_OBJ => {
            let np = py.import("numpy")?;
            let io = py.import("io")?;
            let buffer = io.call_method0("BytesIO")?;
            np.call_method1("save", (&buffer, py_object))?;
            buffer.call_method0("getvalue")?.extract::<Bytes>()?
        }
        tags::STRING => py_object.extract::<String>()?.into_bytes(),
        tags::EXPRESSION => serialize_expression(py_object, qpy_data)?,
        tags::NULL | tags::CASE_DEFAULT => Vec::new(),
        tags::CIRCUIT => serialize_circuit(py, py_object, py.None().bind(py), false, QPY_VERSION)?,
        _ => return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("dumps_value: Unhandled type_key: {}", type_key))),
    };
    println!("Dumped value for {:?} (type {:?}): {:?}", py_object, type_key, hex_string(&value));
    Ok((type_key, value))   
}

pub fn dumps_register(py: Python, register: &Bound<PyAny>) -> PyResult<Bytes>{
    if register.is_instance(CLASSICAL_REGISTER.get_bound(py))? {
        return Ok(register.getattr("name")?.extract::<String>()?.as_bytes().to_vec())
    } else {
        // TODO: explicitly verify this works
        println!("Dumping clbit {:?}", register);
//        let index: usize = register.call_method0("index")?.extract()?;
        let index: usize = register.getattr("_index")?.extract()?;
        println!("Got index {:?}", index);
        //let index_string = index.to_be_bytes().to_vec();
        let index_string = index.to_string().as_bytes().to_vec();
        let mut result = vec![0x00];
        result.extend_from_slice(&index_string);
        println!("Extending the index {:?} results in hex {:?}", index, &result);
        Ok(result)
    }
}

pub fn pack_generic_data(py: Python, py_data: &Bound<PyAny>, qpy_data: &QPYData) -> PyResult<GenericDataPack> {
    let (type_key, data) = dumps_value(py, py_data, qpy_data)?;
    Ok(GenericDataPack { type_key, data_len: data.len() as u64, data})
}

pub fn pack_generic_sequence(py: Python, py_sequence: &Bound<PyAny>, qpy_data: &QPYData) -> PyResult<GenericDataSequencePack> {
    let elements: Vec<GenericDataPack> = py_sequence
    .try_iter()?
    .map(|possible_data_item| {
        let data_item = possible_data_item?;
        pack_generic_data(py, &data_item, qpy_data)
    })
    .collect::<PyResult<_>>()?;
    Ok(GenericDataSequencePack { num_elements: elements.len() as u64, elements })
}

pub mod circuit_instruction_types {
    pub const INSTRUCTION: u8 = b'i';
    pub const GATE: u8 = b'g';
    pub const PAULI_EVOL_GATE: u8 = b'p';
    pub const CONTROLLED_GATE: u8 = b'c';
    pub const ANNOTATED_OPERATION: u8 = b'a';
    
}

pub fn get_circuit_type_key(py: Python, op: &PackedOperation) -> PyResult<u8> {
    match op.view() {
        OperationRef::StandardGate(_) => Ok(circuit_instruction_types::GATE),
        OperationRef::StandardInstruction(_) | OperationRef::Instruction(_) => Ok(circuit_instruction_types::INSTRUCTION),
        OperationRef::Unitary(_) => Ok(circuit_instruction_types::GATE),
        OperationRef::Gate(pygate) => {
            let gate = pygate.gate.bind(py);
            if gate.is_instance(PAULI_EVOLUTION_GATE.get_bound(py))? {
                Ok(circuit_instruction_types::PAULI_EVOL_GATE)
            }
            else if gate.is_instance(CONTROLLED_GATE.get_bound(py))? {
                Ok(circuit_instruction_types::CONTROLLED_GATE)
            } else {
                Ok(circuit_instruction_types::GATE)
            }
        }
        OperationRef::Operation(operation) => {
            if operation.operation.bind(py).is_instance(ANNOTATED_OPERATION.get_bound(py))? {
                Ok(circuit_instruction_types::ANNOTATED_OPERATION)
            } else {
                Err(PyErr::new::<PyValueError, _>(format!("Unable to determine circuit type key for {:?}", operation)))
            }
        }
    }
}

fn serialize_range(py_object: &Bound<PyAny>) -> PyResult<Bytes> {
    let start = py_object.getattr("start")?.extract::<i64>()?;
    let stop = py_object.getattr("stop")?.extract::<i64>()?;
    let step = py_object.getattr("step")?.extract::<i64>()?;
    let range_pack = RangePack {start, stop, step};
    let mut buffer = Cursor::new(Vec::new());
    range_pack.write(&mut buffer).unwrap();
    Ok(buffer.into_inner())
}

fn serialize_expression(py_object: &Bound<PyAny>, qpy_data: &QPYData) -> PyResult<Bytes> {
    println!("About to serialize expression {:?}", py_object);
    let py = py_object.py();
    let clbit_indices = PyDict::new(py);
    for (key, val) in qpy_data.clbit_indices.bind(py).iter() {
        let index = val.getattr("index")?;
        clbit_indices.set_item(key, index)?;
    }
    // TODO: set clbit_indices and standalone_var_indices from inputs
    let standalone_var_indices = PyDict::new(py);
    let io = py.import("io")?;
    let buffer = io.call_method0("BytesIO")?;
    let value = py.import("qiskit.qpy.binary_io.value")?;
    value.call_method1("_write_expr", (&buffer, py_object, clbit_indices, standalone_var_indices, qpy_data.version))?;
    let result = buffer.call_method0("getvalue")?.extract::<Bytes>()?;
    println!("Serialized expression {:?}, got {:?}", py_object, hex_string(&result));
    Ok(result)
}