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

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyAny, PyComplex, PyFloat, PyInt, PyString};
use qiskit_circuit::imports::{PARAMETER, PARAMETER_EXPRESSION, QUANTUM_CIRCUIT, NUMPY_ARRAY, CONTROLLED_GATE, PAULI_EVOLUTION_GATE, ANNOTATED_OPERATION};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::operations::OperationRef;
use crate::params::{serialize_parameter, serialize_parameter_expression};
use crate::circuits::serialize_circuit;
use crate::formats::Bytes;

const QPY_VERSION: u32 = 14;

pub mod tags {
    pub const INTEGER: u8 = b'i';
    pub const FLOAT: u8 = b'f';
    pub const COMPLEX: u8 = b'c';
    pub const CASE_DEFAULT: u8 = b'd';
    pub const REGISTER: u8 = b'R';
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

pub fn get_type_key(py: Python, py_object: &Bound<PyAny>) -> PyResult<u8> {
    if py_object.is_instance(PARAMETER.get_bound(py)).unwrap() {
        return Ok(tags::PARAMETER);
    } else if py_object
        .is_instance(PARAMETER_EXPRESSION.get_bound(py))
        .unwrap()
    {
        return Ok(tags::PARAMETER_EXPRESSION);
    } else if py_object
        .is_instance(QUANTUM_CIRCUIT.get_bound(py))
        .unwrap()
    {
        return Ok(tags::CIRCUIT);
    } else if py_object
        .is_instance(NUMPY_ARRAY.get_bound(py))
        .unwrap()
    {
        return Ok(tags::NUMPY_OBJ);
    } else if py_object.is_instance_of::<PyInt>() {
        return Ok(tags::INTEGER);
    } else if py_object.is_instance_of::<PyFloat>() {
        return Ok(tags::FLOAT);
    } else if py_object.is_instance_of::<PyComplex>() {
        return Ok(tags::COMPLEX);
    } else if py_object.is_instance_of::<PyString>() {
        return Ok(tags::STRING);
    } else if py_object.is_none() {
        return Ok(tags::NULL);
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Unidentified type_key for: {}", py_object)))
}

pub fn dumps_value(py: Python, py_object: &Bound<PyAny>) -> PyResult<(u8, Bytes)> {
    let type_key: u8 = get_type_key(py, py_object)?;
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
        tags::PARAMETER => serialize_parameter(py, py_object)?,
        tags::PARAMETER_EXPRESSION => serialize_parameter_expression(py, py_object)?,
        tags::NUMPY_OBJ => {
            let np = py.import("numpy")?;
            let io = py.import("io")?;
            let buffer = io.call_method0("BytesIO")?;
            np.call_method1("save", (&buffer, py_object))?;
            buffer.call_method0("getvalue")?.extract::<Bytes>()?
        }
        tags::STRING => py_object.extract::<String>()?.into_bytes(),
        tags::NULL => Vec::new(),
        tags::CIRCUIT => serialize_circuit(py, py_object, py.None().bind(py), false, QPY_VERSION)?,
        _ => return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Unhandled type_key: {}", type_key))),
    };
    Ok((type_key, value))   
}

//     elif type_key == type_keys.Value.PARAMETER_VECTOR:
//         binary_data = common.data_to_binary(obj, _write_parameter_vec)

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
