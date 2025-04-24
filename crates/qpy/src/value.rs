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
use pyo3::types::{PyAny, PyComplex, PyFloat, PyInt, PyString};
use qiskit_circuit::imports::{PARAMETER, PARAMETER_EXPRESSION, QUANTUM_CIRCUIT};
use crate::params::{serialize_parameter, serialize_parameter_expression};

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
    pub const NONE: u8 = b'n';
}

pub fn get_type_key(py: Python, py_object: &Bound<PyAny>) -> u8 {
    if py_object.is_instance(PARAMETER.get_bound(py)).unwrap() {
        return tags::PARAMETER;
    } else if py_object
        .is_instance(PARAMETER_EXPRESSION.get_bound(py))
        .unwrap()
    {
        return tags::PARAMETER_EXPRESSION;
    } else if py_object
        .is_instance(QUANTUM_CIRCUIT.get_bound(py))
        .unwrap()
    {
        return tags::CIRCUIT;
    } else if py_object.is_instance_of::<PyInt>() {
        return tags::INTEGER;
    } else if py_object.is_instance_of::<PyFloat>() {
        return tags::FLOAT;
    } else if py_object.is_instance_of::<PyComplex>() {
        return tags::COMPLEX;
    } else if py_object.is_instance_of::<PyString>() {
        return tags::STRING;
    } else if py_object.is_none() {
        return tags::NONE;
    }
    tags::NULL
}

pub fn dumps_value(py: Python, py_object: &Bound<PyAny>) -> PyResult<(u8, Vec<u8>)> {
    //TODO: placeholder for now
    let type_key: u8 = get_type_key(py, py_object);
    let value: Vec<u8> = match type_key {
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
        tags::NUMPY_OBJ => Vec::new(), // TODO: call np.save using pyo3
        tags::STRING => py_object.extract::<String>()?.into_bytes(),
        tags::NULL => Vec::new(),

        _ => Vec::new(),
    };
    Ok((type_key, value))
}

//     elif type_key == type_keys.Value.PARAMETER_VECTOR:
//         binary_data = common.data_to_binary(obj, _write_parameter_vec)
//     elif type_key == type_keys.Value.PARAMETER:
//         binary_data = common.data_to_binary(obj, _write_parameter)
//     elif type_key == type_keys.Value.PARAMETER_EXPRESSION:
//         binary_data = common.data_to_binary(
//             obj, _write_parameter_expression, use_symengine=use_symengine, version=version
//         )
