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
use std::io::Cursor;
use pyo3::prelude::*;
use pyo3::basic::CompareOp;
use pyo3::types::{PyAny, PyInt, PyFloat, PyComplex, PyString, PyDict};
use pyo3::intern;
use binrw::BinWrite;
use qiskit_circuit::operations::Param;
use qiskit_circuit::imports::{PARAMETER, PARAMETER_EXPRESSION, QUANTUM_CIRCUIT};

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

fn get_type_key(py: Python, py_object: &Bound<PyAny>) -> u8 {
    if py_object.is_instance(PARAMETER.get_bound(py)).unwrap() {
        return tags::PARAMETER;
    } else if py_object.is_instance(PARAMETER_EXPRESSION.get_bound(py)).unwrap() {
        return tags::PARAMETER_EXPRESSION;
    } else if py_object.is_instance(QUANTUM_CIRCUIT.get_bound(py)).unwrap() {
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

#[derive(BinWrite)]
#[brw(big)]
pub struct SerializableParam {
    type_key: u8,
    data_len: u64,
    data: Vec<u8>
}

#[derive(BinWrite)]
#[brw(big)]
struct ParameterPack {
    name_length: u16,
    uuid: [u8; 16],
    name: Vec<u8>
}

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
struct ParameterExpressionElementPack {
    op: u8,
    lhs_type: u8,
    lhs_data: [u8; 16],
    rhs_type: u8,
    rhs_data: [u8; 16],
}

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
struct ParameterExpressionPack {
    symbol_table_length: u64,
    expression_data_length: u64,
    expression_data: Vec<u8>,
    symbol_table_data: Vec<u8>,
}

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
struct ParameterExpressionSymbolPack {
    symbol_key: u8,
    value_key: u8,
    value_data_len: u64,
    symbol_data: Vec<u8>,
    value_data: Vec<u8>,
}

fn serialize_parameter_replay_entry(py: Python, inst: &Bound<PyAny>) -> PyResult<(u8, [u8; 16])> {
    let key_type = get_type_key(py, inst);
    let data: [u8; 16] = match key_type {
        tags::PARAMETER => inst.getattr(intern!(py, "uuid"))?.getattr(intern!(py, "bytes"))?.extract::<[u8; 16]>()?,
        tags::NONE => [0u8; 16],
        tags::INTEGER => 0u64.to_be_bytes()
            .into_iter()
            .chain(inst.extract::<i64>()?.to_be_bytes())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(),
        tags::FLOAT => 0u64.to_be_bytes()
            .into_iter()
            .chain(inst.extract::<f64>()?.to_be_bytes())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(),
        _ => [0u8; 16],
    }; // TODO: should also handle complex and ParameterExpression cases
    Ok((key_type, data))
}

fn serialize_parameter_expression_element(py: Python, replay_obj: &Bound<PyAny>) -> PyResult<Vec<u8>> {
    let (lhs_type, lhs_data) = serialize_parameter_replay_entry(py, &replay_obj.getattr(intern!(py, "lhs"))?)?;
    let (rhs_type, rhs_data) = serialize_parameter_replay_entry(py, &replay_obj.getattr(intern!(py, "rhs"))?)?;
    let op = replay_obj.getattr(intern!(py, "op"))?.extract::<u8>()?;
    let packed_element = ParameterExpressionElementPack {
        op,
        lhs_type,
        lhs_data,
        rhs_type,
        rhs_data,
    };
    let mut buffer = Cursor::new(Vec::new());
    packed_element.write(&mut buffer).unwrap();
    Ok(buffer.into_inner())
}
fn write_parameter_expression_elements(py: Python, py_object: &Bound<PyAny>) -> PyResult<Vec<u8>> {
    // TODO: should also collect extra symbol map
    let qpy_replay = py_object.getattr(intern!(py, "_qpy_replay"))?.extract::<Vec<PyObject>>()?;
    let result_data: Vec<u8> = qpy_replay
    .iter()
    .flat_map(|replay_obj| serialize_parameter_expression_element(py, &replay_obj.bind(py)).unwrap())
    .collect();
    Ok(result_data)
}

fn dumps_value(py: Python, py_object: &Bound<PyAny>) -> PyResult<(u8, Vec<u8>)> {
    //TODO: placeholder for now
    let type_key: u8 = 0;
    let value : Vec<u8> = Vec::new();
    Ok((type_key, value))
}
fn write_symbol_table(py: Python, py_object: &Bound<PyAny>) -> PyResult<(u64, Vec<u8>)> {
    let symbol_table_map: Bound<PyDict> = py_object.getattr(intern!(py, "_parameter_symbols"))?.extract()?;
    let data = symbol_table_map.iter()
        .map(|(symbol, value)| {
            let symbol_key = get_type_key(py, &symbol);
           
            let symbol_data: Vec<u8> = match symbol_key {
                tags::PARAMETER => write_parameter(py, &symbol)?,
                _ => Vec::new(),
            };

            let (value_key, value_data) : (u8, Vec<u8>) = match value
                .rich_compare(symbol.getattr("_symbol_expr")?, CompareOp::Eq)?
                .is_truthy()? {
                    true => (symbol_key, Vec::new()),
                    false => dumps_value(py, &value)?,
            };

            let packed_symbol_data = ParameterExpressionSymbolPack{
                symbol_key: symbol_key,
                value_key: value_key,
                value_data_len: value_data.len() as u64,
                symbol_data: symbol_data,
                value_data: value_data,
            };
            let mut buffer = Cursor::new(Vec::new());
            packed_symbol_data.write(&mut buffer).unwrap();
            Ok(buffer.into_inner())
            })
        .collect::<PyResult<Vec<Vec<u8>>>>()?
        .into_iter()
        .flatten()
        .collect();
    Ok((symbol_table_map.len() as u64, data))
    // TODO: should also handle extra symbols
}

fn write_parameter_expression(py: Python, py_object: &Bound<PyAny>) -> PyResult<Vec<u8>> {
    let expression_data = write_parameter_expression_elements(py, py_object)?; // should also collect extra symbol table data
    let (symbol_table_length, symbol_table_data) = write_symbol_table(py, py_object)?;
    let expression_data_length = expression_data.len() as u64;
    let packed_expression = ParameterExpressionPack {
        symbol_table_length: symbol_table_length,
        expression_data_length: expression_data_length,
        expression_data: expression_data,
        symbol_table_data: symbol_table_data,
    };
    let mut buffer = Cursor::new(Vec::new());
    packed_expression.write(&mut buffer).unwrap();
    Ok(buffer.into_inner())
}

fn serialize_parameter_expression(py: Python, py_object: &Bound<PyAny>) -> PyResult<SerializableParam> {
    let data = write_parameter_expression(py, py_object)?;
    let data_len = data.len() as u64;
    Ok(SerializableParam { type_key: (tags::PARAMETER_EXPRESSION), data_len: (data_len), data: (data) })
}

fn write_parameter(py: Python, py_object: &Bound<PyAny>) -> PyResult<Vec<u8>> {
    let name = py_object.getattr(intern!(py, "name"))?.extract::<String>()?.into_bytes();
    let uuid_bytes = py_object.getattr(intern!(py, "uuid"))?.getattr(intern!(py, "bytes"))?.extract::<[u8; 16]>()?;
    let packed_parameter = ParameterPack {
        name_length: name.len() as u16,
        uuid: uuid_bytes,
        name: name,
    };
    let mut parameter_buffer = Cursor::new(Vec::new());
    packed_parameter.write(&mut parameter_buffer).unwrap();
    Ok(parameter_buffer.into_inner())
}
fn serialize_parameter(py: Python, py_object: &Bound<PyAny>) -> PyResult<SerializableParam> {
    let data = write_parameter(py, py_object)?;
    Ok(SerializableParam { type_key: (tags::PARAMETER), data_len: data.len() as u64, data: data })
}


fn serialize_parameters(py: Python, py_object: &Bound<PyAny>) -> PyResult<SerializableParam> {
    match get_type_key(py, py_object){
        tags::PARAMETER => serialize_parameter(py, py_object), // The atomic case
        tags::PARAMETER_EXPRESSION => {
            serialize_parameter_expression(py, py_object)
        }
        _ => {
            Ok(SerializableParam { type_key: (tags::NULL), data_len: (0), data: (Vec::new()) })
        }
    }
}

fn serialize_object(py: Python, py_object: &Py<PyAny>) -> SerializableParam {
    //TODO: placeholder for now
    let type_key = get_type_key(py, py_object.bind(py));
    match type_key {
        tags::CIRCUIT => {
            SerializableParam { type_key: (tags::NULL), data_len: (0), data: (Vec::new()) }
        }
        _ => SerializableParam { type_key: (tags::NULL), data_len: (0), data: (Vec::new()) }
    }
}


pub fn param_to_serializable(py: Python, param: &Param) -> SerializableParam {
    match param {
        Param::Float(val) => SerializableParam { type_key: (tags::FLOAT), data_len: (8), data: (val.to_le_bytes().to_vec()) },
        Param::ParameterExpression(py_object) => serialize_parameters(py, py_object.bind(py)).unwrap(),
        Param::Obj(py_object) => serialize_object(py, py_object),
    }
}

