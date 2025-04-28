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
use pyo3::PyObject;
use crate::formats::{
    Bytes, ParameterExpressionElementPack, ParameterExpressionPack, ParameterExpressionSymbolPack,
    ParameterPack, PackedParam, MappingItem, MappingItemHeader, MappingPack, ExtraSymbolsTablePack
};
use crate::value::{get_type_key, dumps_value, tags};
use binrw::BinWrite;
use pyo3::basic::CompareOp;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyComplex, PyDict, PyFloat, PyInt, PyIterator, PyString};
use qiskit_circuit::imports::{PARAMETER, PARAMETER_EXPRESSION, QUANTUM_CIRCUIT, PARAMETER_SUBS};
use qiskit_circuit::operations::Param;
use std::io::{Cursor, Write};


// For debugging purposes
fn hex_string(bytes: &Bytes) -> String {
    bytes
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
}

fn serialize_parameter_replay_entry(py: Python, inst: &Bound<PyAny>) -> PyResult<(u8, [u8; 16])> {
    // This is different from `dumps_value` since we aim specifically for [u8; 16]
    // This means parameters are not fully stored, only their uuid
    // Also integers and floats are padded with 0
    let key_type = get_type_key(py, inst)?;
    let data = match key_type {
        tags::PARAMETER => inst
            .getattr(intern!(py, "uuid"))?
            .getattr(intern!(py, "bytes"))?
            .extract::<[u8; 16]>()?,
        tags::NULL => [0u8; 16],
        tags::INTEGER => 0u64
            .to_be_bytes()
            .into_iter()
            .chain(inst.extract::<i64>()?.to_be_bytes())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(),
        tags::FLOAT => 0u64
            .to_be_bytes()
            .into_iter()
            .chain(inst.extract::<f64>()?.to_be_bytes())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(),
        _ => return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Unhandled key_type: {}", key_type))),
    }; // TODO: should also handle complex and ParameterExpression cases
    Ok((key_type, data))
}

fn serialize_replay_subs(
    py: Python,
    subs_obj: &Bound<PyAny>,
    extra_symbols: &mut Bound<PyDict>
) -> PyResult<Bytes> {
    let mut buffer = Cursor::new(Vec::new());
    let binds  = subs_obj.getattr("binds")?;
    extra_symbols.call_method1("update", (&binds,))?;

    let num_elements = binds.call_method0("__len__")?.extract::<u64>()?;

    let items: Vec<MappingItem> = PyIterator::from_object(&binds.downcast::<PyDict>()?.items())?
    .map(|item| {
        let (key, value): (PyObject, PyObject) = item?.extract()?;
        let name = key.getattr(py,"name")?.extract::<String>(py)?;
        let key_bytes = name.into_bytes();
        let (item_type, item_bytes) = dumps_value(py, value.bind(py))?;
        let item_header = MappingItemHeader {
            key_size: key_bytes.len() as u16,
            item_type,
            size: item_bytes.len()as u16,
        };
        Ok(MappingItem {
            item_header,
            key_bytes,
            item_bytes,
        })
    }).collect::<PyResult<_>>()?;
    let mapping = MappingPack {
        num_elements,
        items,
    };
    let mut mapping_buffer = Cursor::new(Vec::new());
    mapping.write(&mut mapping_buffer).unwrap();
    let mapping_data = mapping_buffer.into_inner();
    let mapping_data_size: [u8; 8] = (mapping_data.len() as u64).to_be_bytes();
    let mut lhs = [0u8; 16];
    lhs[..8].copy_from_slice(&mapping_data_size);
    let entry = ParameterExpressionElementPack {
        op_code: subs_obj.getattr("op")?.extract::<u8>()?,
        lhs_type: "u".as_bytes()[0],
        lhs,
        rhs_type: "n".as_bytes()[0],
        rhs: [0u8; 16],
    };
    entry.write(&mut buffer).unwrap();
    buffer.write_all(&mapping_data)?;
    Ok(buffer.into_inner())
}

fn serialize_parameter_expression_element(
    py: Python,
    replay_obj: &Bound<PyAny>,
    extra_symbols: &mut Bound<PyDict>
) -> PyResult<Bytes> {
    if replay_obj.is_instance(PARAMETER_SUBS.get_bound(py)).unwrap() {
        return serialize_replay_subs(py, replay_obj, extra_symbols);
    }
    let (lhs_type, lhs) =
        serialize_parameter_replay_entry(py, &replay_obj.getattr(intern!(py, "lhs"))?)?;
    let (rhs_type, rhs) =
        serialize_parameter_replay_entry(py, &replay_obj.getattr(intern!(py, "rhs"))?)?;
    let op_code = replay_obj.getattr(intern!(py, "op"))?.extract::<u8>()?;
    let packed_element = ParameterExpressionElementPack {
        op_code,
        lhs_type,
        lhs,
        rhs_type,
        rhs,
    };
    let mut buffer = Cursor::new(Vec::new());
    packed_element.write(&mut buffer).unwrap();
    Ok(buffer.into_inner())
}
fn serialize_parameter_expression_elements(py: Python, py_object: &Bound<PyAny>, extra_symbols: &mut Bound<PyDict>) -> PyResult<Bytes> {
    let qpy_replay = py_object
        .getattr(intern!(py, "_qpy_replay"))?
        .extract::<Vec<PyObject>>()?;
    let result_data: Bytes = qpy_replay
        .iter()
        .flat_map(|replay_obj| {
            serialize_parameter_expression_element(py, &replay_obj.bind(py), extra_symbols).unwrap()
        })
        .collect();
    Ok(result_data)
}

fn pack_symbol(py: Python, symbol: &Bound<PyAny>, value: &Bound<PyAny>) -> PyResult<ParameterExpressionSymbolPack> {
    let symbol_key = get_type_key(py, &symbol)?;
    let symbol_data: Bytes = match symbol_key {
        tags::PARAMETER => serialize_parameter(py, &symbol)?,
        _ => return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Unhandled symbol_key: {}", symbol_key))),
    };

    let (value_key, value_data): (u8, Bytes) = match value
        .rich_compare(symbol.getattr("_symbol_expr")?, CompareOp::Eq)?
        .is_truthy()?
    {
        true => (symbol_key, Vec::new()),
        false => dumps_value(py, &value)?,
    };

    Ok(ParameterExpressionSymbolPack {
        symbol_key: symbol_key,
        value_key: value_key,
        value_data_len: value_data.len() as u64,
        symbol_data: symbol_data,
        value_data: value_data,
    })
}
fn serialize_symbol_table(py: Python, py_object: &Bound<PyAny>) -> PyResult<(u64, Bytes)> {
    let symbol_table_map: Bound<PyDict> = py_object
        .getattr(intern!(py, "_parameter_symbols"))?
        .extract()?;
    let data = symbol_table_map
        .iter()
        .map(|(symbol, value)| {
            let packed_symbol_data = pack_symbol(py, &symbol, &value)?;
            let mut buffer = Cursor::new(Vec::new());
            packed_symbol_data.write(&mut buffer).unwrap();
            Ok(buffer.into_inner())
        })
        .collect::<PyResult<Vec<Bytes>>>()?
        .into_iter()
        .flatten()
        .collect();
    Ok((symbol_table_map.len() as u64, data))
}

fn serialize_extra_symbol_table(py: Python, extra_symbols: &Bound<PyDict>) -> PyResult<Bytes> {
    let keys = PyIterator::from_object(&extra_symbols.keys())?
    .map(|item| {
        let symbol = item?;
        Ok(pack_symbol(py, &symbol, &symbol)?)
    }).collect::<PyResult<_>>()?;

    let values = PyIterator::from_object(&extra_symbols.values())?
    .map(|item| {
        let symbol = item?;
        Ok(pack_symbol(py, &symbol, &symbol)?)
    }).collect::<PyResult<_>>()?;
    let extra_symbol_table = ExtraSymbolsTablePack{
        keys,
        values
    };
    let mut buffer = Cursor::new(Vec::new());
    extra_symbol_table.write(&mut buffer).unwrap(); // TODO: don't unwrap, propagate the error
    Ok(buffer.into_inner())
}

pub fn serialize_parameter_expression(py: Python, py_object: &Bound<PyAny>) -> PyResult<Bytes> {
    let mut extra_symbols = PyDict::new(py);
    let expression_data = serialize_parameter_expression_elements(py, py_object, &mut extra_symbols)?;
    let (symbol_table_length, symbol_table_data) = serialize_symbol_table(py, py_object)?;
    let extra_symbol_table_data = serialize_extra_symbol_table(py, &extra_symbols)?;
    let symbol_tables_length = symbol_table_length + 2*extra_symbols.call_method0("__len__")?.extract::<u64>()?;
    let expression_data_length = expression_data.len() as u64;
    let packed_expression = ParameterExpressionPack {
        symbol_tables_length,
        expression_data_length,
        expression_data,
        symbol_table_data,
        extra_symbol_table_data
    };
    let mut buffer = Cursor::new(Vec::new());
    packed_expression.write(&mut buffer).unwrap();
    Ok(buffer.into_inner())
}

pub fn serialize_parameter(py: Python, py_object: &Bound<PyAny>) -> PyResult<Bytes> {
    let name = py_object
        .getattr(intern!(py, "name"))?
        .extract::<String>()?
        .into_bytes();
    let uuid_bytes = py_object
        .getattr(intern!(py, "uuid"))?
        .getattr(intern!(py, "bytes"))?
        .extract::<[u8; 16]>()?;
    let packed_parameter = ParameterPack {
        name_length: name.len() as u16,
        uuid: uuid_bytes,
        name: name,
    };
    let mut parameter_buffer = Cursor::new(Vec::new());
    packed_parameter.write(&mut parameter_buffer).unwrap();
    Ok(parameter_buffer.into_inner())
}

pub fn pack_param(py: Python, param: &Param) -> PackedParam {
    let (type_key, data) =
    match param {
        Param::Float(val) => (tags::FLOAT, val.to_le_bytes().to_vec()),
        Param::ParameterExpression(py_object) | Param::Obj(py_object) => dumps_value(py, py_object.bind(py)).unwrap(),
    };
    PackedParam {
        type_key,
        data_len: data.len() as u64,
        data
    }
}
