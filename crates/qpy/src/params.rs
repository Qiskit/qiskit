use pyo3::exceptions::PyAttributeError;
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
use qiskit_circuit::circuit_data;
use crate::formats::{
    Bytes, ParameterExpressionElementPack, ParameterExpressionPack, ParameterExpressionSymbolPack,
    ParameterPack, ParameterVectorPack, PackedParam, MappingItem, MappingItemHeader, MappingPack, ExtraSymbolsTablePack, GenericDataPack, GenericDataSequencePack
};
use crate::value::{get_type_key, dumps_value, serialize, tags, dumps_register, QPYData};
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

fn serialize_parameter_replay_entry(py: Python, inst: &Bound<PyAny>, r_side: bool, qpy_data: &QPYData) -> PyResult<(u8, [u8; 16], Bytes)> {
    // This is different from `dumps_value` since we aim specifically for [u8; 16]
    // This means parameters are not fully stored, only their uuid
    // Also integers and floats are padded with 0
    let mut extra_data = Bytes::new();
    let key_type = get_type_key(py, inst)?; // TODO: get_key_type won't work! it should return "n" for None, but it returns it for numpy
    println!("serialize_parameter_replay_entry, got key type {:}", key_type);
    let data = match key_type {
        tags::PARAMETER | tags::PARAMETER_VECTOR => inst
            .getattr(intern!(py, "uuid"))?
            .getattr(intern!(py, "bytes"))?
            .extract::<[u8; 16]>()?,
        tags::NULL => [0u8; 16],
        tags::COMPLEX => {
            let mut complex_data = [0u8; 16];
            let real = inst.getattr("real")?.extract::<f64>()?.to_be_bytes();
            let imag = inst.getattr("imag")?.extract::<f64>()?.to_be_bytes();
            complex_data[0..8].copy_from_slice(&real);
            complex_data[8..16].copy_from_slice(&imag);
            complex_data
        }
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
        tags::PARAMETER_EXPRESSION => {
            let mut buffer = Cursor::new(Vec::new());
            println!("RUST: printing parameter expression for inst {:?}", inst);
            let entry = if r_side {
                ParameterExpressionElementPack {
                    op_code: 255,
                    lhs_type: "n".as_bytes()[0],
                    lhs: [0u8; 16],
                    rhs_type: "s".as_bytes()[0],
                    rhs: [0u8; 16],
                }
            } else {
                ParameterExpressionElementPack {
                    op_code: 255,
                    lhs_type: "s".as_bytes()[0],
                    lhs: [0u8; 16],
                    rhs_type: "n".as_bytes()[0],
                    rhs: [0u8; 16],
                }
            };
            entry.write(&mut buffer).unwrap();
            println!("RUST: first entry {:?}", hex_string(&buffer.clone().into_inner()));
            let serialized_expression = serialize_parameter_expression_elements(py, inst, &mut PyDict::new(py), qpy_data)?;
            println!("RUST: serialized expression {:?}", hex_string(&serialized_expression));
            buffer.write_all(&serialized_expression).unwrap();
            let entry = if r_side {
                ParameterExpressionElementPack {
                    op_code: 255,
                    lhs_type: "n".as_bytes()[0],
                    lhs: [0u8; 16],
                    rhs_type: "e".as_bytes()[0],
                    rhs: [0u8; 16],
                }
            } else {
                ParameterExpressionElementPack {
                    op_code: 255,
                    lhs_type: "e".as_bytes()[0],
                    lhs: [0u8; 16],
                    rhs_type: "n".as_bytes()[0],
                    rhs: [0u8; 16],
                }
            };
            entry.write(&mut buffer).unwrap();
            extra_data = buffer.into_inner();
            [0u8; 16] // return empty
        } 
        _ => return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Unhandled key_type: {}", key_type))),
    };
    let key_type = match key_type {
        tags::NULL | tags::PARAMETER_EXPRESSION => tags::NUMPY_OBJ, // in parameter replay, none is not stored as 'z' but as 'n'
        tags::PARAMETER_VECTOR => tags::PARAMETER, // in parameter replay, treat parameters and parameter vector elements the same way
        _ => key_type 
    };
    Ok((key_type, data, extra_data))
}

fn serialize_replay_subs(
    py: Python,
    subs_obj: &Bound<PyAny>,
    extra_symbols: &mut Bound<PyDict>,
    qpy_data: &QPYData,
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
        let (item_type, item_bytes) = dumps_value(py, value.bind(py), qpy_data)?;
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

fn getattr_or_none<'py>(py_object: &'py Bound<PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    match py_object.getattr(name) {
        Ok(attr) => Ok(attr),
        Err(err) => {
            if err.is_instance_of::<PyAttributeError>(py_object.py()) {
                Ok(py_object.py().None().bind(py_object.py()).clone())
            }
            else {
                Err(err)
            }
        }
    }
}

fn serialize_parameter_expression_element(
    py: Python,
    replay_obj: &Bound<PyAny>,
    extra_symbols: &mut Bound<PyDict>,
    qpy_data: &QPYData,
) -> PyResult<Bytes> {
    if replay_obj.is_instance(PARAMETER_SUBS.get_bound(py)).unwrap() {
        return serialize_replay_subs(py, replay_obj, extra_symbols, qpy_data);
    }
    let (lhs_type, lhs, extra_lhs_data) =
        serialize_parameter_replay_entry(py, &getattr_or_none(replay_obj,"lhs")?, false, qpy_data)?;
    let (rhs_type, rhs, extra_rhs_data) =
        serialize_parameter_replay_entry(py, &getattr_or_none(replay_obj,"rhs")?, true, qpy_data)?;
    println!("RUST: about to read opcode from {:?}", replay_obj);
    println!("RUST replay_obj.hasattr('op')={:?}",replay_obj.hasattr("op")?);
    println!("RUST replay_obj.__class__={:?}",replay_obj.getattr("__class__")?);
    let op_code = replay_obj.getattr(intern!(py, "op"))?.extract::<u8>()?;
    println!("RUST: Serializing op_code {:?}", op_code);
    println!("RUST: Serializing lhs_type {:?}", lhs_type);
    let packed_element = ParameterExpressionElementPack {
        op_code,
        lhs_type,
        lhs,
        rhs_type,
        rhs,
    };
    let mut buffer = Cursor::new(Vec::new());
    buffer.write_all(&extra_lhs_data).unwrap();
    buffer.write_all(&extra_rhs_data).unwrap();
    packed_element.write(&mut buffer).unwrap();
    Ok(buffer.into_inner())
}
fn serialize_parameter_expression_elements(py: Python, py_object: &Bound<PyAny>, extra_symbols: &mut Bound<PyDict>, qpy_data: &QPYData) -> PyResult<Bytes> {
    let qpy_replay = py_object
        .getattr(intern!(py, "_qpy_replay"))?
        .extract::<Vec<PyObject>>()?;
    let result_data: Bytes = qpy_replay
        .iter()
        .flat_map(|replay_obj| {
            serialize_parameter_expression_element(py, &replay_obj.bind(py), extra_symbols, qpy_data).unwrap()
        })
        .collect();
    Ok(result_data)
}

fn pack_symbol(py: Python, symbol: &Bound<PyAny>, value: &Bound<PyAny>, qpy_data: &QPYData) -> PyResult<ParameterExpressionSymbolPack> {
    println!("Packing symbol {:?}", symbol);
    let symbol_key = get_type_key(py, &symbol)?;
    let symbol_data: Bytes = match symbol_key {
        tags::PARAMETER => serialize_parameter(py, &symbol)?,
        tags::PARAMETER_EXPRESSION => serialize_parameter_expression(py, &symbol, qpy_data)?,
        tags::PARAMETER_VECTOR => serialize_parameter_vector(py, &symbol)?,
        _ => return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Unhandled symbol_key: {}", symbol_key))),
    };

    let (value_key, value_data): (u8, Bytes) = match value
        .rich_compare(symbol.getattr("_symbol_expr")?, CompareOp::Eq)?
        .is_truthy()?
    {
        true => (symbol_key, Vec::new()),
        false => dumps_value(py, &value, qpy_data)?,
    };

    Ok(ParameterExpressionSymbolPack {
        symbol_key: symbol_key,
        value_key: value_key,
        value_data_len: value_data.len() as u64,
        symbol_data: symbol_data,
        value_data: value_data,
    })
}
fn serialize_symbol_table(py: Python, py_object: &Bound<PyAny>, qpy_data: &QPYData) -> PyResult<(u64, Bytes)> {
    let symbol_table_map: Bound<PyDict> = py_object
        .getattr(intern!(py, "_parameter_symbols"))?
        .extract()?;
    let data = symbol_table_map
        .iter()
        .map(|(symbol, value)| {
            let packed_symbol_data = pack_symbol(py, &symbol, &value, qpy_data)?;
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

fn serialize_extra_symbol_table(py: Python, extra_symbols: &Bound<PyDict>, qpy_data: &QPYData) -> PyResult<Bytes> {
    let keys = PyIterator::from_object(&extra_symbols.keys())?
    .map(|item| {
        let symbol = item?;
        Ok(pack_symbol(py, &symbol, &symbol, qpy_data)?)
    }).collect::<PyResult<_>>()?;

    let values = PyIterator::from_object(&extra_symbols.values())?
    .map(|item| {
        let symbol = item?;
        Ok(pack_symbol(py, &symbol, &symbol, qpy_data)?)
    }).collect::<PyResult<_>>()?;
    let extra_symbol_table = ExtraSymbolsTablePack{
        keys,
        values
    };
    let mut buffer = Cursor::new(Vec::new());
    extra_symbol_table.write(&mut buffer).unwrap(); // TODO: don't unwrap, propagate the error
    Ok(buffer.into_inner())
}

pub fn serialize_parameter_expression(py: Python, py_object: &Bound<PyAny>, qpy_data: &QPYData) -> PyResult<Bytes> {
    let mut extra_symbols = PyDict::new(py);
    let expression_data = serialize_parameter_expression_elements(py, py_object, &mut extra_symbols, qpy_data)?;
    let (symbol_table_length, symbol_table_data) = serialize_symbol_table(py, py_object, qpy_data)?;
    let extra_symbol_table_data = serialize_extra_symbol_table(py, &extra_symbols, qpy_data)?;
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

// sadly, we currently need this code duplication to handle the special le encoding for parameters
pub fn pack_generic_instruction_param_data(py: Python, py_data: &Bound<PyAny>, qpy_data: &QPYData) -> PyResult<GenericDataPack> {
    let (type_key, data) = dumps_instruction_param_value(py, py_data, qpy_data)?;
    Ok(GenericDataPack { type_key, data_len: data.len() as u64, data})
}

pub fn pack_generic_instruction_param_sequence(py: Python, py_sequence: &Bound<PyAny>, qpy_data: &QPYData) -> PyResult<GenericDataSequencePack> {
    let elements: Vec<GenericDataPack> = py_sequence
    .try_iter()?
    .map(|possible_data_item| {
        let data_item = possible_data_item?;
        pack_generic_instruction_param_data(py, &data_item,qpy_data)
    })
    .collect::<PyResult<_>>()?;
    Ok(GenericDataSequencePack { num_elements: elements.len() as u64, elements })
}

pub fn dumps_instruction_param_value(py: Python, py_object: &Bound<PyAny>, qpy_data: &QPYData) -> PyResult<(u8, Bytes)> {
    // we need a hack here to encode floats and integers are little endian
    // since for some reason it was done in the original python code
    // TODO This should be fixed in next QPY version.
    println!("Dumping instruction_param_value: {:?}", py_object);
    let type_key: u8 = get_type_key(py, py_object)?;
    println!("Dumping instruction_param_value with type key {:?}", type_key);
    let value: Bytes = match type_key {
        tags::INTEGER => py_object.extract::<i64>()?.to_le_bytes().to_vec(),
        tags::FLOAT => py_object.extract::<f64>()?.to_le_bytes().to_vec(),
        tags::TUPLE => serialize(pack_generic_instruction_param_sequence(py, py_object, qpy_data)?)?,
        tags::REGISTER => dumps_register(py, py_object)?,
        _ => {
            let (_, value) = dumps_value(py, py_object, qpy_data)?;
            value
        }
    };
    println!("Dumped value: {:?}", hex_string(&value));
    Ok((type_key, value))
}

pub fn pack_param(py: Python, param: &Param, qpy_data: &QPYData) -> PyResult<PackedParam> {
    println!("Packing param {:?}", param);
    let (type_key, data) =
    match param {
        Param::Float(val) => (tags::FLOAT, val.to_le_bytes().to_vec()), // using le instead of be for this QPY version
        Param::ParameterExpression(py_object) => dumps_value(py, py_object.bind(py), qpy_data)?,
        Param::Obj(py_object) => dumps_instruction_param_value(py, py_object.bind(py), qpy_data)?
    };
    Ok(PackedParam {
        type_key,
        data_len: data.len() as u64,
        data
    })
}

pub fn serialize_parameter_vector(py: Python, py_object: &Bound<PyAny>) -> PyResult<Bytes> {
    let vector = py_object.getattr("_vector")?;
    let name = vector.getattr("_name")?.extract::<String>()?;
    let name_bytes = name.as_bytes().to_vec();
    let vector_size = vector.call_method0("__len__")?.extract()?;
    let uuid = py_object.getattr("uuid")?.getattr("bytes")?.extract::<[u8; 16]>()?;
    let index = py_object.getattr("_index")?.extract::<u64>()?;
    let packed_parameter_vector = ParameterVectorPack {
        vector_name_size: name_bytes.len() as u16,
        vector_size,
        uuid,
        index,
        name_bytes,
    };
    let mut buffer = Cursor::new(Vec::new());
    packed_parameter_vector.write(&mut buffer).unwrap();
    Ok(buffer.into_inner())
}   

