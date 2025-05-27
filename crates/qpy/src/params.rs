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
use pyo3::basic::CompareOp;
use pyo3::exceptions::PyAttributeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyComplex;
use pyo3::types::{PyAny, PyDict, PyIterator};
use pyo3::PyObject;
use qiskit_circuit::imports::PARAMETER_SUBS;
use qiskit_circuit::operations::Param;
use std::vec;

use crate::bytes::Bytes;
use crate::formats;
use crate::formats::PackedParam;
use crate::value::DumpedValue;
use crate::value::{
    bytes_to_uuid, dumps_register, dumps_value, get_type_key, serialize, tags, QPYData,
};
use crate::value::{deserialize, deserialize_vec};
use hashbrown::HashMap;

pub mod parameter_tags {
    pub const INTEGER: u8 = b'i';
    pub const FLOAT: u8 = b'f';
    pub const COMPLEX: u8 = b'c';
    pub const PARAMETER: u8 = b'p';
    pub const PARAMETER_VECTOR: u8 = b'v';
    pub const NULL: u8 = b'n';
    pub const LHS_EXPRESSION: u8 = b's';
    pub const RHS_EXPRESSION: u8 = b'e';
}

fn pack_parameter_replay_entry(
    py: Python,
    inst: &Bound<PyAny>,
    r_side: bool,
    qpy_data: &QPYData,
) -> PyResult<(u8, [u8; 16], Vec<formats::ParameterExpressionElementPack>)> {
    // This is different from `dumps_value` since we aim specifically for [u8; 16]
    // This means parameters are not fully stored, only their uuid
    // Also integers and floats are padded with 0
    let mut extra_data = Vec::new();
    let key_type = get_type_key(inst)?;
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
            let entry = if r_side {
                formats::ParameterExpressionElementPack::Expression(
                    formats::ParameterExpressionStandardOpPack {
                        lhs_type: parameter_tags::NULL,
                        lhs: [0u8; 16],
                        rhs_type: parameter_tags::LHS_EXPRESSION,
                        rhs: [0u8; 16],
                    },
                )
            } else {
                formats::ParameterExpressionElementPack::Expression(
                    formats::ParameterExpressionStandardOpPack {
                        lhs_type: parameter_tags::LHS_EXPRESSION,
                        lhs: [0u8; 16],
                        rhs_type: parameter_tags::NULL,
                        rhs: [0u8; 16],
                    },
                )
            };
            extra_data.push(entry);
            let packed_expression =
                pack_parameter_expression_elements(inst, &mut PyDict::new(py), qpy_data)?;
            extra_data.extend(packed_expression);
            let entry = if r_side {
                formats::ParameterExpressionElementPack::Expression(
                    formats::ParameterExpressionStandardOpPack {
                        lhs_type: parameter_tags::NULL,
                        lhs: [0u8; 16],
                        rhs_type: parameter_tags::RHS_EXPRESSION,
                        rhs: [0u8; 16],
                    },
                )
            } else {
                formats::ParameterExpressionElementPack::Expression(
                    formats::ParameterExpressionStandardOpPack {
                        lhs_type: parameter_tags::RHS_EXPRESSION,
                        lhs: [0u8; 16],
                        rhs_type: parameter_tags::NULL,
                        rhs: [0u8; 16],
                    },
                )
            };
            extra_data.push(entry);
            [0u8; 16] // return empty
        }
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "Unhandled key_type: {}",
                key_type
            )))
        }
    };
    let key_type = match key_type {
        tags::NULL | tags::PARAMETER_EXPRESSION => tags::NUMPY_OBJ, // in parameter replay, none is not stored as 'z' but as 'n'
        tags::PARAMETER_VECTOR => tags::PARAMETER, // in parameter replay, treat parameters and parameter vector elements the same way
        _ => key_type,
    };
    Ok((key_type, data, extra_data))
}

fn unpack_parameter_replay_entry(
    py: Python,
    opcode: u8,
    value: [u8; 16],
) -> PyResult<Option<PyObject>> {
    // unpacks python data: integers, floats, complex numbers
    match opcode {
        tags::NULL => Ok(Some(py.None())),
        tags::INTEGER => {
            let value = u64::from_be_bytes(value[8..16].try_into()?);
            Ok(Some(value.into_pyobject(py)?.into_any().unbind()))
        }
        tags::FLOAT => {
            let value = f64::from_be_bytes(value[8..16].try_into()?);
            Ok(Some(value.into_pyobject(py)?.into_any().unbind()))
        }
        tags::COMPLEX => {
            let real = f64::from_be_bytes(value[0..8].try_into()?);
            let imag = f64::from_be_bytes(value[8..16].try_into()?);
            let complex_value = PyComplex::from_doubles(py, real, imag);
            Ok(Some(complex_value.into_any().unbind()))
        }
        _ => Ok(None),
    }
}
fn pack_replay_subs(
    subs_obj: &Bound<PyAny>,
    extra_symbols: &mut Bound<PyDict>,
    qpy_data: &QPYData,
) -> PyResult<formats::ParameterExpressionElementPack> {
    let py = subs_obj.py();
    let binds = subs_obj.getattr("binds")?;
    extra_symbols.call_method1("update", (&binds,))?;

    let items: Vec<formats::MappingItem> =
        PyIterator::from_object(&binds.downcast::<PyDict>()?.items())?
            .map(|item| {
                let (key, value): (PyObject, PyObject) = item?.extract()?;
                let name = key
                    .getattr(py, intern!(py, "name"))?
                    .extract::<String>(py)?;
                let key_bytes: Bytes = name.into();
                let (item_type, item_bytes) = dumps_value(value.bind(py), qpy_data)?;
                Ok(formats::MappingItem {
                    item_type,
                    key_bytes,
                    item_bytes,
                })
            })
            .collect::<PyResult<_>>()?;
    let mapping = formats::MappingPack { items };
    let mapping_data = serialize(&mapping)?;
    let entry = formats::ParameterExpressionSubsOpPack { mapping_data };
    Ok(formats::ParameterExpressionElementPack::Substitute(entry))
}

fn unpack_mapping<'py>(
    py: Python<'py>,
    mapping_data: &formats::MappingPack,
    qpy_data: &mut QPYData,
) -> PyResult<Bound<'py, PyDict>> {
    let py_dict = PyDict::new(py);
    for item in &mapping_data.items {
        // the key type is always assumed to be a string
        let key: String = (&item.key_bytes).try_into()?;
        let value = DumpedValue {
            data_type: item.item_type,
            data: Bytes(item.item_bytes.clone()),
        };
        let value_py = value.to_python(py, qpy_data)?;
        py_dict.set_item(key, value_py)?;
    }
    Ok(py_dict)
}

fn getattr_or_none<'py>(py_object: &'py Bound<PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    match py_object.getattr(name) {
        Ok(attr) => Ok(attr),
        Err(err) => {
            if err.is_instance_of::<PyAttributeError>(py_object.py()) {
                Ok(py_object.py().None().bind(py_object.py()).clone())
            } else {
                Err(err)
            }
        }
    }
}

fn pack_parameter_expression_element(
    replay_obj: &Bound<PyAny>,
    extra_symbols: &mut Bound<PyDict>,
    qpy_data: &QPYData,
) -> PyResult<Vec<formats::ParameterExpressionElementPack>> {
    let py = replay_obj.py();
    let mut result = Vec::new();
    if replay_obj
        .is_instance(PARAMETER_SUBS.get_bound(py))
        .unwrap()
    {
        return Ok(vec![pack_replay_subs(replay_obj, extra_symbols, qpy_data)?]);
    }
    let (lhs_type, lhs, extra_lhs_data) =
        pack_parameter_replay_entry(py, &getattr_or_none(replay_obj, "lhs")?, false, qpy_data)?;
    let (rhs_type, rhs, extra_rhs_data) =
        pack_parameter_replay_entry(py, &getattr_or_none(replay_obj, "rhs")?, true, qpy_data)?;
    let op_code = replay_obj.getattr(intern!(py, "op"))?.extract::<u8>()?;
    let entry = formats::ParameterExpressionStandardOpPack {
        lhs_type,
        lhs,
        rhs_type,
        rhs,
    };
    let packed_parameter = vec![pack_parameter_expression_by_op(op_code, entry)?];
    result.extend(extra_lhs_data);
    result.extend(extra_rhs_data);
    result.extend(packed_parameter);
    Ok(result)
}

fn pack_parameter_expression_by_op(
    opcode: u8,
    data: formats::ParameterExpressionStandardOpPack,
) -> PyResult<formats::ParameterExpressionElementPack> {
    use formats::ParameterExpressionElementPack::*;
    match opcode {
        0 => Ok(Add(data)),
        1 => Ok(Sub(data)),
        2 => Ok(Mul(data)),
        3 => Ok(Div(data)),
        4 => Ok(Pow(data)),
        5 => Ok(Sin(data)),
        6 => Ok(Cos(data)),
        7 => Ok(Tan(data)),
        8 => Ok(Asin(data)),
        9 => Ok(Acos(data)),
        10 => Ok(Exp(data)),
        11 => Ok(Log(data)),
        12 => Ok(Sign(data)),
        13 => Ok(Grad(data)),
        14 => Ok(Conj(data)),
        16 => Ok(Abs(data)),
        17 => Ok(Atan(data)),
        18 => Ok(Rsub(data)),
        19 => Ok(Rdiv(data)),
        20 => Ok(Rpow(data)),
        255 => Ok(Expression(data)),
        _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Invalid opcode: {}",
            opcode
        ))),
    }
}

pub fn unpack_parameter_expression_standard_op(
    packed_parameter: formats::ParameterExpressionElementPack,
) -> PyResult<(u8, formats::ParameterExpressionStandardOpPack)> {
    use formats::ParameterExpressionElementPack::*;
    match packed_parameter {
        Add(op) => Ok((0, op)),
        Sub(op) => Ok((1, op)),
        Mul(op) => Ok((2, op)),
        Div(op) => Ok((3, op)),
        Pow(op) => Ok((4, op)),
        Sin(op) => Ok((5, op)),
        Cos(op) => Ok((6, op)),
        Tan(op) => Ok((7, op)),
        Asin(op) => Ok((8, op)),
        Acos(op) => Ok((9, op)),
        Exp(op) => Ok((10, op)),
        Log(op) => Ok((11, op)),
        Sign(op) => Ok((12, op)),
        Grad(op) => Ok((13, op)),
        Conj(op) => Ok((14, op)),
        Abs(op) => Ok((16, op)),
        Atan(op) => Ok((17, op)),
        Rsub(op) => Ok((18, op)),
        Rdiv(op) => Ok((19, op)),
        Rpow(op) => Ok((20, op)),
        Expression(op) => Ok((255, op)),
        _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Non standard operation {:?}",
            packed_parameter
        ))),
    }
}

fn pack_parameter_expression_elements(
    py_object: &Bound<PyAny>,
    extra_symbols: &mut Bound<PyDict>,
    qpy_data: &QPYData,
) -> PyResult<Vec<formats::ParameterExpressionElementPack>> {
    let py = py_object.py();
    let qpy_replay = py_object
        .getattr(intern!(py, "_qpy_replay"))?
        .extract::<Vec<PyObject>>()?;
    let mut result = Vec::new();
    for replay_obj in qpy_replay.iter() {
        let packed_parameter =
            pack_parameter_expression_element(replay_obj.bind(py), extra_symbols, qpy_data)?;
        result.extend(packed_parameter);
    }
    Ok(result)
}

fn pack_symbol(
    symbol: &Bound<PyAny>,
    value: &Bound<PyAny>,
    qpy_data: &QPYData,
) -> PyResult<formats::ParameterExpressionSymbolPack> {
    let symbol_key = get_type_key(symbol)?;
    let (value_key, value_data): (u8, Bytes) = match value
        .rich_compare(symbol.getattr("_symbol_expr")?, CompareOp::Eq)?
        .is_truthy()?
    {
        true => (symbol_key, Bytes::new()),
        false => dumps_value(value, qpy_data)?,
    };
    match symbol_key {
        tags::PARAMETER_EXPRESSION => {
            let symbol_data = pack_parameter_expression(symbol, qpy_data)?;
            Ok(formats::ParameterExpressionSymbolPack::ParameterExpression(
                formats::ParameterExpressionParameterExpressionSymbolPack {
                    value_key,
                    symbol_data,
                    value_data,
                },
            ))
        }
        tags::PARAMETER => {
            let symbol_data = pack_parameter(symbol)?;
            Ok(formats::ParameterExpressionSymbolPack::Parameter(
                formats::ParameterExpressionParameterSymbolPack {
                    value_key,
                    symbol_data,
                    value_data,
                },
            ))
        }
        tags::PARAMETER_VECTOR => {
            let symbol_data = pack_parameter_vector(symbol)?;
            Ok(formats::ParameterExpressionSymbolPack::ParameterVector(
                formats::ParameterExpressionParameterVectorSymbolPack {
                    value_key,
                    symbol_data,
                    value_data,
                },
            ))
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Unhandled symbol_key: {}",
            symbol_key
        ))),
    }
}

fn pack_symbol_table(
    py: Python,
    py_object: &Bound<PyAny>,
    qpy_data: &QPYData,
) -> PyResult<Vec<formats::ParameterExpressionSymbolPack>> {
    py_object
        .getattr(intern!(py, "_parameter_symbols"))?
        .extract::<Bound<PyDict>>()?
        .iter()
        .map(|(symbol, value)| pack_symbol(&symbol, &value, qpy_data))
        .collect::<PyResult<_>>()
}

fn pack_extra_symbol_table(
    extra_symbols: &Bound<PyDict>,
    qpy_data: &QPYData,
) -> PyResult<(
    Vec<formats::ParameterExpressionSymbolPack>,
    Vec<formats::ParameterExpressionSymbolPack>,
)> {
    let keys = PyIterator::from_object(&extra_symbols.keys())?
        .map(|item| {
            let symbol = item?;
            pack_symbol(&symbol, &symbol, qpy_data)
        })
        .collect::<PyResult<_>>()?;
    let values = PyIterator::from_object(&extra_symbols.values())?
        .map(|item| {
            let symbol = item?;
            pack_symbol(&symbol, &symbol, qpy_data)
        })
        .collect::<PyResult<_>>()?;
    Ok((keys, values))
}

pub fn pack_parameter_expression(
    py_object: &Bound<PyAny>,
    qpy_data: &QPYData,
) -> PyResult<formats::ParameterExpressionPack> {
    let py = py_object.py();
    let mut extra_symbols = PyDict::new(py);
    let packed_expression_data =
        pack_parameter_expression_elements(py_object, &mut extra_symbols, qpy_data)?;
    let expression_data = serialize(&packed_expression_data)?;
    let mut symbol_table_data = pack_symbol_table(py, py_object, qpy_data)?;
    let (extra_symbols_keys, extra_symbols_values) =
        pack_extra_symbol_table(&extra_symbols, qpy_data)?;
    symbol_table_data.extend(extra_symbols_keys);
    symbol_table_data.extend(extra_symbols_values);
    Ok(formats::ParameterExpressionPack {
        expression_data,
        symbol_table_data,
    })
}

fn op_code_to_method(opcode: u8) -> PyResult<&'static str> {
    let method = match opcode {
        0 => "__add__",
        1 => "__sub__",
        2 => "__mul__",
        3 => "__truediv__",
        4 => "__pow__",
        5 => "sin",
        6 => "cos",
        7 => "tan",
        8 => "arcsin",
        9 => "arccos",
        10 => "exp",
        11 => "log",
        12 => "sign",
        13 => "gradient",
        14 => "conjugate",
        15 => "subs",
        16 => "abs",
        17 => "arctan",
        18 => "__rsub__",
        19 => "__rtruediv__",
        20 => "__rpow__",
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid opcode: {}",
                opcode
            )))
        }
    };
    Ok(method)
}
pub fn unpack_parameter_expression(
    py: Python,
    parameter_expression: formats::ParameterExpressionPack,
    qpy_data: &mut QPYData,
) -> PyResult<PyObject> {
    let mut param_uuid_map: HashMap<[u8; 16], PyObject> = HashMap::new();
    let mut name_map: HashMap<String, PyObject> = HashMap::new();

    let mut stack: Vec<PyObject> = Vec::new();
    for item in &parameter_expression.symbol_table_data {
        let (symbol_uuid, symbol, value) = match item {
            formats::ParameterExpressionSymbolPack::ParameterExpression(_) => {
                continue;
            }
            formats::ParameterExpressionSymbolPack::Parameter(symbol_pack) => {
                let symbol = unpack_parameter(py, &symbol_pack.symbol_data)?;
                let value = if symbol_pack.value_key != parameter_tags::PARAMETER {
                    let dumped_value = DumpedValue {
                        data_type: symbol_pack.value_key,
                        data: Bytes(symbol_pack.value_data.clone()),
                    };
                    dumped_value.to_python(py, qpy_data)?
                } else {
                    symbol.clone()
                };
                (symbol_pack.symbol_data.uuid, symbol, value)
            }
            formats::ParameterExpressionSymbolPack::ParameterVector(symbol_pack) => {
                let symbol = unpack_parameter_vector(py, &symbol_pack.symbol_data, qpy_data)?;
                let value = if symbol_pack.value_key != parameter_tags::PARAMETER_VECTOR {
                    let dumped_value = DumpedValue {
                        data_type: symbol_pack.value_key,
                        data: Bytes(symbol_pack.value_data.clone()),
                    };
                    dumped_value.to_python(py, qpy_data)?
                } else {
                    symbol.clone()
                };
                (symbol_pack.symbol_data.uuid, symbol, value)
            }
        };
        param_uuid_map.insert(symbol_uuid, value.clone());
        name_map.insert(
            value
                .bind(py)
                .call_method0("__str__")?
                .extract::<String>()?,
            symbol,
        );
    }
    let parameter_expression_data = deserialize_vec::<formats::ParameterExpressionElementPack>(
        &parameter_expression.expression_data,
    )?;

    for element in parameter_expression_data {
        let opcode = if let formats::ParameterExpressionElementPack::Substitute(subs) = element {
            // we construct a pydictionary describing the substitution and letting the python Parameter class handle it
            let (mapping_pack, _) = deserialize::<formats::MappingPack>(&subs.mapping_data)?;
            let mapping = unpack_mapping(py, &mapping_pack, qpy_data)?;
            let subs_mapping = PyDict::new(py);
            for item in mapping.iter() {
                let key: String = item.0.extract()?;
                let value = item.1;
                subs_mapping.set_item(name_map.get(&key), value)?;
            }
            stack.push(subs_mapping.unbind().as_any().clone());
            15 // return substitution opcode
        } else {
            let (opcode, op) = unpack_parameter_expression_standard_op(element)?;
            // LHS
            match op.lhs_type {
                parameter_tags::PARAMETER | parameter_tags::PARAMETER_VECTOR => {
                    if let Some(value) = param_uuid_map.get(&op.lhs) {
                        stack.push(value.clone());
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Parameter UUID not found: {:?}",
                            op.lhs
                        )));
                    }
                }
                parameter_tags::FLOAT | parameter_tags::INTEGER | parameter_tags::COMPLEX => {
                    if let Some(value) = unpack_parameter_replay_entry(py, op.lhs_type, op.lhs)? {
                        stack.push(value);
                    }
                }
                parameter_tags::NULL => (), // pass
                parameter_tags::LHS_EXPRESSION | parameter_tags::RHS_EXPRESSION => continue,
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unknown ParameterExpression operation type: {}",
                        op.lhs_type
                    )))
                }
            }
            // RHS
            match op.rhs_type {
                parameter_tags::PARAMETER | parameter_tags::PARAMETER_VECTOR => {
                    if let Some(value) = param_uuid_map.get(&op.rhs) {
                        stack.push(value.clone());
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Parameter UUID not found: {:?}",
                            op.rhs
                        )));
                    }
                }
                parameter_tags::FLOAT | parameter_tags::INTEGER | parameter_tags::COMPLEX => {
                    if let Some(value) = unpack_parameter_replay_entry(py, op.rhs_type, op.rhs)? {
                        stack.push(value);
                    }
                }
                parameter_tags::NULL => (), // pass
                parameter_tags::LHS_EXPRESSION | parameter_tags::RHS_EXPRESSION => continue,
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                        "Unknown ParameterExpression operation type: {}",
                        op.rhs_type
                    )))
                }
            }
            if opcode == 255 {
                continue;
            }
            opcode
        };
        let method_str = op_code_to_method(opcode)?;

        if [0, 1, 2, 3, 4, 13, 15, 18, 19, 20].contains(&opcode) {
            let rhs = stack.pop().ok_or(pyo3::exceptions::PyValueError::new_err(
                "Stack underflow while parsing parameter expression",
            ))?;
            let lhs = stack.pop().ok_or(pyo3::exceptions::PyValueError::new_err(
                "Stack underflow while parsing parameter expression",
            ))?;
            // TODO: in the python code they switch order for commutatve operations - why?

            stack.push(lhs.getattr(py, method_str)?.call1(py, (rhs,))?);
        } else {
            // unary op
            let lhs = stack.pop().ok_or(pyo3::exceptions::PyValueError::new_err(
                "Stack underflow while parsing parameter expression",
            ))?;
            stack.push(lhs.getattr(py, method_str)?.call0(py)?);
        }
    }

    let result = stack.pop().ok_or(pyo3::exceptions::PyValueError::new_err(
        "Stack underflow while parsing parameter expression",
    ))?;
    Ok(result)
}

pub fn pack_parameter(py_object: &Bound<PyAny>) -> PyResult<formats::ParameterPack> {
    let py = py_object.py();
    let name = py_object
        .getattr(intern!(py, "name"))?
        .extract::<String>()?;
    let uuid = py_object
        .getattr(intern!(py, "uuid"))?
        .getattr(intern!(py, "bytes"))?
        .extract::<[u8; 16]>()?;
    Ok(formats::ParameterPack { uuid, name })
}

pub fn unpack_parameter(py: Python, parameter: &formats::ParameterPack) -> PyResult<PyObject> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("name", parameter.name.clone())?;
    kwargs.set_item("uuid", bytes_to_uuid(py, parameter.uuid)?)?;
    Ok(py
        .import("qiskit.circuit.parameter")?
        .getattr("Parameter")?
        .call((), Some(&kwargs))?
        .unbind())
}

// sadly, we currently need this code duplication to handle the special le encoding for parameters
pub fn pack_generic_instruction_param_data(
    py_data: &Bound<PyAny>,
    qpy_data: &QPYData,
) -> PyResult<formats::GenericDataPack> {
    let (type_key, data) = dumps_instruction_param_value(py_data, qpy_data)?;
    Ok(formats::GenericDataPack { type_key, data })
}

pub fn pack_generic_instruction_param_sequence(
    py_sequence: &Bound<PyAny>,
    qpy_data: &QPYData,
) -> PyResult<formats::GenericDataSequencePack> {
    let elements: Vec<formats::GenericDataPack> = py_sequence
        .try_iter()?
        .map(|possible_data_item| {
            let data_item = possible_data_item?;
            pack_generic_instruction_param_data(&data_item, qpy_data)
        })
        .collect::<PyResult<_>>()?;
    Ok(formats::GenericDataSequencePack {
        num_elements: elements.len() as u64,
        elements,
    })
}

pub fn dumps_instruction_param_value(
    py_object: &Bound<PyAny>,
    qpy_data: &QPYData,
) -> PyResult<(u8, Bytes)> {
    // we need a hack here to encode floats and integers are little endian
    // since for some reason it was done in the original python code
    // TODO This should be fixed in next QPY version.
    let type_key: u8 = get_type_key(py_object)?;
    let value: Bytes = match type_key {
        tags::INTEGER => py_object.extract::<i64>()?.to_le_bytes().into(),
        tags::FLOAT => py_object.extract::<f64>()?.to_le_bytes().into(),
        tags::TUPLE => serialize(&pack_generic_instruction_param_sequence(
            py_object, qpy_data,
        )?)?,
        tags::REGISTER => dumps_register(py_object)?,
        _ => {
            let (_, value) = dumps_value(py_object, qpy_data)?;
            value
        }
    };
    Ok((type_key, value))
}

pub fn pack_param(py: Python, param: &Param, qpy_data: &QPYData) -> PyResult<formats::PackedParam> {
    let (type_key, data) = match param {
        Param::Float(val) => (tags::FLOAT, val.to_le_bytes().into()), // using le instead of be for this QPY version
        Param::ParameterExpression(py_object) => dumps_value(py_object.bind(py), qpy_data)?,
        Param::Obj(py_object) => dumps_instruction_param_value(py_object.bind(py), qpy_data)?,
    };
    Ok(formats::PackedParam { type_key, data })
}

pub fn unpack_param(
    py: Python,
    packed_param: &PackedParam,
    qpy_data: &mut QPYData,
) -> PyResult<Param> {
    match packed_param.type_key {
        tags::FLOAT => Ok(Param::Float(packed_param.data.try_to_le_f64()?)),
        tags::PARAMETER_EXPRESSION | tags::PARAMETER => {
            // TODO - should we also do this for parameter vector?
            let dumped_value = DumpedValue {
                data_type: packed_param.type_key,
                data: Bytes(packed_param.data.clone()),
            };
            Ok(Param::ParameterExpression(
                dumped_value.to_python(py, qpy_data)?,
            ))
        }
        _ => {
            // TODO cloning the data in order to leverage DumpedValue's converter is far from optimal, we should find
            // as way to use some common interface for both DumpedValue and PackedParam conversions
            let dumped_value = DumpedValue {
                data_type: packed_param.type_key,
                data: Bytes(packed_param.data.clone()),
            };
            Ok(Param::Obj(dumped_value.to_python(py, qpy_data)?))
        }
    }
}
pub fn pack_parameter_vector(py_object: &Bound<PyAny>) -> PyResult<formats::ParameterVectorPack> {
    let vector = py_object.getattr("_vector")?;
    let name = vector.getattr("_name")?.extract::<String>()?;
    let vector_size = vector.call_method0("__len__")?.extract()?;
    let uuid = py_object
        .getattr("uuid")?
        .getattr("bytes")?
        .extract::<[u8; 16]>()?;
    let index = py_object.getattr("_index")?.extract::<u64>()?;
    Ok(formats::ParameterVectorPack {
        vector_size,
        uuid,
        index,
        name,
    })
}

pub fn unpack_parameter_vector(
    py: Python,
    pack: &formats::ParameterVectorPack,
    qpy_data: &mut QPYData,
) -> PyResult<PyObject> {
    let vector = match qpy_data.vectors.get(&pack.name) {
        Some(vector) => vector,
        None => {
            let vector = py
                .import("qiskit.circuit.parametervector")?
                .getattr("ParameterVector")?
                .call1((pack.name.clone(), pack.vector_size))?
                .unbind();
            qpy_data.vectors.insert(pack.name.clone(), vector);
            qpy_data.vectors.get(&pack.name).unwrap()
        }
    }
    .bind(py);
    let vector_element = vector.get_item(pack.index)?;
    let vector_element_uuid = vector_element
        .getattr(intern!(py, "uuid"))?
        .getattr(intern!(py, "bytes"))?
        .extract::<[u8; 16]>()?;
    if vector_element_uuid != pack.uuid {
        // we need to create a new parameter vector element and hack it into the vector
        let param_vector_element = py
            .import("qiskit.circuit.parametervector")?
            .getattr("ParameterVectorElement")?
            .call1((vector, pack.index, bytes_to_uuid(py, pack.uuid)?))?
            .unbind();
        vector
            .getattr("_params")?
            .set_item(pack.index, param_vector_element)?;
    }
    Ok(vector.get_item(pack.index)?.unbind().clone())
}
