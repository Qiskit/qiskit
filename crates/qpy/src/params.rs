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
use pyo3::exceptions::{PyAttributeError, PyTypeError, PyValueError};
use pyo3::types::{PyAny, PyComplex, PyDict, PyFloat, PyInt, PyIterator, PySet, PyTuple};
use pyo3::{intern, prelude::*, IntoPyObjectExt};
use qiskit_circuit::imports;
use qiskit_circuit::operations::Param;
use qiskit_circuit::parameter::parameter_expression::{self, OPReplay, OpCode, ParameterExpression, ParameterValueType, PyParameter};
use qiskit_circuit::parameter::symbol_expr::Symbol;
use std::vec;
use std::sync::Arc;
use uuid::Uuid;

use crate::bytes::Bytes;
use crate::circuit_reader::load_register;
use crate::formats;
use crate::value::{
    DumpedPyValue, GenericValue, QPYReadData, QPYWriteData, bytes_to_py_uuid, deserialize, deserialize_vec, dumps_py_value, get_type_key, load_value, serialize, tags
};
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
    qpy_data: &QPYWriteData,
) -> PyResult<(u8, [u8; 16], Vec<formats::ParameterExpressionElementPack>)> {
    // This is different from `dumps_py_value` since we aim specifically for [u8; 16]
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
                pack_py_parameter_expression_elements(inst, &mut PyDict::new(py), qpy_data)?;
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
            return Err(PyTypeError::new_err(format!(
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

// fn unpack_parameter_replay_entry(
//     py: Python,
//     opcode: u8,
//     value: [u8; 16],
// ) -> PyResult<Option<Py<PyAny>>> {
//     // unpacks python data: integers, floats, complex numbers
//     match opcode {
//         tags::NULL => Ok(Some(py.None())),
//         tags::INTEGER => {
//             let value = i64::from_be_bytes(value[8..16].try_into()?);
//             Ok(Some(value.into_py_any(py)?))
//         }
//         tags::FLOAT => {
//             let value = f64::from_be_bytes(value[8..16].try_into()?);
//             Ok(Some(value.into_py_any(py)?))
//         }
//         tags::COMPLEX => {
//             let real = f64::from_be_bytes(value[0..8].try_into()?);
//             let imag = f64::from_be_bytes(value[8..16].try_into()?);
//             let complex_value = PyComplex::from_doubles(py, real, imag);
//             Ok(Some(complex_value.into_py_any(py)?))
//         }
//         _ => Ok(None),
//     }
// }

fn pack_replay_subs(
    subs_obj: &Bound<PyAny>,
    extra_symbols: &mut Bound<PyDict>,
    qpy_data: &QPYWriteData,
) -> PyResult<formats::ParameterExpressionElementPack> {
    let py = subs_obj.py();
    let binds = subs_obj.getattr("binds")?;
    extra_symbols.call_method1("update", (&binds,))?;

    let items: Vec<formats::MappingItem> =
        PyIterator::from_object(&binds.cast::<PyDict>()?.items())?
            .map(|item| {
                let (key, value): (Py<PyAny>, Py<PyAny>) = item?.extract()?;
                let key_bytes = key
                    .bind(py)
                    .getattr(intern!(py, "uuid"))?
                    .getattr(intern!(py, "bytes"))?
                    .extract::<Bytes>()?;
                let (item_type, item_bytes) = dumps_py_value(value, qpy_data)?;
                Ok(formats::MappingItem {
                    item_type,
                    key_bytes,
                    item_bytes,
                })
            })
            .collect::<PyResult<_>>()?;
    let mapping = formats::MappingPack { items };
    let mapping_data = serialize(&mapping);
    let entry = formats::ParameterExpressionSubsOpPack { mapping_data };
    Ok(formats::ParameterExpressionElementPack::Substitute(entry))
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
    replay_py_obj: &Bound<PyAny>,
    extra_symbols: &mut Bound<PyDict>,
    qpy_data: &QPYWriteData,
) -> PyResult<Vec<formats::ParameterExpressionElementPack>> {
    let py = replay_py_obj.py();
    let mut result = Vec::new();
    let replay_obj = replay_py_obj.extract::<OPReplay>()?;
    if replay_obj.op == OpCode::SUBSTITUTE {
        return Ok(vec![pack_replay_subs(
            replay_py_obj,
            extra_symbols,
            qpy_data,
        )?]);
    }
    let (lhs_type, lhs, extra_lhs_data) =
        pack_parameter_replay_entry(py, &getattr_or_none(replay_py_obj, "lhs")?, false, qpy_data)?;
    let (rhs_type, rhs, extra_rhs_data) =
        pack_parameter_replay_entry(py, &getattr_or_none(replay_py_obj, "rhs")?, true, qpy_data)?;
    let op_code = replay_obj.op as u8;
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
    match opcode {
        0 => Ok(formats::ParameterExpressionElementPack::Add(data)),
        1 => Ok(formats::ParameterExpressionElementPack::Sub(data)),
        2 => Ok(formats::ParameterExpressionElementPack::Mul(data)),
        3 => Ok(formats::ParameterExpressionElementPack::Div(data)),
        4 => Ok(formats::ParameterExpressionElementPack::Pow(data)),
        5 => Ok(formats::ParameterExpressionElementPack::Sin(data)),
        6 => Ok(formats::ParameterExpressionElementPack::Cos(data)),
        7 => Ok(formats::ParameterExpressionElementPack::Tan(data)),
        8 => Ok(formats::ParameterExpressionElementPack::Asin(data)),
        9 => Ok(formats::ParameterExpressionElementPack::Acos(data)),
        10 => Ok(formats::ParameterExpressionElementPack::Exp(data)),
        11 => Ok(formats::ParameterExpressionElementPack::Log(data)),
        12 => Ok(formats::ParameterExpressionElementPack::Sign(data)),
        13 => Ok(formats::ParameterExpressionElementPack::Grad(data)),
        14 => Ok(formats::ParameterExpressionElementPack::Conj(data)),
        16 => Ok(formats::ParameterExpressionElementPack::Abs(data)),
        17 => Ok(formats::ParameterExpressionElementPack::Atan(data)),
        18 => Ok(formats::ParameterExpressionElementPack::Rsub(data)),
        19 => Ok(formats::ParameterExpressionElementPack::Rdiv(data)),
        20 => Ok(formats::ParameterExpressionElementPack::Rpow(data)),
        255 => Ok(formats::ParameterExpressionElementPack::Expression(data)),
        _ => Err(PyTypeError::new_err(format!("Invalid opcode: {}", opcode))),
    }
}

pub fn unpack_parameter_expression_standard_op(
    packed_parameter: formats::ParameterExpressionElementPack,
) -> PyResult<(u8, formats::ParameterExpressionStandardOpPack)> {
    match packed_parameter {
        formats::ParameterExpressionElementPack::Add(op) => Ok((0, op)),
        formats::ParameterExpressionElementPack::Sub(op) => Ok((1, op)),
        formats::ParameterExpressionElementPack::Mul(op) => Ok((2, op)),
        formats::ParameterExpressionElementPack::Div(op) => Ok((3, op)),
        formats::ParameterExpressionElementPack::Pow(op) => Ok((4, op)),
        formats::ParameterExpressionElementPack::Sin(op) => Ok((5, op)),
        formats::ParameterExpressionElementPack::Cos(op) => Ok((6, op)),
        formats::ParameterExpressionElementPack::Tan(op) => Ok((7, op)),
        formats::ParameterExpressionElementPack::Asin(op) => Ok((8, op)),
        formats::ParameterExpressionElementPack::Acos(op) => Ok((9, op)),
        formats::ParameterExpressionElementPack::Exp(op) => Ok((10, op)),
        formats::ParameterExpressionElementPack::Log(op) => Ok((11, op)),
        formats::ParameterExpressionElementPack::Sign(op) => Ok((12, op)),
        formats::ParameterExpressionElementPack::Grad(op) => Ok((13, op)),
        formats::ParameterExpressionElementPack::Conj(op) => Ok((14, op)),
        formats::ParameterExpressionElementPack::Abs(op) => Ok((16, op)),
        formats::ParameterExpressionElementPack::Atan(op) => Ok((17, op)),
        formats::ParameterExpressionElementPack::Rsub(op) => Ok((18, op)),
        formats::ParameterExpressionElementPack::Rdiv(op) => Ok((19, op)),
        formats::ParameterExpressionElementPack::Rpow(op) => Ok((20, op)),
        formats::ParameterExpressionElementPack::Expression(op) => Ok((255, op)),
        _ => Err(PyTypeError::new_err(format!(
            "Non standard operation {:?}",
            packed_parameter
        ))),
    }
}

fn pack_py_parameter_expression_elements(
    py_object: &Bound<PyAny>,
    extra_symbols: &mut Bound<PyDict>,
    qpy_data: &QPYWriteData,
) -> PyResult<Vec<formats::ParameterExpressionElementPack>> {
    let py = py_object.py();
    let qpy_replay = py_object
        .getattr(intern!(py, "_qpy_replay"))?
        .extract::<Vec<Py<PyAny>>>()?;
    let mut result = Vec::new();
    for replay_obj in qpy_replay.iter() {
        let packed_parameter =
            pack_parameter_expression_element(replay_obj.bind(py), extra_symbols, qpy_data)?;
        result.extend(packed_parameter);
    }
    Ok(result)
}

fn pack_py_symbol(
    symbol: &Bound<PyAny>,
    value: Option<&Bound<PyAny>>,
    qpy_data: &QPYWriteData,
) -> PyResult<formats::ParameterExpressionSymbolPack> {
    let symbol_key = get_type_key(symbol)?;
    let (value_key, value_data): (u8, Bytes) = match value {
        None => (symbol_key, Bytes::new()),
        Some(py_value) => dumps_py_value(py_value.clone().unbind(), qpy_data)?,
    };
    match symbol_key {
        tags::PARAMETER_EXPRESSION => {
            let symbol_data = pack_py_parameter_expression(symbol, qpy_data)?;
            Ok(formats::ParameterExpressionSymbolPack::ParameterExpression(
                formats::ParameterExpressionParameterExpressionSymbolPack {
                    value_key,
                    symbol_data,
                    value_data,
                },
            ))
        }
        tags::PARAMETER => {
            let symbol_data = pack_py_parameter(symbol)?;
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
        _ => Err(PyTypeError::new_err(format!(
            "Unhandled symbol_key: {}",
            symbol_key
        ))),
    }
}

fn pack_symbol_table(
    py: Python,
    py_object: &Bound<PyAny>,
    qpy_data: &QPYWriteData,
) -> PyResult<Vec<formats::ParameterExpressionSymbolPack>> {
    py_object
        .getattr(intern!(py, "parameters"))?
        .extract::<Bound<PySet>>()?
        .iter()
        .map(|symbol| pack_py_symbol(&symbol, None, qpy_data))
        .collect::<PyResult<_>>()
}

fn pack_extra_symbol_table(
    extra_symbols: &Bound<PyDict>,
    qpy_data: &QPYWriteData,
) -> PyResult<(
    Vec<formats::ParameterExpressionSymbolPack>,
    Vec<formats::ParameterExpressionSymbolPack>,
)> {
    let keys = PyIterator::from_object(&extra_symbols.keys())?
        .map(|item| {
            let symbol = item?;
            pack_py_symbol(&symbol, Some(&symbol), qpy_data)
        })
        .collect::<PyResult<_>>()?;
    let values = PyIterator::from_object(&extra_symbols.values())?
        .map(|item| {
            let symbol = item?;
            pack_py_symbol(&symbol, Some(&symbol), qpy_data)
        })
        .collect::<PyResult<_>>()?;
    Ok((keys, values))
}

pub fn pack_py_parameter_expression(
    py_object: &Bound<PyAny>,
    qpy_data: &QPYWriteData,
) -> PyResult<formats::ParameterExpressionPack> {
    println!("**************pack_py_parameter_expression*****************");
    let py = py_object.py();
    let mut extra_symbols = PyDict::new(py);
    println!("py object: {}", py_object);
    let packed_expression_data =
        pack_py_parameter_expression_elements(py_object, &mut extra_symbols, qpy_data)?;
    println!("packed expression data: {:?}", packed_expression_data);
    let expression_data = serialize(&packed_expression_data);
    let mut symbol_table_data: Vec<formats::ParameterExpressionSymbolPack> = pack_symbol_table(py, py_object, qpy_data)?;
    let (extra_symbols_keys, extra_symbols_values) =
        pack_extra_symbol_table(&extra_symbols, qpy_data)?;
    symbol_table_data.extend(extra_symbols_keys);
    symbol_table_data.extend(extra_symbols_values);
    println!("**************pack_py_parameter_expression DONE*****************");
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
        _ => return Err(PyValueError::new_err(format!("Invalid opcode: {}", opcode))),
    };
    Ok(method)
}

fn parameter_value_type_from_generic_value(value: &GenericValue) -> PyResult<ParameterValueType> {
    match value {
        GenericValue::Complex64(complex) => Ok(ParameterValueType::Complex(*complex)),
        GenericValue::Int64(int) => Ok(ParameterValueType::Int(*int)),
        GenericValue::Float64(float) => Ok(ParameterValueType::Float(*float)),
        GenericValue::ParameterExpressionSymbol(symbol) => Ok(ParameterValueType::Parameter(PyParameter{symbol: symbol.clone()}))
        // TODO: need to treat parameter vector differentely
    }
}
pub fn unpack_parameter_expression(
    py: Python,
    parameter_expression: formats::ParameterExpressionPack,
    qpy_data: &mut QPYReadData,
) -> PyResult<ParameterExpression> {
// ) -> PyResult<Py<PyAny>> {
    println!("*************unpacking parameter expression************");
    println!("{:?}", parameter_expression);
    let mut param_uuid_map: HashMap<[u8; 16], GenericValue> = HashMap::new();
    // let mut name_map: HashMap<String, Py<PyAny>> = HashMap::new();

    for item in &parameter_expression.symbol_table_data {
        println!("Going over symbol item: {:?}", item);
        let (symbol_uuid, symbol, value) = match item {
            formats::ParameterExpressionSymbolPack::ParameterExpression(_) => {
                continue;
            }
            formats::ParameterExpressionSymbolPack::Parameter(symbol_pack) => {
                let symbol = unpack_symbol(&symbol_pack.symbol_data);
                let value = match symbol_pack.value_key {
                    parameter_tags::PARAMETER => GenericValue::ParameterExpressionSymbol(symbol.clone()),
                    _ => load_value(symbol_pack.value_key, &symbol_pack.value_data, qpy_data)?
                };
                (symbol_pack.symbol_data.uuid, symbol, value)
            }
            formats::ParameterExpressionSymbolPack::ParameterVector(symbol_pack) => {
                // this call will also create the corresponding vector and update qpy_data if needed
                let symbol = unpack_parameter_vector_symbol(&symbol_pack.symbol_data, qpy_data)?;
                let value = match symbol_pack.value_key {
                    parameter_tags::PARAMETER_VECTOR => GenericValue::ParameterExpressionSymbol(symbol.clone()),
                    _ => load_value(symbol_pack.value_key, &symbol_pack.value_data, qpy_data)?
                };
                (symbol_pack.symbol_data.uuid, symbol, value)
            }
        };
        param_uuid_map.insert(symbol_uuid, value.clone());
        // name_map should only be used for version < 15
        // name_map.insert(
        //     value
        //         .bind(py)
        //         .call_method0("__str__")?
        //         .extract::<String>()?,
        //     symbol,
        // );
    }
    let parameter_expression_data = deserialize_vec::<formats::ParameterExpressionElementPack>(
        &parameter_expression.expression_data,
    )?;
    println!("Unpacked parameter_expression_data = {:?}", parameter_expression_data);

    // we now convert the parameter_expression_data into Vec<OPReplay> that can be used via ParameterExpression::from_qpy
    let mut replay: Vec<OPReplay> = Vec::new();

    for element in parameter_expression_data {
        println!("handling element {:?}", element);
        if let formats::ParameterExpressionElementPack::Substitute(subs) = element {
            // In the python code, substitutions were put on the stack with the rest of the operations
            // And applied during the Parameter Expression construction phase. This seems to be unsupported in the current
            // Rust implementation, so we assume every 

            // we construct a pydictionary describing the substitution and letting the python Parameter class handle it
            // let subs_mapping = PyDict::new(py);
            let mapping_pack = deserialize::<formats::MappingPack>(&subs.mapping_data)?.0;
            let mut subs_mapping: HashMap<Symbol, ParameterExpression> = HashMap::new();
            for item in mapping_pack.items {
                let key_uuid: [u8; 16] = (&item.key_bytes).try_into()?;
                let value_generic_item = load_value(item.item_type, &item.item_bytes, qpy_data)?;
                let key_generic_item = param_uuid_map.get(&key_uuid).ok_or_else(|| {
                    PyValueError::new_err(format!("Parameter UUID not found: {:?}", &key_uuid))
                })?;
                let key = if let GenericValue::ParameterExpressionSymbol(symbol) = key_generic_item {
                    symbol
                } else {
                    return Err(PyValueError::new_err(format!("Substitution command used left operand {:?} which is not a symbol", &key_generic_item)));
                };

                let value = match value_generic_item {
                    GenericValue::ParameterExpressionSymbol(symbol) => ParameterExpression::from_symbol(symbol),
                    _ => {
                        return Err(PyValueError::new_err(format!("Substitution command used right operand {:?} which is not a parameter expression", &value_generic_item)));
                    }
                };                
                subs_mapping.insert(key.clone(), value);
            }
            let opcode = OpCode::SUBSTITUTE;
            // TODO: we can't push this into the stack in the current implementation
            // but maybe after implementing the correct mechanism we can do replay.append(OPReplay {op, lhs: None, rhs: None, Some(subs_mapping)})
            // stack.push(subs_mapping);
            
        } else {
            let (opcode, op) = unpack_parameter_expression_standard_op(element)?;
            println!("opcode: {:?}, op: {:?}", opcode, op);
            // loading values from replay pack is tricky, since everything is stored using 16-bytes, even 8-byte ints and floats
            // LHS
            let lhs: Option<ParameterValueType> = match op.lhs_type {
                parameter_tags::PARAMETER | parameter_tags::PARAMETER_VECTOR => {
                    if let Some(value) = param_uuid_map.get(&op.lhs) {
                        Some(parameter_value_type_from_generic_value(value)?)
                    } else {
                        return Err(PyValueError::new_err(format!(
                            "Parameter UUID not found: {:?}",
                            op.lhs
                        )));
                    }
                }
                parameter_tags::FLOAT | parameter_tags::INTEGER | parameter_tags::COMPLEX => {
                    let value = load_value(op.lhs_type, &op.lhs.into(), qpy_data)?;
                    Some(parameter_value_type_from_generic_value(&value)?)
                }
                parameter_tags::NULL => None, // pass
                parameter_tags::LHS_EXPRESSION | parameter_tags::RHS_EXPRESSION => continue,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown ParameterExpression operation type: {}",
                        op.lhs_type
                    )))
                }
            };
            println!("got lhs: {:?}", lhs);
            // RHS
            let rhs: Option<ParameterValueType> = match op.rhs_type {
                parameter_tags::PARAMETER | parameter_tags::PARAMETER_VECTOR => {
                    if let Some(value) = param_uuid_map.get(&op.rhs) {
                        Some(parameter_value_type_from_generic_value(value)?)
                    } else {
                        return Err(PyValueError::new_err(format!(
                            "Parameter UUID not found: {:?}",
                            op.rhs
                        )));
                    }
                }
                parameter_tags::FLOAT | parameter_tags::INTEGER | parameter_tags::COMPLEX => {
                    let value = load_value(op.rhs_type, &op.rhs.into(), qpy_data)?;
                    Some(parameter_value_type_from_generic_value(&value)?)
                }
                parameter_tags::NULL => None, // pass
                parameter_tags::LHS_EXPRESSION | parameter_tags::RHS_EXPRESSION => continue,
                _ => {
                    return Err(PyTypeError::new_err(format!(
                        "Unknown ParameterExpression operation type: {}",
                        op.rhs_type
                    )))
                }
            };
            println!("got rhs: {:?}", rhs);
            // if opcode == 255 {
            //     continue;
            // }
            let op = OpCode::from_u8(opcode)?;
            replay.push(OPReplay { op, lhs, rhs });
        };
        println!("Replay: {:?}", replay);
        // we don't need the rest of this, it's covered in from_qpy
        // let method_str = op_code_to_method(opcode)?;

        // if [0, 1, 2, 3, 4, 13, 15, 18, 19, 20].contains(&opcode) {
        //     let rhs = stack.pop().ok_or(PyTypeError::new_err(
        //         "Stack underflow while parsing parameter expression",
        //     ))?;
        //     let lhs = stack.pop().ok_or(PyTypeError::new_err(
        //         "Stack underflow while parsing parameter expression",
        //     ))?;
        //     // Reverse ops for commutative ops, which are add, mul (0 and 2 respectively)
        //     // op codes 13 and 15 can never be reversed and 18, 19, 20
        //     // are the reversed versions of non-commuative operations
        //     // so 1, 3, 4 and 18, 19, 20 handle this explicitly.
        //     if [0, 2].contains(&opcode)
        //         && !lhs
        //             .bind(py)
        //             .is_instance(imports::PARAMETER_EXPRESSION.get_bound(py))?
        //         && rhs
        //             .bind(py)
        //             .is_instance(imports::PARAMETER_EXPRESSION.get_bound(py))?
        //     {
        //         let method_str = match &opcode {
        //             0 => "__radd__",
        //             2 => "__rmul__",
        //             _ => method_str,
        //         };
        //         stack.push(rhs.getattr(py, method_str)?.call1(py, (lhs,))?);
        //     } else {
        //         stack.push(lhs.getattr(py, method_str)?.call1(py, (rhs,))?);
        //     }
        // } else {
        //     // unary op
        //     let lhs = stack.pop().ok_or(PyValueError::new_err(
        //         "Stack underflow while parsing parameter expression",
        //     ))?;
        //     stack.push(lhs.getattr(py, method_str)?.call0(py)?);
        // }
    }
    let result = ParameterExpression::from_qpy(&replay).map_err(|_| {
        PyValueError::new_err(format!(
            "Failure while loading parameter expression"
        ))
    });
    println!("Got result {:?}", result);
    result

    // let result = stack.pop().ok_or(PyValueError::new_err(
    //     "Stack underflow while parsing parameter expression",
    // ))?;
    // Ok(result)
}

pub fn pack_py_parameter(py_object: &Bound<PyAny>) -> PyResult<formats::ParameterPack> {
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

// pub fn unpack_parameter(py: Python, parameter: &formats::ParameterPack) -> PyResult<Py<PyAny>> {
//     let kwargs = PyDict::new(py);
//     kwargs.set_item("name", parameter.name.clone())?;
//     kwargs.set_item("uuid", bytes_to_uuid(py, parameter.uuid)?)?;
//     Ok(imports::PARAMETER
//         .get_bound(py)
//         .call((), Some(&kwargs))?
//         .unbind())
// }
pub fn pack_symbol(symbol: Symbol) -> formats::ParameterPack {
    let uuid = symbol.uuid.as_bytes().clone();
    let name = symbol.name.clone();
    formats::ParameterPack {uuid, name}
}

pub fn unpack_symbol(parameter_pack: &formats::ParameterPack) -> Symbol {
    let name = parameter_pack.name.clone();
    let uuid = Uuid::from_bytes(parameter_pack.uuid);
    Symbol{name, uuid, index: None, vector: None}
}

// parameter vector symbols are currently much more tricky than standalone symbols
// since we don't have a rust-space concept of ParameterVector; it is a pure python object
// which we must manage while creating its elements. Moreover, the vector itself is not stored anywhere
// so we need to create it in an ad-hoc fashion as we encounter its elements during the parsing of the
// qpy file. In particular we need to keep our qpy_data nearby so we can update the vector list as needed
// and we must use python calls to create and modify the python-space ParameterVector
pub fn unpack_parameter_vector_symbol(parameter_vector_pack: &formats::ParameterVectorPack, qpy_data: &mut QPYReadData) -> PyResult<Symbol> {
    let name = parameter_vector_pack.name.clone();
    let uuid = Uuid::from_bytes(parameter_vector_pack.uuid);
    let index = parameter_vector_pack.index as u32; // sadly, the `Symbol` class does not conform to the qpy u64 format
    // we have extracted the rust-space data, but now we must deal with the python-space vector class

    // first get the uuid for the vector's "root" (it's first element)
    // we rely on the convention that the uuid's for the vector elements are sequential
    let root_uuid_int = uuid.as_u128() - (index as u128);
    let root_uuid = Uuid::from_bytes(root_uuid_int.to_be_bytes());

    let vector = Python::attach(|py| -> PyResult<_> {  // we use python-space to interface with the ParameterVector data
        let vector_data = match qpy_data.vectors.get_mut(&root_uuid) {
            Some(value) => value,
            None => Python::attach(|py| -> PyResult<_> {  // we use python-space to create a new parameter vector
                let vector = imports::PARAMETER_VECTOR
                    .get_bound(py)
                    .call1((name.clone(), parameter_vector_pack.vector_size))?
                    .unbind();
                qpy_data.vectors.insert(root_uuid, (vector, Vec::new()));
                Ok(qpy_data.vectors.get_mut(&root_uuid).unwrap())
            })?
        };
        let vector = vector_data.0.bind(py);
        let vector_element = vector.get_item(index)?;
        let vector_element_uuid = Uuid::from_bytes(
            vector_element
                .getattr(intern!(py, "uuid"))?
                .getattr(intern!(py, "bytes"))?
                .extract::<[u8; 16]>()?,
        );
        if vector_element_uuid != uuid {
            // we need to create a new parameter vector element and hack it into the vector
            vector_data.1.push(index);
            let param_vector_element = imports::PARAMETER_VECTOR_ELEMENT
                .get_bound(py)
                .call1((vector, index, bytes_to_py_uuid(py, parameter_vector_pack.uuid)?))?
                .unbind();
            vector
                .getattr("_params")?
                .set_item(index, param_vector_element)?;
        }
        Ok(vector.clone().unbind())
    })?;

    Ok(Symbol{name, uuid, index: Some(index), vector: Some(vector)})
}


// sadly, we currently need this code duplication to handle the special le encoding for parameters
pub fn pack_generic_instruction_param_data(
    py_data: &Bound<PyAny>,
    qpy_data: &QPYWriteData,
) -> PyResult<formats::GenericDataPack> {
    let (type_key, data) = dumps_instruction_param_value(py_data, qpy_data)?;
    Ok(formats::GenericDataPack { type_key, data })
}

pub fn pack_generic_instruction_param_sequence(
    py_sequence: &Bound<PyAny>,
    qpy_data: &QPYWriteData,
) -> PyResult<formats::GenericDataSequencePack> {
    let elements: Vec<formats::GenericDataPack> = py_sequence
        .try_iter()?
        .map(|possible_data_item| {
            let data_item = possible_data_item?;
            pack_generic_instruction_param_data(&data_item, qpy_data)
        })
        .collect::<PyResult<_>>()?;
    Ok(formats::GenericDataSequencePack { elements })
}

pub fn unpack_generic_instruction_param_sequence_to_tuple(
    py: Python,
    packed_sequence: formats::GenericDataSequencePack,
    qpy_data: &mut QPYReadData,
) -> PyResult<Py<PyAny>> {
    let elements: Vec<Py<PyAny>> = packed_sequence
        .elements
        .iter()
        .map(|data_pack| {
            load_instruction_param_value(py, data_pack.type_key, &data_pack.data, qpy_data)
        })
        .collect::<PyResult<_>>()?;
    PyTuple::new(py, elements)?.into_py_any(py)
}

pub fn dumps_param_expression(
    exp: &ParameterExpression,
    qpy_data: &QPYWriteData,
) -> PyResult<(u8, Bytes)> {
    println!("dumps_param_expression called with exp {:?}", exp);
    // if the parameter expression is a single symbol, we should treat it like a parameter
    let result = if let Ok(symbol) = exp.try_to_symbol() {
        let packed_symbol = pack_symbol(symbol);
        (tags::PARAMETER, serialize(&packed_symbol))
    } else  {
        Python::attach(|py| {
            dumps_py_value(exp.clone().into_py_any(py)?, qpy_data)}
        )?
    };    
    Ok(result)
}

pub fn dumps_instruction_param_value(
    py_object: &Bound<PyAny>,
    qpy_data: &QPYWriteData,
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
        )?),
        tags::REGISTER => dumps_register_param(py_object)?,
        _ => {
            let (_, value) = dumps_py_value(py_object.clone().unbind(), qpy_data)?;
            value
        }
    };
    Ok((type_key, value))
}

pub fn load_instruction_param_value(
    py: Python,
    type_key: u8,
    data: &Bytes,
    qpy_data: &mut QPYReadData,
) -> PyResult<Py<PyAny>> {
Ok(match type_key {
        tags::INTEGER => {
            let value = i64::from_le_bytes(data[..8].try_into()?);
            PyInt::new(py, value).into()
        }
        tags::FLOAT => {
            let value = f64::from_le_bytes(data[..8].try_into()?);
            PyFloat::new(py, value).into()
        }
        tags::TUPLE => unpack_generic_instruction_param_sequence_to_tuple(
            py,
            deserialize::<formats::GenericDataSequencePack>(data)?.0,
            qpy_data,
        )?,
        tags::REGISTER => load_register(py, data.clone(), qpy_data.circuit_data)?,
        _ => DumpedPyValue {
            data_type: type_key,
            data: data.clone(),
        }
        .to_python(py, qpy_data)?,
    })
}

pub fn dumps_register_param(register: &Bound<PyAny>) -> PyResult<Bytes> {
    let py = register.py();
    if register.is_instance(imports::CLASSICAL_REGISTER.get_bound(py))? {
        Ok(register.getattr("name")?.extract::<String>()?.into())
    } else {
        let index: usize = register.getattr("_index")?.extract()?;
        let index_string = index.to_string().as_bytes().to_vec();
        let mut result = Bytes(vec![0x00]);
        result.extend_from_slice(&index_string);
        Ok(result)
    }
}

pub fn pack_param_obj(
    py: Python,
    param: &Param,
    qpy_data: &QPYWriteData,
) -> PyResult<formats::PackedParam> {
    println!("Hello world from pack_param_obj");
    let (type_key, data) = match param {
        Param::Float(val) => (tags::FLOAT, val.to_le_bytes().into()), // using le instead of be for this QPY version
        Param::ParameterExpression(exp) => {
            println!("got exp = {:?}", exp);
            dumps_param_expression(exp, qpy_data)?
        }
        Param::Obj(py_object) => dumps_instruction_param_value(py_object.bind(py), qpy_data)?,
    };
    Ok(formats::PackedParam { type_key, data })
}

pub fn pack_param(
    py_object: &Bound<PyAny>,
    qpy_data: &QPYWriteData,
) -> PyResult<formats::PackedParam> {
    let (type_key, data) = dumps_instruction_param_value(py_object, qpy_data)?;
    Ok(formats::PackedParam { type_key, data })
}

pub fn unpack_param(
    py: Python,
    packed_param: &formats::PackedParam,
    qpy_data: &mut QPYReadData,
) -> PyResult<Param> {
    match packed_param.type_key {
        tags::FLOAT => Ok(Param::Float(packed_param.data.try_to_le_f64()?)),
        tags::PARAMETER => {
            let (packed_symbol, _) = deserialize::<formats::ParameterPack>(&packed_param.data)?;
            let parameter_expression = ParameterExpression::from_symbol(unpack_symbol(&packed_symbol));
            Ok(Param::ParameterExpression(
                Arc::new(parameter_expression),
            ))
        }
        tags::PARAMETER_EXPRESSION | tags::PARAMETER_VECTOR => {
            let dumped_value = DumpedPyValue {
                data_type: packed_param.type_key,
                data: packed_param.data.clone(),
            };
            println!("unpack_param called for tag {:?} and data {:?}", packed_param.type_key, dumped_value.data);
            let (packed_parameter_expression, _) =
                    deserialize::<formats::ParameterExpressionPack>(&dumped_value.data)?;
            println!("parameter expression pack: {:?}", packed_parameter_expression);
            let parameter_expression = unpack_parameter_expression(py, packed_parameter_expression, qpy_data)?;
            Ok(Param::ParameterExpression(
                // Arc::new(parameter_expression),
                Arc::new(parameter_expression),
            ))
        }
        _ => {
            let param_value = load_instruction_param_value(
                py,
                packed_param.type_key,
                &packed_param.data,
                qpy_data,
            )?;
            Ok(Param::Obj(param_value))
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
    // let index = py_object.getattr("_index")?.extract::<u64>()?;
    let index = py_object.getattr("index")?.extract::<u64>()?;
    Ok(formats::ParameterVectorPack {
        vector_size,
        uuid,
        index,
        name,
    })
}

// pub fn unpack_parameter_vector(
//     py: Python,
//     pack: &formats::ParameterVectorPack,
//     qpy_data: &mut QPYReadData,
// ) -> PyResult<Py<PyAny>> {
//     let root_uuid_int = u128::from_be_bytes(pack.uuid) - (pack.index as u128);
//     let root_uuid = Uuid::from_bytes(root_uuid_int.to_be_bytes());
//     let vector_data = match qpy_data.vectors.get_mut(&root_uuid) {
//         Some(value) => value,
//         None => {
//             let vector = imports::PARAMETER_VECTOR
//                 .get_bound(py)
//                 .call1((pack.name.clone(), pack.vector_size))?
//                 .unbind();
//             qpy_data.vectors.insert(root_uuid, (vector, Vec::new()));
//             qpy_data.vectors.get_mut(&root_uuid).unwrap()
//         }
//     };
//     let vector = vector_data.0.bind(py);
//     let vector_element = vector.get_item(pack.index)?;
//     let vector_element_uuid = Uuid::from_bytes(
//         vector_element
//             .getattr(intern!(py, "uuid"))?
//             .getattr(intern!(py, "bytes"))?
//             .extract::<[u8; 16]>()?,
//     );
//     if vector_element_uuid != Uuid::from_bytes(pack.uuid) {
//         // we need to create a new parameter vector element and hack it into the vector
//         vector_data.1.push(pack.index);
//         let param_vector_element = imports::PARAMETER_VECTOR_ELEMENT
//             .get_bound(py)
//             .call1((vector, pack.index, bytes_to_uuid(py, pack.uuid)?))?
//             .unbind();
//         vector
//             .getattr("_params")?
//             .set_item(pack.index, param_vector_element)?;
//     }
//     Ok(vector.get_item(pack.index)?.unbind().clone())
// }
