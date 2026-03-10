// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
use binrw::Endian;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use qiskit_circuit::imports;
use qiskit_circuit::operations::Param;
use qiskit_circuit::parameter::parameter_expression::{
    OPReplay, OpCode, ParameterExpression, ParameterValueType, PyParameter,
};
use qiskit_circuit::parameter::symbol_expr::Symbol;
use std::sync::Arc;
use uuid::Uuid;

use crate::bytes::Bytes;
use crate::formats;
use crate::py_methods::{py_convert_from_generic_value, py_pack_param};
use crate::value::{
    GenericValue, QPYReadData, QPYWriteData, ValueType, deserialize, deserialize_vec, load_value,
    pack_generic_value, serialize,
};
use binrw::binrw;
use hashbrown::HashMap;

// The various values of values that can exist in a parameter expression node
// This data is stored inside the parent of the node, not in the node itself
// So it has two "dummy" values, LhsExpression and RhsExpression indicating that
// The node has a non-leaf child that the expression reconstruction algorithm should recurse into
// In addition null represents a missing child.
// The nodes can have concrete integer/float/complex values, or be symbols (standalone/part of vector)
#[binrw]
#[brw(repr = u8)]
#[repr(u8)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ParameterType {
    Integer = b'i',
    Float = b'f',
    Complex = b'c',
    Parameter = b'p',
    ParameterVector = b'v',
    Null = b'n',
    LhsExpression = b's',
    RhsExpression = b'e',
}

fn parameter_type_name(type_key: &ParameterType) -> String {
    String::from(match type_key {
        ParameterType::Integer => "integer",
        ParameterType::Float => "float",
        ParameterType::Complex => "complex",
        ParameterType::Parameter => "parameter",
        ParameterType::ParameterVector => "parameter vector",
        ParameterType::Null => "null",
        ParameterType::LhsExpression => "lhs expression",
        ParameterType::RhsExpression => "rhs expression",
    })
}

impl std::fmt::Display for ParameterType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", parameter_type_name(self),)
    }
}

impl TryFrom<ParameterType> for ValueType {
    type Error = PyErr;
    fn try_from(value: ParameterType) -> Result<Self, Self::Error> {
        match value {
            ParameterType::Complex => Ok(ValueType::Complex),
            ParameterType::Float => Ok(ValueType::Float),
            ParameterType::Integer => Ok(ValueType::Integer),
            ParameterType::Null => Ok(ValueType::Null),
            ParameterType::ParameterVector => Ok(ValueType::ParameterVector),
            ParameterType::Parameter => Ok(ValueType::Parameter),
            _ => Err(PyValueError::new_err(
                "Cannot convert to value type {value}",
            )),
        }
    }
}

pub(crate) fn pack_parameter_expression_by_op(
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

pub(crate) fn unpack_parameter_expression_standard_op(
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

fn parameter_value_type_from_generic_value(value: &GenericValue) -> PyResult<ParameterValueType> {
    match value {
        GenericValue::Complex64(complex) => Ok(ParameterValueType::Complex(*complex)),
        GenericValue::Int64(int) => Ok(ParameterValueType::Int(*int)),
        GenericValue::Float64(float) => Ok(ParameterValueType::Float(*float)),
        GenericValue::ParameterExpressionSymbol(symbol) => {
            Ok(ParameterValueType::Parameter(PyParameter {
                symbol: symbol.clone(),
            }))
        }
        _ => Err(PyValueError::new_err(
            "Data value that cannot be stored as a parameter value",
        )),
    }
}

// To store a parameter expression, we keep two pieces of data:
// 1) The **expression data** which is an already serialized Vec<formats::ParameterExpressionElementPack>
// which is an encoding of the replay used to reconstruct the expression, where each element encodes a specific operation
// with the operand data stored in ParameterExpressionStandardOpPack
// 2) The **symbol_table_data** where the symbols appearing anywhere in the expression are stored; only their uuid values
// are referred to in the expression data
// in older QPY versions, parameter expressions could have substitute commands, which made packing more complex
// this is no longer used in the rust-based parameter expressions, so we do not fully utilize the formats
pub(crate) fn pack_parameter_expression(
    exp: &ParameterExpression,
) -> PyResult<formats::ParameterExpressionPack> {
    let packed_expression_data = pack_parameter_expression_elements(exp)?;
    let expression_data = serialize(&packed_expression_data);
    let symbol_table_data: Vec<formats::ParameterExpressionSymbolPack> = exp
        .iter_symbols()
        .map(pack_symbol_table_element)
        .collect::<PyResult<_>>()?;
    Ok(formats::ParameterExpressionPack {
        expression_data,
        symbol_table_data,
    })
}

fn pack_symbol_table_element(symbol: &Symbol) -> PyResult<formats::ParameterExpressionSymbolPack> {
    let value_data = Bytes::new(); // this was used only when packing symbol tables related to substitution commands and no longer relevant
    if symbol.is_vector_element() {
        let value_key = ValueType::ParameterVector;
        let symbol_data = pack_parameter_vector(symbol)?;
        let symbol_pack = formats::ParameterExpressionParameterVectorSymbolPack {
            value_key,
            value_data,
            symbol_data,
        };
        Ok(formats::ParameterExpressionSymbolPack::ParameterVector(
            symbol_pack,
        ))
    } else {
        let value_key = ValueType::Parameter;
        let symbol_data = pack_symbol(symbol);
        let symbol_pack = formats::ParameterExpressionParameterSymbolPack {
            value_key,
            value_data,
            symbol_data,
        };
        Ok(formats::ParameterExpressionSymbolPack::Parameter(
            symbol_pack,
        ))
    }
}

fn pack_parameter_expression_elements(
    exp: &ParameterExpression,
) -> PyResult<Vec<formats::ParameterExpressionElementPack>> {
    let mut result = Vec::new();
    for replay_obj in exp.qpy_replay().iter() {
        let packed_parameter = pack_parameter_expression_element(replay_obj)?;
        result.extend(packed_parameter);
    }
    Ok(result)
}

fn pack_parameter_expression_element(
    replay_obj: &OPReplay,
) -> PyResult<Vec<formats::ParameterExpressionElementPack>> {
    let mut result = Vec::new();
    let (lhs_type, lhs) = pack_parameter_replay_entry(&replay_obj.lhs)?;
    let (rhs_type, rhs) = pack_parameter_replay_entry(&replay_obj.rhs)?;
    let op_code = replay_obj.op as u8;
    let entry = formats::ParameterExpressionStandardOpPack {
        lhs_type,
        lhs,
        rhs_type,
        rhs,
    };
    let packed_parameter = vec![pack_parameter_expression_by_op(op_code, entry)?];
    result.extend(packed_parameter);
    Ok(result)
}

// this function identifies the data type of the parameter replay entry
// and returns the u8 for the type, the [u8; 16] encoding for the data (which stores)
// numbers explicitly, not using 8 bytes for f64 or u64, and storing uuid for more complex vals
// subexpressions are packed using the empty [0u8; 16], followed by "extra data" of the expression's encoding
fn pack_parameter_replay_entry(
    inst: &Option<ParameterValueType>,
) -> PyResult<(ParameterType, [u8; 16])> {
    // This is different from `py_dumps_value` since we aim specifically for [u8; 16]
    // This means parameters are not fully stored, only their uuid
    // Also integers and floats are padded with 0
    let value = match inst {
        None => return Ok((ParameterType::Null, [0u8; 16])),
        Some(val) => val,
    };
    Ok(match value {
        ParameterValueType::Int(val) => (
            ParameterType::Integer,
            Bytes::from(val).try_to_16_byte_slice()?,
        ),
        ParameterValueType::Float(val) => (
            ParameterType::Float,
            Bytes::from(val).try_to_16_byte_slice()?,
        ),
        ParameterValueType::Complex(val) => (
            ParameterType::Complex,
            Bytes::from(val).try_to_16_byte_slice()?,
        ),
        ParameterValueType::Parameter(parameter) => {
            (ParameterType::Parameter, *parameter.symbol.uuid.as_bytes())
        }
        ParameterValueType::VectorElement(element) => (
            ParameterType::ParameterVector,
            *element.symbol.uuid.as_bytes(),
        ),
    })
}

pub(crate) fn unpack_parameter_expression(
    parameter_expression_pack: &formats::ParameterExpressionPack,
    qpy_data: &mut QPYReadData,
) -> PyResult<ParameterExpression> {
    // we begin by loading the symbol table data and hashing it according to each symbol's uuid
    let mut param_uuid_map: HashMap<[u8; 16], GenericValue> = HashMap::new();
    for item in &parameter_expression_pack.symbol_table_data {
        let (symbol_uuid, _, value) = match item {
            formats::ParameterExpressionSymbolPack::ParameterExpression(_) => {
                continue;
            }
            formats::ParameterExpressionSymbolPack::Parameter(symbol_pack) => {
                let symbol = unpack_symbol(&symbol_pack.symbol_data);
                let value = match symbol_pack.value_key {
                    ValueType::Parameter => GenericValue::ParameterExpressionSymbol(symbol.clone()),
                    _ => load_value(symbol_pack.value_key, &symbol_pack.value_data, qpy_data)?,
                };
                (symbol_pack.symbol_data.uuid, symbol, value)
            }
            formats::ParameterExpressionSymbolPack::ParameterVector(symbol_pack) => {
                // this call will also create the corresponding vector and update qpy_data if needed
                let symbol = unpack_parameter_vector(&symbol_pack.symbol_data, qpy_data)?;
                let value = match symbol_pack.value_key {
                    ValueType::ParameterVector => {
                        GenericValue::ParameterExpressionSymbol(symbol.clone())
                    }
                    _ => load_value(symbol_pack.value_key, &symbol_pack.value_data, qpy_data)?,
                };
                (symbol_pack.symbol_data.uuid, symbol, value)
            }
        };
        param_uuid_map.insert(symbol_uuid, value.clone());
    }
    let parameter_expression_data = deserialize_vec::<formats::ParameterExpressionElementPack>(
        &parameter_expression_pack.expression_data,
    )?;

    // we now convert the parameter_expression_data into Vec<OPReplay> that can be used via ParameterExpression::from_qpy
    let mut replay: Vec<OPReplay> = Vec::new();
    // Due to sub operations being different than the other elements of the replay, we store them separately, with an index
    // indicating when to perform them
    let mut sub_operations: Vec<(usize, HashMap<Symbol, ParameterExpression>)> = Vec::new();
    for element in parameter_expression_data {
        if let formats::ParameterExpressionElementPack::Substitute(subs) = element {
            // In the python code, substitutions were put on the stack with the rest of the operations
            // And applied during the Parameter Expression construction phase. This seems to be unsupported in the current
            // Rust implementation, so we assume every

            // we construct a pydictionary describing the substitution and letting the python Parameter class handle it
            let mapping_pack = deserialize::<formats::MappingPack>(&subs.mapping_data)?.0;
            let mut subs_mapping: HashMap<Symbol, ParameterExpression> = HashMap::new();

            for item in mapping_pack.items {
                let key_uuid: [u8; 16] = (&item.key_bytes).try_into()?;
                let value_generic_item = load_value(item.item_type, &item.item_bytes, qpy_data)?;
                let key_generic_item = param_uuid_map.get(&key_uuid).ok_or_else(|| {
                    PyValueError::new_err(format!("Parameter UUID not found: {:?}", &key_uuid))
                })?;
                let key = if let GenericValue::ParameterExpressionSymbol(symbol) = key_generic_item
                {
                    symbol
                } else {
                    return Err(PyValueError::new_err(format!(
                        "Substitution command used left operand {:?} which is not a symbol",
                        &key_generic_item
                    )));
                };

                let value = match value_generic_item {
                    GenericValue::ParameterExpressionSymbol(symbol) => {
                        ParameterExpression::from_symbol(symbol)
                    }
                    GenericValue::ParameterExpressionVectorSymbol(symbol) => {
                        ParameterExpression::from_symbol(symbol)
                    }
                    GenericValue::ParameterExpression(exp) => exp.as_ref().clone(),
                    _ => {
                        return Err(PyValueError::new_err(format!(
                            "Substitution command used right operand {:?} which is not a parameter expression",
                            &value_generic_item
                        )));
                    }
                };
                subs_mapping.insert(key.clone(), value);
            }
            let _opcode = OpCode::SUBSTITUTE;
            sub_operations.push((replay.len(), subs_mapping));
        } else {
            let (opcode, op) = unpack_parameter_expression_standard_op(element)?;
            // loading values from replay pack is tricky, since everything is stored using 16-bytes, even 8-byte ints and floats
            // LHS
            let lhs: Option<ParameterValueType> = match op.lhs_type {
                ParameterType::Parameter | ParameterType::ParameterVector => {
                    if let Some(value) = param_uuid_map.get(&op.lhs) {
                        Some(parameter_value_type_from_generic_value(value)?)
                    } else {
                        return Err(PyValueError::new_err(format!(
                            "Parameter UUID not found: {:?}",
                            op.lhs
                        )));
                    }
                }
                ParameterType::Float | ParameterType::Integer | ParameterType::Complex => {
                    let value =
                        load_value(ValueType::try_from(op.lhs_type)?, &op.lhs.into(), qpy_data)?;
                    Some(parameter_value_type_from_generic_value(&value)?)
                }
                ParameterType::Null => None, // pass
                ParameterType::LhsExpression | ParameterType::RhsExpression => continue,
            };
            // RHS
            let rhs: Option<ParameterValueType> = match op.rhs_type {
                ParameterType::Parameter | ParameterType::ParameterVector => {
                    if let Some(value) = param_uuid_map.get(&op.rhs) {
                        Some(parameter_value_type_from_generic_value(value)?)
                    } else {
                        return Err(PyValueError::new_err(format!(
                            "Parameter UUID not found: {:?}",
                            op.rhs
                        )));
                    }
                }
                ParameterType::Float | ParameterType::Integer | ParameterType::Complex => {
                    let value =
                        load_value(ValueType::try_from(op.rhs_type)?, &op.rhs.into(), qpy_data)?;
                    Some(parameter_value_type_from_generic_value(&value)?)
                }
                ParameterType::Null => None, // pass
                ParameterType::LhsExpression | ParameterType::RhsExpression => continue,
            };
            let op = OpCode::from_u8(opcode)?;
            replay.push(OPReplay { op, lhs, rhs });
        };
    }
    ParameterExpression::from_qpy(&replay, Some(sub_operations))
        .map_err(|_| PyValueError::new_err("Failure while loading parameter expression"))
}

pub(crate) fn pack_symbol(symbol: &Symbol) -> formats::ParameterSymbolPack {
    let uuid = *symbol.uuid.as_bytes();
    let name = symbol.name.clone();
    formats::ParameterSymbolPack { uuid, name }
}

pub(crate) fn unpack_symbol(parameter_pack: &formats::ParameterSymbolPack) -> Symbol {
    let name = parameter_pack.name.clone();
    let uuid = Uuid::from_bytes(parameter_pack.uuid);
    Symbol {
        name,
        uuid,
        index: None,
        vector: None,
    }
}

// currently, the only way to extract the length of the vector the symbol belongs to
// is via python space since the vector is stored as a python reference in the symbol
pub(crate) fn pack_parameter_vector(
    symbol: &Symbol,
) -> PyResult<formats::ParameterVectorElementPack> {
    let vector_size = Python::attach(|py| -> PyResult<_> {
        match &symbol.vector {
            None => Err(PyValueError::new_err(
                "No vector data for parameter vector element",
            )),
            Some(vector) => vector.bind(py).call_method0("__len__")?.extract(),
        }
    })?;
    let index = match symbol.index {
        None => {
            return Err(PyValueError::new_err(
                "No index data for parameter vector element",
            ));
        }
        Some(index_value) => index_value as u64,
    };
    Ok(formats::ParameterVectorElementPack {
        vector_size,
        uuid: *symbol.uuid.as_bytes(),
        index,
        name: symbol.name.clone(),
    })
}

// parameter vector symbols are currently much more tricky than standalone symbols
// since we don't have a rust-space concept of ParameterVector; it is a pure python object
// which we must manage while creating its elements. Moreover, the vector itself is not stored anywhere
// so we need to create it in an ad-hoc fashion as we encounter its elements during the parsing of the
// qpy file. In particular we need to keep our qpy_data nearby so we can update the vector list as needed
// and we must use python calls to create and modify the python-space ParameterVector
pub(crate) fn unpack_parameter_vector(
    parameter_vector_pack: &formats::ParameterVectorElementPack,
    qpy_data: &mut QPYReadData,
) -> PyResult<Symbol> {
    let name = parameter_vector_pack.name.clone();
    let uuid = Uuid::from_bytes(parameter_vector_pack.uuid);
    let index = parameter_vector_pack.index as u32; // sadly, the `Symbol` class does not conform to the qpy u64 format
    // we have extracted the rust-space data, but now we must deal with the python-space vector class

    // first get the uuid for the vector's "root" (it's first element)
    // we rely on the convention that the uuid's for the vector elements are sequential
    let root_uuid_int = uuid.as_u128() - (index as u128);
    let root_uuid = Uuid::from_bytes(root_uuid_int.to_be_bytes());

    let vector = Python::attach(|py| -> PyResult<_> {
        // we use python-space to interface with the ParameterVector data
        let vector_data = match qpy_data.vectors.get_mut(&root_uuid) {
            Some(value) => value,
            None => Python::attach(|py| -> PyResult<_> {
                // we use python-space to create a new parameter vector
                let vector = imports::PARAMETER_VECTOR
                    .get_bound(py)
                    .call1((name.clone(), parameter_vector_pack.vector_size))?
                    .unbind();
                qpy_data.vectors.insert(root_uuid, (vector, Vec::new()));
                Ok(qpy_data.vectors.get_mut(&root_uuid).unwrap())
            })?,
        };
        let vector = vector_data.0.bind(py);
        let vector_name = vector.getattr("name")?.extract::<String>()?;
        let vector_element = vector.get_item(index)?.extract::<Symbol>()?;
        if vector_element.uuid != uuid {
            // we need to create a new parameter vector element and hack it into the vector
            vector_data.1.push(index);
            // let param_vector_element = PyParameterVectorElement::py_new(py, vector, index, parameter_vector_pack.uuid)
            let param_vector_element = Symbol::py_new(
                &vector_name,
                Some(uuid.as_u128()),
                Some(index),
                Some(vector.clone().unbind()),
            )?;
            vector
                .getattr("_params")?
                .set_item(index, param_vector_element)?;
        }
        Ok(vector.clone().unbind())
    })?;

    Ok(Symbol {
        name,
        uuid,
        index: Some(index),
        vector: Some(vector),
    })
}

pub(crate) fn pack_param_expression(
    exp: &ParameterExpression,
    qpy_data: &QPYWriteData,
) -> PyResult<formats::GenericDataPack> {
    // if the parameter expression is a single symbol, we should treat it like a parameter
    // or a parameter vector, depending on whether the `vector` field exists
    if let Ok(symbol) = exp.try_to_symbol() {
        match symbol.vector {
            None => pack_generic_value(&GenericValue::ParameterExpressionSymbol(symbol), qpy_data),
            Some(_) => pack_generic_value(
                &GenericValue::ParameterExpressionVectorSymbol(symbol),
                qpy_data,
            ),
        }
    } else {
        pack_generic_value(
            &GenericValue::ParameterExpression(Arc::new(exp.clone())),
            qpy_data,
        )
    }
}

pub(crate) fn pack_param_obj(
    param: &Param,
    qpy_data: &QPYWriteData,
    endian: Endian,
) -> PyResult<formats::GenericDataPack> {
    Ok(match param {
        Param::Float(val) => match endian {
            Endian::Little => formats::GenericDataPack {
                type_key: ValueType::Float,
                data: val.to_le_bytes().into(),
            },
            Endian::Big => formats::GenericDataPack {
                type_key: ValueType::Float,
                data: val.to_be_bytes().into(),
            },
        },
        Param::ParameterExpression(exp) => pack_param_expression(exp, qpy_data)?,
        Param::Obj(py_object) => {
            Python::attach(|py| py_pack_param(py_object.bind(py), qpy_data, endian))?
        }
    })
}

pub(crate) fn generic_value_to_param(value: &GenericValue, endian: Endian) -> PyResult<Param> {
    let value = match endian {
        Endian::Big => value,
        Endian::Little => &value.as_le(),
    };
    match value {
        GenericValue::Float64(float_val) => Ok(Param::Float(*float_val)),
        GenericValue::ParameterExpressionSymbol(symbol) => {
            let parameter_expression = ParameterExpression::from_symbol(symbol.clone());
            Ok(Param::ParameterExpression(Arc::new(parameter_expression)))
        }
        GenericValue::ParameterExpressionVectorSymbol(symbol) => {
            let parameter_expression = ParameterExpression::from_symbol(symbol.clone());
            Ok(Param::ParameterExpression(Arc::new(parameter_expression)))
        }
        GenericValue::ParameterExpression(exp) => Ok(Param::ParameterExpression(exp.clone())),
        _ => Ok(Param::Obj(py_convert_from_generic_value(value)?)),
    }
}
