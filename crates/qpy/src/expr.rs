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

// methods for serialization/deserialization of Expression
use crate::error::{QpyError, to_binrw_error};
use crate::formats::{
    ArrayValuePack, ExpressionElementPack, ExpressionTypePack, ExpressionValueElementPack,
    ExpressionVarElementPack, ExpressionVarRegisterPack,
};
use crate::value::{
    ArrayTypePack, QPYReadData, QPYWriteData, clbit_at, clbit_index, creg_by_name, pack_biguint,
    pack_duration, unpack_biguint, unpack_duration,
};
use binrw::{BinRead, BinResult, BinWrite, Endian, Error};
use num_bigint::BigUint;
use qiskit_circuit::classical::expr::{
    Binary, BinaryOp, Cast, Expr, Index, Unary, UnaryOp, Value, Var,
};
use qiskit_circuit::classical::types::Type;
use qiskit_circuit::duration::Duration;
use std::io::{Read, Seek, Write};

// packed expression types implicitly contain the magic number identifying them in the qpy file
pub(crate) fn pack_expression_type(ty: &Type, version: u8) -> Result<ExpressionTypePack, QpyError> {
    match ty {
        Type::Bool => Ok(ExpressionTypePack::Bool),
        Type::Uint(width) => Ok(ExpressionTypePack::Int(*width)),
        Type::Duration => Ok(ExpressionTypePack::Duration),
        Type::Float => Ok(ExpressionTypePack::Float),
        Type::Array {
            elem,
            elem_width,
            size,
        } => {
            if version < 18 {
                return Err(QpyError::UnsupportedFeatureForVersion {
                    feature: "array-typed expressions".to_string(),
                    version,
                    min_version: 18,
                });
            }
            Ok(ExpressionTypePack::Array(ArrayTypePack::from_parts(
                *elem,
                *elem_width,
                *size,
            )))
        }
    }
}

pub(crate) fn unpack_expression_type(type_pack: ExpressionTypePack) -> Type {
    match type_pack {
        ExpressionTypePack::Bool => Type::Bool,
        ExpressionTypePack::Duration => Type::Duration,
        ExpressionTypePack::Float => Type::Float,
        ExpressionTypePack::Int(width) => Type::Uint(width),
        ExpressionTypePack::Array(pack) => pack.to_type(),
    }
}

fn pack_scalar_value_element(
    value: &Value,
    qpy_data: &QPYWriteData,
) -> Result<ExpressionValueElementPack, QpyError> {
    match value {
        Value::Uint { raw, ty } => match ty {
            Type::Bool => Ok(ExpressionValueElementPack::Bool(raw.to_bytes_le()[0])), // effectively truncating modulo 256
            Type::Uint(_) => Ok(ExpressionValueElementPack::Int(pack_biguint(raw))),
            _ => Ok(ExpressionValueElementPack::Bool(raw.to_bytes_le()[0])), // TODO: should this be different?
        },
        Value::Float { raw, .. } => Ok(ExpressionValueElementPack::Float(*raw)),
        Value::Duration(duration) => {
            if qpy_data.version < 16 && matches!(duration, Duration::ps(_)) {
                return Err(QpyError::UnsupportedFeatureForVersion {
                    feature: "Duration variant 'Duration.ps'".to_string(),
                    version: 16,
                    min_version: qpy_data.version,
                });
            }
            Ok(ExpressionValueElementPack::Duration(pack_duration(
                duration,
            )))
        }
        Value::Array { .. } => Err(QpyError::SerializationError(
            "nested array values are not supported in QPY".to_string(),
        )),
    }
}

fn unpack_scalar_value_element(
    ty: Type,
    value_element_pack: ExpressionValueElementPack,
) -> Result<Value, QpyError> {
    match value_element_pack {
        ExpressionValueElementPack::Bool(val) => Ok(Value::Uint {
            raw: BigUint::from_bytes_le(&[val]),
            ty,
        }),
        ExpressionValueElementPack::Int(val) => Ok(Value::Uint {
            raw: unpack_biguint(val),
            ty,
        }),
        ExpressionValueElementPack::Duration(duration) => {
            Ok(Value::Duration(unpack_duration(duration)))
        }
        ExpressionValueElementPack::Float(val) => Ok(Value::Float { raw: val, ty }),
        ExpressionValueElementPack::Array(_) => Err(QpyError::DeserializationError(
            "nested array values are not supported in QPY".to_string(),
        )),
    }
}

pub(crate) fn pack_expression_value(
    value: &Value,
    qpy_data: &QPYWriteData,
) -> Result<ExpressionElementPack, QpyError> {
    let (ty, value_pack) = match value {
        Value::Array { elems, ty } => {
            if qpy_data.version < 18 {
                return Err(QpyError::UnsupportedFeatureForVersion {
                    feature: "array-typed expressions".to_string(),
                    version: qpy_data.version,
                    min_version: 18,
                });
            }
            let packed_elems = elems
                .iter()
                .map(|elem| pack_scalar_value_element(elem, qpy_data))
                .collect::<Result<Vec<_>, _>>()?;
            (
                ty,
                ExpressionValueElementPack::Array(ArrayValuePack {
                    elems: packed_elems,
                }),
            )
        }
        scalar => (
            scalar_value_type(scalar),
            pack_scalar_value_element(scalar, qpy_data)?,
        ),
    };
    Ok(ExpressionElementPack::Value(
        pack_expression_type(ty, qpy_data.version)?,
        value_pack,
    ))
}

fn scalar_value_type(value: &Value) -> &Type {
    match value {
        Value::Uint { ty, .. } | Value::Float { ty, .. } => ty,
        Value::Duration(_) => &Type::Duration,
        Value::Array { ty, .. } => ty,
    }
}

pub(crate) fn unpack_expression_value(
    value_type_pack: ExpressionTypePack,
    value_element_pack: ExpressionValueElementPack,
) -> Result<Value, QpyError> {
    let ty = unpack_expression_type(value_type_pack);
    if let ExpressionValueElementPack::Array(array_pack) = value_element_pack {
        let Type::Array {
            elem,
            elem_width,
            size,
        } = ty
        else {
            return Err(QpyError::DeserializationError(
                "array EXPR_VALUE with a non-array EXPR_TYPE".to_string(),
            ));
        };
        if array_pack.elems.len() != size as usize {
            return Err(QpyError::DeserializationError(format!(
                "array EXPR_VALUE has {} elements but type size is {size}",
                array_pack.elems.len(),
            )));
        }
        let elem_ty = Type::from_scalar(elem, elem_width);
        let mut elems = Vec::with_capacity(array_pack.elems.len());
        for pack in array_pack.elems {
            elems.push(unpack_scalar_value_element(elem_ty, pack)?);
        }
        return Ok(Value::Array { elems, ty });
    }
    unpack_scalar_value_element(ty, value_element_pack)
}

pub(crate) fn pack_expression_var(
    var: &Var,
    qpy_data: &QPYWriteData,
) -> Result<ExpressionElementPack, QpyError> {
    let (ty, value_pack) = match var {
        Var::Bit { bit } => (
            &Type::Bool,
            ExpressionVarElementPack::Clbit(clbit_index(bit, qpy_data)?),
        ),
        Var::Register { register, ty } => (
            ty,
            ExpressionVarElementPack::Register(ExpressionVarRegisterPack {
                name: register.name().to_string(),
            }),
        ),
        Var::Standalone { uuid, name, ty } => (
            ty,
            ExpressionVarElementPack::Uuid(*qpy_data.standalone_var_indices.get(uuid).ok_or_else(
                || {
                    QpyError::InvalidParameter(format!(
                        "Could not find standalone variable {:?} in the qpy data",
                        name
                    ))
                },
            )?),
        ),
    };
    Ok(ExpressionElementPack::Var(
        pack_expression_type(ty, qpy_data.version)?,
        value_pack,
    ))
}

pub(crate) fn unpack_expression_var(
    var_type_pack: ExpressionTypePack,
    var_element_pack: ExpressionVarElementPack,
    qpy_data: &QPYReadData,
) -> Result<Var, QpyError> {
    let ty = unpack_expression_type(var_type_pack);
    match var_element_pack {
        ExpressionVarElementPack::Clbit(index) => Ok(Var::Bit {
            bit: clbit_at(index, qpy_data)?,
        }),
        ExpressionVarElementPack::Register(packed_register) => Ok(Var::Register {
            register: creg_by_name(&packed_register.name, qpy_data)?,
            ty,
        }),
        ExpressionVarElementPack::Uuid(key) => {
            let var = qpy_data.standalone_vars.get(&key).ok_or_else(|| {
                QpyError::InvalidParameter("Standalone var not found in qpy data".to_string())
            })?; // note: this is not an actual expr::Var; merely a key for this var inside the circuit data
            Ok(qpy_data
                .circuit_data
                .vars_stretches_view()
                .vars()
                .get(*var)
                .ok_or_else(|| {
                    QpyError::InvalidParameter(
                        "Standalone var not found in circuit data".to_string(),
                    )
                })?
                .clone()) // TODO: can we avoid cloning?
        }
    }
}

pub(crate) fn write_expression<W: Write + Seek>(
    exp: &Expr,
    writer: &mut W,
    endian: Endian,
    (qpy_data,): (&QPYWriteData,),
) -> binrw::BinResult<()> {
    match exp {
        Expr::Value(val) => {
            pack_expression_value(val, qpy_data)
                .map_err(|e| to_binrw_error(writer, e))?
                .write_options(writer, endian, ())?;
        }
        Expr::Var(var) => {
            pack_expression_var(var, qpy_data)
                .map_err(|e| to_binrw_error(writer, e))?
                .write_options(writer, endian, ())?;
        }
        Expr::Stretch(stretch) => {
            ExpressionElementPack::Stretch(
                ExpressionTypePack::Duration,
                qpy_data.standalone_var_indices[&stretch.uuid],
            )
            .write_options(writer, endian, ())?;
        }
        Expr::Index(index_node) => {
            ExpressionElementPack::Index(
                pack_expression_type(&index_node.ty, qpy_data.version)
                    .map_err(|e| to_binrw_error(writer, e))?,
            )
            .write_options(writer, endian, ())?;
            write_expression(&index_node.target, writer, endian, (qpy_data,))?;
            write_expression(&index_node.index, writer, endian, (qpy_data,))?;
        }
        Expr::Cast(cast_node) => {
            ExpressionElementPack::Cast(
                pack_expression_type(&cast_node.ty, qpy_data.version)
                    .map_err(|e| to_binrw_error(writer, e))?,
                cast_node.implicit as u8,
            )
            .write_options(writer, endian, ())?;
            write_expression(&cast_node.operand, writer, endian, (qpy_data,))?;
        }
        Expr::Unary(unary_node) => {
            ExpressionElementPack::Unary(
                pack_expression_type(&unary_node.ty, qpy_data.version)
                    .map_err(|e| to_binrw_error(writer, e))?,
                unary_node.op as u8,
            )
            .write_options(writer, endian, ())?;
            write_expression(&unary_node.operand, writer, endian, (qpy_data,))?;
        }
        Expr::Binary(binary_node) => {
            ExpressionElementPack::Binary(
                pack_expression_type(&binary_node.ty, qpy_data.version)
                    .map_err(|e| to_binrw_error(writer, e))?,
                binary_node.op as u8,
            )
            .write_options(writer, endian, ())?;
            write_expression(&binary_node.left, writer, endian, (qpy_data,))?;
            write_expression(&binary_node.right, writer, endian, (qpy_data,))?;
        }
    };
    Ok(())
}

pub(crate) fn read_expression<R: Read + Seek>(
    reader: &mut R,
    endian: Endian,
    (qpy_data,): (&QPYReadData,),
) -> BinResult<Expr> {
    let exp_element = ExpressionElementPack::read_options(reader, endian, ())?;
    match exp_element {
        ExpressionElementPack::Value(value_type_pack, value_element_pack) => Ok(Expr::Value(
            unpack_expression_value(value_type_pack, value_element_pack)
                .map_err(|e| to_binrw_error(reader, e))?,
        )),
        ExpressionElementPack::Var(var_type_pack, var_element_pack) => Ok(Expr::Var(
            unpack_expression_var(var_type_pack, var_element_pack, qpy_data)
                .map_err(|e| to_binrw_error(reader, e))?,
        )),
        ExpressionElementPack::Stretch(_stretch_type_pack, key) => {
            let stretch = qpy_data.standalone_stretches.get(&key).ok_or_else(|| {
                to_binrw_error(
                    reader,
                    QpyError::InvalidParameter(format!(
                        "Standalone stretch with key {} not found in qpy data",
                        key
                    )),
                )
            })?;
            Ok(Expr::Stretch(
                qpy_data
                    .circuit_data
                    .vars_stretches_view()
                    .stretches()
                    .get(*stretch)
                    .ok_or_else(|| {
                        to_binrw_error(
                            reader,
                            QpyError::InvalidParameter(
                                "Stretch not found in circuit data".to_string(),
                            ),
                        )
                    })?
                    .clone(),
            )) // TODO: can we avoid cloning?
        }
        ExpressionElementPack::Index(index_type_pack) => {
            let target = read_expression(reader, endian, (qpy_data,))?;
            let index = read_expression(reader, endian, (qpy_data,))?;
            let constant = target.is_const() && index.is_const();
            Ok(Expr::Index(Box::new(Index {
                target,
                index,
                ty: unpack_expression_type(index_type_pack),
                constant,
            })))
        }
        ExpressionElementPack::Cast(cast_type_pack, implicit) => {
            let operand = read_expression(reader, endian, (qpy_data,))?;
            let constant = operand.is_const();
            Ok(Expr::Cast(Box::new(Cast {
                operand,
                ty: unpack_expression_type(cast_type_pack),
                constant,
                implicit: implicit != 0,
            })))
        }
        ExpressionElementPack::Unary(unary_type_pack, op) => {
            let operand = read_expression(reader, endian, (qpy_data,))?;
            let constant = operand.is_const();
            Ok(Expr::Unary(Box::new(Unary {
                op: UnaryOp::from_u8(op).map_err(|_| Error::NoVariantMatch { pos: (0) })?,
                operand,
                ty: unpack_expression_type(unary_type_pack),
                constant,
            })))
        }
        ExpressionElementPack::Binary(binary_type_pack, op) => {
            let left = read_expression(reader, endian, (qpy_data,))?;
            let right = read_expression(reader, endian, (qpy_data,))?;
            let constant = left.is_const() && right.is_const();
            Ok(Expr::Binary(Box::new(Binary {
                op: BinaryOp::from_u8(op).map_err(|_| Error::NoVariantMatch { pos: (0) })?,
                left,
                right,
                ty: unpack_expression_type(binary_type_pack),
                constant,
            })))
        }
    }
}
