# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Binary IO for alphanumeric objects"""

import struct
import uuid

import numpy as np

from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.parametervector import ParameterVector, ParameterVectorElement
from qiskit.qpy import common, formats
from qiskit.qpy.common import AlphanumericTypeKey as TypeKey, ENCODE

try:
    import symengine

    HAS_SYMENGINE = True
except ImportError:
    HAS_SYMENGINE = False


def _write_parameter(file_obj, obj):
    name_bytes = obj._name.encode("utf8")
    file_obj.write(struct.pack(formats.PARAMETER_PACK, len(name_bytes), obj._uuid.bytes))
    file_obj.write(name_bytes)


def _write_parameter_vec(file_obj, obj):
    name_bytes = obj._vector._name.encode(ENCODE)
    file_obj.write(
        struct.pack(
            formats.PARAMETER_VECTOR_ELEMENT_PACK,
            len(name_bytes),
            obj._vector._size,
            obj._uuid.bytes,
            obj._index,
        )
    )
    file_obj.write(name_bytes)


def _write_parameter_expression(file_obj, obj):
    from sympy import srepr, sympify

    expr_bytes = srepr(sympify(obj._symbol_expr)).encode(ENCODE)
    param_expr_header_raw = struct.pack(
        formats.PARAMETER_EXPR_PACK, len(obj._parameter_symbols), len(expr_bytes)
    )
    file_obj.write(param_expr_header_raw)
    file_obj.write(expr_bytes)

    for symbol, value in obj._parameter_symbols.items():
        symbol_key = TypeKey.assign(symbol)

        # serialize key
        if symbol_key == TypeKey.PARAMETER_VECTOR:
            symbol_data = common.to_binary(symbol, _write_parameter_vec)
        else:
            symbol_data = common.to_binary(symbol, _write_parameter)

        # serialize value
        if value == symbol._symbol_expr:
            value_key = symbol_key
            value_data = bytes()
        else:
            value_key, value_data = dumps(value)

        elem_header = struct.pack(
            formats.PARAM_EXPR_MAP_ELEM_PACK_V3,
            symbol_key,
            value_key,
            len(value_data),
        )
        file_obj.write(elem_header)
        file_obj.write(symbol_data)
        file_obj.write(value_data)


def _read_parameter(file_obj):
    data = formats.PARAMETER(
        *struct.unpack(formats.PARAMETER_PACK, file_obj.read(formats.PARAMETER_SIZE))
    )
    param_uuid = uuid.UUID(bytes=data.uuid)
    name = file_obj.read(data.name_size).decode(ENCODE)
    param = Parameter.__new__(Parameter, name, uuid=param_uuid)
    param.__init__(name)
    return param


def _read_parameter_vec(file_obj, vectors):
    data = formats.PARAMETER_VECTOR_ELEMENT(
        *struct.unpack(
            formats.PARAMETER_VECTOR_ELEMENT_PACK,
            file_obj.read(formats.PARAMETER_VECTOR_ELEMENT_SIZE),
        ),
    )
    param_uuid = uuid.UUID(bytes=data.uuid)
    name = file_obj.read(data.vector_name_size).decode(ENCODE)
    if name not in vectors:
        vectors[name] = (ParameterVector(name, data.vector_size), set())
    vector = vectors[name][0]
    if vector[data.index]._uuid != param_uuid:
        vectors[name][1].add(data.index)
        vector._params[data.index] = ParameterVectorElement.__new__(
            ParameterVectorElement, vector, data.index, uuid=param_uuid
        )
        vector._params[data.index].__init__(vector, data.index)
    return vector[data.index]


def _read_parameter_expression(file_obj):
    data = formats.PARAMETER_EXPR(
        *struct.unpack(formats.PARAMETER_EXPR_PACK, file_obj.read(formats.PARAMETER_EXPR_SIZE))
    )
    from sympy.parsing.sympy_parser import parse_expr

    if HAS_SYMENGINE:
        expr = symengine.sympify(parse_expr(file_obj.read(data.expr_size).decode(ENCODE)))
    else:
        expr = parse_expr(file_obj.read(data.expr_size).decode(ENCODE))
    symbol_map = {}
    for _ in range(data.map_elements):
        elem_data = formats.PARAM_EXPR_MAP_ELEM(
            *struct.unpack(
                formats.PARAM_EXPR_MAP_ELEM_PACK,
                file_obj.read(formats.PARAM_EXPR_MAP_ELEM_SIZE),
            )
        )
        symbol = _read_parameter(file_obj)

        elem_key = TypeKey(elem_data.type)
        binary_data = file_obj.read(elem_data.size)
        if elem_key == TypeKey.INTEGER:
            value = struct.unpack("!q", binary_data)
        elif elem_key == TypeKey.FLOAT:
            value = struct.unpack("!d", binary_data)
        elif elem_key == TypeKey.COMPLEX:
            value = complex(*struct.unpack(formats.COMPLEX_PACK, binary_data))
        elif elem_key == TypeKey.PARAMETER:
            value = symbol._symbol_expr
        elif elem_key == TypeKey.PARAMETER_EXPRESSION:
            value = common.from_binary(binary_data, _read_parameter_expression)
        else:
            raise TypeError("Invalid parameter expression map type: %s" % elem_key)
        symbol_map[symbol] = value

    return ParameterExpression(symbol_map, expr)


def _read_parameter_expression_v3(file_obj, vectors):
    data = formats.PARAMETER_EXPR(
        *struct.unpack(formats.PARAMETER_EXPR_PACK, file_obj.read(formats.PARAMETER_EXPR_SIZE))
    )
    from sympy.parsing.sympy_parser import parse_expr

    if HAS_SYMENGINE:
        expr = symengine.sympify(parse_expr(file_obj.read(data.expr_size).decode(ENCODE)))
    else:
        expr = parse_expr(file_obj.read(data.expr_size).decode(ENCODE))
    symbol_map = {}
    for _ in range(data.map_elements):
        elem_data = formats.PARAM_EXPR_MAP_ELEM_V3(
            *struct.unpack(
                formats.PARAM_EXPR_MAP_ELEM_PACK_V3,
                file_obj.read(formats.PARAM_EXPR_MAP_ELEM_SIZE_V3),
            )
        )
        symbol_key = TypeKey(elem_data.symbol_type)

        if symbol_key == TypeKey.PARAMETER:
            symbol = _read_parameter(file_obj)
        elif symbol_key == TypeKey.PARAMETER_VECTOR:
            symbol = _read_parameter_vec(file_obj, vectors)
        else:
            raise TypeError("Invalid parameter expression map type: %s" % symbol_key)

        elem_key = TypeKey(elem_data.type)
        binary_data = file_obj.read(elem_data.size)
        if elem_key == TypeKey.INTEGER:
            value = struct.unpack("!q", binary_data)
        elif elem_key == TypeKey.FLOAT:
            value = struct.unpack("!d", binary_data)
        elif elem_key == TypeKey.COMPLEX:
            value = complex(*struct.unpack(formats.COMPLEX_PACK, binary_data))
        elif elem_key in (TypeKey.PARAMETER, TypeKey.PARAMETER_VECTOR):
            value = symbol._symbol_expr
        elif elem_key == TypeKey.PARAMETER_EXPRESSION:
            value = common.from_binary(binary_data, _read_parameter_expression_v3, vectors=vectors)
        else:
            raise TypeError("Invalid parameter expression map type: %s" % elem_key)
        symbol_map[symbol] = value

    return ParameterExpression(symbol_map, expr)


def dumps(obj):
    """Serialize input alphanumeric object.

    Args:
        obj (any): Arbitrary alphanumeric object to serialize.

    Returns:
        tuple: TypeKey and binary data.

    Raises:
        NotImplementedError: Serializer for given format is not ready.
    """
    type_key = TypeKey.assign(obj)

    if type_key == TypeKey.INTEGER:
        binary_data = struct.pack("!q", obj)
    elif type_key == TypeKey.FLOAT:
        binary_data = struct.pack("!d", obj)
    elif type_key == TypeKey.COMPLEX:
        binary_data = struct.pack(formats.COMPLEX_PACK, obj.real, obj.imag)
    elif type_key == TypeKey.NUMPY_OBJ:
        binary_data = common.to_binary(obj, np.save)
    elif type_key == TypeKey.STRING:
        binary_data = obj.encode(ENCODE)
    elif type_key == TypeKey.PARAMETER_VECTOR:
        binary_data = common.to_binary(obj, _write_parameter_vec)
    elif type_key == TypeKey.PARAMETER:
        binary_data = common.to_binary(obj, _write_parameter)
    elif type_key == TypeKey.PARAMETER_EXPRESSION:
        binary_data = common.to_binary(obj, _write_parameter_expression)
    else:
        raise NotImplementedError(f"Serialization for {type_key} is not implemented.")

    return type_key, binary_data


def loads(type_key, binary_data, version, vectors):
    """Deserialize input binary data to alphanumeric object.

    Args:
        type_key (ValueTypeKey): Type enum information.
        binary_data (bytes): Data to deserialize.
        version (int): QPY version.
        vectors (dict): ParameterVector in current scope.

    Returns:
        any: Deserialized alphanumeric object.

    Raises:
        NotImplementedError: Serializer for given format is not ready.
    """
    if isinstance(type_key, bytes):
        type_key = common.AlphanumericTypeKey(type_key)

    if type_key == TypeKey.INTEGER:
        obj = struct.unpack("!q", binary_data)[0]
    elif type_key == TypeKey.FLOAT:
        obj = struct.unpack("!d", binary_data)[0]
    elif type_key == TypeKey.COMPLEX:
        obj = complex(*struct.unpack(formats.COMPLEX_PACK, binary_data))
    elif type_key == TypeKey.NUMPY_OBJ:
        obj = common.from_binary(binary_data, np.load)
    elif type_key == TypeKey.STRING:
        obj = binary_data.decode(ENCODE)
    elif type_key == TypeKey.PARAMETER_VECTOR:
        obj = common.from_binary(binary_data, _read_parameter_vec, vectors=vectors)
    elif type_key == TypeKey.PARAMETER:
        obj = common.from_binary(binary_data, _read_parameter)
    elif type_key == TypeKey.PARAMETER_EXPRESSION:
        if version < 3:
            obj = common.from_binary(binary_data, _read_parameter_expression)
        else:
            obj = common.from_binary(binary_data, _read_parameter_expression_v3, vectors=vectors)
    else:
        raise NotImplementedError(f"Serialization for {type_key} is not implemented.")

    return obj
