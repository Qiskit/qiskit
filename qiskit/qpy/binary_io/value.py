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

"""Binary IO for any value objects, such as numbers, string, parameters."""

import struct
import uuid

import numpy as np

from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.parametervector import ParameterVector, ParameterVectorElement
from qiskit.qpy import common, formats, exceptions, type_keys
from qiskit.utils import optionals as _optional


def _write_parameter(file_obj, obj):
    name_bytes = obj._name.encode(common.ENCODE)
    file_obj.write(struct.pack(formats.PARAMETER_PACK, len(name_bytes), obj._uuid.bytes))
    file_obj.write(name_bytes)


def _write_parameter_vec(file_obj, obj):
    name_bytes = obj._vector._name.encode(common.ENCODE)
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

    expr_bytes = srepr(sympify(obj._symbol_expr)).encode(common.ENCODE)
    param_expr_header_raw = struct.pack(
        formats.PARAMETER_EXPR_PACK, len(obj._parameter_symbols), len(expr_bytes)
    )
    file_obj.write(param_expr_header_raw)
    file_obj.write(expr_bytes)

    for symbol, value in obj._parameter_symbols.items():
        symbol_key = type_keys.Value.assign(symbol)

        # serialize key
        if symbol_key == type_keys.Value.PARAMETER_VECTOR:
            symbol_data = common.data_to_binary(symbol, _write_parameter_vec)
        else:
            symbol_data = common.data_to_binary(symbol, _write_parameter)

        # serialize value
        if value == symbol._symbol_expr:
            value_key = symbol_key
            value_data = bytes()
        else:
            value_key, value_data = dumps_value(value)

        elem_header = struct.pack(
            formats.PARAM_EXPR_MAP_ELEM_V3_PACK,
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
    name = file_obj.read(data.name_size).decode(common.ENCODE)
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
    name = file_obj.read(data.vector_name_size).decode(common.ENCODE)
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

    if _optional.HAS_SYMENGINE:
        import symengine

        expr = symengine.sympify(parse_expr(file_obj.read(data.expr_size).decode(common.ENCODE)))
    else:
        expr = parse_expr(file_obj.read(data.expr_size).decode(common.ENCODE))
    symbol_map = {}
    for _ in range(data.map_elements):
        elem_data = formats.PARAM_EXPR_MAP_ELEM(
            *struct.unpack(
                formats.PARAM_EXPR_MAP_ELEM_PACK,
                file_obj.read(formats.PARAM_EXPR_MAP_ELEM_SIZE),
            )
        )
        symbol = _read_parameter(file_obj)

        elem_key = type_keys.Value(elem_data.type)
        binary_data = file_obj.read(elem_data.size)
        if elem_key == type_keys.Value.INTEGER:
            value = struct.unpack("!q", binary_data)
        elif elem_key == type_keys.Value.FLOAT:
            value = struct.unpack("!d", binary_data)
        elif elem_key == type_keys.Value.COMPLEX:
            value = complex(*struct.unpack(formats.COMPLEX_PACK, binary_data))
        elif elem_key == type_keys.Value.PARAMETER:
            value = symbol._symbol_expr
        elif elem_key == type_keys.Value.PARAMETER_EXPRESSION:
            value = common.data_from_binary(binary_data, _read_parameter_expression)
        else:
            raise exceptions.QpyError("Invalid parameter expression map type: %s" % elem_key)
        symbol_map[symbol] = value

    return ParameterExpression(symbol_map, expr)


def _read_parameter_expression_v3(file_obj, vectors):
    data = formats.PARAMETER_EXPR(
        *struct.unpack(formats.PARAMETER_EXPR_PACK, file_obj.read(formats.PARAMETER_EXPR_SIZE))
    )
    from sympy.parsing.sympy_parser import parse_expr

    if _optional.HAS_SYMENGINE:
        import symengine

        expr = symengine.sympify(parse_expr(file_obj.read(data.expr_size).decode(common.ENCODE)))
    else:
        expr = parse_expr(file_obj.read(data.expr_size).decode(common.ENCODE))
    symbol_map = {}
    for _ in range(data.map_elements):
        elem_data = formats.PARAM_EXPR_MAP_ELEM_V3(
            *struct.unpack(
                formats.PARAM_EXPR_MAP_ELEM_V3_PACK,
                file_obj.read(formats.PARAM_EXPR_MAP_ELEM_V3_SIZE),
            )
        )
        symbol_key = type_keys.Value(elem_data.symbol_type)

        if symbol_key == type_keys.Value.PARAMETER:
            symbol = _read_parameter(file_obj)
        elif symbol_key == type_keys.Value.PARAMETER_VECTOR:
            symbol = _read_parameter_vec(file_obj, vectors)
        else:
            raise exceptions.QpyError("Invalid parameter expression map type: %s" % symbol_key)

        elem_key = type_keys.Value(elem_data.type)
        binary_data = file_obj.read(elem_data.size)
        if elem_key == type_keys.Value.INTEGER:
            value = struct.unpack("!q", binary_data)
        elif elem_key == type_keys.Value.FLOAT:
            value = struct.unpack("!d", binary_data)
        elif elem_key == type_keys.Value.COMPLEX:
            value = complex(*struct.unpack(formats.COMPLEX_PACK, binary_data))
        elif elem_key in (type_keys.Value.PARAMETER, type_keys.Value.PARAMETER_VECTOR):
            value = symbol._symbol_expr
        elif elem_key == type_keys.Value.PARAMETER_EXPRESSION:
            value = common.data_from_binary(
                binary_data, _read_parameter_expression_v3, vectors=vectors
            )
        else:
            raise exceptions.QpyError("Invalid parameter expression map type: %s" % elem_key)
        symbol_map[symbol] = value

    return ParameterExpression(symbol_map, expr)


def dumps_value(obj):
    """Serialize input value object.

    Args:
        obj (any): Arbitrary value object to serialize.

    Returns:
        tuple: TypeKey and binary data.

    Raises:
        QpyError: Serializer for given format is not ready.
    """
    type_key = type_keys.Value.assign(obj)

    if type_key == type_keys.Value.INTEGER:
        binary_data = struct.pack("!q", obj)
    elif type_key == type_keys.Value.FLOAT:
        binary_data = struct.pack("!d", obj)
    elif type_key == type_keys.Value.COMPLEX:
        binary_data = struct.pack(formats.COMPLEX_PACK, obj.real, obj.imag)
    elif type_key == type_keys.Value.NUMPY_OBJ:
        binary_data = common.data_to_binary(obj, np.save)
    elif type_key == type_keys.Value.STRING:
        binary_data = obj.encode(common.ENCODE)
    elif type_key == type_keys.Value.NULL:
        binary_data = b""
    elif type_key == type_keys.Value.PARAMETER_VECTOR:
        binary_data = common.data_to_binary(obj, _write_parameter_vec)
    elif type_key == type_keys.Value.PARAMETER:
        binary_data = common.data_to_binary(obj, _write_parameter)
    elif type_key == type_keys.Value.PARAMETER_EXPRESSION:
        binary_data = common.data_to_binary(obj, _write_parameter_expression)
    else:
        raise exceptions.QpyError(f"Serialization for {type_key} is not implemented in value I/O.")

    return type_key, binary_data


def write_value(file_obj, obj):
    """Write a value to the file like object.

    Args:
        file_obj (File): A file like object to write data.
        obj (any): Value to write.
    """
    type_key, data = dumps_value(obj)
    common.write_generic_typed_data(file_obj, type_key, data)


def loads_value(type_key, binary_data, version, vectors):
    """Deserialize input binary data to value object.

    Args:
        type_key (ValueTypeKey): Type enum information.
        binary_data (bytes): Data to deserialize.
        version (int): QPY version.
        vectors (dict): ParameterVector in current scope.

    Returns:
        any: Deserialized value object.

    Raises:
        QpyError: Serializer for given format is not ready.
    """
    # pylint: disable=too-many-return-statements

    if isinstance(type_key, bytes):
        type_key = type_keys.Value(type_key)

    if type_key == type_keys.Value.INTEGER:
        return struct.unpack("!q", binary_data)[0]
    if type_key == type_keys.Value.FLOAT:
        return struct.unpack("!d", binary_data)[0]
    if type_key == type_keys.Value.COMPLEX:
        return complex(*struct.unpack(formats.COMPLEX_PACK, binary_data))
    if type_key == type_keys.Value.NUMPY_OBJ:
        return common.data_from_binary(binary_data, np.load)
    if type_key == type_keys.Value.STRING:
        return binary_data.decode(common.ENCODE)
    if type_key == type_keys.Value.NULL:
        return None
    if type_key == type_keys.Value.PARAMETER_VECTOR:
        return common.data_from_binary(binary_data, _read_parameter_vec, vectors=vectors)
    if type_key == type_keys.Value.PARAMETER:
        return common.data_from_binary(binary_data, _read_parameter)
    if type_key == type_keys.Value.PARAMETER_EXPRESSION:
        if version < 3:
            return common.data_from_binary(binary_data, _read_parameter_expression)
        else:
            return common.data_from_binary(
                binary_data, _read_parameter_expression_v3, vectors=vectors
            )

    raise exceptions.QpyError(f"Serialization for {type_key} is not implemented in value I/O.")


def read_value(file_obj, version, vectors):
    """Read a value from the file like object.

    Args:
        file_obj (File): A file like object to write data.
        version (int): QPY version.
        vectors (dict): ParameterVector in current scope.

    Returns:
        any: Deserialized value object.
    """
    type_key, data = common.read_generic_typed_data(file_obj)

    return loads_value(type_key, data, version, vectors)
