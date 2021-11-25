# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Read and write parameter values."""

import io
import struct
import uuid
from collections import namedtuple

import numpy as np

from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.qpy.common import (
    assign_key,
    TypeKey,
    OBJECT_PACK,
    OBJECT_PACK_SIZE,
)

try:
    import symengine

    HAS_SYMENGINE = True
except ImportError:
    HAS_SYMENGINE = False


# PARAMETER
PARAMETER = namedtuple("PARAMETER", ["name_size", "uuid"])
PARAMETER_PACK = "!H16s"
PARAMETER_SIZE = struct.calcsize(PARAMETER_PACK)

# PARAMETER_EXPR
PARAMETER_EXPR = namedtuple("PARAMETER_EXPR", ["map_elements", "expr_size"])
PARAMETER_EXPR_PACK = "!QQ"
PARAMETER_EXPR_SIZE = struct.calcsize(PARAMETER_EXPR_PACK)

# COMPLEX binary format
COMPLEX = namedtuple("COMPLEX", ["real", "imag"])
COMPLEX_PACK = "!dd"


def _read_parameter(file_obj):
    param_raw = struct.unpack(PARAMETER_PACK, file_obj.read(PARAMETER_SIZE))
    name_size = param_raw[0]
    param_uuid = uuid.UUID(bytes=param_raw[1])
    name = file_obj.read(name_size).decode("utf8")
    param = Parameter.__new__(Parameter, name, uuid=param_uuid)
    param.__init__(name)
    return param


def _read_parameter_expression(file_obj):
    param_expr_raw = struct.unpack(PARAMETER_EXPR_PACK, file_obj.read(PARAMETER_EXPR_SIZE))
    map_elements = param_expr_raw[0]
    from sympy.parsing.sympy_parser import parse_expr

    if HAS_SYMENGINE:
        expr = symengine.sympify(parse_expr(file_obj.read(param_expr_raw[1]).decode("utf8")))
    else:
        expr = parse_expr(file_obj.read(param_expr_raw[1]).decode("utf8"))
    symbol_map = {}
    for _ in range(map_elements):
        elem_raw = file_obj.read(OBJECT_PACK_SIZE)
        elem = struct.unpack(OBJECT_PACK, elem_raw)
        param = _read_parameter(file_obj)
        elem_type = TypeKey(elem[0].decode("utf8"))
        elem_data = file_obj.read(elem[1]).decode("utf8")
        if TypeKey.is_number(elem_type):
            value = loads_numbers(elem_type, elem_data)
        elif elem_type == TypeKey.PARAMETER:
            value = param._symbol_expr
        elif elem_type == TypeKey.PARAMETER_EXPRESSION:
            with io.BytesIO(elem_data) as container:
                value = _read_parameter_expression(container)
        else:
            raise TypeError("Invalid parameter expression map type: %s" % elem_type)
        symbol_map[param] = value
    return ParameterExpression(symbol_map, expr)


def loads_numbers(type_key, data_binary):
    if type_key == TypeKey.FLOAT:
        return struct.unpack("!d", data_binary)[0]
    if type_key == TypeKey.INTEGER:
        return struct.unpack("!q", data_binary)[0]
    if type_key == TypeKey.COMPLEX:
        return complex(*struct.unpack(COMPLEX_PACK, data_binary))
    if type_key == TypeKey.NUMPY:
        with io.BytesIO(data_binary) as container:
            value = np.load(container)
        return value

    raise TypeError(f"Invalid number type {type_key} for value {data_binary}")


def loads_parameter_value(type_key, data_binary):

    if TypeKey.is_number(type_key):
        value = loads_numbers(type_key, data_binary)
    elif type_key == TypeKey.PARAMETER:
        with io.BytesIO(data_binary) as container:
            value = _read_parameter(container)
    elif type_key == TypeKey.PARAMETER_EXPRESSION:
        with io.BytesIO(data_binary) as container:
            value = _read_parameter_expression(container)
    else:
        raise TypeError("Invalid value type: %s" % type_key)

    return value


def _write_parameter(file_obj, param):
    name_bytes = param._name.encode("utf8")
    file_obj.write(struct.pack(PARAMETER_PACK, len(name_bytes), param._uuid.bytes))
    file_obj.write(name_bytes)


def _write_parameter_expression(file_obj, param):
    from sympy import srepr, sympify

    expr_bytes = srepr(sympify(param._symbol_expr)).encode("utf8")
    param_expr_header_raw = struct.pack(
        PARAMETER_EXPR_PACK, len(param._parameter_symbols), len(expr_bytes)
    )
    file_obj.write(param_expr_header_raw)
    file_obj.write(expr_bytes)
    for parameter, value in param._parameter_symbols.items():
        # serialize key
        with io.BytesIO() as parameter_container:
            _write_parameter(parameter_container, parameter)
            parameter_container.seek(0)
            parameter_data = parameter_container.read()

        # serialize value
        if value == parameter._symbol_expr:
            type_key = TypeKey.PARAMETER
            data = bytes()
        else:
            type_key = assign_key(value)

            if TypeKey.is_number(type_key):
                data = dumps_numbers(type_key, value)
            elif type_key == TypeKey.PARAMETER_EXPRESSION:
                with io.BytesIO() as container:
                    _write_parameter_expression(container, value)
                    container.seek(0)
                    data = container.read()
            else:
                raise TypeError(
                    f"Invalid parameter expression map type {type_key} for value {value}."
                )

        elem_header = struct.pack(OBJECT_PACK, type_key.value.encode("utf8"), len(data))
        file_obj.write(elem_header)
        file_obj.write(parameter_data)
        file_obj.write(data)


def dumps_numbers(type_key, value):
    if type_key == TypeKey.FLOAT:
        return struct.pack("!d", value)
    if type_key == TypeKey.INTEGER:
        return struct.pack("!q", value)
    if type_key == TypeKey.COMPLEX:
        return struct.pack(COMPLEX_PACK, value.real, value.imag)
    if type_key == TypeKey.NUMPY:
        with io.BytesIO() as container:
            np.save(container, value)
            container.seek(0)
            data_binary = container.read()
        return data_binary

    raise TypeError(f"Invalid number type for {value}: {type(value)}")


def dumps_parameter_value(type_key, value):
    if TypeKey.is_number(type_key):
        data_binary = dumps_numbers(type_key, value)
    elif type_key == TypeKey.PARAMETER:
        with io.BytesIO() as container:
            _write_parameter(container, value)
            container.seek(0)
            data_binary = container.read()
    elif type_key == TypeKey.PARAMETER_EXPRESSION:
        with io.BytesIO() as container:
            _write_parameter_expression(container, value)
            container.seek(0)
            data_binary = container.read()
    else:
        raise TypeError(
            f"Invalid parameter value type {type_key} for value {value}."
        )

    return data_binary
