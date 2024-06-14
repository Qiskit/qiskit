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

from __future__ import annotations

import collections.abc
import struct
import uuid

import numpy as np
import symengine
from symengine.lib.symengine_wrapper import (  # pylint: disable = no-name-in-module
    load_basic,
)


from qiskit.circuit import CASE_DEFAULT, Clbit, ClassicalRegister
from qiskit.circuit.classical import expr, types
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.parametervector import ParameterVector, ParameterVectorElement
from qiskit.qpy import common, formats, exceptions, type_keys


def _write_parameter(file_obj, obj):
    name_bytes = obj.name.encode(common.ENCODE)
    file_obj.write(struct.pack(formats.PARAMETER_PACK, len(name_bytes), obj.uuid.bytes))
    file_obj.write(name_bytes)


def _write_parameter_vec(file_obj, obj):
    name_bytes = obj._vector._name.encode(common.ENCODE)
    file_obj.write(
        struct.pack(
            formats.PARAMETER_VECTOR_ELEMENT_PACK,
            len(name_bytes),
            len(obj._vector),
            obj.uuid.bytes,
            obj._index,
        )
    )
    file_obj.write(name_bytes)


def _write_parameter_expression(file_obj, obj, use_symengine, *, version):
    if use_symengine:
        expr_bytes = obj._symbol_expr.__reduce__()[1][0]
    else:
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
            value_key, value_data = dumps_value(value, version=version, use_symengine=use_symengine)

        elem_header = struct.pack(
            formats.PARAM_EXPR_MAP_ELEM_V3_PACK,
            symbol_key,
            value_key,
            len(value_data),
        )
        file_obj.write(elem_header)
        file_obj.write(symbol_data)
        file_obj.write(value_data)


class _ExprWriter(expr.ExprVisitor[None]):
    __slots__ = ("file_obj", "clbit_indices", "standalone_var_indices", "version")

    def __init__(self, file_obj, clbit_indices, standalone_var_indices, version):
        self.file_obj = file_obj
        self.clbit_indices = clbit_indices
        self.standalone_var_indices = standalone_var_indices
        self.version = version

    def visit_generic(self, node, /):
        raise exceptions.QpyError(f"unhandled Expr object '{node}'")

    def visit_var(self, node, /):
        self.file_obj.write(type_keys.Expression.VAR)
        _write_expr_type(self.file_obj, node.type)
        if node.standalone:
            self.file_obj.write(type_keys.ExprVar.UUID)
            self.file_obj.write(
                struct.pack(
                    formats.EXPR_VAR_UUID_PACK,
                    *formats.EXPR_VAR_UUID(self.standalone_var_indices[node]),
                )
            )
        elif isinstance(node.var, Clbit):
            self.file_obj.write(type_keys.ExprVar.CLBIT)
            self.file_obj.write(
                struct.pack(
                    formats.EXPR_VAR_CLBIT_PACK,
                    *formats.EXPR_VAR_CLBIT(self.clbit_indices[node.var]),
                )
            )
        elif isinstance(node.var, ClassicalRegister):
            self.file_obj.write(type_keys.ExprVar.REGISTER)
            self.file_obj.write(
                struct.pack(
                    formats.EXPR_VAR_REGISTER_PACK, *formats.EXPR_VAR_REGISTER(len(node.var.name))
                )
            )
            self.file_obj.write(node.var.name.encode(common.ENCODE))
        else:
            raise exceptions.QpyError(f"unhandled Var object '{node.var}'")

    def visit_value(self, node, /):
        self.file_obj.write(type_keys.Expression.VALUE)
        _write_expr_type(self.file_obj, node.type)
        if node.value is True or node.value is False:
            self.file_obj.write(type_keys.ExprValue.BOOL)
            self.file_obj.write(
                struct.pack(formats.EXPR_VALUE_BOOL_PACK, *formats.EXPR_VALUE_BOOL(node.value))
            )
        elif isinstance(node.value, int):
            self.file_obj.write(type_keys.ExprValue.INT)
            if node.value == 0:
                num_bytes = 0
                buffer = b""
            else:
                # This wastes a byte for `-(2 ** (8*n - 1))` for natural `n`, but they'll still
                # decode fine so it's not worth another special case.  They'll encode to
                # b"\xff\x80\x00\x00...", but we could encode them to b"\x80\x00\x00...".
                num_bytes = (node.value.bit_length() // 8) + 1
                buffer = node.value.to_bytes(num_bytes, "big", signed=True)
            self.file_obj.write(
                struct.pack(formats.EXPR_VALUE_INT_PACK, *formats.EXPR_VALUE_INT(num_bytes))
            )
            self.file_obj.write(buffer)
        else:
            raise exceptions.QpyError(f"unhandled Value object '{node.value}'")

    def visit_cast(self, node, /):
        self.file_obj.write(type_keys.Expression.CAST)
        _write_expr_type(self.file_obj, node.type)
        self.file_obj.write(
            struct.pack(formats.EXPRESSION_CAST_PACK, *formats.EXPRESSION_CAST(node.implicit))
        )
        node.operand.accept(self)

    def visit_unary(self, node, /):
        self.file_obj.write(type_keys.Expression.UNARY)
        _write_expr_type(self.file_obj, node.type)
        self.file_obj.write(
            struct.pack(formats.EXPRESSION_UNARY_PACK, *formats.EXPRESSION_UNARY(node.op.value))
        )
        node.operand.accept(self)

    def visit_binary(self, node, /):
        self.file_obj.write(type_keys.Expression.BINARY)
        _write_expr_type(self.file_obj, node.type)
        self.file_obj.write(
            struct.pack(formats.EXPRESSION_BINARY_PACK, *formats.EXPRESSION_BINARY(node.op.value))
        )
        node.left.accept(self)
        node.right.accept(self)

    def visit_index(self, node, /):
        if self.version < 12:
            raise exceptions.UnsupportedFeatureForVersion(
                "the 'Index' expression", required=12, target=self.version
            )
        self.file_obj.write(type_keys.Expression.INDEX)
        _write_expr_type(self.file_obj, node.type)
        node.target.accept(self)
        node.index.accept(self)


def _write_expr(
    file_obj,
    node: expr.Expr,
    clbit_indices: collections.abc.Mapping[Clbit, int],
    standalone_var_indices: collections.abc.Mapping[expr.Var, int],
    version: int,
):
    node.accept(_ExprWriter(file_obj, clbit_indices, standalone_var_indices, version))


def _write_expr_type(file_obj, type_: types.Type):
    if type_.kind is types.Bool:
        file_obj.write(type_keys.ExprType.BOOL)
    elif type_.kind is types.Uint:
        file_obj.write(type_keys.ExprType.UINT)
        file_obj.write(
            struct.pack(formats.EXPR_TYPE_UINT_PACK, *formats.EXPR_TYPE_UINT(type_.width))
        )
    else:
        raise exceptions.QpyError(f"unhandled Type object '{type_};")


def _read_parameter(file_obj):
    data = formats.PARAMETER(
        *struct.unpack(formats.PARAMETER_PACK, file_obj.read(formats.PARAMETER_SIZE))
    )
    param_uuid = uuid.UUID(bytes=data.uuid)
    name = file_obj.read(data.name_size).decode(common.ENCODE)
    return Parameter(name, uuid=param_uuid)


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
    if vector[data.index].uuid != param_uuid:
        vectors[name][1].add(data.index)
        vector._params[data.index] = ParameterVectorElement(vector, data.index, uuid=param_uuid)
    return vector[data.index]


def _read_parameter_expression(file_obj):
    data = formats.PARAMETER_EXPR(
        *struct.unpack(formats.PARAMETER_EXPR_PACK, file_obj.read(formats.PARAMETER_EXPR_SIZE))
    )
    from sympy.parsing.sympy_parser import parse_expr

    expr_ = symengine.sympify(parse_expr(file_obj.read(data.expr_size).decode(common.ENCODE)))
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

    return ParameterExpression(symbol_map, expr_)


def _read_parameter_expression_v3(file_obj, vectors, use_symengine):
    data = formats.PARAMETER_EXPR(
        *struct.unpack(formats.PARAMETER_EXPR_PACK, file_obj.read(formats.PARAMETER_EXPR_SIZE))
    )

    payload = file_obj.read(data.expr_size)
    if use_symengine:
        expr_ = load_basic(payload)
    else:
        from sympy.parsing.sympy_parser import parse_expr

        expr_ = symengine.sympify(parse_expr(payload.decode(common.ENCODE)))

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
                binary_data,
                _read_parameter_expression_v3,
                vectors=vectors,
                use_symengine=use_symengine,
            )
        else:
            raise exceptions.QpyError("Invalid parameter expression map type: %s" % elem_key)
        symbol_map[symbol] = value

    return ParameterExpression(symbol_map, expr_)


def _read_expr(
    file_obj,
    clbits: collections.abc.Sequence[Clbit],
    cregs: collections.abc.Mapping[str, ClassicalRegister],
    standalone_vars: collections.abc.Sequence[expr.Var],
) -> expr.Expr:
    # pylint: disable=too-many-return-statements
    type_key = file_obj.read(formats.EXPRESSION_DISCRIMINATOR_SIZE)
    type_ = _read_expr_type(file_obj)
    if type_key == type_keys.Expression.VAR:
        var_type_key = file_obj.read(formats.EXPR_VAR_DISCRIMINATOR_SIZE)
        if var_type_key == type_keys.ExprVar.UUID:
            payload = formats.EXPR_VAR_UUID._make(
                struct.unpack(formats.EXPR_VAR_UUID_PACK, file_obj.read(formats.EXPR_VAR_UUID_SIZE))
            )
            return standalone_vars[payload.var_index]
        if var_type_key == type_keys.ExprVar.CLBIT:
            payload = formats.EXPR_VAR_CLBIT._make(
                struct.unpack(
                    formats.EXPR_VAR_CLBIT_PACK, file_obj.read(formats.EXPR_VAR_CLBIT_SIZE)
                )
            )
            return expr.Var(clbits[payload.index], type_)
        if var_type_key == type_keys.ExprVar.REGISTER:
            payload = formats.EXPR_VAR_REGISTER._make(
                struct.unpack(
                    formats.EXPR_VAR_REGISTER_PACK, file_obj.read(formats.EXPR_VAR_REGISTER_SIZE)
                )
            )
            name = file_obj.read(payload.reg_name_size).decode(common.ENCODE)
            return expr.Var(cregs[name], type_)
        raise exceptions.QpyError("Invalid classical-expression Var key '{var_type_key}'")
    if type_key == type_keys.Expression.VALUE:
        value_type_key = file_obj.read(formats.EXPR_VALUE_DISCRIMINATOR_SIZE)
        if value_type_key == type_keys.ExprValue.BOOL:
            payload = formats.EXPR_VALUE_BOOL._make(
                struct.unpack(
                    formats.EXPR_VALUE_BOOL_PACK, file_obj.read(formats.EXPR_VALUE_BOOL_SIZE)
                )
            )
            return expr.Value(payload.value, type_)
        if value_type_key == type_keys.ExprValue.INT:
            payload = formats.EXPR_VALUE_INT._make(
                struct.unpack(
                    formats.EXPR_VALUE_INT_PACK, file_obj.read(formats.EXPR_VALUE_INT_SIZE)
                )
            )
            return expr.Value(
                int.from_bytes(file_obj.read(payload.num_bytes), "big", signed=True), type_
            )
        raise exceptions.QpyError("Invalid classical-expression Value key '{value_type_key}'")
    if type_key == type_keys.Expression.CAST:
        payload = formats.EXPRESSION_CAST._make(
            struct.unpack(formats.EXPRESSION_CAST_PACK, file_obj.read(formats.EXPRESSION_CAST_SIZE))
        )
        return expr.Cast(
            _read_expr(file_obj, clbits, cregs, standalone_vars), type_, implicit=payload.implicit
        )
    if type_key == type_keys.Expression.UNARY:
        payload = formats.EXPRESSION_UNARY._make(
            struct.unpack(
                formats.EXPRESSION_UNARY_PACK, file_obj.read(formats.EXPRESSION_UNARY_SIZE)
            )
        )
        return expr.Unary(
            expr.Unary.Op(payload.opcode),
            _read_expr(file_obj, clbits, cregs, standalone_vars),
            type_,
        )
    if type_key == type_keys.Expression.BINARY:
        payload = formats.EXPRESSION_BINARY._make(
            struct.unpack(
                formats.EXPRESSION_BINARY_PACK, file_obj.read(formats.EXPRESSION_BINARY_SIZE)
            )
        )
        return expr.Binary(
            expr.Binary.Op(payload.opcode),
            _read_expr(file_obj, clbits, cregs, standalone_vars),
            _read_expr(file_obj, clbits, cregs, standalone_vars),
            type_,
        )
    if type_key == type_keys.Expression.INDEX:
        return expr.Index(
            _read_expr(file_obj, clbits, cregs, standalone_vars),
            _read_expr(file_obj, clbits, cregs, standalone_vars),
            type_,
        )
    raise exceptions.QpyError(f"Invalid classical-expression Expr key '{type_key}'")


def _read_expr_type(file_obj) -> types.Type:
    type_key = file_obj.read(formats.EXPR_TYPE_DISCRIMINATOR_SIZE)
    if type_key == type_keys.ExprType.BOOL:
        return types.Bool()
    if type_key == type_keys.ExprType.UINT:
        elem = formats.EXPR_TYPE_UINT._make(
            struct.unpack(formats.EXPR_TYPE_UINT_PACK, file_obj.read(formats.EXPR_TYPE_UINT_SIZE))
        )
        return types.Uint(elem.width)
    raise exceptions.QpyError(f"Invalid classical-expression Type key '{type_key}'")


def read_standalone_vars(file_obj, num_vars):
    """Read the ``num_vars`` standalone variable declarations from the file.

    Args:
        file_obj (File): a file-like object to read from.
        num_vars (int): the number of variables to read.

    Returns:
        tuple[dict, list]: the first item is a mapping of the ``ExprVarDeclaration`` type keys to
        the variables defined by that type key, and the second is the total order of variable
        declarations.
    """
    read_vars = {
        type_keys.ExprVarDeclaration.INPUT: [],
        type_keys.ExprVarDeclaration.CAPTURE: [],
        type_keys.ExprVarDeclaration.LOCAL: [],
    }
    var_order = []
    for _ in range(num_vars):
        data = formats.EXPR_VAR_DECLARATION._make(
            struct.unpack(
                formats.EXPR_VAR_DECLARATION_PACK,
                file_obj.read(formats.EXPR_VAR_DECLARATION_SIZE),
            )
        )
        type_ = _read_expr_type(file_obj)
        name = file_obj.read(data.name_size).decode(common.ENCODE)
        var = expr.Var(uuid.UUID(bytes=data.uuid_bytes), type_, name=name)
        read_vars[data.usage].append(var)
        var_order.append(var)
    return read_vars, var_order


def _write_standalone_var(file_obj, var, type_key):
    name = var.name.encode(common.ENCODE)
    file_obj.write(
        struct.pack(
            formats.EXPR_VAR_DECLARATION_PACK,
            *formats.EXPR_VAR_DECLARATION(var.var.bytes, type_key, len(name)),
        )
    )
    _write_expr_type(file_obj, var.type)
    file_obj.write(name)


def write_standalone_vars(file_obj, circuit):
    """Write the standalone variables out from a circuit.

    Args:
        file_obj (File): the file-like object to write to.
        circuit (QuantumCircuit): the circuit to take the variables from.

    Returns:
        dict[expr.Var, int]: a mapping of the variables written to the index that they were written
        at.
    """
    index = 0
    out = {}
    for var in circuit.iter_input_vars():
        _write_standalone_var(file_obj, var, type_keys.ExprVarDeclaration.INPUT)
        out[var] = index
        index += 1
    for var in circuit.iter_captured_vars():
        _write_standalone_var(file_obj, var, type_keys.ExprVarDeclaration.CAPTURE)
        out[var] = index
        index += 1
    for var in circuit.iter_declared_vars():
        _write_standalone_var(file_obj, var, type_keys.ExprVarDeclaration.LOCAL)
        out[var] = index
        index += 1
    return out


def dumps_value(
    obj,
    *,
    version,
    index_map=None,
    use_symengine=False,
    standalone_var_indices=None,
):
    """Serialize input value object.

    Args:
        obj (any): Arbitrary value object to serialize.
        version (int): the target QPY version for the dump.
        index_map (dict): Dictionary with two keys, "q" and "c".  Each key has a value that is a
            dictionary mapping :class:`.Qubit` or :class:`.Clbit` instances (respectively) to their
            integer indices.
        use_symengine (bool): If True, symbolic objects will be serialized using symengine's
            native mechanism. This is a faster serialization alternative, but not supported in all
            platforms. Please check that your target platform is supported by the symengine library
            before setting this option, as it will be required by qpy to deserialize the payload.
        standalone_var_indices (dict): Dictionary that maps standalone :class:`.expr.Var` entries to
            the index that should be used to refer to them.

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
    elif type_key in (type_keys.Value.NULL, type_keys.Value.CASE_DEFAULT):
        binary_data = b""
    elif type_key == type_keys.Value.PARAMETER_VECTOR:
        binary_data = common.data_to_binary(obj, _write_parameter_vec)
    elif type_key == type_keys.Value.PARAMETER:
        binary_data = common.data_to_binary(obj, _write_parameter)
    elif type_key == type_keys.Value.PARAMETER_EXPRESSION:
        binary_data = common.data_to_binary(
            obj, _write_parameter_expression, use_symengine=use_symengine, version=version
        )
    elif type_key == type_keys.Value.EXPRESSION:
        clbit_indices = {} if index_map is None else index_map["c"]
        standalone_var_indices = {} if standalone_var_indices is None else standalone_var_indices
        binary_data = common.data_to_binary(
            obj,
            _write_expr,
            clbit_indices=clbit_indices,
            standalone_var_indices=standalone_var_indices,
            version=version,
        )
    else:
        raise exceptions.QpyError(f"Serialization for {type_key} is not implemented in value I/O.")

    return type_key, binary_data


def write_value(
    file_obj, obj, *, version, index_map=None, use_symengine=False, standalone_var_indices=None
):
    """Write a value to the file like object.

    Args:
        file_obj (File): A file like object to write data.
        obj (any): Value to write.
        version (int): the target QPY version for the dump.
        index_map (dict): Dictionary with two keys, "q" and "c".  Each key has a value that is a
            dictionary mapping :class:`.Qubit` or :class:`.Clbit` instances (respectively) to their
            integer indices.
        use_symengine (bool): If True, symbolic objects will be serialized using symengine's
            native mechanism. This is a faster serialization alternative, but not supported in all
            platforms. Please check that your target platform is supported by the symengine library
            before setting this option, as it will be required by qpy to deserialize the payload.
        standalone_var_indices (dict): Dictionary that maps standalone :class:`.expr.Var` entries to
            the index that should be used to refer to them.
    """
    type_key, data = dumps_value(
        obj,
        version=version,
        index_map=index_map,
        use_symengine=use_symengine,
        standalone_var_indices=standalone_var_indices,
    )
    common.write_generic_typed_data(file_obj, type_key, data)


def loads_value(
    type_key,
    binary_data,
    version,
    vectors,
    *,
    clbits=(),
    cregs=None,
    use_symengine=False,
    standalone_vars=(),
):
    """Deserialize input binary data to value object.

    Args:
        type_key (ValueTypeKey): Type enum information.
        binary_data (bytes): Data to deserialize.
        version (int): QPY version.
        vectors (dict): ParameterVector in current scope.
        clbits (Sequence[Clbit]): Clbits in the current scope.
        cregs (Mapping[str, ClassicalRegister]): Classical registers in the current scope.
        use_symengine (bool): If True, symbolic objects will be de-serialized using symengine's
            native mechanism. This is a faster serialization alternative, but not supported in all
            platforms. Please check that your target platform is supported by the symengine library
            before setting this option, as it will be required by qpy to deserialize the payload.
        standalone_vars (Sequence[Var]): standalone :class:`.expr.Var` nodes in the order that they
            were declared by the circuit header.

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
    if type_key == type_keys.Value.CASE_DEFAULT:
        return CASE_DEFAULT
    if type_key == type_keys.Value.PARAMETER_VECTOR:
        return common.data_from_binary(binary_data, _read_parameter_vec, vectors=vectors)
    if type_key == type_keys.Value.PARAMETER:
        return common.data_from_binary(binary_data, _read_parameter)
    if type_key == type_keys.Value.PARAMETER_EXPRESSION:
        if version < 3:
            return common.data_from_binary(binary_data, _read_parameter_expression)
        else:
            return common.data_from_binary(
                binary_data,
                _read_parameter_expression_v3,
                vectors=vectors,
                use_symengine=use_symengine,
            )
    if type_key == type_keys.Value.EXPRESSION:
        return common.data_from_binary(
            binary_data,
            _read_expr,
            clbits=clbits,
            cregs=cregs or {},
            standalone_vars=standalone_vars,
        )

    raise exceptions.QpyError(f"Serialization for {type_key} is not implemented in value I/O.")


def read_value(
    file_obj,
    version,
    vectors,
    *,
    clbits=(),
    cregs=None,
    use_symengine=False,
    standalone_vars=(),
):
    """Read a value from the file like object.

    Args:
        file_obj (File): A file like object to write data.
        version (int): QPY version.
        vectors (dict): ParameterVector in current scope.
        clbits (Sequence[Clbit]): Clbits in the current scope.
        cregs (Mapping[str, ClassicalRegister]): Classical registers in the current scope.
        use_symengine (bool): If True, symbolic objects will be de-serialized using symengine's
            native mechanism. This is a faster serialization alternative, but not supported in all
            platforms. Please check that your target platform is supported by the symengine library
            before setting this option, as it will be required by qpy to deserialize the payload.
        standalone_vars (Sequence[expr.Var]): standalone variables in the order they were defined in
            the QPY payload.

    Returns:
        any: Deserialized value object.
    """
    type_key, data = common.read_generic_typed_data(file_obj)

    return loads_value(
        type_key,
        data,
        version,
        vectors,
        clbits=clbits,
        cregs=cregs,
        use_symengine=use_symengine,
        standalone_vars=standalone_vars,
    )
