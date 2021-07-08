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

# pylint: disable=invalid-name,too-many-boolean-expressions,redundant-keyword-arg

"""
===========================================================
QPY serialization (:mod:`qiskit.circuit.qpy_serialization`)
===========================================================

.. currentmodule:: qiskit.circuit.qpy_serialization

.. autosummary::

    load
    dump

QPY Format
==========

The QPY serialization format is a portable cross-platform binary
serialization format for :class:`~qiskit.circuit.QuantumCircuit` objects in Qiskit. The basic
file format is as follows:

A QPY file (or memory object) always starts with the following 7
byte UTF8 string: ``QISKIT`` which is immediately followed by the overall
file header. The contents of thie file header as defined as a C struct are:

.. code-block:: c

    struct {
        unsigned char qpy_version;
        unsigned char qiskit_major_version;
        unsigned char qiskit_minor_version;
        unsigned char qiskit_patch_version;
        uint64_t num_circuits;
    }

All values use network byte order [#f1]_ (big endian) for cross platform
compatibility.

The file header is immediately followed by the circuit payloads.
Each individual circuit is composed of the following parts:

``HEADER | METADATA | REGISTERS | CUSTOM_DEFINITIONS | INSTRUCTIONS``

There is a circuit payload for each circuit (where the total number is dictated
by ``num_circuits`` in the file header). There is no padding between the
circuits in the data.

HEADER
------

The contents of HEADER as defined as a C struct are:

.. code-block:: c

    struct {
        unit16_t name_size;
        double global_phase;
        uint32_t num_qubits;
        uint32_t num_clbits;
        uint64_t metadata_size;
        uint32_t num_registers;
        uint64_t num_instructions;
        uint64_t num_custom_gates;
    }

This is immediately followed by ``name_size`` bytes of utf8 data for the name
of the circuit.

METADATA
--------

The METADATA field is a UTF8 encoded JSON string. After reading the HEADER
(which is a fixed size at the start of the QPY file) and the ``name`` string
you then read the`metadata_size`` number of bytes and parse the JSON to get
the metadata for the circuit.

REGISTERS
---------

The contents of REGISTERS is a number of REGISTER object. If num_registers is
> 0 then after reading METADATA you read that number of REGISTER structs defined
as:

.. code-block:: c

    struct {
        char type;
        _Bool standalone;
        uint32_t size;
        unit16_t name_size;
    }

``type`` can be ``'q'`` or ``'c'``.

Immediately following the REGISTER struct is the utf8 encoded register name of
size ``name_size``. After the ``name`` utf8 bytes there is then an array of
uint32_t values of size ``size`` that contains a map of the register's index to
the circuit's qubit index. For example, array element 0's value is the index
of the ``register[0]``'s position in the containing circuit's qubits list.

The standalone boolean determines whether the register is constructed as a
standalone register that was added to the circuit or was created from existing
bits. A register is considered standalone if it has bits constructed solely
as part of it, for example::

    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr)

the register ``qr`` would be a standalone register. While something like::

    bits = [Qubit(), Qubit()]
    qr = QuantumRegister(bits=bits)
    qc = QuantumCircuit(bits=bits)

``qr`` would have ``standalone`` set to ``False``.


CUSTOM_DEFINITIONS
------------------

If the circuit contains custom defitions for any of the instruction in the circuit.
this section

CUSTOM_DEFINITION_HEADER contents are defined as:

.. code-block:: c

    struct {
        uint64_t size;
    }

If size is greater than 0 that means the circuit contains custom instruction(s).
Each custom instruction is defined with a CUSTOM_INSTRUCTION block defined as:

.. code-block:: c

    struct {
        uint16_t name_size;
        char type;
        _Bool custom_definition;
        uint64_t size
    }

Immediately following the CUSTOM_INSTRUCTION struct is the utf8 encoded name
of size ``name_size``.

If ``custom_definition`` is ``True`` that means that the immediately following
``size`` bytes contains a QPY circuit data which can be used for the custom
definition of that gate. If ``custom_definition`` is ``False`` than the
instruction can be considered opaque (ie no definition).

INSTRUCTIONS
------------

The contents of INSTRUCTIONS is a list of INSTRUCTION metadata objects

.. code-block:: c

    struct {
        uint16_t name_size;
        uint16_t label_size;
        uint16_t num_parameters;
        uint32_t num_qargs;
        uint32_t num_cargs;
        _Bool has_conditionl
        uint16_t conditonal_reg_name_size;
        long long conditional_value;
    }

This metadata object is immediately followed by ``name_size`` bytes of utf8 bytes
for the ``name``. ``name`` here is the Qiskit class name for the Instruction
class if it's defined in Qiskit. Otherwise it falls back to the custom
instruction name. Following the ``name`` bytes there are ``label_size`` bytes of
utf8 data for the label if one was set on the instruction. Following the label
bytes if ``has_conditional`` is ``True`` then there are
``conditonal_reg_name_size`` bytes of utf8 data for the name of the condtional
register name.

This is immediately followed by the INSTRUCTION_ARG structs for the list of
arguments of that instruction. These are in the order of all quantum arguments
(there are num_qargs of these) followed by all classical arguments (num_cargs
of these).

The contents of each INSTRUCTION_ARG is:

.. code-block:: c

    struct {
        char type;
        unisgned int index;
    }

``type`` can be ``'q'`` or ``'c'``.

After all arguments for an instruction the parameters are specified with
``num_parameters`` INSTRUCTION_PARAM structs.

The contents of each INSTRUCTION_PARAM is:

.. code-block:: c

    struct {
        char type;
        uint64_t size;
    }

After each INSTRUCTION_PARAM the next ``size`` bytes are the parameter's data.
The ``type`` field can be ``'i'``, ``'f'``, ``'p'``, ``'e'``, ``'s'``,
or ``'n'`` which dictate the format. For ``'i'`` it's an integer, ``'f'`` it's
a double, ``'s'`` if it's a string (encoded as utf8), ``'p'`` defines a
:class:`~qiskit.circuit.Parameter` object  which is represented by a PARAM
struct (see below), ``e`` defines a :class:`~qiskit.circuit.ParameterExpression`
object (that's not a :class:`~qiskit.circuit.Paramter`) which is represented by
a PARAM_EXPR struct (see below), and ``'n'`` represents an object from numpy
(either an ``ndarray`` or a numpy type) which means the data is .npy
format [#f2]_ data.


PARAMETER
---------

A PARAMETER represents a :class:`~qiskit.circuit.Parameter` object the data for
a INSTRUCTION_PARAM. The contents of the PARAMETER are defined as:

.. code-block:: c

    struct {
        uint16_t name_size;
        char uuid[16];
    }

which is immediately followed by ``name_size`` utf8 bytes representing the
parameter name.

PARAMETER_EXPR
--------------

A PARAMETER_EXPR represents a :class:`~qiskit.circuit.ParameterExpression`
object that the data for an INSTRUCTION_PARAM. The contents of a PARAMETER_EXPR
are defined as:

The PARAMETER_EXPR data starts with a header:

.. code-block:: c

    struct {
        uint64_t map_elements,
        uint64_t expr_size,
    }

Immediately following the header is ``expr_size`` bytes of utf8 data containing
the expression string, which is the sympy srepr of the expression for the
parameter expression. Follwing that is a symbol map which contains
``map_elements`` elements with the format

.. code-block:: c

    struct {
        char type;
        uint64_t size;
    }

Which is followed immediately by ``PARAMETER`` object (both the struct and utf8
name bytes) for the symbol map key. That is followed by ``size`` bytes for the
data of the symbol. The data format is dependent on the value of ``type``. If
``type`` is ``p`` then it represents a :class:`~qiskit.circuit.Parameter` and
size will be 0, the value will just be the same as the key. If
``type`` is ``f`` then it represents a double precision float. If ``type`` is
``c`` it represents a double precision complex, which is represented by:

.. code-block:: c

    struct {
        double real;
        double imag;
    }

this matches the internal C representation of Python's complex type. [#f3]_
Finally, if type is ``i`` it represents an integer which is an ``int64_t``.


.. [#f1] https://tools.ietf.org/html/rfc1700
.. [#f2] https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
.. [#f3] https://docs.python.org/3/c-api/complex.html#c.Py_complex
"""
from collections import namedtuple
import io
import json
import struct
import uuid
import warnings

import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.gate import Gate
from qiskit.circuit.instruction import Instruction
from qiskit.circuit import library
from qiskit import circuit as circuit_mod
from qiskit import extensions
from qiskit.extensions import quantum_initializer
from qiskit.version import __version__
from qiskit.exceptions import QiskitError

try:
    import symengine

    HAS_SYMENGINE = True
except ImportError:
    HAS_SYMENGINE = False


# v1 Binary Format
# ----------------
# FILE_HEADER
FILE_HEADER = namedtuple(
    "FILE_HEADER",
    ["preface", "qpy_version", "major_version", "minor_version", "patch_version", "num_circuits"],
)
FILE_HEADER_PACK = "!6sBBBBQ"
FILE_HEADER_SIZE = struct.calcsize(FILE_HEADER_PACK)

# HEADER binary format
HEADER = namedtuple(
    "HEADER",
    [
        "name_size",
        "global_phase",
        "num_qubits",
        "num_clbits",
        "metadata_size",
        "num_registers",
        "num_instructions",
    ],
)
HEADER_PACK = "!HdIIQIQ"
HEADER_SIZE = struct.calcsize(HEADER_PACK)

# CUSTOM_DEFINITIONS
# CUSTOM DEFINITION HEADER
CUSTOM_DEFINITION_HEADER = namedtuple("CUSTOM_DEFINITION_HEADER", ["size"])
CUSTOM_DEFINITION_HEADER_PACK = "!Q"
CUSTOM_DEFINITION_HEADER_SIZE = struct.calcsize(CUSTOM_DEFINITION_HEADER_PACK)

# CUSTOM_DEFINITION
CUSTOM_DEFINITION = namedtuple(
    "CUSTOM_DEFINITON",
    ["gate_name_size", "type", "num_qubits", "num_clbits", "custom_definition", "size"],
)
CUSTOM_DEFINITION_PACK = "!H1cII?Q"
CUSTOM_DEFINITION_SIZE = struct.calcsize(CUSTOM_DEFINITION_PACK)


# REGISTER binary format
REGISTER = namedtuple("REGISTER", ["type", "standalone", "size", "name_size"])
REGISTER_PACK = "!1c?IH"
REGISTER_SIZE = struct.calcsize(REGISTER_PACK)

# INSTRUCTION binary format
INSTRUCTION = namedtuple(
    "INSTRUCTION",
    [
        "name_size",
        "label_size",
        "num_parameters",
        "num_qargs",
        "num_cargs",
        "has_condition",
        "condition_register_size",
        "value",
    ],
)
INSTRUCTION_PACK = "!HHHII?Hq"
INSTRUCTION_SIZE = struct.calcsize(INSTRUCTION_PACK)
# Instruction argument format
INSTRUCTION_ARG = namedtuple("INSTRUCTION_ARG", ["type", "size"])
INSTRUCTION_ARG_PACK = "!1cI"
INSTRUCTION_ARG_SIZE = struct.calcsize(INSTRUCTION_ARG_PACK)
# INSTRUCTION parameter format
INSTRUCTION_PARAM = namedtuple("INSTRUCTION_PARAM", ["type", "size"])
INSTRUCTION_PARAM_PACK = "!1cQ"
INSTRUCTION_PARAM_SIZE = struct.calcsize(INSTRUCTION_PARAM_PACK)
# PARAMETER
PARAMETER = namedtuple("PARAMETER", ["name_size", "uuid"])
PARAMETER_PACK = "!H16s"
PARAMETER_SIZE = struct.calcsize(PARAMETER_PACK)
# PARAMETER_EXPR
PARAMETER_EXPR = namedtuple("PARAMETER_EXPR", ["map_elements", "expr_size"])
PARAMETER_EXPR_PACK = "!QQ"
PARAMETER_EXPR_SIZE = struct.calcsize(PARAMETER_EXPR_PACK)
# PARAMETER_EXPR_MAP_ELEM
PARAM_EXPR_MAP_ELEM = namedtuple("PARAMETER_EXPR_MAP_ELEM", ["type", "size"])
PARAM_EXPR_MAP_ELEM_PACK = "!cQ"
PARAM_EXPR_MAP_ELEM_SIZE = struct.calcsize(PARAM_EXPR_MAP_ELEM_PACK)
# Complex
COMPLEX = namedtuple("COMPLEX", ["real", "imag"])
COMPLEX_PACK = "!dd"
COMPLEX_SIZE = struct.calcsize(COMPLEX_PACK)


def _read_header(file_obj):
    header_raw = struct.unpack(HEADER_PACK, file_obj.read(HEADER_SIZE))
    header = HEADER._make(header_raw)
    name = file_obj.read(header[0]).decode("utf8")
    metadata_raw = file_obj.read(header[4])
    metadata = json.loads(metadata_raw)
    return header, name, metadata


def _read_registers(file_obj, num_registers):
    registers = {"q": {}, "c": {}}
    for _reg in range(num_registers):
        register_raw = file_obj.read(REGISTER_SIZE)
        register = struct.unpack(REGISTER_PACK, register_raw)
        name = file_obj.read(register[3]).decode("utf8")
        standalone = register[1]
        REGISTER_ARRAY_PACK = "!%sI" % register[2]
        bit_indices_raw = file_obj.read(struct.calcsize(REGISTER_ARRAY_PACK))
        bit_indices = list(struct.unpack(REGISTER_ARRAY_PACK, bit_indices_raw))
        if register[0].decode("utf8") == "q":
            registers["q"][name] = (standalone, bit_indices)
        else:
            registers["c"][name] = (standalone, bit_indices)
    return registers


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
        elem_raw = file_obj.read(PARAM_EXPR_MAP_ELEM_SIZE)
        elem = struct.unpack(PARAM_EXPR_MAP_ELEM_PACK, elem_raw)
        param = _read_parameter(file_obj)
        elem_type = elem[0].decode("utf8")
        elem_data = file_obj.read(elem[1])
        if elem_type == "f":
            value = struct.unpack("!d", elem_data)
        elif elem_type == "i":
            value = struct.unpack("!q", elem_data)
        elif elem_type == "c":
            value = complex(*struct.unpack(COMPLEX_PACK, elem_data))
        elif elem_type == "p":
            value = param._symbol_expr
        elif elem_type == "e":
            value = _read_parameter_expression(io.BytesIO(elem_data))
        else:
            raise TypeError("Invalid parameter expression map type: %s" % elem_type)
        symbol_map[param] = value
    return ParameterExpression(symbol_map, expr)


def _read_instruction(file_obj, circuit, registers, custom_instructions):
    instruction_raw = file_obj.read(INSTRUCTION_SIZE)
    instruction = struct.unpack(INSTRUCTION_PACK, instruction_raw)
    name_size = instruction[0]
    label_size = instruction[1]
    qargs = []
    cargs = []
    params = []
    gate_name = file_obj.read(name_size).decode("utf8")
    label = file_obj.read(label_size).decode("utf8")
    num_qargs = instruction[3]
    num_cargs = instruction[4]
    num_params = instruction[2]
    has_condition = instruction[5]
    register_name_size = instruction[6]
    condition_register = file_obj.read(register_name_size).decode("utf8")
    condition_value = instruction[7]
    condition_tuple = None
    if has_condition:
        condition_tuple = (registers["c"][condition_register], condition_value)
    qubit_indices = dict(enumerate(circuit.qubits))
    clbit_indices = dict(enumerate(circuit.clbits))
    # Load Arguments
    for _qarg in range(num_qargs):
        qarg_raw = file_obj.read(INSTRUCTION_ARG_SIZE)
        qarg = struct.unpack(INSTRUCTION_ARG_PACK, qarg_raw)
        if qarg[0].decode("utf8") == "c":
            raise TypeError("Invalid input carg prior to all qargs")
        qargs.append(qubit_indices[qarg[1]])
    for _carg in range(num_cargs):
        carg_raw = file_obj.read(INSTRUCTION_ARG_SIZE)
        carg = struct.unpack(INSTRUCTION_ARG_PACK, carg_raw)
        if carg[0].decode("utf8") == "q":
            raise TypeError("Invalid input qarg after all qargs")
        cargs.append(clbit_indices[carg[1]])
    # Load Parameters
    for _param in range(num_params):
        param_raw = file_obj.read(INSTRUCTION_PARAM_SIZE)
        param = struct.unpack(INSTRUCTION_PARAM_PACK, param_raw)
        data = file_obj.read(param[1])
        type_str = param[0].decode("utf8")
        param = None
        if type_str == "i":
            param = struct.unpack("<q", data)[0]
        elif type_str == "f":
            param = struct.unpack("<d", data)[0]
        elif type_str == "n":
            container = io.BytesIO(data)
            param = np.load(container)
        elif type_str == "s":
            param = data.decode("utf8")
        elif type_str == "p":
            container = io.BytesIO(data)
            param = _read_parameter(container)
        elif type_str == "e":
            container = io.BytesIO(data)
            param = _read_parameter_expression(container)
        else:
            raise TypeError("Invalid parameter type: %s" % type_str)
        params.append(param)
    # Load Gate object
    gate_class = None
    if gate_name in ("Gate", "Instruction"):
        inst_obj = _parse_custom_instruction(custom_instructions, gate_name, params)
        inst_obj.condition = condition_tuple
        if label_size > 0:
            inst_obj.label = label
        circuit._append(inst_obj, qargs, cargs)
        return
    elif hasattr(library, gate_name):
        gate_class = getattr(library, gate_name)
    elif hasattr(circuit_mod, gate_name):
        gate_class = getattr(circuit_mod, gate_name)
    elif hasattr(extensions, gate_name):
        gate_class = getattr(extensions, gate_name)
    elif hasattr(quantum_initializer, gate_name):
        gate_class = getattr(quantum_initializer, gate_name)
    elif gate_name in custom_instructions:
        inst_obj = _parse_custom_instruction(custom_instructions, gate_name, params)
        inst_obj.condition = condition_tuple
        if label_size > 0:
            inst_obj.label = label
        circuit._append(inst_obj, qargs, cargs)
        return
    else:
        raise AttributeError("Invalid instruction type: %s" % gate_name)
    if gate_name == "Barrier":
        params = [len(qargs)]
    gate = gate_class(*params)
    gate.condition = condition_tuple
    if label_size > 0:
        gate.label = label
    circuit._append(gate, qargs, cargs)


def _parse_custom_instruction(custom_instructions, gate_name, params):
    (type_str, num_qubits, num_clbits, definition) = custom_instructions[gate_name]
    if type_str == "i":
        inst_obj = Instruction(gate_name, num_qubits, num_clbits, params)
        if definition:
            inst_obj.definition = definition
    elif type_str == "g":
        inst_obj = Gate(gate_name, num_qubits, params)
        inst_obj.definition = definition
    else:
        raise ValueError("Invalid custom instruction type '%s'" % type_str)
    return inst_obj


def _read_custom_instructions(file_obj):
    custom_instructions = {}
    custom_definition_header_raw = file_obj.read(CUSTOM_DEFINITION_HEADER_SIZE)
    custom_definition_header = struct.unpack(
        CUSTOM_DEFINITION_HEADER_PACK, custom_definition_header_raw
    )
    if custom_definition_header[0] > 0:
        for _ in range(custom_definition_header[0]):
            custom_definition_raw = file_obj.read(CUSTOM_DEFINITION_SIZE)
            custom_definition = struct.unpack(CUSTOM_DEFINITION_PACK, custom_definition_raw)
            (
                name_size,
                type_str,
                num_qubits,
                num_clbits,
                has_custom_definition,
                size,
            ) = custom_definition
            name = file_obj.read(name_size).decode("utf8")
            type_str = type_str.decode("utf8")
            definition_circuit = None
            if has_custom_definition:
                definition_buffer = io.BytesIO(file_obj.read(size))
                definition_circuit = _read_circuit(definition_buffer)
            custom_instructions[name] = (type_str, num_qubits, num_clbits, definition_circuit)
    return custom_instructions


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
        parameter_container = io.BytesIO()
        _write_parameter(parameter_container, parameter)
        parameter_container.seek(0)
        parameter_data = parameter_container.read()
        if isinstance(value, float):
            type_str = "f"
            data = struct.pack("!d", value)
        elif isinstance(value, complex):
            type_str = "c"
            data = struct.pack(COMPLEX_PACK, value.real, value.imag)
        elif isinstance(value, int):
            type_str = "i"
            data = struct.pack("!q", value)
        elif value == parameter._symbol_expr:
            type_str = "p"
            data = bytes()
        elif isinstance(value, ParameterExpression):
            type_str = "e"
            container = io.BytesIO()
            _write_parameter_expression(container, value)
            container.seek(0)
            data = container.read()
        else:
            raise TypeError(f"Invalid expression type in symbol map for {param}: {type(value)}")

        elem_header = struct.pack(PARAM_EXPR_MAP_ELEM_PACK, type_str.encode("utf8"), len(data))
        file_obj.write(elem_header)
        file_obj.write(parameter_data)
        file_obj.write(data)


def _write_instruction(file_obj, instruction_tuple, custom_instructions, index_map):
    gate_class_name = instruction_tuple[0].__class__.__name__
    if (
        (
            not hasattr(library, gate_class_name)
            and not hasattr(circuit_mod, gate_class_name)
            and not hasattr(extensions, gate_class_name)
            and not hasattr(quantum_initializer, gate_class_name)
        )
        or gate_class_name == "Gate"
        or gate_class_name == "Instruction"
    ):
        if instruction_tuple[0].name not in custom_instructions:
            custom_instructions[instruction_tuple[0].name] = instruction_tuple[0]
        gate_class_name = instruction_tuple[0].name

    has_condition = False
    condition_register = b""
    condition_value = 0
    if instruction_tuple[0].condition:
        has_condition = True
        condition_register = instruction_tuple[0].condition[0].name.encode("utf8")
        condition_value = instruction_tuple[0].condition[1]

    gate_class_name = gate_class_name.encode("utf8")
    label = getattr(instruction_tuple[0], "label")
    if label:
        label_raw = label.encode("utf8")
    else:
        label_raw = b""
    instruction_raw = struct.pack(
        INSTRUCTION_PACK,
        len(gate_class_name),
        len(label_raw),
        len(instruction_tuple[0].params),
        instruction_tuple[0].num_qubits,
        instruction_tuple[0].num_clbits,
        has_condition,
        len(condition_register),
        condition_value,
    )
    file_obj.write(instruction_raw)
    file_obj.write(gate_class_name)
    file_obj.write(label_raw)
    file_obj.write(condition_register)
    # Encode instruciton args
    for qbit in instruction_tuple[1]:
        instruction_arg_raw = struct.pack(INSTRUCTION_ARG_PACK, b"q", index_map["q"][qbit])
        file_obj.write(instruction_arg_raw)
    for clbit in instruction_tuple[2]:
        instruction_arg_raw = struct.pack(INSTRUCTION_ARG_PACK, b"c", index_map["c"][clbit])
        file_obj.write(instruction_arg_raw)
    # Encode instruction params
    for param in instruction_tuple[0].params:
        container = io.BytesIO()
        if isinstance(param, int):
            type_key = "i"
            data = struct.pack("<q", param)
            size = struct.calcsize("<q")
        elif isinstance(param, float):
            type_key = "f"
            data = struct.pack("<d", param)
            size = struct.calcsize("<d")
        elif isinstance(param, str):
            type_key = "s"
            data = param.encode("utf8")
            size = len(data)
        elif isinstance(param, Parameter):
            type_key = "p"
            _write_parameter(container, param)
            container.seek(0)
            data = container.read()
            size = len(data)
        elif isinstance(param, ParameterExpression):
            type_key = "e"
            _write_parameter_expression(container, param)
            container.seek(0)
            data = container.read()
            size = len(data)
        elif isinstance(param, (np.integer, np.floating, np.ndarray)):
            type_key = "n"
            np.save(container, param)
            container.seek(0)
            data = container.read()
            size = len(data)
        else:
            raise TypeError(
                f"Invalid parameter type {instruction_tuple[0]} for gate {type(param)},"
            )
        instruction_param_raw = struct.pack(INSTRUCTION_PARAM_PACK, type_key.encode("utf8"), size)
        file_obj.write(instruction_param_raw)
        file_obj.write(data)
        container.close()


def _write_custom_instruction(file_obj, name, instruction):
    if isinstance(instruction, Gate):
        type_str = b"g"
    else:
        type_str = b"i"
    has_definition = False
    size = 0
    data = None
    num_qubits = instruction.num_qubits
    num_clbits = instruction.num_clbits
    if instruction.definition:
        has_definition = True
        definition_buffer = io.BytesIO()
        _write_circuit(definition_buffer, instruction.definition)
        definition_buffer.seek(0)
        data = definition_buffer.read()
        definition_buffer.close()
        size = len(data)
    name_raw = name.encode("utf8")
    custom_instruction_raw = struct.pack(
        CUSTOM_DEFINITION_PACK,
        len(name_raw),
        type_str,
        num_qubits,
        num_clbits,
        has_definition,
        size,
    )
    file_obj.write(custom_instruction_raw)
    file_obj.write(name_raw)
    if data:
        file_obj.write(data)


def dump(circuits, file_obj):
    """Write QPY binary data to a file

    This function is used to save a circuit to a file for later use or transfer
    between machines. The QPY format is backwards compatible and can be
    loaded with future versions of Qiskit.

    For example:

    .. code-block:: python

        from qiskit.circuit import QuantumCircuit
        from qiskit.circuit import qpy_serialization

        qc = QuantumCircuit(2, name='Bell', metadata={'test': True})
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

    from this you can write the qpy data to a file:

    .. code-block:: python

        with open('bell.qpy', 'wb') as fd:
            qpy_serialization.dump(qc, fd)

    or a gzip compressed file:

    .. code-block:: python

        import gzip

        with gzip.open('bell.qpy.gz', 'wb') as fd:
            qpy_serialization.dump(qc, fd)

    Which will save the qpy serialized circuit to the provided file.

    Args:
        circuits (list or QuantumCircuit): The quantum circuit object(s) to
            store in the specified file like object. This can either be a
            single QuantumCircuit object or a list of QuantumCircuits.
        file_obj (file): The file like object to write the QPY data too
    """
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]
    version_parts = [int(x) for x in __version__.split(".")[0:3]]
    header = struct.pack(
        FILE_HEADER_PACK,
        b"QISKIT",
        1,
        version_parts[0],
        version_parts[1],
        version_parts[2],
        len(circuits),
    )
    file_obj.write(header)
    for circuit in circuits:
        _write_circuit(file_obj, circuit)


def _write_circuit(file_obj, circuit):
    metadata_raw = json.dumps(circuit.metadata, separators=(",", ":")).encode("utf8")
    metadata_size = len(metadata_raw)
    num_registers = len(circuit.qregs) + len(circuit.cregs)
    num_instructions = len(circuit)
    circuit_name = circuit.name.encode("utf8")
    header_raw = HEADER(
        name_size=len(circuit_name),
        global_phase=circuit.global_phase,
        num_qubits=circuit.num_qubits,
        num_clbits=circuit.num_clbits,
        metadata_size=metadata_size,
        num_registers=num_registers,
        num_instructions=num_instructions,
    )
    header = struct.pack(HEADER_PACK, *header_raw)
    file_obj.write(header)
    file_obj.write(circuit_name)
    file_obj.write(metadata_raw)
    qubit_indices = {bit: index for index, bit in enumerate(circuit.qubits)}
    clbit_indices = {bit: index for index, bit in enumerate(circuit.clbits)}
    if num_registers > 0:
        for reg in circuit.qregs:
            standalone = all(bit._register is reg for bit in reg)
            reg_name = reg.name.encode("utf8")
            file_obj.write(struct.pack(REGISTER_PACK, b"q", standalone, reg.size, len(reg_name)))
            file_obj.write(reg_name)
            REGISTER_ARRAY_PACK = "!%sI" % reg.size
            file_obj.write(struct.pack(REGISTER_ARRAY_PACK, *(qubit_indices[bit] for bit in reg)))
        for reg in circuit.cregs:
            standalone = all(bit._register is reg for bit in reg)
            reg_name = reg.name.encode("utf8")
            file_obj.write(struct.pack(REGISTER_PACK, b"c", standalone, reg.size, len(reg_name)))
            file_obj.write(reg_name)
            REGISTER_ARRAY_PACK = "!%sI" % reg.size
            file_obj.write(struct.pack(REGISTER_ARRAY_PACK, *(clbit_indices[bit] for bit in reg)))
    instruction_buffer = io.BytesIO()
    custom_instructions = {}
    index_map = {}
    index_map["q"] = qubit_indices
    index_map["c"] = clbit_indices
    for instruction in circuit.data:
        _write_instruction(instruction_buffer, instruction, custom_instructions, index_map)
    file_obj.write(struct.pack(CUSTOM_DEFINITION_HEADER_PACK, len(custom_instructions)))

    for name, instruction in custom_instructions.items():
        _write_custom_instruction(file_obj, name, instruction)

    instruction_buffer.seek(0)
    file_obj.write(instruction_buffer.read())
    instruction_buffer.close()


def load(file_obj):
    """Load a QPY binary file

    This function is used to load a serialized QPY circuit file and create
    :class:`~qiskit.circuit.QuantumCircuit` objects from its contents.
    For example:

    .. code-block:: python

        from qiskit.circuit import qpy_serialization

        with open('bell.qpy', 'rb') as fd:
            circuits = qpy_serialization.load(fd)

    or with a gzip compressed file:

    .. code-block:: python

        import gzip
        from qiskit.circuit import qpy_serialization

        with gzip.open('bell.qpy.gz', 'rb') as fd:
            circuits = qpy_serialization.load(fd)

    which will read the contents of the qpy and return a list of
    :class:`~qiskit.circuit.QuantumCircuit` objects from the file.

    Args:
        file_obj (File): A file like object that contains the QPY binary
            data for a circuit
    Returns:
        list: A list of the QuantumCircuit objects from the QPY data. A list
            is always returned, even if there is only 1 circuit in the QPY
            data.
    Raises:
        QiskitError: if ``file_obj`` is not a valid QPY file
    """
    file_header_raw = file_obj.read(FILE_HEADER_SIZE)
    file_header = struct.unpack(FILE_HEADER_PACK, file_header_raw)
    if file_header[0].decode("utf8") != "QISKIT":
        raise QiskitError("Input file is not a valid QPY file")
    version_parts = [int(x) for x in __version__.split(".")[0:3]]
    header_version_parts = [file_header[2], file_header[3], file_header[4]]
    if (
        version_parts[0] < header_version_parts[0]
        or (
            version_parts[0] == header_version_parts[0]
            and header_version_parts[1] > version_parts[1]
        )
        or (
            version_parts[0] == header_version_parts[0]
            and header_version_parts[1] == version_parts[1]
            and header_version_parts[2] > version_parts[2]
        )
    ):
        warnings.warn(
            "The qiskit version used to generate the provided QPY "
            "file, %s, is newer than the current qiskit version %s. "
            "This may result in an error if the QPY file uses "
            "instructions not present in this current qiskit "
            "version" % (".".join([str(x) for x in header_version_parts]), __version__)
        )
    circuits = []
    for _ in range(file_header[5]):
        circuits.append(_read_circuit(file_obj))
    return circuits


def _read_circuit(file_obj):
    header, name, metadata = _read_header(file_obj)
    (
        name_size,  # pylint: disable=unused-variable
        global_phase,
        num_qubits,
        num_clbits,
        metadata_size,  # pylint: disable=unused-variable
        num_registers,
        num_instructions,
    ) = header
    out_registers = {"q": {}, "c": {}}
    if num_registers > 0:
        circ = QuantumCircuit(name=name, global_phase=global_phase, metadata=metadata)
        registers = _read_registers(file_obj, num_registers)

        for bit_type_label, bit_type, reg_type in [
            ("q", Qubit, QuantumRegister),
            ("c", Clbit, ClassicalRegister),
        ]:
            register_bits = set()
            # Add quantum registers and bits
            for register_name in registers[bit_type_label]:
                standalone, indices = registers[bit_type_label][register_name]
                if standalone:
                    start = min(indices)
                    count = start
                    out_of_order = False
                    for index in indices:
                        if not out_of_order and index != count:
                            out_of_order = True
                        count += 1
                        if index in register_bits:
                            raise QiskitError("Duplicate register bits found")
                        register_bits.add(index)

                    num_reg_bits = len(indices)
                    # Create a standlone register of the appropriate length (from
                    # the number of indices in the qpy data) and add it to the circuit
                    reg = reg_type(num_reg_bits, register_name)
                    # If any bits from qreg are out of order in the circuit handle
                    # is case
                    if out_of_order:
                        sorted_indices = np.argsort(indices)
                        for index in sorted_indices:
                            pos = indices[index]
                            if bit_type_label == "q":
                                bit_len = len(circ.qubits)
                            else:
                                bit_len = len(circ.clbits)
                            # Fill any holes between the current register bit and the
                            # next one
                            if pos > bit_len:
                                bits = [bit_type() for _ in range(pos - bit_len)]
                                circ.add_bits(bits)
                            circ.add_bits([reg[index]])
                        circ.add_register(reg)
                    else:
                        if bit_type_label == "q":
                            bit_len = len(circ.qubits)
                        else:
                            bit_len = len(circ.clbits)
                        # If there is a hole between the start of the register and the
                        # current bits and standalone bits to fill the gap.
                        if start > len(circ.qubits):
                            bits = [bit_type() for _ in range(start - bit_len)]
                            circ.add_bits(bit_len)
                        circ.add_register(reg)
                        out_registers[bit_type_label][register_name] = reg
                else:
                    for index in indices:
                        if bit_type_label == "q":
                            bit_len = len(circ.qubits)
                        else:
                            bit_len = len(circ.clbits)
                        # Add any missing bits
                        bits = [bit_type() for _ in range(index + 1 - bit_len)]
                        circ.add_bits(bits)
                        if index in register_bits:
                            raise QiskitError("Duplicate register bits found")
                        register_bits.add(index)
                    if bit_type_label == "q":
                        bits = [circ.qubits[i] for i in indices]
                    else:
                        bits = [circ.clbits[i] for i in indices]
                    reg = reg_type(name=register_name, bits=bits)
                    circ.add_register(reg)
                    out_registers[bit_type_label][register_name] = reg
        # If we don't have sufficient bits in the circuit after adding
        # all the registers add more bits to fill the circuit
        if len(circ.qubits) < num_qubits:
            qubits = [Qubit() for _ in range(num_qubits - len(circ.qubits))]
            circ.add_bits(qubits)
        if len(circ.clbits) < num_clbits:
            clbits = [Clbit() for _ in range(num_qubits - len(circ.clbits))]
            circ.add_bits(clbits)
    else:
        circ = QuantumCircuit(
            num_qubits,
            num_clbits,
            name=name,
            global_phase=global_phase,
            metadata=metadata,
        )
    custom_instructions = _read_custom_instructions(file_obj)
    for _instruction in range(num_instructions):
        _read_instruction(file_obj, circ, out_registers, custom_instructions)

    return circ
