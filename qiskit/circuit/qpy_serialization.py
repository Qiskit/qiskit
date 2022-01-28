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

# pylint: disable=invalid-name,too-many-boolean-expressions

"""
###########################################################
QPY serialization (:mod:`qiskit.circuit.qpy_serialization`)
###########################################################

.. currentmodule:: qiskit.circuit.qpy_serialization

*********
Using QPY
*********

Using QPY is defined to be straightforward and mirror the user API of the
serializers in Python's standard library, ``pickle`` and ``json``. There are
2 user facing functions: :func:`qiskit.circuit.qpy_serialization.dump` and
:func:`qiskit.circuit.qpy_serialization.load` which are used to dump QPY data
to a file object and load circuits from QPY data in a file object respectively.
For example::

    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit import qpy_serialization

    qc = QuantumCircuit(2, name='Bell', metadata={'test': True})
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    with open('bell.qpy', 'wb') as fd:
        qpy_serialization.dump(qc, fd)

    with open('bell.qpy', 'rb') as fd:
        new_qc = qpy_serialization.load(fd)[0]

API documentation
=================

.. autosummary::
   :toctree: ../stubs/

   load
   dump

QPY Compatibility
=================

The QPY format is designed to be backwards compatible moving forward. This means
you should be able to load a QPY with any newer Qiskit version than the one
that generated it. However, loading a QPY file with an older Qiskit version is
not supported and may not work.

For example, if you generated a QPY file using qiskit-terra 0.18.1 you could
load that QPY file with qiskit-terra 0.19.0 and a hypothetical qiskit-terra
0.29.0. However, loading that QPY file with 0.18.0 is not supported and may not
work.

**********
QPY Format
**********

The QPY serialization format is a portable cross-platform binary
serialization format for :class:`~qiskit.circuit.QuantumCircuit` objects in Qiskit. The basic
file format is as follows:

A QPY file (or memory object) always starts with the following 7
byte UTF8 string: ``QISKIT`` which is immediately followed by the overall
file header. The contents of the file header as defined as a C struct are:

.. code-block:: c

    struct {
        uint8_t qpy_version;
        uint8_t qiskit_major_version;
        uint8_t qiskit_minor_version;
        uint8_t qiskit_patch_version;
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

.. _version_3:

Version 3
=========

Version 3 of the QPY format is identical to :ref:`version_2` except that it defines
a struct format to represent a :class:`~qiskit.circuit.library.PauliEvolutionGate`
natively in QPY. To accomplish this the :ref:`custom_definition` struct now supports
a new type value ``'p'`` to represent a :class:`~qiskit.circuit.library.PauliEvolutionGate`.
Enties in the custom instructions tables have unique name generated that start with the
string ``"###PauliEvolutionGate_"`` followed by a uuid string. This gate name is reservered
in QPY and if you have a custom :class:`~qiskit.circuit.Instruction` object with a definition
set and that name prefix it will error. If it's of type ``'p'`` the data payload is defined
as follows:

.. _pauli_evo_qpy:

PAULI_EVOLUTION
---------------

This represents the high level :class:`~qiskit.circuit.library.PauliEvolutionGate`

.. code-block:: c

    struct {
        uint64_t operator_count;
        _Bool standalone_op;
        char time_type;
        uint64_t time_size;
        uint64_t synthesis_size;
    }

This is immediately followed by ``operator_count`` elements defined by the :ref:`pauli_sum_op`
payload.  Following that we have ``time_size`` bytes representing the ``time`` attribute. If
``standalone_op`` is ``True`` then there must only be a single operator. The
encoding of these bytes is determined by the value of ``time_type``. Possible values of
``time_type`` are ``'f'``, ``'p'``, and ``'e'``. If ``time_type`` is ``'f'`` it's a double,
``'p'`` defines a :class:`~qiskit.circuit.Parameter` object  which is represented by a
:ref:`param_struct`, ``e`` defines a :class:`~qiskit.circuit.ParameterExpression` object
(that's not a :class:`~qiskit.circuit.Parameter`) which is represented by a :ref:`param_expr`.
Following that is ``synthesis_size`` bytes which is a utf8 encoded json payload representing
the :class:`.EvolutionSynthesis` class used by the gate.

.. _pauli_sum_op:

SPARSE_PAULI_OP_LIST_ELEM
-------------------------

This represents an instance of :class:`.PauliSumOp`.


.. code-block:: c

    struct {
        uint32_t pauli_op_size;
    }

which is immediately followed by ``pauli_op_size`` bytes which are .npy format [#f2]_
data which represents the :class:`~qiskit.quantum_info.SparsePauliOp`.

Version 3 of the QPY format also defines a struct format to represent a
:class:`~qiskit.circuit.ParameterVectorElement` as a distinct subclass from
a :class:`~qiskit.circuit.Parameter`. This adds a new parameter type char ``'v'``
to represent a :class:`~qiskit.circuit.ParameterVectorElement` which is now
supported as a type string value for an INSTRUCTION_PARAM. The payload for these
parameters are defined below as :ref:`param_vector`.

.. _param_vector:


PARAMETER_VECTOR_ELEMENT
------------------------

A PARAMETER_VECTOR_ELEMENT represents a :class:`~qiskit.circuit.ParameterVectorElement`
object the data for a INSTRUCTION_PARAM. The contents of the PARAMETER_VECTOR_ELEMENT are
defined as:

.. code-block:: c

    struct {
        uint16_t vector_name_size;
        uint64_t vector_size;
        char uuid[16];
        uint64_t index;
    }

which is immediately followed by ``vector_name_size`` utf8 bytes representing
the parameter's vector name.

.. _param_expr_v3:


PARAMETER_EXPR
--------------

Additionally, since QPY format version v3 distinguishes between a
:class:`~qiskit.circuit.Parameter` and :class:`~qiskit.circuit.ParameterVectorElement`
the payload for a :class:`~qiskit.circuit.ParameterExpression` needs to be updated
to distinguish between the types. The following is the modified payload format
which is mostly identical to the format in Version 1 and :ref:`version_2` but just
modifies the ``map_elements`` struct to include a symbol type field.

A PARAMETER_EXPR represents a :class:`~qiskit.circuit.ParameterExpression`
object that the data for an INSTRUCTION_PARAM. The contents of a PARAMETER_EXPR
are defined as:

.. code-block:: c

    struct {
        uint64_t map_elements;
        uint64_t expr_size;
    }

Immediately following the header is ``expr_size`` bytes of utf8 data containing
the expression string, which is the sympy srepr of the expression for the
parameter expression. Following that is a symbol map which contains
``map_elements`` elements with the format

.. code-block:: c

    struct {
        char symbol_type;
        char type;
        uint64_t size;
    }

The ``symbol_type`` key determines the payload type of the symbol representation
for the element. If it's ``p`` it represents a :class:`~qiskit.circuit.Parameter`
and if it's ``v`` it represents a :class:`~qiskit.circuit.ParameterVectorElement`.
The map element struct is immediately followed by the symbol map key payload, if
``symbol_type`` is ``p`` then it is followed immediately by a :ref:`param_struct`
object (both the struct and utf8 name bytes) and if ``symbol_type`` is ``v``
then the struct is imediately followed by :ref:`param_vector` (both the struct
and utf8 name bytes). That is followed by ``size`` bytes for the
data of the symbol. The data format is dependent on the value of ``type``. If
``type`` is ``p`` then it represents a :class:`~qiskit.circuit.Parameter` and
size will be 0, the value will just be the same as the key. Similarly if the
``type`` is ``v`` then it represents a :class:`~qiskit.circuit.ParameterVectorElement`
and size will be 0 as the value will just be the same as the key. If
``type`` is ``f`` then it represents a double precision float. If ``type`` is
``c`` it represents a double precision complex, which is represented by the
:ref:`complex`. Finally, if type is ``i`` it represents an integer which is an
``int64_t``.

.. _version_2:

Version 2
=========

Version 2 of the QPY format is identical to version 1 except for the HEADER
section is slightly different. You can refer to the :ref:`version_1` section
for the details on the rest of the payload format.

HEADER
------

The contents of HEADER are defined as a C struct are:

.. code-block:: c

    struct {
        uint16_t name_size;
        char global_phase_type;
        uint16_t global_phase_size;
        uint32_t num_qubits;
        uint32_t num_clbits;
        uint64_t metadata_size;
        uint32_t num_registers;
        uint64_t num_instructions;
        uint64_t num_custom_gates;
    }

This is immediately followed by ``name_size`` bytes of utf8 data for the name
of the circuit. Following this is immediately ``global_phase_size`` bytes
representing the global phase. The content of that data is dictated by the
value of ``global_phase_type``. If it's ``'f'`` the data is a float and is the
size of a ``double``. If it's ``'p'`` defines a :class:`~qiskit.circuit.Parameter`
object  which is represented by a PARAM struct (see below), ``e`` defines a
:class:`~qiskit.circuit.ParameterExpression` object (that's not a
:class:`~qiskit.circuit.Parameter`) which is represented by a PARAM_EXPR struct
(see below).

.. _version_1:

Version 1
=========

HEADER
------

The contents of HEADER as defined as a C struct are:

.. code-block:: c

    struct {
        uint16_t name_size;
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
        uint16_t name_size;
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


.. _custom_definition:

CUSTOM_DEFINITIONS
------------------

This section specifies custom definitions for any of the instructions in the circuit.

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
        uint64_t size;
    }

Immediately following the CUSTOM_INSTRUCTION struct is the utf8 encoded name
of size ``name_size``.

If ``custom_definition`` is ``True`` that means that the immediately following
``size`` bytes contains a QPY circuit data which can be used for the custom
definition of that gate. If ``custom_definition`` is ``False`` then the
instruction can be considered opaque (ie no definition). The ``type`` field
determines what type of object will get created with the custom definition.
If it's ``'g'`` it will be a :class:`~qiskit.circuit.Gate` object, ``'i'``
it will be a :class:`~qiskit.circuit.Instruction` object.

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
        _Bool has_conditional;
        uint16_t conditional_reg_name_size;
        int64_t conditional_value;
    }

This metadata object is immediately followed by ``name_size`` bytes of utf8 bytes
for the ``name``. ``name`` here is the Qiskit class name for the Instruction
class if it's defined in Qiskit. Otherwise it falls back to the custom
instruction name. Following the ``name`` bytes there are ``label_size`` bytes of
utf8 data for the label if one was set on the instruction. Following the label
bytes if ``has_conditional`` is ``True`` then there are
``conditional_reg_name_size`` bytes of utf8 data for the name of the conditional
register name. In case of single classical bit conditions the register name
utf8 data will be prefixed with a null character "\\x00" and then a utf8 string
integer representing the classical bit index in the circuit that the condition
is on.

This is immediately followed by the INSTRUCTION_ARG structs for the list of
arguments of that instruction. These are in the order of all quantum arguments
(there are num_qargs of these) followed by all classical arguments (num_cargs
of these).

The contents of each INSTRUCTION_ARG is:

.. code-block:: c

    struct {
        char type;
        uint32_t index;
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
The ``type`` field can be ``'i'``, ``'f'``, ``'p'``, ``'e'``, ``'s'``, ``'c'``
or ``'n'`` which dictate the format. For ``'i'`` it's an integer, ``'f'`` it's
a double, ``'s'`` if it's a string (encoded as utf8), ``'c'`` is a complex and
the data is represented by the struct format in the :ref:`param_expr` section.
``'p'`` defines a :class:`~qiskit.circuit.Parameter` object  which is
represented by a :ref:`param_struct` struct, ``e`` defines a
:class:`~qiskit.circuit.ParameterExpression` object (that's not a
:class:`~qiskit.circuit.Parameter`) which is represented by a :ref:`param_expr`
struct (on QPY format :ref:`version_3` the format is tweak slightly see:
:ref:`param_expr_v3`), ``'n'`` represents an object from numpy (either an
``ndarray`` or a numpy type) which means the data is .npy format [#f2]_ data,
and in QPY :ref:`version_3` ``'v'`` represents a
:class:`~qiskit.circuit.ParameterVectorElement` which is represented by a
:ref:`param_vector` struct.

.. _param_struct:

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

.. _param_expr:

PARAMETER_EXPR
--------------

A PARAMETER_EXPR represents a :class:`~qiskit.circuit.ParameterExpression`
object that the data for an INSTRUCTION_PARAM. The contents of a PARAMETER_EXPR
are defined as:

The PARAMETER_EXPR data starts with a header:

.. code-block:: c

    struct {
        uint64_t map_elements;
        uint64_t expr_size;
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
``c`` it represents a double precision complex, which is represented by :ref:`complex`.
Finally, if type is ``i`` it represents an integer which is an ``int64_t``.

.. _complex:

COMPLEX
-------

When representing a double precision complex value in QPY the following
struct is used:


.. code-block:: c

    struct {
        double real;
        double imag;
    }

this matches the internal C representation of Python's complex type. [#f3]_


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
from qiskit.circuit.parametervector import ParameterVector, ParameterVectorElement
from qiskit.circuit.gate import Gate
from qiskit.circuit.instruction import Instruction
from qiskit.circuit import library
from qiskit import circuit as circuit_mod
from qiskit import extensions
from qiskit.extensions import quantum_initializer
from qiskit.version import __version__
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.synthesis import evolution as evo_synth

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
HEADER_V2 = namedtuple(
    "HEADER",
    [
        "name_size",
        "global_phase_type",
        "global_phase_size",
        "num_qubits",
        "num_clbits",
        "metadata_size",
        "num_registers",
        "num_instructions",
    ],
)
HEADER_V2_PACK = "!H1cHIIQIQ"
HEADER_V2_SIZE = struct.calcsize(HEADER_V2_PACK)

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
# PARAMETER_EXPR_MAP_ELEM_V3
PARAM_EXPR_MAP_ELEM_V3 = namedtuple("PARAMETER_EXPR_MAP_ELEM", ["symbol_type", "type", "size"])
PARAM_EXPR_MAP_ELEM_PACK_V3 = "!ccQ"
PARAM_EXPR_MAP_ELEM_SIZE_V3 = struct.calcsize(PARAM_EXPR_MAP_ELEM_PACK_V3)
# Complex
COMPLEX = namedtuple("COMPLEX", ["real", "imag"])
COMPLEX_PACK = "!dd"
COMPLEX_SIZE = struct.calcsize(COMPLEX_PACK)
# PARAMETER_VECTOR_ELEMENT
PARAMETER_VECTOR_ELEMENT = namedtuple(
    "PARAMETER_VECTOR_ELEMENT", ["vector_name_size", "vector_size", "uuid", "index"]
)
PARAMETER_VECTOR_ELEMENT_PACK = "!HQ16sQ"
PARAMETER_VECTOR_ELEMENT_SIZE = struct.calcsize(PARAMETER_VECTOR_ELEMENT_PACK)
# Pauli Evolution Gate
PAULI_EVOLUTION_DEF = namedtuple(
    "PAULI_EVOLUTION_DEF",
    ["operator_size", "standalone_op", "time_type", "time_size", "synth_method_size"],
)
PAULI_EVOLUTION_DEF_PACK = "!Q?1cQQ"
PAULI_EVOLUTION_DEF_SIZE = struct.calcsize(PAULI_EVOLUTION_DEF_PACK)
# SparsePauliOp List
SPARSE_PAULI_OP_LIST_ELEM = namedtuple("SPARSE_PAULI_OP_LIST_ELEMENT", ["size"])
SPARSE_PAULI_OP_LIST_ELEM_PACK = "!Q"
SPARSE_PAULI_OP_LIST_ELEM_SIZE = struct.calcsize(SPARSE_PAULI_OP_LIST_ELEM_PACK)


def _read_header_v2(file_obj, version, vectors):
    header_raw = struct.unpack(HEADER_V2_PACK, file_obj.read(HEADER_V2_SIZE))
    header_tuple = HEADER_V2._make(header_raw)
    name = file_obj.read(header_tuple[0]).decode("utf8")
    global_phase_type_str = header_tuple[1].decode("utf8")
    data = file_obj.read(header_tuple[2])
    if global_phase_type_str == "f":
        global_phase = struct.unpack("!d", data)[0]
    elif global_phase_type_str == "i":
        global_phase = struct.unpack("!q", data)[0]
    elif global_phase_type_str == "p":
        with io.BytesIO(data) as container:
            global_phase = _read_parameter(container)
    elif global_phase_type_str == "v":
        with io.BytesIO(data) as container:
            global_phase = _read_parameter_vec(container, vectors)
    elif global_phase_type_str == "e":
        with io.BytesIO(data) as container:
            if version < 3:
                global_phase = _read_parameter_expression(container)
            else:
                global_phase = _read_parameter_expression_v3(container, vectors)
    else:
        raise TypeError("Invalid global phase type: %s" % global_phase_type_str)
    header = {
        "global_phase": global_phase,
        "num_qubits": header_tuple[3],
        "num_clbits": header_tuple[4],
        "num_registers": header_tuple[6],
        "num_instructions": header_tuple[7],
    }
    metadata_raw = file_obj.read(header_tuple[5])
    metadata = json.loads(metadata_raw)
    return header, name, metadata


def _read_header(file_obj):
    header_raw = struct.unpack(HEADER_PACK, file_obj.read(HEADER_SIZE))
    header_tuple = HEADER._make(header_raw)
    name = file_obj.read(header_tuple[0]).decode("utf8")
    header = {
        "global_phase": header_tuple[1],
        "num_qubits": header_tuple[2],
        "num_clbits": header_tuple[3],
        "num_registers": header_tuple[5],
        "num_instructions": header_tuple[6],
    }
    metadata_raw = file_obj.read(header_tuple[4])
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


def _read_parameter_vec(file_obj, vectors):
    param_raw = struct.unpack(
        PARAMETER_VECTOR_ELEMENT_PACK, file_obj.read(PARAMETER_VECTOR_ELEMENT_SIZE)
    )
    vec_name_size = param_raw[0]
    param_uuid = uuid.UUID(bytes=param_raw[2])
    param_index = param_raw[3]
    name = file_obj.read(vec_name_size).decode("utf8")
    if name not in vectors:
        vectors[name] = (ParameterVector(name, param_raw[1]), set())
    vector = vectors[name][0]
    if vector[param_index]._uuid != param_uuid:
        vectors[name][1].add(param_index)
        vector._params[param_index] = ParameterVectorElement.__new__(
            ParameterVectorElement, vector, param_index, uuid=param_uuid
        )
        vector._params[param_index].__init__(vector, param_index)
    return vector[param_index]


def _read_parameter_expression_v3(file_obj, vectors):
    param_expr_raw = struct.unpack(PARAMETER_EXPR_PACK, file_obj.read(PARAMETER_EXPR_SIZE))
    map_elements = param_expr_raw[0]
    from sympy.parsing.sympy_parser import parse_expr

    if HAS_SYMENGINE:
        expr = symengine.sympify(parse_expr(file_obj.read(param_expr_raw[1]).decode("utf8")))
    else:
        expr = parse_expr(file_obj.read(param_expr_raw[1]).decode("utf8"))
    symbol_map = {}
    for _ in range(map_elements):
        elem_raw = file_obj.read(PARAM_EXPR_MAP_ELEM_SIZE_V3)
        elem = struct.unpack(PARAM_EXPR_MAP_ELEM_PACK_V3, elem_raw)
        symbol_type = elem[0].decode("utf8")
        if symbol_type == "p":
            param = _read_parameter(file_obj)
        elif symbol_type == "v":
            param = _read_parameter_vec(file_obj, vectors)
        elem_type = elem[1].decode("utf8")
        elem_data = file_obj.read(elem[2])
        if elem_type == "f":
            value = struct.unpack("!d", elem_data)
        elif elem_type == "i":
            value = struct.unpack("!q", elem_data)
        elif elem_type == "c":
            value = complex(*struct.unpack(COMPLEX_PACK, elem_data))
        elif elem_type in ("p", "v"):
            value = param._symbol_expr
        elif elem_type == "e":
            value = _read_parameter_expression_v3(io.BytesIO(elem_data), vectors)
        else:
            raise TypeError("Invalid parameter expression map type: %s" % elem_type)
        symbol_map[param] = value
    return ParameterExpression(symbol_map, expr)


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


def _read_instruction(file_obj, circuit, registers, custom_instructions, version, vectors):
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
        # If an invalid register name is used assume it's a single bit
        # condition and treat the register name as a string of the clbit index
        if ClassicalRegister.name_format.match(condition_register) is None:
            # If invalid register prefixed with null character it's a clbit
            # index for single bit condition
            if condition_register[0] == "\x00":
                conditional_bit = int(condition_register[1:])
                condition_tuple = (circuit.clbits[conditional_bit], condition_value)
            else:
                raise ValueError(
                    f"Invalid register name: {condition_register} for condition register of "
                    f"instruction: {gate_name}"
                )
        else:
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
        elif type_str == "c":
            param = complex(*struct.unpack(COMPLEX_PACK, data))
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
            if version < 3:
                param = _read_parameter_expression(container)
            else:
                param = _read_parameter_expression_v3(container, vectors)
        elif type_str == "v":
            container = io.BytesIO(data)
            param = _read_parameter_vec(container, vectors)
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
    elif gate_name in custom_instructions:
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
    else:
        raise AttributeError("Invalid instruction type: %s" % gate_name)
    if gate_name == "Initialize":
        gate = gate_class(params)
    else:
        if gate_name == "Barrier":
            params = [len(qargs)]
        gate = gate_class(*params)
    gate.condition = condition_tuple
    if label_size > 0:
        gate.label = label
    if not isinstance(gate, Instruction):
        circuit.append(gate, qargs, cargs)
    else:
        circuit._append(gate, qargs, cargs)


def _parse_custom_instruction(custom_instructions, gate_name, params):
    (type_str, num_qubits, num_clbits, definition) = custom_instructions[gate_name]
    if type_str == "i":
        inst_obj = Instruction(gate_name, num_qubits, num_clbits, params)
        if definition is not None:
            inst_obj.definition = definition
    elif type_str == "g":
        inst_obj = Gate(gate_name, num_qubits, params)
        inst_obj.definition = definition
    elif type_str == "p":
        inst_obj = definition
    else:
        raise ValueError("Invalid custom instruction type '%s'" % type_str)
    return inst_obj


def _read_custom_instructions(file_obj, version, vectors):
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
                if version < 3 or not name.startswith(r"###PauliEvolutionGate_"):
                    definition_circuit = _read_circuit(definition_buffer, version)
                elif name.startswith(r"###PauliEvolutionGate_"):
                    definition_circuit = _read_pauli_evolution_gate(definition_buffer, vectors)
            custom_instructions[name] = (type_str, num_qubits, num_clbits, definition_circuit)
    return custom_instructions


def _write_parameter(file_obj, param):
    name_bytes = param._name.encode("utf8")
    file_obj.write(struct.pack(PARAMETER_PACK, len(name_bytes), param._uuid.bytes))
    file_obj.write(name_bytes)


def _write_parameter_vec(file_obj, param):
    name_bytes = param._vector._name.encode("utf8")
    file_obj.write(
        struct.pack(
            PARAMETER_VECTOR_ELEMENT_PACK,
            len(name_bytes),
            param._vector._size,
            param._uuid.bytes,
            param._index,
        )
    )
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
        with io.BytesIO() as parameter_container:
            if isinstance(parameter, ParameterVectorElement):
                symbol_type_str = "v"
                _write_parameter_vec(parameter_container, parameter)
            else:
                symbol_type_str = "p"
                _write_parameter(parameter_container, parameter)
            parameter_data = parameter_container.getvalue()
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
            type_str = symbol_type_str
            data = bytes()
        elif isinstance(value, ParameterExpression):
            type_str = "e"
            container = io.BytesIO()
            _write_parameter_expression(container, value)
            container.seek(0)
            data = container.read()
        else:
            raise TypeError(f"Invalid expression type in symbol map for {param}: {type(value)}")

        elem_header = struct.pack(
            PARAM_EXPR_MAP_ELEM_PACK_V3,
            symbol_type_str.encode("utf8"),
            type_str.encode("utf8"),
            len(data),
        )
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
        or isinstance(instruction_tuple[0], library.BlueprintCircuit)
    ):
        if instruction_tuple[0].name not in custom_instructions:
            custom_instructions[instruction_tuple[0].name] = instruction_tuple[0]
        gate_class_name = instruction_tuple[0].name

    elif isinstance(instruction_tuple[0], library.PauliEvolutionGate):
        gate_class_name = r"###PauliEvolutionGate_" + str(uuid.uuid4())
        custom_instructions[gate_class_name] = instruction_tuple[0]

    has_condition = False
    condition_register = b""
    condition_value = 0
    if instruction_tuple[0].condition:
        has_condition = True
        if isinstance(instruction_tuple[0].condition[0], Clbit):
            bit_index = index_map["c"][instruction_tuple[0].condition[0]]
            condition_register = b"\x00" + str(bit_index).encode("utf8")
            condition_value = int(instruction_tuple[0].condition[1])
        else:
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
        elif isinstance(param, ParameterVectorElement):
            type_key = "v"
            _write_parameter_vec(container, param)
            container.seek(0)
            data = container.read()
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
        elif isinstance(param, complex):
            type_key = "c"
            data = struct.pack(COMPLEX_PACK, param.real, param.imag)
            size = struct.calcsize(COMPLEX_PACK)
        elif isinstance(param, (np.integer, np.floating, np.ndarray, np.complexfloating)):
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


def _write_pauli_evolution_gate(file_obj, evolution_gate):
    operator_list = evolution_gate.operator
    standalone = False
    if not isinstance(operator_list, list):
        operator_list = [operator_list]
        standalone = True
    num_operators = len(operator_list)
    pauli_data_buf = io.BytesIO()
    for operator in operator_list:
        with io.BytesIO() as element_buf:
            with io.BytesIO() as buf:
                pauli_list = operator.to_list(array=True)
                np.save(buf, pauli_list)
                data = buf.getvalue()
            element_metadata = struct.pack(SPARSE_PAULI_OP_LIST_ELEM_PACK, len(data))
            element_buf.write(element_metadata)
            element_buf.write(data)
            pauli_data_buf.write(element_buf.getvalue())
    time = evolution_gate.time
    if isinstance(time, float):
        time_type = b"f"
        time_data = struct.pack("!d", time)
        time_size = struct.calcsize("!d")
    elif isinstance(time, ParameterVectorElement):
        time_type = b"v"
        with io.BytesIO() as buf:
            _write_parameter_vec(buf, time)
            time_data = buf.getvalue()
            time_size = len(time_data)
    elif isinstance(time, Parameter):
        time_type = b"p"
        with io.BytesIO() as buf:
            _write_parameter(buf, time)
            time_data = buf.getvalue()
            time_size = len(time_data)
    elif isinstance(time, ParameterExpression):
        time_type = b"e"
        with io.BytesIO() as buf:
            _write_parameter_expression(buf, time)
            time_data = buf.getvalue()
            time_size = len(time_data)
    else:
        raise TypeError(f"Invalid time type {time} for PauliEvolutionGate")

    synth_class = str(type(evolution_gate.synthesis).__name__)
    settings_dict = evolution_gate.synthesis.settings
    synth_data = json.dumps({"class": synth_class, "settings": settings_dict}).encode("utf8")
    synth_size = len(synth_data)
    pauli_evolution_raw = struct.pack(
        PAULI_EVOLUTION_DEF_PACK, num_operators, standalone, time_type, time_size, synth_size
    )
    file_obj.write(pauli_evolution_raw)
    file_obj.write(pauli_data_buf.getvalue())
    pauli_data_buf.close()
    file_obj.write(time_data)
    file_obj.write(synth_data)


def _read_pauli_evolution_gate(file_obj, vectors):
    pauli_evolution_raw = struct.unpack(
        PAULI_EVOLUTION_DEF_PACK, file_obj.read(PAULI_EVOLUTION_DEF_SIZE)
    )
    if pauli_evolution_raw[0] != 1 and pauli_evolution_raw[1]:
        raise ValueError(
            "Can't have a standalone operator with {pauli_evolution_raw[0]} operators in the payload"
        )
    operator_list = []
    for _ in range(pauli_evolution_raw[0]):
        op_size = struct.unpack(
            SPARSE_PAULI_OP_LIST_ELEM_PACK, file_obj.read(SPARSE_PAULI_OP_LIST_ELEM_SIZE)
        )[0]
        operator_list.append(SparsePauliOp.from_list(np.load(io.BytesIO(file_obj.read(op_size)))))
    if pauli_evolution_raw[1]:
        pauli_op = operator_list[0]
    else:
        pauli_op = operator_list

    time_type = pauli_evolution_raw[2]
    time_data = file_obj.read(pauli_evolution_raw[3])
    if time_type == b"f":
        time = struct.unpack("!d", time_data)[0]
    elif time_type == b"p":
        with io.BytesIO(time_data) as buf:
            time = _read_parameter(buf)
    elif time_type == b"e":
        with io.BytesIO(time_data) as buf:
            time = _read_parameter_expression_v3(buf, vectors)
    elif time_type == b"v":
        with io.BytesIO(time_data) as buf:
            time = _read_parameter_vec(buf, vectors)
    synth_data = json.loads(file_obj.read(pauli_evolution_raw[4]))
    synthesis = getattr(evo_synth, synth_data["class"])(**synth_data["settings"])
    return_gate = library.PauliEvolutionGate(pauli_op, time=time, synthesis=synthesis)
    return return_gate


def _write_custom_instruction(file_obj, name, instruction):
    if isinstance(instruction, library.PauliEvolutionGate):
        type_str = b"p"
    elif isinstance(instruction, Gate):
        type_str = b"g"
    else:
        type_str = b"i"
    has_definition = False
    size = 0
    data = None
    num_qubits = instruction.num_qubits
    num_clbits = instruction.num_clbits
    if instruction.definition is not None or type_str == b"p":
        has_definition = True
        definition_buffer = io.BytesIO()
        if type_str == b"p":
            _write_pauli_evolution_gate(definition_buffer, instruction)
        else:
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
        3,
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
    if isinstance(circuit.global_phase, float):
        global_phase_type = b"f"
        global_phase_data = struct.pack("!d", circuit.global_phase)
    elif isinstance(circuit.global_phase, int):
        global_phase_type = b"i"
        global_phase_data = struct.pack("!q", circuit.global_phase)
    elif isinstance(circuit.global_phase, ParameterVectorElement):
        global_phase_type = b"v"
        with io.BytesIO() as container:
            _write_parameter_vec(container, circuit.global_phase)
            global_phase_data = container.getvalue()
    elif isinstance(circuit.global_phase, Parameter):
        global_phase_type = b"p"
        with io.BytesIO() as container:
            _write_parameter(container, circuit.global_phase)
            global_phase_data = container.getvalue()
    elif isinstance(circuit.global_phase, ParameterExpression):
        global_phase_type = b"e"
        with io.BytesIO() as container:
            _write_parameter_expression(container, circuit.global_phase)
            global_phase_data = container.getvalue()
    else:
        raise TypeError("unsupported global phase type %s" % type(circuit.global_phase))
    header_raw = HEADER_V2(
        name_size=len(circuit_name),
        global_phase_type=global_phase_type,
        global_phase_size=len(global_phase_data),
        num_qubits=circuit.num_qubits,
        num_clbits=circuit.num_clbits,
        metadata_size=metadata_size,
        num_registers=num_registers,
        num_instructions=num_instructions,
    )
    header = struct.pack(HEADER_V2_PACK, *header_raw)
    file_obj.write(header)
    file_obj.write(circuit_name)
    file_obj.write(global_phase_data)
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
        list: List of ``QuantumCircuit``
            The list of :class:`~qiskit.circuit.QuantumCircuit` objects
            contained in the QPY data. A list is always returned, even if there
            is only 1 circuit in the QPY data.
    Raises:
        QiskitError: if ``file_obj`` is not a valid QPY file
    """
    file_header_raw = file_obj.read(FILE_HEADER_SIZE)
    file_header = struct.unpack(FILE_HEADER_PACK, file_header_raw)
    if file_header[0].decode("utf8") != "QISKIT":
        raise QiskitError("Input file is not a valid QPY file")
    qpy_version = file_header[1]
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
        circuits.append(_read_circuit(file_obj, qpy_version))
    return circuits


def _read_circuit(file_obj, version):
    vectors = {}
    if version < 2:
        header, name, metadata = _read_header(file_obj)
    else:
        header, name, metadata = _read_header_v2(file_obj, version, vectors)
    global_phase = header["global_phase"]
    num_qubits = header["num_qubits"]
    num_clbits = header["num_clbits"]
    num_registers = header["num_registers"]
    num_instructions = header["num_instructions"]
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
    custom_instructions = _read_custom_instructions(file_obj, version, vectors)
    for _instruction in range(num_instructions):
        _read_instruction(file_obj, circ, out_registers, custom_instructions, version, vectors)
    for vec_name, (vector, initialized_params) in vectors.items():
        if len(initialized_params) != len(vector):
            warnings.warn(
                f"The ParameterVector: '{vec_name}' is not fully identical to its "
                "pre-serialization state. Elements "
                f"{', '.join([str(x) for x in set(range(len(vector))) - initialized_params])} "
                "in the ParameterVector will be not equal to the pre-serialized ParameterVector "
                f"as they weren't used in the circuit: {circ.name}",
                UserWarning,
            )

    return circ
