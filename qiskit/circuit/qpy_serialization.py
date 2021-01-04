# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name,too-many-boolean-expressions

"""QPY serialization module


The QPY serialization format is a a portable cross-platform binary
serialization format for QuantumCircuit objects in Qiskit. The basic
format for the file format is as follows:

HEADER | METADATA | REGISTERS | INSTRUCTIONS

All values use network byte order [1] (big endian) for cross platform
compatibility. All strings will be padded with ``0x00`` after the string
if the string is shorter than the max size.

The contents of HEADER as defined as a C struct are:

.. code-block:: c

    struct {
        unsigned char qpy_version;
        unsigned char qiskit_major_version;
        unsigned char qiskit_minor_version;
        unsigned char qiskit_patch_version;
        char name[64];
        double global_phase;
        unsigned int num_qubits;
        unsigned int num_clbits;
        unsigned long long metadata_size;
        unsigned int num_registers;
        unsigned long long num_instructions;
    }

The METADATA field is a utf8 encoded json string. After reading the HEADER
(which is a fixed size at the start of the QPY file you then read the
``metadata_size`` number of bytes and parse the JSON to get the metadata for
the circuit.

The contents of REGISTERS is a number of REGISTER object. If num_registers is
> 0 then after reading METADATA you read that number of REGISTER structs defined
as:

.. code-block:: c

    struct {
        char type;
        unsigned int size;
        char name[10];
    }

``type`` can be ``'q'`` or ``'c'``.

The contents of INSTRUCTIONS is a list of INSTRUCTION metadata objects

.. code-block:: c
    struct {
        char name[256];
        unsigned short num_parameters;
        unsigned int num_qargs;
        unsigned int num_cargs;
    }

``name`` here is the Qiskit class name for the Instruction class if it's
defined in Qiskit. Otherwise it falls back to the custom instruction name.


which is immediately followed by the INSTRUCTION_ARG structs for the list of
arguments of that instruction. These are in the order of all quantum arguments
(there are num_qargs of these) followed by all classical arguments (num_cargs
of these).

The contents of each INSTRUCTION_ARG is:

.. code-block:: c

    struct {
        char type;
        unisgned int size;
        char name[10];
    }

``type`` can be ``'q'`` or ``'c'``.

After all arguments for an instruction the parameters are specified with
``num_parameters`` INSTRUCTION_PARAM structs.

The contents of each INSTRUCTION_PARAM is:

.. code-block:: c

    struct {
        char type;
        unsigned long long size;
    }

After each INSTRUCTIOn_PARAM the next ``size`` bytes are the parameter's data.
The ``type`` field can be ``'i'``, ``'f'``, ``'p'``, or ``'n'`` which dictate
the format. For ``'i'`` it's an integer, ``'f'`` it's a double, ``'p'`` defines
a paramter expression which is represented by a 256 character utf8 string, and
``'n'`` represents an object from numpy (either an ``ndarray`` or a numpy type)
which means the data is .npy format [1] data.

[1] https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
"""
from collections import namedtuple
import io
import json
import struct
import warnings

import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit import library
from qiskit import circuit as circuit_mod
from qiskit import extensions
from qiskit.extensions import quantum_initializer
from qiskit.version import __version__

# v1 Binary Format
# ----------------
# HEADER binary format
HEADER = namedtuple('HEADER', ['qpy_version', 'major_version', 'minor_version',
                               'patch_version', 'name', 'global_phase',
                               'num_qubits', 'num_clbits', 'metadata_size',
                               'num_registers', 'num_instructions'])
HEADER_PACK = '!BBBB64sdIIQIQ'
HEADER_SIZE = struct.calcsize(HEADER_PACK)

# REGISTER binary format
REGISTER = namedtuple('REGISTER', ['type', 'size', 'name'])
REGISTER_PACK = '!1cI10s'
REGISTER_SIZE = struct.calcsize(REGISTER_PACK)

# INSTRUCTION binary format
INSTRUCTION = namedtuple('INSTRUCTION', ['name', 'num_parameters', 'num_qargs',
                                         'num_cargs'])
INSTRUCTION_PACK = '!256sHII'
INSTRUCTION_SIZE = struct.calcsize(INSTRUCTION_PACK)
# Instruction argument format
INSTRUCTION_ARG = namedtuple('INSTRUCTION_ARG', ['type', 'size', 'name'])
INSTRUCTION_ARG_PACK = '!1cI10s'
INSTRUCTION_ARG_SIZE = struct.calcsize(INSTRUCTION_ARG_PACK)
# INSTRUCTION parameter format
INSTRUCTION_PARAM = namedtuple('INSTRUCTION_PARAM', ['type', 'size'])
INSTRUCTION_PARAM_PACK = '!1cQ'
INSTRUCTION_PARAM_SIZE = struct.calcsize(INSTRUCTION_PARAM_PACK)


def _read_header(file_obj):
    header_raw = struct.unpack(HEADER_PACK, file_obj.read(HEADER_SIZE))
    header = HEADER._make(header_raw)
    metadata_raw = file_obj.read(header[8])
    metadata = json.loads(metadata_raw)
    return header, metadata


def _read_registers(file_obj, num_registers):
    registers = {'q': {}, 'c': {}}
    for _reg in range(num_registers):
        register_raw = file_obj.read(REGISTER_SIZE)
        register = struct.unpack(REGISTER_PACK, register_raw)
        name = register[2].decode('utf8').rstrip('\x00')
        if register[0].decode('utf8') == 'q':
            registers['q'][name] = QuantumRegister(register[1], name)
        else:
            registers['c'][name] = ClassicalRegister(register[1], name)
    return registers


def _read_instruction(file_obj, circuit, registers):
    instruction_raw = file_obj.read(INSTRUCTION_SIZE)
    instruction = struct.unpack(INSTRUCTION_PACK, instruction_raw)
    qargs = []
    cargs = []
    params = []
    gate_name = instruction[0].decode('utf8').rstrip('\x00')
    num_qargs = instruction[2]
    num_cargs = instruction[3]
    num_params = instruction[1]
    # Load Arguments
    for _qarg in range(num_qargs):
        qarg_raw = file_obj.read(INSTRUCTION_ARG_SIZE)
        qarg = struct.unpack(INSTRUCTION_ARG_PACK, qarg_raw)
        if qarg[0].decode('utf8').rstrip('\x00') == 'c':
            raise TypeError('Invalid input carg prior to all qargs')
        name = qarg[2].decode('utf8').rstrip('\x00')
        qargs.append(registers['q'][name][qarg[1]])
    for _carg in range(num_cargs):
        carg_raw = file_obj.read(INSTRUCTION_ARG_SIZE)
        carg = struct.unpack(INSTRUCTION_ARG_PACK, carg_raw)
        if carg[0].decode('utf8').rstrip('\x00') == 'q':
            raise TypeError('Invalid input qarg after all qargs')
        name = carg[2].decode('utf8').rstrip('\x00')
        cargs.append(registers['c'][name][carg[1]])
    # Load Parameters
    for _param in range(num_params):
        param_raw = file_obj.read(INSTRUCTION_PARAM_SIZE)
        param = struct.unpack(INSTRUCTION_PARAM_PACK, param_raw)
        data = file_obj.read(param[1])
        type_str = param[0].decode('utf8')
        param = None
        if type_str == 'i':
            param = struct.unpack('<q', data)
        elif type_str == 'f':
            param = struct.unpack('<d', param)
        elif type_str == 'n':
            container = io.BytesIO()
            container.write(data)
            container.seek(0)
            param = np.load(container)
        else:
            raise TypeError("Invalid parameter type: %s" % type_str)
        params.append(param)
    # Load Gate object
    gate_class = None
    if hasattr(library, gate_name):
        gate_class = getattr(library, gate_name)
    elif hasattr(circuit_mod, gate_name):
        gate_class = getattr(circuit_mod, gate_name)
    elif hasattr(extensions, gate_name):
        gate_class = getattr(extensions, gate_name)
    elif hasattr(quantum_initializer, gate_name):
        gate_class = getattr(quantum_initializer, gate_name)
    # TODO: Handle custom gates and custom instructions
    else:
        raise AttributeError("Invalid gate type: %s" % gate_name)
    if gate_name == 'Barrier':
        params = [len(qargs)]
    gate = gate_class(*params)
    circuit.append(gate, qargs, cargs)


def _write_instruction(file_obj, instruction_tuple):
    gate_class_name = instruction_tuple[0].__class__.__name__.encode('utf8')
    instruction_raw = struct.pack(INSTRUCTION_PACK, gate_class_name,
                                  len(instruction_tuple[0].params),
                                  instruction_tuple[0].num_qubits,
                                  instruction_tuple[0].num_clbits)
    file_obj.write(instruction_raw)
    # Encode instruciton args
    for qbit in instruction_tuple[1]:
        instruction_arg_raw = struct.pack(INSTRUCTION_ARG_PACK, 'q'.encode('utf8'),
                                          qbit.index, qbit.register.name.encode('utf8'))
        file_obj.write(instruction_arg_raw)
    for clbit in instruction_tuple[2]:
        instruction_arg_raw = struct.pack(INSTRUCTION_ARG_PACK, 'c'.encode('utf8'),
                                          clbit.index, clbit.register.name.encode('utf8'))
        file_obj.write(instruction_arg_raw)
    # Encode instruction params
    for param in instruction_tuple[0].params:
        container = io.BytesIO()
        if isinstance(param, int):
            type_key = 'i'
            data = struct.pack('<q', param)
            size = struct.calcsize('<q')
        elif isinstance(param, float):
            type_key = 'f'
            data = struct.pack('<d', param)
            size = struct.calcsize('<d')
        elif isinstance(param, ParameterExpression):
            # TODO handle this
            pass
        elif isinstance(np.integer, np.floating, np.ndarray):
            type_key = 'n'
            np.savez(container, param)
            container.seek(0)
            data = container.read()
            size = len(data)
        else:
            raise TypeError("Invalid parameter type %s for gate %s," % (
                instruction_tuple[0], type(param)))
        instruction_param_raw = struct.pack(INSTRUCTION_PARAM_PACK,
                                            type_key.encode('utf8'), size)
        file_obj.write(instruction_param_raw)
        file_obj.write(data)
        container.close()


def dump(file_obj, circuit):
    """Write QPY binary data to a file

    Args:
        file_obj (file): The file like object to write the QPY data too
        circuit (QuantumCircuit): The quantum circuit object to store
    """
    version_parts = [int(x) for x in __version__.split('.')[0:3]]
    metadata_raw = json.dumps(circuit.metadata).encode('utf8')
    metadata_size = len(metadata_raw)
    num_registers = len(circuit.qregs) + len(circuit.cregs)
    num_instructions = len(circuit)
    header = struct.pack(HEADER_PACK, 1, version_parts[0], version_parts[1],
                         version_parts[2], circuit.name.encode('utf8'),
                         circuit.global_phase, circuit.num_qubits,
                         circuit.num_clbits, metadata_size, num_registers,
                         num_instructions)
    file_obj.write(header)
    file_obj.write(metadata_raw)
    if num_registers > 0:
        for reg in circuit.qregs:
            file_obj.write(struct.pack(REGISTER_PACK, 'q'.encode('utf8'),
                                       reg.size, reg.name.encode('utf8')))
        for reg in circuit.cregs:
            file_obj.write(struct.pack(REGISTER_PACK, 'c'.encode('utf8'),
                                       reg.size, reg.name.encode('utf8')))
    for instruction in circuit.data:
        _write_instruction(file_obj, instruction)


def load(file_obj):
    """Load a QPY binary file

    Args:
        file_obj (File): A file like object that contains the QPY binary
            data for a circuit
    Returns:
        QuantumCircuit: The QuantumCircuit from the QPY data
    """
    header, metadata = _read_header(file_obj)
    version_parts = [int(x) for x in __version__.split('.')[0:3]]
    header_version_parts = [header[1], header[2], header[3]]
    if version_parts[0] < header_version_parts[0] or (
            version_parts[0] == header_version_parts[0] and
            header_version_parts[1] > version_parts[1]) or (
                version_parts[0] == header_version_parts[0] and
                header_version_parts[1] == version_parts[1] and
                header_version_parts[2] > version_parts[2]):
        warnings.warn('The qiskit version used to generate the provided QPY '
                      'file, %s, is newer than the current qiskit version %s. '
                      'This may result in an error if the QPY file uses '
                      'instructions not present in this current qiskit version' % (
                          '.'.join(header_version_parts), __version__))
    registers = {}
    if header[8] > 0:
        circ = QuantumCircuit(name=header[4].decode('utf8').rstrip('\x00'),
                              global_phase=header[5],
                              metadata=metadata)
        registers = _read_registers(file_obj, header[9])
        for qreg in registers['q'].values():
            circ.add_register(qreg)
        for creg in registers['c'].values():
            circ.add_register(creg)
    else:
        circ = QuantumCircuit(header[5], header[6],
                              name=header[4].decode('utf8').rstrip('\x00'),
                              global_phase=header[5],
                              metadata=metadata)
    for _instruction in range(header[10]):
        _read_instruction(file_obj, circ, registers)

    return circ
