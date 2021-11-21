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

from qiskit.pulse import channels, instructions

from qiskit.pulse.channels import Channel
from qiskit.qpy.common import write_binary, read_binary, assign_key, TypeKey
from .parameter_values import dumps_parameter_value, loads_parameter_value


# CHANNEL binary format
CHANNEL = namedtuple("CHANNEL", ["name_size", "index_type", "index_size"])
CHANNEL_PACK = "!H1cH"
CHANNEL_PACK_SIZE = struct.calcsize(CHANNEL_PACK)

# INSTRUCTION binary format
INSTRUCTION = namedtuple("INSTRUCTION", ["name_size", "label_size", "num_operands"])
INSTRUCTION_PACK = "!HHH"
INSTRUCTION_PACK_SIZE = struct.calcsize(INSTRUCTION_PACK)


def _read_channel(file_obj):
    channel_header = struct.unpack(CHANNEL_PACK, file_obj.read(CHANNEL_PACK_SIZE))
    channel_class_name = file_obj.read(channel_header[0]).decode("utf8")
    index_type_key = TypeKey(channel_header[1].decode("utf8"))
    index_binary = file_obj.read(channel_header[2])
    index = loads_parameter_value(index_type_key, index_binary)

    if hasattr(channels, channel_class_name):
        channel_type = getattr(channels, channel_class_name)
        return channel_type(index)

    raise TypeError(f"Invalid class name {channel_class_name} for channel object.")


def _read_instruction(file_obj):
    instruction_header = struct.unpack(INSTRUCTION_PACK, file_obj.read(INSTRUCTION_PACK_SIZE))
    instruction_class_name = file_obj.read(instruction_header[0]).decode("utf8")
    label = file_obj.read(instruction_header[1]).decode("utf8")

    operands = []
    for _ in range(instruction_header[2]):
        type_key, data = read_binary(file_obj)
        if TypeKey.is_number(type_key) or TypeKey.is_variable(type_key):
            value = loads_parameter_value(type_key, data)
        elif type_key == TypeKey.CHANNEL:
            with io.BytesIO(data) as container:
                value = _read_channel(container)
        else:
            raise TypeError(f"Invalid instruction operand type {type_key} for value {data}.")
        operands.append(value)

    if hasattr(instructions, instruction_class_name):
        instruction_type = getattr(instructions, instruction_class_name)
        return instruction_type(*operands, name=label)

    raise TypeError(f"Invalid instruction class name {instruction_class_name}.")


def _write_channel(file_obj, data):
    channel_class_name = data.__class__.__name__.encode("utf8")
    index_type_key = assign_key(data.index)
    index_binany = dumps_parameter_value(index_type_key, data.index)

    channel_header = struct.pack(
        CHANNEL_PACK,
        len(channel_class_name),
        index_type_key.value.encode("utf8"),
        len(index_binany),
    )
    file_obj.write(channel_header)
    file_obj.write(channel_class_name)
    file_obj.write(index_binany)


def _write_instruction(file_obj, instruction):
    instruction_class_name = instruction.__class__.__name__.encode("utf8")

    label = instruction.name
    if label:
        label_raw = label.encode("utf8")
    else:
        label_raw = b""

    instruction_header = struct.pack(
        INSTRUCTION_PACK,
        len(instruction_class_name),
        len(label_raw),
        len(instruction.operands)
    )
    file_obj.write(instruction_header)
    file_obj.write(instruction_class_name)
    file_obj.write(label_raw)

    for operand in instruction.operands:
        type_key = assign_key(operand)
        if TypeKey.is_number(type_key) or TypeKey.is_variable(type_key):
            data = dumps_parameter_value(type_key, operand)
        elif type_key == TypeKey.CHANNEL:
            with io.BytesIO() as container:
                _write_channel(container, operand)
                container.seek(0)
                data = container.read()
        else:
            raise TypeError(
                f"Invalid instruction operand type {type(operand)} for {operand}."
            )

        write_binary(file_obj, data, type_key)
