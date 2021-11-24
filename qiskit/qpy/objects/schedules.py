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
import importlib
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

from qiskit.pulse import channels, instructions, library
from qiskit.pulse.transforms import alignments

from qiskit.pulse.channels import Channel
from qiskit.qpy.common import (
    write_binary,
    read_binary,
    assign_key,
    TypeKey,
)
from .parameter_values import (
    dumps_parameter_value,
    dumps_numbers,
    loads_parameter_value,
    loads_numbers,
)
from .mapping import write_mapping, read_mapping


# CHANNEL binary format
CHANNEL = namedtuple("CHANNEL", ["name_size", "index_type", "index_size"])
CHANNEL_PACK = "!H1cH"
CHANNEL_PACK_SIZE = struct.calcsize(CHANNEL_PACK)

# WAVEFORM binary format
WAVEFORM = namedtuple("WAVEFORM", ["label_size", "epsilon", "data_size", "amp_limited"])
WAVEFORM_PACK = "!HfQ?"
WAVEFORM_PACK_SIZE = struct.calcsize(WAVEFORM_PACK)

# PARAMETRIC PULSE binary format
PARAMETRIC_PULSE = namedtuple(
    "PARAM_PULSE",
    [
        "name_size",
        "module_path_size",
        "label_size",
        "amp_limited",
    ]
)
PARAMETRIC_PULSE_PACK = "!HHHH?"
PARAMETRIC_PULSE_PACK_SIZE = struct.calcsize(PARAMETRIC_PULSE_PACK)

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


def _read_waveform(file_obj):
    waveform_header = struct.unpack(WAVEFORM_PACK, file_obj.read(WAVEFORM_PACK_SIZE))
    label = file_obj.read(waveform_header[0]).decode("utf8")
    epsilon = float(waveform_header[1])
    amp_limited = bool(waveform_header[3])
    samples = loads_numbers(TypeKey.NUMPY, file_obj.read(waveform_header[2]))

    return library.Waveform(
        samples=samples, name=label, epsilon=epsilon, limit_amplitude=amp_limited
    )


def _read_parametric_pulse(file_obj):
    param_pulse_header = struct.unpack(
        PARAMETRIC_PULSE_PACK, file_obj.read(PARAMETRIC_PULSE_PACK_SIZE)
    )
    pulse_class_name = file_obj.read(param_pulse_header[0]).decode("utf8")
    module_path = file_obj.read(param_pulse_header[1]).encode("utf8")
    label = file_obj.read(param_pulse_header[2]).decode("utf8")
    amp_limited = bool(param_pulse_header[3])
    parameters = read_mapping(file_obj)

    if "duration" not in parameters:
        raise KeyError(
            f"Parametric pulse parameter duration is missing in {list(parameters.keys())}."
        )

    if hasattr(library, pulse_class_name) and not module_path:
        pulse_type = getattr(library, pulse_class_name)
    else:
        # user defined parametric pulse
        try:
            custom_module = importlib.import_module(module_path)
            pulse_type = getattr(custom_module, pulse_class_name)
        except ModuleNotFoundError:
            raise TypeError(f"Invalid parametric pulse class name {pulse_class_name}.")

    return pulse_type(
        name=label, limit_amplitude=amp_limited, **parameters
    )


def _read_instruction(file_obj):
    instruction_header = struct.unpack(INSTRUCTION_PACK, file_obj.read(INSTRUCTION_PACK_SIZE))
    instruction_class_name = file_obj.read(instruction_header[0]).decode("utf8")
    label = file_obj.read(instruction_header[1]).decode("utf8")

    operands = []
    for _ in range(instruction_header[2]):
        type_key, data = read_binary(file_obj)
        if TypeKey.is_number(type_key) or TypeKey.is_variable(type_key):
            value = loads_parameter_value(type_key, data)
        elif type_key == TypeKey.STRING:
            value = data.decode("utf8")
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


def _read_alignment_context(file_obj):
    name_size_raw = file_obj.read(struct.calcsize("!H"))
    name_size = struct.unpack("!H", name_size_raw)
    alignment_class_name = file_obj.read(name_size).decode("utf8")
    kwargs = read_mapping(file_obj)

    if hasattr(alignments, alignment_class_name):
        alignment_type = getattr(alignments, alignment_class_name)
        return alignment_type(**kwargs)

    raise TypeError(f"Invalid alignment context name {alignment_class_name}.")


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


def _write_waveform(file_obj, data):
    samples_binary = dumps_numbers(TypeKey.NUMPY, data.samples)
    label = data.name
    if label:
        label_binary = label.encode("utf8")
    else:
        label_binary = bytes()

    wavefom_header = struct.pack(
        WAVEFORM_PACK,
        len(label_binary),
        data.epsilon,
        len(samples_binary),
        data.limit_amplitude,
    )
    file_obj.write(wavefom_header)
    file_obj.write(label_binary)
    file_obj.write(samples_binary)


def _write_parametric_pulse(file_obj, data):
    pulse_class_name = data.__class__.__name__.encode("utf8")

    if hasattr(library, pulse_class_name):
        module_path = bytes()
    else:
        # user defined parametric pulse
        module_path = data.__class__.__module__.encode("utf8")

    if data.name:
        label = data.name.encode("utf8")
    else:
        label = bytes()

    param_pulse_header = struct.pack(
        PARAMETRIC_PULSE_PACK,
        len(pulse_class_name),
        len(module_path),
        len(label),
        data.limit_amplitude,
    )
    file_obj.write(param_pulse_header)
    file_obj.write(module_path)
    file_obj.write(pulse_class_name)
    file_obj.write(label)
    write_mapping(file_obj, data.parameters)


def _write_instruction(file_obj, instruction):
    instruction_class_name = instruction.__class__.__name__.encode("utf8")

    label = instruction.name
    if label:
        label_binary = label.encode("utf8")
    else:
        label_binary = bytes()

    instruction_header = struct.pack(
        INSTRUCTION_PACK,
        len(instruction_class_name),
        len(label_binary),
        len(instruction.operands)
    )
    file_obj.write(instruction_header)
    file_obj.write(instruction_class_name)
    file_obj.write(label_binary)

    for operand in instruction.operands:
        type_key = assign_key(operand)
        if TypeKey.is_number(type_key) or TypeKey.is_variable(type_key):
            data = dumps_parameter_value(type_key, operand)
        elif type_key == TypeKey.STRING:
            data = operand.encode("utf8")
        elif type_key == TypeKey.CHANNEL:
            with io.BytesIO() as container:
                _write_channel(container, operand)
                container.seek(0)
                data = container.read()
        elif type_key == TypeKey.WAVEFORM:
            with io.BytesIO as container:
                _write_waveform(file_obj, operand)
                container.seek(0)
                data = container.read()
        elif type_key == TypeKey.PARAMETRIC_PULSE:
            with io.BytesIO as container:
                _write_parametric_pulse(file_obj, operand)
                container.seek(0)
                data = container.read()
        else:
            raise TypeError(
                f"Invalid instruction operand type {type(operand)} for {operand}."
            )

        write_binary(file_obj, data, type_key)


def _write_alignment_context(file_obj, context):
    alignment_class_name = context.__class__.__name__.encode("utf8")
    name_size = struct.pack("!H", len(alignment_class_name))
    file_obj.write(name_size)
    file_obj.write(alignment_class_name)
    write_mapping(file_obj, context.to_dict())
