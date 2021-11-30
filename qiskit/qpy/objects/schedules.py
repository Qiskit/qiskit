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

# pylint: disable=invalid-name

"""Read and write schedule and schedule instructions."""

import importlib
import io
import json
import struct
from collections import namedtuple

from qiskit.pulse import channels, instructions, library
from qiskit.pulse.schedule import ScheduleBlock
from qiskit.pulse.configuration import Kernel, Discriminator
from qiskit.pulse.transforms import alignments
from qiskit.qpy.common import (
    write_binary,
    read_binary,
    assign_key,
    TypeKey,
)
from .mapping import write_mapping, read_mapping
from .parameter_values import (
    dumps_parameter_value,
    dumps_numbers,
    loads_parameter_value,
    loads_numbers,
)

# SCHEDULE_BLOCK binary format
SCHEDULE_BLOCK = namedtuple(
    "SCHEDULE_BLOCK",
    [
        "name_size",
        "metadata_size",
        "alignment_data_size",
        "num_elements",
    ],
)
SCHEDULE_BLOCK_PACK = "!HQQQ"
SCHEDULE_BLOCK_PACK_SIZE = struct.calcsize(SCHEDULE_BLOCK_PACK)

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
    ],
)
PARAMETRIC_PULSE_PACK = "!HHH?"
PARAMETRIC_PULSE_PACK_SIZE = struct.calcsize(PARAMETRIC_PULSE_PACK)

# INSTRUCTION binary format
INSTRUCTION = namedtuple("INSTRUCTION", ["name_size", "label_size", "num_operands"])
INSTRUCTION_PACK = "!HHH"
INSTRUCTION_PACK_SIZE = struct.calcsize(INSTRUCTION_PACK)

# MEASURE_PROCESSOR binary format
MEASURE_PROCESSOR = namedtuple("MEASURE_PROCESSOR", ["name_size", "params_size"])
MEASURE_PROCESSOR_PACK = "!HQ"
MEASURE_PROCESSOR_PACK_SIZE = struct.calcsize(MEASURE_PROCESSOR_PACK)


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

    kwargs = {
        "samples": samples,
        "epsilon": epsilon,
        "limit_amplitude": amp_limited,
    }
    if label:
        kwargs["name"] = label

    return library.Waveform(**kwargs)


def _read_parametric_pulse(file_obj):
    param_pulse_header = struct.unpack(
        PARAMETRIC_PULSE_PACK, file_obj.read(PARAMETRIC_PULSE_PACK_SIZE)
    )
    pulse_class_name = file_obj.read(param_pulse_header[0]).decode("utf8")
    module_path = file_obj.read(param_pulse_header[1]).decode("utf8")
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
        except ModuleNotFoundError as ex:
            raise TypeError(f"Invalid parametric pulse class name {pulse_class_name}.") from ex

    kwargs = parameters
    kwargs["limit_amplitude"] = amp_limited
    if label:
        kwargs["name"] = label

    return pulse_type(**kwargs)


def _read_measure_processor(file_obj, processor):
    processor_header = struct.unpack(MEASURE_PROCESSOR_PACK, file_obj.read(MEASURE_PROCESSOR_PACK_SIZE))
    name = file_obj.read(processor_header[0]).encode("utf8")
    params = json.loads(file_obj.read(processor_header[1]).decode("utf8"))

    return processor(name, **params)


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
        elif type_key == TypeKey.WAVEFORM:
            with io.BytesIO(data) as container:
                value = _read_waveform(container)
        elif type_key == TypeKey.PARAMETRIC_PULSE:
            with io.BytesIO(data) as container:
                value = _read_parametric_pulse(container)
        elif type_key == TypeKey.KERNEL:
            with io.BytesIO(data) as container:
                value = _read_measure_processor(container, Kernel)
        elif type_key == TypeKey.DISCRIMINATOR:
            with io.BytesIO(data) as container:
                value = _read_measure_processor(container, Discriminator)
        else:
            raise TypeError(f"Invalid instruction operand type {type_key} for value {data}.")
        operands.append(value)

    if hasattr(instructions, instruction_class_name):
        instruction_type = getattr(instructions, instruction_class_name)
        if label:
            return instruction_type(*operands, name=label)
        return instruction_type(*operands)

    raise TypeError(f"Invalid instruction class name {instruction_class_name}.")


def _read_alignment_context(file_obj):
    name_size_raw = file_obj.read(struct.calcsize("!H"))
    name_size = struct.unpack("!H", name_size_raw)[0]
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
        index_type_key.encode("utf8"),
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

    if hasattr(library, data.__class__.__name__):
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
    file_obj.write(pulse_class_name)
    file_obj.write(module_path)
    file_obj.write(label)
    write_mapping(file_obj, data.parameters)


def _write_measure_processor(file_obj, data):
    name = data.name.encode("utf8")
    params = json.dumps(data.params, separators=(",", ":")).encode("utf8")

    processor_header = struct.pack(
        MEASURE_PROCESSOR_PACK,
        len(name),
        len(params)
    )
    file_obj.write(processor_header)
    file_obj.write(name)
    file_obj.write(params)


def _write_instruction(file_obj, instruction):
    instruction_class_name = instruction.__class__.__name__.encode("utf8")

    label = instruction.name
    if label:
        label_binary = label.encode("utf8")
    else:
        label_binary = bytes()

    instruction_header = struct.pack(
        INSTRUCTION_PACK, len(instruction_class_name), len(label_binary), len(instruction.operands)
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
            with io.BytesIO() as container:
                _write_waveform(container, operand)
                container.seek(0)
                data = container.read()
        elif type_key == TypeKey.PARAMETRIC_PULSE:
            with io.BytesIO() as container:
                _write_parametric_pulse(container, operand)
                container.seek(0)
                data = container.read()
        elif type_key in [TypeKey.KERNEL, TypeKey.DISCRIMINATOR]:
            with io.BytesIO() as container:
                _write_measure_processor(container, operand)
                container.seek(0)
                data = container.read()
        else:
            raise TypeError(f"Invalid instruction operand type {type(operand)} for {operand}.")

        write_binary(file_obj, data, type_key)


def _write_alignment_context(file_obj, context):
    alignment_class_name = context.__class__.__name__.encode("utf8")
    name_size = struct.pack("!H", len(alignment_class_name))
    file_obj.write(name_size)
    file_obj.write(alignment_class_name)
    write_mapping(file_obj, context.to_dict())


def read_schedule_block(file_obj):
    """Read a single schedule block from the file like object.

    Args:
        file_obj (File): A file like object that contains the QPY binary data.

    Returns:
        ScheduleBlock: Deserialized schedule block object.

    Raises:
        TypeError: If any of the instructions is invalid data format.
    """
    block_header = struct.unpack(SCHEDULE_BLOCK_PACK, file_obj.read(SCHEDULE_BLOCK_PACK_SIZE))
    block_name = file_obj.read(block_header[0]).decode("utf8")
    metadata = json.loads(file_obj.read(block_header[1]).decode("utf8"))
    with io.BytesIO(file_obj.read(block_header[2])) as alignment_container:
        alignment_data = _read_alignment_context(alignment_container)

    blocks = []
    for _ in range(block_header[3]):
        type_key, data_binary = read_binary(file_obj)
        if type_key == TypeKey.SCHEDULE_BLOCK:
            with io.BytesIO(data_binary) as block_container:
                block_elem = read_schedule_block(block_container)
        elif type_key == TypeKey.INSTRUCTION:
            with io.BytesIO(data_binary) as block_container:
                block_elem = _read_instruction(block_container)
        else:
            raise TypeError(f"Invalid block component type {type_key} for {data_binary}.")
        blocks.append(block_elem)

    deser_block = ScheduleBlock(
        name=block_name,
        metadata=metadata,
        alignment_context=alignment_data,
    )
    for block in blocks:
        deser_block.append(block, inplace=True)

    return deser_block


def write_schedule_block(file_obj, block):
    """Write a single schedule block to the file like object.

    Args:
        file_obj (File): A file like object to write schedule block data.
        block (ScheduleBlock): A pulse program to write.

    Raises:
        TypeError: If any of the instructions is invalid data format.
    """
    block_name = block.name.encode("utf8")
    metadata = json.dumps(block.metadata, separators=(",", ":")).encode("utf8")
    with io.BytesIO() as alignment_container:
        _write_alignment_context(alignment_container, block.alignment_context)
        alignment_container.seek(0)
        alignment_data = alignment_container.read()

    block_header = struct.pack(
        SCHEDULE_BLOCK_PACK,
        len(block_name),
        len(metadata),
        len(alignment_data),
        len(block),
    )
    file_obj.write(block_header)
    file_obj.write(block_name)
    file_obj.write(metadata)
    file_obj.write(alignment_data)

    for block_elem in block.blocks:
        if isinstance(block_elem, ScheduleBlock):
            type_key = TypeKey.SCHEDULE_BLOCK
            with io.BytesIO() as block_container:
                write_schedule_block(block_container, block_elem)
                block_container.seek(0)
                block_data = block_container.read()
        elif isinstance(block_elem, instructions.Instruction):
            type_key = TypeKey.INSTRUCTION
            with io.BytesIO() as block_container:
                _write_instruction(block_container, block_elem)
                block_container.seek(0)
                block_data = block_container.read()
        else:
            raise TypeError(f"Invalid block component {type(block_elem)}.")

        write_binary(file_obj, block_data, type_key)
