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
import json
import struct
import zlib
import warnings

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.pulse import library, channels
from qiskit.pulse.schedule import ScheduleBlock
from qiskit.qpy import formats, common, type_keys
from qiskit.qpy.binary_io import value
from qiskit.utils import optionals as _optional

if _optional.HAS_SYMENGINE:
    import symengine as sym
else:
    import sympy as sym


def _read_channel(file_obj, version):
    type_key = common.read_type_key(file_obj)
    index = value.read_value(file_obj, version, {})

    channel_cls = type_keys.ScheduleChannel.retrieve(type_key)

    return channel_cls(index)


def _read_waveform(file_obj, version):
    header = formats.WAVEFORM._make(
        struct.unpack(
            formats.WAVEFORM_PACK,
            file_obj.read(formats.WAVEFORM_SIZE),
        )
    )
    samples_raw = file_obj.read(header.data_size)
    samples = common.data_from_binary(samples_raw, np.load)
    name = value.read_value(file_obj, version, {})

    return library.Waveform(
        samples=samples,
        name=name,
        epsilon=header.epsilon,
        limit_amplitude=header.amp_limited,
    )


def _loads_symbolic_expr(expr_bytes):
    from sympy import parse_expr

    if expr_bytes == b"":
        return None

    expr_txt = zlib.decompress(expr_bytes).decode(common.ENCODE)
    expr = parse_expr(expr_txt)

    if _optional.HAS_SYMENGINE:
        from symengine import sympify

        return sympify(expr)
    return expr


def _format_legacy_qiskit_pulse(pulse_type, envelope, parameters):
    # In the transition to Qiskit Terra 0.23, the representation of library pulses was changed from
    # complex "amp" to float "amp" and "angle". The existing library pulses in previous versions are
    # handled here separately to conform with the new representation. To avoid role assumption for
    # "amp" for custom pulses, only the library pulses are handled this way.

    # Note that parameters is mutated during the function call

    # List of pulses in the library in QPY version 5 and below:
    legacy_library_pulses = ["Gaussian", "GaussianSquare", "Drag", "Constant"]

    if pulse_type in legacy_library_pulses:
        # Once complex amp support will be deprecated we will need:
        # parameters["angle"] = np.angle(parameters["amp"])
        # parameters["amp"] = np.abs(parameters["amp"])

        # In the meanwhile we simply add:
        parameters["angle"] = 0
        _amp, _angle = sym.symbols("amp, angle")
        envelope = envelope.subs(_amp, _amp * sym.exp(sym.I * _angle))

        # And warn that this will change in future releases:
        warnings.warn(
            "Complex amp support for symbolic library pulses will be deprecated. "
            "Once deprecated, library pulses loaded from old QPY files (Terra version <=0.22.2),"
            " will be converted automatically to float (amp,angle) representation.",
            PendingDeprecationWarning,
        )
    return envelope


def _read_symbolic_pulse(file_obj, version, qiskit_version):
    header = formats.SYMBOLIC_PULSE._make(
        struct.unpack(
            formats.SYMBOLIC_PULSE_PACK,
            file_obj.read(formats.SYMBOLIC_PULSE_SIZE),
        )
    )
    pulse_type = file_obj.read(header.type_size).decode(common.ENCODE)
    envelope = _loads_symbolic_expr(file_obj.read(header.envelope_size))
    constraints = _loads_symbolic_expr(file_obj.read(header.constraints_size))
    valid_amp_conditions = _loads_symbolic_expr(file_obj.read(header.valid_amp_conditions_size))
    parameters = common.read_mapping(
        file_obj,
        deserializer=value.loads_value,
        version=version,
        vectors={},
    )
    if qiskit_version < (0, 23, 0):
        envelope = _format_legacy_qiskit_pulse(pulse_type, envelope, parameters)
        # Note that parameters is mutated during the function call

    duration = value.read_value(file_obj, version, {})
    name = value.read_value(file_obj, version, {})

    return library.SymbolicPulse(
        pulse_type=pulse_type,
        duration=duration,
        parameters=parameters,
        name=name,
        limit_amplitude=header.amp_limited,
        envelope=envelope,
        constraints=constraints,
        valid_amp_conditions=valid_amp_conditions,
    )


def _read_alignment_context(file_obj, version):
    type_key = common.read_type_key(file_obj)

    context_params = common.read_sequence(
        file_obj,
        deserializer=value.loads_value,
        version=version,
        vectors={},
    )
    context_cls = type_keys.ScheduleAlignment.retrieve(type_key)

    instance = object.__new__(context_cls)
    instance._context_params = tuple(context_params)

    return instance


def _loads_operand(type_key, data_bytes, version, qiskit_version):
    if type_key == type_keys.ScheduleOperand.WAVEFORM:
        return common.data_from_binary(data_bytes, _read_waveform, version=version)
    if type_key == type_keys.ScheduleOperand.SYMBOLIC_PULSE:
        return common.data_from_binary(
            data_bytes, _read_symbolic_pulse, version=version, qiskit_version=qiskit_version
        )
    if type_key == type_keys.ScheduleOperand.CHANNEL:
        return common.data_from_binary(data_bytes, _read_channel, version=version)

    return value.loads_value(type_key, data_bytes, version, {})


def _read_element(file_obj, version, metadata_deserializer, qiskit_version=None):
    type_key = common.read_type_key(file_obj)

    if type_key == type_keys.Program.SCHEDULE_BLOCK:
        return read_schedule_block(
            file_obj, version, metadata_deserializer, qiskit_version=qiskit_version
        )

    operands = common.read_sequence(
        file_obj, deserializer=_loads_operand, version=version, qiskit_version=qiskit_version
    )
    name = value.read_value(file_obj, version, {})

    instance = object.__new__(type_keys.ScheduleInstruction.retrieve(type_key))
    instance._operands = tuple(operands)
    instance._name = name
    instance._hash = None

    return instance


def _write_channel(file_obj, data):
    type_key = type_keys.ScheduleChannel.assign(data)
    common.write_type_key(file_obj, type_key)
    value.write_value(file_obj, data.index)


def _write_waveform(file_obj, data):
    samples_bytes = common.data_to_binary(data.samples, np.save)

    header = struct.pack(
        formats.WAVEFORM_PACK,
        data.epsilon,
        len(samples_bytes),
        data._limit_amplitude,
    )
    file_obj.write(header)
    file_obj.write(samples_bytes)
    value.write_value(file_obj, data.name)


def _dumps_symbolic_expr(expr):
    from sympy import srepr, sympify

    if expr is None:
        return b""

    expr_bytes = srepr(sympify(expr)).encode(common.ENCODE)
    return zlib.compress(expr_bytes)


def _write_symbolic_pulse(file_obj, data):
    pulse_type_bytes = data.pulse_type.encode(common.ENCODE)
    envelope_bytes = _dumps_symbolic_expr(data.envelope)
    constraints_bytes = _dumps_symbolic_expr(data.constraints)
    valid_amp_conditions_bytes = _dumps_symbolic_expr(data.valid_amp_conditions)

    header_bytes = struct.pack(
        formats.SYMBOLIC_PULSE_PACK,
        len(pulse_type_bytes),
        len(envelope_bytes),
        len(constraints_bytes),
        len(valid_amp_conditions_bytes),
        data._limit_amplitude,
    )
    file_obj.write(header_bytes)
    file_obj.write(pulse_type_bytes)
    file_obj.write(envelope_bytes)
    file_obj.write(constraints_bytes)
    file_obj.write(valid_amp_conditions_bytes)
    common.write_mapping(
        file_obj,
        mapping=data._params,
        serializer=value.dumps_value,
    )
    value.write_value(file_obj, data.duration)
    value.write_value(file_obj, data.name)


def _write_alignment_context(file_obj, context):
    type_key = type_keys.ScheduleAlignment.assign(context)
    common.write_type_key(file_obj, type_key)
    common.write_sequence(
        file_obj,
        sequence=context._context_params,
        serializer=value.dumps_value,
    )


def _dumps_operand(operand):
    if isinstance(operand, library.Waveform):
        type_key = type_keys.ScheduleOperand.WAVEFORM
        data_bytes = common.data_to_binary(operand, _write_waveform)
    elif isinstance(operand, library.SymbolicPulse):
        type_key = type_keys.ScheduleOperand.SYMBOLIC_PULSE
        data_bytes = common.data_to_binary(operand, _write_symbolic_pulse)
    elif isinstance(operand, channels.Channel):
        type_key = type_keys.ScheduleOperand.CHANNEL
        data_bytes = common.data_to_binary(operand, _write_channel)
    else:
        type_key, data_bytes = value.dumps_value(operand)

    return type_key, data_bytes


def _write_element(file_obj, element, metadata_serializer):
    if isinstance(element, ScheduleBlock):
        common.write_type_key(file_obj, type_keys.Program.SCHEDULE_BLOCK)
        write_schedule_block(file_obj, element, metadata_serializer)
    else:
        type_key = type_keys.ScheduleInstruction.assign(element)
        common.write_type_key(file_obj, type_key)
        common.write_sequence(
            file_obj,
            sequence=element.operands,
            serializer=_dumps_operand,
        )
        value.write_value(file_obj, element.name)


def read_schedule_block(file_obj, version, metadata_deserializer=None, qiskit_version=None):
    """Read a single ScheduleBlock from the file like object.

    Args:
        file_obj (File): A file like object that contains the QPY binary data.
        version (int): QPY version.
        metadata_deserializer (JSONDecoder): An optional JSONDecoder class
            that will be used for the ``cls`` kwarg on the internal
            ``json.load`` call used to deserialize the JSON payload used for
            the :attr:`.ScheduleBlock.metadata` attribute for a schdule block
            in the file-like object. If this is not specified the circuit metadata will
            be parsed as JSON with the stdlib ``json.load()`` function using
            the default ``JSONDecoder`` class.
        qiskit_version (tuple): tuple with major, minor and patch versions of qiskit.

    Returns:
        ScheduleBlock: The schedule block object from the file.

    Raises:
        TypeError: If any of the instructions is invalid data format.
        QiskitError: QPY version is earlier than block support.
    """

    if version < 5:
        QiskitError(f"QPY version {version} does not support ScheduleBlock.")

    data = formats.SCHEDULE_BLOCK_HEADER._make(
        struct.unpack(
            formats.SCHEDULE_BLOCK_HEADER_PACK,
            file_obj.read(formats.SCHEDULE_BLOCK_HEADER_SIZE),
        )
    )
    name = file_obj.read(data.name_size).decode(common.ENCODE)
    metadata_raw = file_obj.read(data.metadata_size)
    metadata = json.loads(metadata_raw, cls=metadata_deserializer)
    context = _read_alignment_context(file_obj, version)

    block = ScheduleBlock(
        name=name,
        metadata=metadata,
        alignment_context=context,
    )
    for _ in range(data.num_elements):
        block_elm = _read_element(
            file_obj, version, metadata_deserializer, qiskit_version=qiskit_version
        )
        block.append(block_elm, inplace=True)

    return block


def write_schedule_block(file_obj, block, metadata_serializer=None):
    """Write a single ScheduleBlock object in the file like object.

    Args:
        file_obj (File): The file like object to write the circuit data in.
        block (ScheduleBlock): A schedule block data to write.
        metadata_serializer (JSONEncoder): An optional JSONEncoder class that
            will be passed the :attr:`.ScheduleBlock.metadata` dictionary for
            ``block`` and will be used as the ``cls`` kwarg
            on the ``json.dump()`` call to JSON serialize that dictionary.

    Raises:
        TypeError: If any of the instructions is invalid data format.
    """
    metadata = json.dumps(block.metadata, separators=(",", ":"), cls=metadata_serializer).encode(
        common.ENCODE
    )
    block_name = block.name.encode(common.ENCODE)

    # Write schedule block header
    header_raw = formats.SCHEDULE_BLOCK_HEADER(
        name_size=len(block_name),
        metadata_size=len(metadata),
        num_elements=len(block),
    )
    header = struct.pack(formats.SCHEDULE_BLOCK_HEADER_PACK, *header_raw)
    file_obj.write(header)
    file_obj.write(block_name)
    file_obj.write(metadata)

    _write_alignment_context(file_obj, block.alignment_context)
    for block_elm in block.blocks:
        _write_element(file_obj, block_elm, metadata_serializer)
