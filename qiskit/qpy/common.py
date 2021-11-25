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
"""Common functions across several serialization and deserialization modules."""

import struct
from collections import namedtuple
from enum import Enum

import numpy as np

from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.channels import Channel
from qiskit.pulse.library import Waveform, ParametricPulse
from qiskit.pulse.schedule import ScheduleBlock
from qiskit.pulse.instructions import Instruction as PulseInstruction

# OBJECT binary format
OBJECT = namedtuple("OBJECT", ["type", "size"])
OBJECT_PACK = "!1cQ"
OBJECT_PACK_SIZE = struct.calcsize(OBJECT_PACK)


class TypeKey(str, Enum):
    """Reserved type string for binary format header."""

    FLOAT = "f"
    INTEGER = "i"
    COMPLEX = "c"
    NUMPY = "n"
    PARAMETER = "p"
    PARAMETER_EXPRESSION = "e"
    STRING = "s"
    CHANNEL = "d"
    PARAMETRIC_PULSE = "r"
    WAVEFORM = "w"
    QUANTUM_CIRCUIT = "t"
    SCHEDULE_BLOCK = "b"
    INSTRUCTION = "j"

    @classmethod
    def is_number(cls, type_key) -> bool:
        """Return if input type is representation of numerical value."""
        if type_key in [cls.FLOAT, cls.INTEGER, cls.COMPLEX, cls.NUMPY]:
            return True
        return False

    @classmethod
    def is_variable(cls, type_key) -> bool:
        """Return if input type is representation of parameter value."""
        if type_key in [cls.PARAMETER, cls.PARAMETER_EXPRESSION]:
            return True
        return False


# pylint: disable=too-many-return-statements
def assign_key(obj) -> TypeKey:
    """Assign type key to input object.

    Args:
        obj (Any): Input object to evaluate.

    Returns:
        Corresponding type key.

    Raises:
        TypeError: if object is not valid data type in QPY module.
    """
    if isinstance(obj, float):
        return TypeKey.FLOAT
    if isinstance(obj, int):
        return TypeKey.INTEGER
    if isinstance(obj, complex):
        return TypeKey.COMPLEX
    if isinstance(obj, (np.integer, np.floating, np.ndarray, np.complexfloating)):
        return TypeKey.NUMPY
    if isinstance(obj, Parameter):
        return TypeKey.PARAMETER
    if isinstance(obj, ParameterExpression):
        return TypeKey.PARAMETER_EXPRESSION
    if isinstance(obj, str):
        return TypeKey.STRING
    if isinstance(obj, Channel):
        return TypeKey.CHANNEL
    if isinstance(obj, Waveform):
        return TypeKey.WAVEFORM
    if isinstance(obj, ParametricPulse):
        return TypeKey.PARAMETRIC_PULSE
    if isinstance(obj, ScheduleBlock):
        return TypeKey.SCHEDULE_BLOCK
    if isinstance(obj, PulseInstruction):
        return TypeKey.INSTRUCTION

    raise TypeError(f"Object type {type(obj)} is not supported.")


def read_binary(file_obj):
    """Read a single data chunk from the file like object.

    Args:
        file_obj (File): A file like object that contains the QPY binary data.

    Returns:
        tuple: Tuple of ``TypeKey`` and the bytes object of the single data.
    """
    data_chunk = struct.unpack(OBJECT_PACK, file_obj.read(OBJECT_PACK_SIZE))
    type_key = data_chunk[0].decode("utf8")
    data_binary = file_obj.read(data_chunk[1])

    return TypeKey(type_key), data_binary


def write_binary(file_obj, data_binary, type_key):
    """Write a single binary data to the file like object.

    Args:
        file_obj (File): A file like object to write data.
        data_binary (bytes): Binary data to write.
        type_key (TypeKey): Object type of the data.
    """
    data_header = struct.pack(OBJECT_PACK, type_key.encode("utf8"), len(data_binary))
    file_obj.write(data_header)
    file_obj.write(data_binary)
