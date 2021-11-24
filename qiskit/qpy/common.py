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
from enum import Flag

import numpy as np

from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.channels import Channel
from qiskit.pulse.library import Waveform, ParametricPulse

# OBJECT binary format
OBJECT = namedtuple("OBJECT", ["type", "size"])
OBJECT_PACK = "!1cQ"
OBJECT_PACK_SIZE = struct.calcsize(OBJECT_PACK)


class TypeKey(Flag):
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

    @classmethod
    def is_number(cls, type_key):
        if type_key in [cls.FLOAT, cls.INTEGER, cls.COMPLEX, cls.NUMPY]:
            return True
        return False

    @classmethod
    def is_variable(cls, type_key):
        if type_key in [cls.PARAMETER, cls.PARAMETER_EXPRESSION]:
            return True
        return False


def assign_key(obj):
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

    raise TypeError(f"Object type {type(obj)} is not supported.")


def read_binary(file_obj):
    data_chunk = struct.unpack(OBJECT_PACK, file_obj.read(OBJECT_PACK_SIZE))
    type_key = data_chunk[0].decode("utf8")
    data_binary = file_obj.read(data_chunk[1])

    return TypeKey(type_key), data_binary


def write_binary(file_obj, data_binary, type_key):
    data_header = struct.pack(OBJECT_PACK, type_key.value.encode("utf8"), len(data_binary))
    file_obj.write(data_header)
    file_obj.write(data_binary)
