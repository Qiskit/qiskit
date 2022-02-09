# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=too-many-return-statements

"""
Common functions across several serialization and deserialization modules.
"""

import io
import struct
from enum import Enum

import numpy as np

from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit import Gate, Instruction as CircuitInstruction
from qiskit.qpy import formats

QPY_VERSION = 3
ENCODE = "utf8"


class CircuitInstructionTypeKey(bytes, Enum):
    """Type key enum for circuit instruction object."""

    INSTRUCTION = b"i"
    GATE = b"g"
    PAULI_EVOL_GATE = b"p"

    @classmethod
    def assign(cls, obj):
        """Assign type key to given object.

        Args:
            obj (any): Arbitrary object to evaluate.

        Returns:
            CircuitInstructionTypeKey: Corresponding key object.

        Raises:
            TypeError: if object type is not defined in QPY. Likely not supported.
        """
        if isinstance(obj, PauliEvolutionGate):
            return cls.PAULI_EVOL_GATE
        if isinstance(obj, Gate):
            return cls.GATE
        if isinstance(obj, CircuitInstruction):
            return cls.INSTRUCTION

        raise TypeError(f"Object type {type(obj)} is not supported.")


class AlphanumericTypeKey(bytes, Enum):
    """Type key enum for alphanumeric object."""

    INTEGER = b"i"
    FLOAT = b"f"
    COMPLEX = b"c"
    NUMPY_OBJ = b"n"
    PARAMETER = b"p"
    PARAMETER_VECTOR = b"v"
    PARAMETER_EXPRESSION = b"e"
    STRING = b"s"

    @classmethod
    def assign(cls, obj):
        """Assign type key to given object.

        Args:
            obj (any): Arbitrary object to evaluate.

        Returns:
            AlphanumericTypeKey: Corresponding key object.

        Raises:
            TypeError: if object type is not defined in QPY. Likely not supported.
        """
        if isinstance(obj, int):
            return cls.INTEGER
        if isinstance(obj, float):
            return cls.FLOAT
        if isinstance(obj, complex):
            return cls.COMPLEX
        if isinstance(obj, (np.integer, np.floating, np.ndarray, np.complexfloating)):
            return cls.NUMPY_OBJ
        if isinstance(obj, ParameterVectorElement):
            return cls.PARAMETER_VECTOR
        if isinstance(obj, Parameter):
            return cls.PARAMETER
        if isinstance(obj, ParameterExpression):
            return cls.PARAMETER_EXPRESSION
        if isinstance(obj, str):
            return cls.STRING

        raise TypeError(f"Object type {type(obj)} is not supported.")


def read_typed_data(file_obj, key_scope):
    """Read a single data chunk from the file like object.

    Args:
        file_obj (File): A file like object that contains the QPY binary data.
        key_scope (EnumMeta): Expected type of this data.

    Returns:
        tuple: Tuple of ``TypeKey`` and the bytes object of the single data.
    """
    data = formats.TYPED_OBJECT(
        *struct.unpack(
            formats.TYPED_OBJECT_PACK,
            file_obj.read(formats.TYPED_OBJECT_PACK_SIZE),
        )
    )

    return key_scope(data.type), file_obj.read(data.size)


def write_typed_data(file_obj, type_key, data_binary):
    """Write statically typed binary data to the file like object.

    Args:
        file_obj (File): A file like object to write data.
        type_key (Enum): Object type of the data.
        data_binary (bytes): Binary data to write.
    """
    data_header = struct.pack(formats.TYPED_OBJECT_PACK, type_key, len(data_binary))
    file_obj.write(data_header)
    file_obj.write(data_binary)


def to_binary(obj, serializer, **kwargs):
    """Convert object into binary data with specified serializer.

    Args:
        obj (any): Object to serialize.
        serializer (Callable): Serializer callback that can handle input object type.
        kwargs: Options set to the serializer.

    Returns:
        bytes: Binary data.
    """
    with io.BytesIO() as container:
        serializer(container, obj, **kwargs)
        container.seek(0)
        binary_data = container.read()

    return binary_data


def from_binary(binary_data, deserializer, **kwargs):
    """Load object from binary data with specified serializer.

    Args:
        binary_data (bytes): Binary data to deserialize.
        deserializer (Callable): Deserializer callback that can handle input object type.
        kwargs: Options set to the deserializer.

    Returns:
        any: Deserialized object.
    """
    with io.BytesIO(binary_data) as container:
        obj = deserializer(container, **kwargs)
    return obj
