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

from qiskit.circuit import Gate, Instruction as CircuitInstruction, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.pulse.channels import Channel
from qiskit.pulse.channels import (
    DriveChannel,
    MeasureChannel,
    ControlChannel,
    MemorySlot,
    RegisterSlot,
)
from qiskit.pulse.instructions import (
    Acquire,
    Play,
    Delay,
    SetFrequency,
    ShiftFrequency,
    SetPhase,
    ShiftPhase,
    RelativeBarrier,
)
from qiskit.pulse.library import Waveform, SymbolicPulse
from qiskit.pulse.schedule import ScheduleBlock
from qiskit.pulse.transforms.alignments import (
    AlignLeft,
    AlignRight,
    AlignSequential,
    AlignEquispaced,
)
from qiskit.qpy import formats, exceptions

QPY_VERSION = 6
ENCODE = "utf8"


class ValueTypeKey(bytes, Enum):
    """Type key enum for value object, e.g. numbers, string, null, parameters."""

    INTEGER = b"i"
    FLOAT = b"f"
    COMPLEX = b"c"
    NUMPY_OBJ = b"n"
    PARAMETER = b"p"
    PARAMETER_VECTOR = b"v"
    PARAMETER_EXPRESSION = b"e"
    STRING = b"s"
    NULL = b"z"

    @classmethod
    def assign(cls, obj):
        """Assign type key to given object.

        Args:
            obj (any): Arbitrary object to evaluate.

        Returns:
            ValueTypeKey: Corresponding key object.

        Raises:
            QpyError: if object type is not defined in QPY. Likely not supported.
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
        if obj is None:
            return cls.NULL

        raise exceptions.QpyError(
            f"Object type {type(obj)} is not supported in {cls.__name__} namespace."
        )


class ContainerTypeKey(bytes, Enum):
    """Typle key enum for container-like object."""

    RANGE = b"r"
    TUPLE = b"t"

    @classmethod
    def assign(cls, obj):
        """Assign type key to given object.

        Args:
            obj (any): Arbitrary object to evaluate.

        Returns:
            ContainerTypeKey: Corresponding key object.

        Raises:
            QpyError: if object type is not defined in QPY. Likely not supported.
        """
        if isinstance(obj, range):
            return cls.RANGE
        if isinstance(obj, tuple):
            return cls.TUPLE

        raise exceptions.QpyError(
            f"Object type {type(obj)} is not supported in {cls.__name__} namespace."
        )


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
            QpyError: if object type is not defined in QPY. Likely not supported.
        """
        if isinstance(obj, PauliEvolutionGate):
            return cls.PAULI_EVOL_GATE
        if isinstance(obj, Gate):
            return cls.GATE
        if isinstance(obj, CircuitInstruction):
            return cls.INSTRUCTION

        raise exceptions.QpyError(
            f"Object type {type(obj)} is not supported in {cls.__name__} namespace."
        )


class ScheduleAlignmentTypeKey(bytes, Enum):
    """Type key enum for schedule block alignment context object."""

    LEFT = b"l"
    RIGHT = b"r"
    SEQUENTIAL = b"s"
    EQUISPACED = b"e"

    # AlignFunc is not serializable due to the callable in context parameter

    @classmethod
    def assign(cls, obj):
        """Assign type key to given object.

        Args:
            obj (any): Arbitrary object to evaluate.

        Returns:
            ScheduleAlignmentTypeKey: Corresponding key object.

        Raises:
            QpyError: if object type is not defined in QPY. Likely not supported.
        """
        if isinstance(obj, AlignLeft):
            return cls.LEFT
        if isinstance(obj, AlignRight):
            return cls.RIGHT
        if isinstance(obj, AlignSequential):
            return cls.SEQUENTIAL
        if isinstance(obj, AlignEquispaced):
            return cls.EQUISPACED

        raise exceptions.QpyError(
            f"Object type {type(obj)} is not supported in {cls.__name__} namespace."
        )

    @classmethod
    def retrieve(cls, type_key):
        """Get context class from given type key.

        Args:
            type_key (bytes): Object type key.

        Returns:
            AlignmentKind: Schedule alignment context.

        Raises:
            QpyError: When key is not defined.
        """
        if type_key == cls.LEFT:
            return AlignLeft
        if type_key == cls.RIGHT:
            return AlignRight
        if type_key == cls.SEQUENTIAL:
            return AlignSequential
        if type_key == cls.EQUISPACED:
            return AlignEquispaced

        raise exceptions.QpyError(
            f"Type key {type_key} is not defined in {cls.__name__} namespace."
        )


class ScheduleElementTypeKey(bytes, Enum):
    """Type key enum for schedule block element object."""

    ACQUIRE = b"a"
    PLAY = b"p"
    DELAY = b"d"
    SET_FREQUENCY = b"f"
    SHIFT_FREQUENCY = b"g"
    SET_PHASE = b"q"
    SHIFT_PHASE = b"r"
    BARRIER = b"b"

    # 's' is reserved by ScheduleBlock, i.e. block can be nested as an element.
    # Call instructon is not supported by QPY.
    # This instruction is excluded from ScheduleBlock instructions with
    # qiskit-terra/#8005 and new instruction Reference will be added instead.
    # Call is only applied to Schedule which is not supported by QPY.
    # Also snapshot is not suppored because of its limited usecase.

    @classmethod
    def assign(cls, obj):
        """Assign type key to given object.

        Args:
            obj (any): Arbitrary object to evaluate.

        Returns:
            ScheduleElementTypeKey: Corresponding key object.

        Raises:
            QpyError: if object type is not defined in QPY. Likely not supported.
        """
        if isinstance(obj, Acquire):
            return (cls.ACQUIRE,)
        if isinstance(obj, Play):
            return cls.PLAY
        if isinstance(obj, Delay):
            return cls.DELAY
        if isinstance(obj, SetFrequency):
            return cls.SET_FREQUENCY
        if isinstance(obj, ShiftFrequency):
            return cls.SHIFT_FREQUENCY
        if isinstance(obj, SetPhase):
            return cls.SET_PHASE
        if isinstance(obj, ShiftPhase):
            return cls.SHIFT_PHASE
        if isinstance(obj, RelativeBarrier):
            return cls.BARRIER

        raise exceptions.QpyError(
            f"Object type {type(obj)} is not supported in {cls.__name__} namespace."
        )

    @classmethod
    def retrieve(cls, type_key):
        """Get instruction class from given type key.

        Args:
            type_key (bytes): Object type key.

        Returns:
            Instruction: Pulse instruction.

        Raises:
            QpyError: When key is not defined.
        """
        if type_key == cls.ACQUIRE:
            return Acquire
        if type_key == cls.PLAY:
            return Play
        if type_key == cls.DELAY:
            return Delay
        if type_key == cls.SET_FREQUENCY:
            return SetFrequency
        if type_key == cls.SHIFT_FREQUENCY:
            return ShiftFrequency
        if type_key == cls.SET_PHASE:
            return SetPhase
        if type_key == cls.SHIFT_PHASE:
            return ShiftPhase
        if type_key == cls.BARRIER:
            return RelativeBarrier

        raise exceptions.QpyError(
            f"Type key {type_key} is not defined in {cls.__name__} namespace."
        )


class ScheduleOperandTypeKey(bytes, Enum):
    """Type key enum for schedule instruction operand object."""

    INTEGER = b"i"
    FLOAT = b"f"
    NUMPY_OBJ = b"n"
    PARAMETER = b"p"
    PARAMETER_EXPRESSION = b"e"
    WAVEFORM = b"w"
    SYMBOLIC_PULSE = b"s"
    CHANNEL = b"c"
    NULL = b"z"

    # Discriminator and Acquire instance are not serialzied.
    # Data format of these object is somewhat opaque and not defiend well.
    # It's rarely used in the Qiskit experiements. Of course these can be added later.

    @classmethod
    def assign(cls, obj):
        """Assign type key to given object.

        Args:
            obj (any): Arbitrary object to evaluate.

        Returns:
            ScheduleOperandTypeKey: Corresponding key object.

        Raises:
            QpyError: if object type is not defined in QPY. Likely not supported.
        """
        if isinstance(obj, int):
            return cls.INTEGER
        if isinstance(obj, float):
            return cls.FLOAT
        if isinstance(obj, (np.integer, np.floating, np.ndarray, np.complexfloating)):
            return cls.NUMPY_OBJ
        if isinstance(obj, Parameter):
            return cls.PARAMETER
        if isinstance(obj, ParameterExpression):
            return cls.PARAMETER_EXPRESSION
        if isinstance(obj, Waveform):
            return cls.WAVEFORM
        if isinstance(obj, SymbolicPulse):
            return cls.SYMBOLIC_PULSE
        if isinstance(obj, Channel):
            return cls.CHANNEL
        if obj is None:
            return cls.NULL

        raise exceptions.QpyError(
            f"Object type {type(obj)} is not supported in {cls.__name__} namespace."
        )


class ScheduleChannelTypeKey(bytes, Enum):
    """Type key enum for schedule channel object."""

    DRIVE = b"d"
    CONTROL = b"c"
    MEASURE = b"m"
    ACQURE = b"a"
    MEM_SLOT = b"e"
    REG_SLOT = b"r"

    # SnapShot channel is not defined because of its limited usecase.

    @classmethod
    def assign(cls, obj):
        """Assign type key to given object.

        Args:
            obj (any): Arbitrary object to evaluate.

        Returns:
            ScheduleChannelTypeKey: Corresponding key object.

        Raises:
            QpyError: if object type is not defined in QPY. Likely not supported.
        """
        if isinstance(obj, DriveChannel):
            return cls.DRIVE
        if isinstance(obj, ControlChannel):
            return cls.CONTROL
        if isinstance(obj, MeasureChannel):
            return cls.MEASURE
        if isinstance(obj, Acquire):
            return cls.ACQURE
        if isinstance(obj, MemorySlot):
            return cls.MEM_SLOT
        if isinstance(obj, RegisterSlot):
            return cls.REG_SLOT

        raise exceptions.QpyError(
            f"Object type {type(obj)} is not supported in {cls.__name__} namespace."
        )

    @classmethod
    def retrieve(cls, type_key):
        """Get context class from given type key.

        Args:
            type_key (bytes): Object type key.

        Returns:
            Channel: Schedule channel.

        Raises:
            QpyError: When key is not defined.
        """
        if type_key == cls.DRIVE:
            return DriveChannel
        if type_key == cls.CONTROL:
            return ControlChannel
        if type_key == cls.MEASURE:
            return MeasureChannel
        if type_key == cls.ACQURE:
            return Acquire
        if type_key == cls.MEM_SLOT:
            return MemorySlot
        if type_key == cls.REG_SLOT:
            return RegisterSlot

        raise exceptions.QpyError(
            f"Type key {type_key} is not defined in {cls.__name__} namespace."
        )


class ProgramTypeKey(bytes, Enum):
    """Typle key enum for program that Qiskit generates."""

    CIRCUIT = b"q"
    SCHEDULE_BLOCK = b"s"

    @classmethod
    def assign(cls, obj):
        """Assign type key to given object.

        Args:
            obj (any): Arbitrary object to evaluate.

        Returns:
            ProgramTypeKey: Corresponding key object.

        Raises:
            QpyError: if object type is not defined in QPY. Likely not supported.
        """
        if isinstance(obj, QuantumCircuit):
            return cls.CIRCUIT
        if isinstance(obj, ScheduleBlock):
            return cls.SCHEDULE_BLOCK

        raise exceptions.QpyError(
            f"Object type {type(obj)} is not supported in {cls.__name__} namespace."
        )


def read_generic_typed_data(file_obj):
    """Read a single data chunk from the file like object.

    Args:
        file_obj (File): A file like object that contains the QPY binary data.

    Returns:
        tuple: Tuple of type key binary and the bytes object of the single data.
    """
    data = formats.TYPED_DATA(
        *struct.unpack(
            formats.TYPED_DATA_PACK,
            file_obj.read(formats.TYPED_DATA_SIZE),
        )
    )

    return data.type, file_obj.read(data.size)


def write_generic_typed_data(file_obj, type_key, data_binary):
    """Write statically typed binary data to the file like object.

    Args:
        file_obj (File): A file like object to write data.
        type_key (Enum): Object type of the data.
        data_binary (bytes): Binary data to write.
    """
    data_header = struct.pack(formats.TYPED_DATA_PACK, type_key, len(data_binary))
    file_obj.write(data_header)
    file_obj.write(data_binary)


def read_sequence(file_obj, deserializer, **kwargs):
    """Read a sequence of data from the file like object.

    Args:
        file_obj (File): A file like object that contains the QPY binary data.
        deserializer (Callable): Deserializer callback that can handle input object type.
        kwargs: Options set to the deserializer.

    Returns:
        Sequence: Deserialized object.
    """
    sequence = []

    data = formats.SEQUENCE._make(
        struct.unpack(
            formats.SEQUENCE_PACK,
            file_obj.read(formats.SEQUENCE_SIZE),
        )
    )
    for _ in range(data.num_elements):
        sequence.append(deserializer(file_obj, **kwargs))

    return sequence


def write_sequence(file_obj, sequence, serializer, **kwargs):
    """Write a sequence of data in the file like object.

    Args:
        file_obj (File): A file like object to write data.
        sequence (Sequence): Object to serialize.
        serializer (Callable): Serializer callback that can handle input object type.
        kwargs: Options set to the serializer.
    """
    num_elements = len(sequence)

    file_obj.write(struct.pack(formats.SEQUENCE_PACK, num_elements))
    for datum in sequence:
        serializer(file_obj, datum, **kwargs)


def read_type_key(file_obj):
    """Read a type key from the file like object.

    Args:
        file_obj (File): A file like object that contains the QPY binary data.

    Returns:
        bytes: Type key.
    """
    key_size = struct.calcsize("!1c")
    return struct.unpack("!1c", file_obj.read(key_size))[0]


def write_type_key(file_obj, type_key):
    """Write a type key in the file like object.

    Args:
        file_obj (File): A file like object that contains the QPY binary data.
        type_key (bytes): Type key to write.
    """
    file_obj.write(struct.pack("!1c", type_key))


def data_to_binary(obj, serializer, **kwargs):
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
        binary_data = container.getvalue()

    return binary_data


def sequence_to_binary(sequence, serializer, **kwargs):
    """Convert sequence into binary data with specified serializer.

    Args:
        sequence (Sequence): Object to serialize.
        serializer (Callable): Serializer callback that can handle input object type.
        kwargs: Options set to the serializer.

    Returns:
        bytes: Binary data.
    """
    with io.BytesIO() as container:
        write_sequence(container, sequence, serializer, **kwargs)
        binary_data = container.getvalue()

    return binary_data


def data_from_binary(binary_data, deserializer, **kwargs):
    """Load object from binary data with specified deserializer.

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


def sequence_from_binary(binary_data, deserializer, **kwargs):
    """Load object from binary sequence with specified deserializer.

    Args:
        binary_data (bytes): Binary data to deserialize.
        deserializer (Callable): Deserializer callback that can handle input object type.
        kwargs: Options set to the deserializer.

    Returns:
        any: Deserialized sequence.
    """
    with io.BytesIO(binary_data) as container:
        sequence = read_sequence(container, deserializer, **kwargs)

    return sequence
