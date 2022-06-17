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

from qiskit.qpy import formats

QPY_VERSION = 6
ENCODE = "utf8"


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
