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

"""Read and write mapping data. Keys are strings and values are numbers or strings."""

import struct
from collections import namedtuple

from qiskit.qpy.common import assign_key, TypeKey
from .parameter_values import dumps_parameter_value, loads_parameter_value

# MAPPING_ITEM binary format
MAPPING_ITEM = namedtuple("MAPPING_ITEM", ["name_size", "type", "data_size"])
MAPPING_ITEM_PACK = "!H1cQ"
MAPPING_ITEM_PACK_SIZE = struct.calcsize(MAPPING_ITEM_PACK)


def read_mapping(file_obj):
    map_size_raw = file_obj.read(struct.calcsize("!Q"))
    map_size = struct.unpack("!Q", map_size_raw)[0]

    mapping = dict()
    for _ in range(map_size):
        mapping_item_header = struct.unpack(
            MAPPING_ITEM_PACK, file_obj.read(MAPPING_ITEM_PACK_SIZE)
        )
        param_name = file_obj.read(mapping_item_header[0]).decode("utf8")
        type_key = TypeKey(mapping_item_header[1].decode("utf8"))
        value_binary = file_obj.read(mapping_item_header[2])
        if TypeKey.is_number(type_key) or TypeKey.is_variable(type_key):
            value = loads_parameter_value(type_key, value_binary)
        elif type_key == TypeKey.STRING:
            value = value_binary.decode("utf8")
        else:
            raise TypeError(f"Invalid mapping value type {type_key} for value {value_binary}.")
        mapping[param_name] = value

    return mapping


def write_mapping(file_obj, data):
    map_size = struct.pack("!Q", len(data))
    file_obj.write(map_size)

    for name, value in data.items():
        name_binary = name.encode("utf8")
        type_key = assign_key(value)
        if TypeKey.is_number(type_key) or TypeKey.is_variable(type_key):
            value_binary = dumps_parameter_value(type_key, value)
        elif type_key == TypeKey.STRING:
            value_binary = value.encode("utf8")
        else:
            raise TypeError(f"Invalid mapping value type {type_key} for value {value}.")
        param_item_header = struct.pack(
            MAPPING_ITEM_PACK,
            len(name_binary),
            type_key.value.encode("utf8"),
            len(value_binary),
        )
        file_obj.write(param_item_header)
        file_obj.write(name_binary)
        file_obj.write(value_binary)
