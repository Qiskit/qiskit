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

import struct
import warnings
from collections import namedtuple

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.pulse.schedule import ScheduleBlock
from qiskit.version import __version__
from .common import TypeKey
from .objects.schedules import write_schedule_block, read_schedule_block

# FILE_HEADER
FILE_HEADER = namedtuple(
    "FILE_HEADER",
    ["preface", "qpy_version", "major_version", "minor_version", "patch_version", "num_circuits"],
)
FILE_HEADER_PACK = "!6sBBBBQ"
FILE_HEADER_SIZE = struct.calcsize(FILE_HEADER_PACK)


def dump(programs, file_obj):
    if not isinstance(programs, list):
        programs = [programs]

    version_parts = [int(x) for x in __version__.split(".")[0:3]]
    file_header = struct.pack(
        FILE_HEADER_PACK,
        b"QISKIT",
        3,
        version_parts[0],
        version_parts[1],
        version_parts[2],
        len(programs),
    )
    file_obj.write(file_header)

    for program in programs:
        if isinstance(program, QuantumCircuit):
            type_key = TypeKey.QUANTUM_CIRCUIT
            writer = None
        elif isinstance(program, ScheduleBlock):
            type_key = TypeKey.SCHEDULE_BLOCK
            writer = write_schedule_block
        else:
            raise TypeError(f"Program format {type(program)} is not supported.")
        file_obj.write(struct.pack("!1c", type_key.value.encode("utf8")))
        writer(file_obj, program)


def load(file_obj, verbose=False):
    file_header = struct.unpack(FILE_HEADER_PACK, file_obj.read(FILE_HEADER_SIZE))
    if file_header[0].decode("utf8") != "QISKIT":
        raise QiskitError("Input file is not a valid QPY file")
    qpy_version = file_header[1]
    version_parts = [int(x) for x in __version__.split(".")[0:3]]
    header_version_parts = [file_header[2], file_header[3], file_header[4]]
    if (
        verbose and (
            version_parts[0] < header_version_parts[0]
            or (
                version_parts[0] == header_version_parts[0]
                and header_version_parts[1] > version_parts[1]
            )
            or (
                version_parts[0] == header_version_parts[0]
                and header_version_parts[1] == version_parts[1]
                and header_version_parts[2] > version_parts[2]
            )
        )
    ):
        warnings.warn(
            "The qiskit version used to generate the provided QPY "
            "file, %s, is newer than the current qiskit version %s. "
            "This may result in an error if the QPY file uses "
            "instructions not present in this current qiskit "
            "version" % (".".join([str(x) for x in header_version_parts]), __version__)
        )

    programs = []
    for _ in range(file_header[5]):
        if qpy_version < 2:
            # Only quantum circuit is supported by QPY v < 2
            type_key = TypeKey.QUANTUM_CIRCUIT
        else:
            type_key_raw = struct.unpack("!1c", file_obj.read(struct.calcsize("!1c")))
            type_key = TypeKey(type_key_raw[0].decode("utf8"))

        if type_key == TypeKey.QUANTUM_CIRCUIT:
            loader = None
        elif type_key == TypeKey.SCHEDULE_BLOCK:
            loader = read_schedule_block
        else:
            raise TypeError(f"Invalid payload format data kind {type_key}")
        programs.append(loader(file_obj))

    return programs
