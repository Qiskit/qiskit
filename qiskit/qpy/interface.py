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

"""User interface of qpy serializer."""

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
    """Write QPY binary data to a file

    This function is used to save a Qiskit program to a file for later use or transfer
    between machines. The QPY format is backwards compatible and can be
    loaded with future versions of Qiskit.

    For example:

    .. code-block:: python

        from qiskit.circuit import QuantumCircuit
        from qiskit.circuit import qpy_serialization

        qc = QuantumCircuit(2, name='Bell', metadata={'test': True})
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

    from this you can write the qpy data to a file:

    .. code-block:: python

        with open('bell.qpy', 'wb') as fd:
            qpy_serialization.dump(qc, fd)

    or a gzip compressed file:

    .. code-block:: python

        import gzip

        with gzip.open('bell.qpy.gz', 'wb') as fd:
            qpy_serialization.dump(qc, fd)

    Which will save the qpy serialized circuit to the provided file.
    You can also write pulse programs to a file in the samy way.

    Args:
        programs (list or QuantumCircuit or ScheduleBlock):
            The quantum circuit object(s) or ScheduleBlock object(s) to
            store in the specified file like object. This can either be a
            single program object or a list of programs.
        file_obj (file): The file like object to write the QPY data to

    Raises:
        TypeError: if any of the entries is not supported data format.
    """
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
        file_obj.write(struct.pack("!1c", type_key.encode("utf8")))
        writer(file_obj, program)


def load(file_obj, verbose=False):
    """Load a QPY binary file

    This function is used to load a serialized QPY Qiskit program file and create
    :class:`~qiskit.circuit.QuantumCircuit` objects or
    :class:`~qiskit.pulse.schedule.ScheduleBlock` objects from its contents.
    For example:

    .. code-block:: python

        from qiskit.circuit import qpy_serialization

        with open('bell.qpy', 'rb') as fd:
            circuits = qpy_serialization.load(fd)

    or with a gzip compressed file:

    .. code-block:: python

        import gzip
        from qiskit.circuit import qpy_serialization

        with gzip.open('bell.qpy.gz', 'rb') as fd:
            circuits = qpy_serialization.load(fd)

    which will read the contents of the qpy and return a list of
    :class:`~qiskit.circuit.QuantumCircuit` objects or
    :class:`~qiskit.pulse.schedule.ScheduleBlock` objects from the file.

    Args:
        file_obj (File): A file like object that contains the QPY binary
            data for a circuit or pulse schedule
        verbose (bool): Raise a user warning if the Qiskit version in the QPY file and
            the currently loaded package are different.
    Returns:
        list: List of ``QuantumCircuit`` or ``ScheduleBlock``
            The list of :class:`~qiskit.circuit.QuantumCircuit` objects
            or :class:`~qiskit.pulse.schedule.ScheduleBlock` objects
            contained in the QPY data. A list is always returned, even if there
            is only 1 program in the QPY data.
    Raises:
        QiskitError: if ``file_obj`` is not a valid QPY file
        TypeError: if any of the entries is not supported data format.
    """
    file_header = struct.unpack(FILE_HEADER_PACK, file_obj.read(FILE_HEADER_SIZE))
    if file_header[0].decode("utf8") != "QISKIT":
        raise QiskitError("Input file is not a valid QPY file")
    qpy_version = file_header[1]
    version_parts = [int(x) for x in __version__.split(".")[0:3]]
    header_version_parts = [file_header[2], file_header[3], file_header[4]]

    # pylint: disable=too-many-boolean-expressions
    if verbose and (
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
    ):
        warnings.warn(
            "The qiskit version used to generate the provided QPY "
            "file, %s, is newer than the current qiskit version %s. "
            "This may result in an error if the QPY file uses "
            "instructions not present in this current qiskit "
            "version" % (".".join([str(x) for x in header_version_parts]), __version__),
            UserWarning,
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
