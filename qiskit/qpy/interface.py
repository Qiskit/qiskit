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

"""User interface of qpy serializer."""

from json import JSONEncoder, JSONDecoder
from typing import Union, List, BinaryIO
from collections.abc import Iterable
import struct
import warnings

from qiskit.circuit import QuantumCircuit
from qiskit.pulse import ScheduleBlock
from qiskit.exceptions import QiskitError
from qiskit.qpy import formats, common, binary_io, type_keys
from qiskit.qpy.exceptions import QpyError
from qiskit.version import __version__


# pylint: disable=invalid-name
QPY_SUPPORTED_TYPES = Union[QuantumCircuit, ScheduleBlock]


def dump(
    programs: Union[List[QPY_SUPPORTED_TYPES], QPY_SUPPORTED_TYPES],
    file_obj: BinaryIO,
    metadata_serializer: JSONEncoder = None,
    circuits: QuantumCircuit = None,
):
    """Write QPY binary data to a file

    This function is used to save a circuit to a file for later use or transfer
    between machines. The QPY format is backwards compatible and can be
    loaded with future versions of Qiskit.

    For example:

    .. code-block:: python

        from qiskit.circuit import QuantumCircuit
        from qiskit import qpy

        qc = QuantumCircuit(2, name='Bell', metadata={'test': True})
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

    from this you can write the qpy data to a file:

    .. code-block:: python

        with open('bell.qpy', 'wb') as fd:
            qpy.dump(qc, fd)

    or a gzip compressed file:

    .. code-block:: python

        import gzip

        with gzip.open('bell.qpy.gz', 'wb') as fd:
            qpy.dump(qc, fd)

    Which will save the qpy serialized circuit to the provided file.

    Args:
        programs: QPY supported object(s) to store in the specified file like object.
            QPY supports :class:`.QuantumCircuit` and :class:`.ScheduleBlock`.
            Different data types must be separately serialized.
        file_obj: The file like object to write the QPY data too
        metadata_serializer: An optional JSONEncoder class that
            will be passed the ``.metadata`` attribute for each program in ``programs`` and will be
            used as the ``cls`` kwarg on the `json.dump()`` call to JSON serialize that dictionary.
        circuits: Deprecated. Use ``programs`` instead.

    Raises:
        QpyError: When multiple data format is mixed in the output.
        TypeError: When invalid data type is input.
    """
    if circuits is not None:
        warnings.warn(
            "'circuits' has been deprecated. Use 'programs' instead.",
            DeprecationWarning,
        )
        programs = circuits

    if not isinstance(programs, Iterable):
        programs = [programs]

    program_types = set()
    for program in programs:
        program_types.add(type(program))

    if len(program_types) > 1:
        raise QpyError(
            "Input programs contain multiple data types. "
            "Different data type must be serialized separately."
        )
    program_type = next(iter(program_types))

    if issubclass(program_type, QuantumCircuit):
        type_key = type_keys.Program.CIRCUIT
        writer = binary_io.write_circuit
    elif program_type is ScheduleBlock:
        type_key = type_keys.Program.SCHEDULE_BLOCK
        writer = binary_io.write_schedule_block
    else:
        raise TypeError(f"'{program_type}' is not supported data type.")

    version_parts = [int(x) for x in __version__.split(".")[0:3]]
    header = struct.pack(
        formats.FILE_HEADER_PACK,
        b"QISKIT",
        common.QPY_VERSION,
        version_parts[0],
        version_parts[1],
        version_parts[2],
        len(programs),
    )
    file_obj.write(header)
    common.write_type_key(file_obj, type_key)

    for program in programs:
        writer(file_obj, program, metadata_serializer=metadata_serializer)


def load(
    file_obj: BinaryIO,
    metadata_deserializer: JSONDecoder = None,
) -> List[QPY_SUPPORTED_TYPES]:
    """Load a QPY binary file

    This function is used to load a serialized QPY Qiskit program file and create
    :class:`~qiskit.circuit.QuantumCircuit` objects or
    :class:`~qiskit.pulse.schedule.ScheduleBlock` objects from its contents.
    For example:

    .. code-block:: python

        from qiskit import qpy

        with open('bell.qpy', 'rb') as fd:
            circuits = qpy.load(fd)

    or with a gzip compressed file:

    .. code-block:: python

        import gzip
        from qiskit import qpy

        with gzip.open('bell.qpy.gz', 'rb') as fd:
            circuits = qpy.load(fd)

    which will read the contents of the qpy and return a list of
    :class:`~qiskit.circuit.QuantumCircuit` objects or
    :class:`~qiskit.pulse.schedule.ScheduleBlock` objects from the file.

    Args:
        file_obj: A file like object that contains the QPY binary
            data for a circuit or pulse schedule.
        metadata_deserializer: An optional JSONDecoder class
            that will be used for the ``cls`` kwarg on the internal
            ``json.load`` call used to deserialize the JSON payload used for
            the ``.metadata`` attribute for any programs in the QPY file.
            If this is not specified the circuit metadata will
            be parsed as JSON with the stdlib ``json.load()`` function using
            the default ``JSONDecoder`` class.

    Returns:
        The list of Qiskit programs contained in the QPY data.
        A list is always returned, even if there is only 1 program in the QPY data.

    Raises:
        QiskitError: if ``file_obj`` is not a valid QPY file
        TypeError: When invalid data type is loaded.
    """
    data = formats.FILE_HEADER._make(
        struct.unpack(
            formats.FILE_HEADER_PACK,
            file_obj.read(formats.FILE_HEADER_SIZE),
        )
    )
    if data.preface.decode(common.ENCODE) != "QISKIT":
        raise QiskitError("Input file is not a valid QPY file")
    version_parts = [int(x) for x in __version__.split(".")[0:3]]
    header_version_parts = [data.major_version, data.minor_version, data.patch_version]

    # pylint: disable=too-many-boolean-expressions
    if (
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
            "version" % (".".join([str(x) for x in header_version_parts]), __version__)
        )

    if data.qpy_version < 5:
        type_key = type_keys.Program.CIRCUIT
    else:
        type_key = common.read_type_key(file_obj)

    if type_key == type_keys.Program.CIRCUIT:
        loader = binary_io.read_circuit
    elif type_key == type_keys.Program.SCHEDULE_BLOCK:
        loader = binary_io.read_schedule_block
    else:
        raise TypeError(f"Invalid payload format data kind '{type_key}'.")

    programs = []
    for _ in range(data.num_programs):
        programs.append(
            loader(file_obj, data.qpy_version, metadata_deserializer=metadata_deserializer)
        )
    return programs
