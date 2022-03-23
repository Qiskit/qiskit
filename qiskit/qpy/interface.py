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

import struct
import warnings

from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.qpy import formats, common, binary_io
from qiskit.version import __version__


def dump(circuits, file_obj, metadata_serializer=None):
    """Write QPY binary data to a file

    This function is used to save a circuit to a file for later use or transfer
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

    Args:
        circuits (list or QuantumCircuit): The quantum circuit object(s) to
            store in the specified file like object. This can either be a
            single QuantumCircuit object or a list of QuantumCircuits.
        file_obj (file): The file like object to write the QPY data too
        metadata_serializer (JSONEncoder): An optional JSONEncoder class that
            will be passed the :attr:`.QuantumCircuit.metadata` dictionary for
            each circuit in ``circuits`` and will be used as the ``cls`` kwarg
            on the ``json.dump()`` call to JSON serialize that dictionary.
    """
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]
    version_parts = [int(x) for x in __version__.split(".")[0:3]]
    header = struct.pack(
        formats.FILE_HEADER_PACK,
        b"QISKIT",
        common.QPY_VERSION,
        version_parts[0],
        version_parts[1],
        version_parts[2],
        len(circuits),
    )
    file_obj.write(header)
    for circuit in circuits:
        binary_io.write_circuit(file_obj, circuit, metadata_serializer=metadata_serializer)


def load(file_obj, metadata_deserializer=None):
    """Load a QPY binary file

    This function is used to load a serialized QPY circuit file and create
    :class:`~qiskit.circuit.QuantumCircuit` objects from its contents.
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
    :class:`~qiskit.circuit.QuantumCircuit` objects from the file.

    Args:
        file_obj (File): A file like object that contains the QPY binary
            data for a circuit
        metadata_deserializer (JSONDecoder): An optional JSONDecoder class
            that will be used for the ``cls`` kwarg on the internal
            ``json.load`` call used to deserialize the JSON payload used for
            the :attr:`.QuantumCircuit.metadata` attribute for any circuits
            in the QPY file. If this is not specified the circuit metadata will
            be parsed as JSON with the stdlib ``json.load()`` function using
            the default ``JSONDecoder`` class.
    Returns:
        list: List of ``QuantumCircuit``
            The list of :class:`~qiskit.circuit.QuantumCircuit` objects
            contained in the QPY data. A list is always returned, even if there
            is only 1 circuit in the QPY data.
    Raises:
        QiskitError: if ``file_obj`` is not a valid QPY file
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
    circuits = []
    for _ in range(data.num_circuits):
        circuits.append(
            binary_io.read_circuit(
                file_obj, data.qpy_version, metadata_deserializer=metadata_deserializer
            )
        )
    return circuits
