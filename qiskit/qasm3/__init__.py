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

"""
==========================
Qasm (:mod:`qiskit.qasm3`)
==========================

.. currentmodule:: qiskit.qasm3

.. autosummary::
    :toctree: ../stubs/

    Exporter
    dumps
    dump
"""

from .exporter import Exporter
from .exceptions import QASM3Error, QASM3ExporterError


def dumps(circuit, **kwargs) -> str:
    """Serialize a :class:`~qiskit.circuit.QuantumCircuit` object in an OpenQASM3 string.

    .. note::

        This is a quick interface to the main :obj:`.Exporter` interface.  All keyword arguments to
        this function are inherited from the constructor of that class, and if you have multiple
        circuits to export, it will be faster to create an :obj:`.Exporter` instance, and use its
        :obj:`.Exporter.dumps` method.

    Args:
        circuit (QuantumCircuit): Circuit to serialize.
        **kwargs: Arguments for the :obj:`.Exporter` constructor.

    Returns:
        str: The OpenQASM3 serialization
    """
    return Exporter(**kwargs).dumps(circuit)


def dump(circuit, stream, **kwargs) -> None:
    """Serialize a :class:`~qiskit.circuit.QuantumCircuit` object as a OpenQASM3 stream to file-like
    object.

    .. note::

        This is a quick interface to the main :obj:`.Exporter` interface.  All keyword arguments to
        this function are inherited from the constructor of that class, and if you have multiple
        circuits to export, it will be faster to create an :obj:`.Exporter` instance, and use its
        :obj:`.Exporter.dump` method.

    Args:
        circuit (QuantumCircuit): Circuit to serialize.
        stream (TextIOBase): stream-like object to dump the OpenQASM3 serialization
        **kwargs: Arguments for the :obj:`.Exporter` constructor.

    """
    Exporter(**kwargs).dump(circuit, stream)
