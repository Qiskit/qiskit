# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quantum Operation Mixin."""

from abc import ABC, abstractmethod


class Operation(ABC):
    """Quantum Operation Interface Class.
    For objects that can be added to a :class:`~qiskit.circuit.QuantumCircuit`.
    These objects include :class:`~qiskit.circuit.Gate`, :class:`~qiskit.circuit.Reset`,
    :class:`~qiskit.circuit.Barrier`, :class:`~qiskit.circuit.Measure`,
    and operators such as :class:`~qiskit.quantum_info.Clifford`.
    The main purpose is to add an :class:`~qiskit.circuit.Operation` to a
    :class:`~qiskit.circuit.QuantumCircuit` without synthesizing it before the transpilation.

    Example:

        Add a Clifford and a Toffoli gate to a QuantumCircuit.

        .. plot::
           :include-source:

           from qiskit import QuantumCircuit
           from qiskit.quantum_info import Clifford, random_clifford

           qc = QuantumCircuit(3)
           cliff = random_clifford(2)
           qc.append(cliff, [0, 1])
           qc.ccx(0, 1, 2)
           qc.draw('mpl')
    """

    __slots__ = ()

    @property
    @abstractmethod
    def name(self):
        """Unique string identifier for operation type."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_qubits(self):
        """Number of qubits."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_clbits(self):
        """Number of classical bits."""
        raise NotImplementedError
