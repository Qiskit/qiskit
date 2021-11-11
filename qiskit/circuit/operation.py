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

from abc import ABC
from qiskit.circuit.exceptions import CircuitError


class Operation(ABC):
    """Quantum Operation Mixin Class.
    For objects that can be added to a :class:`~qiskit.circuit.QuantumCircuit`.
    These objects include :class:`~qiskit.circuit.Gate`, :class:`~qiskit.circuit.Reset`,
    :class:`~qiskit.circuit.Barrier`, :class:`~qiskit.circuit.Measure`,
    and operators such as :class:`~qiskit.quantum_info.Clifford`.
    """

    # pylint: disable=unused-argument
    def __new__(cls, *args, **kwargs):
        if cls is Operation:
            raise CircuitError("An Operation mixin should not be instantiated directly.")
        return super().__new__(cls)

    def __init__(self, name, num_qubits, num_clbits, operands):
        self._name = name
        self._num_qubits = num_qubits
        self._num_clbits = num_clbits
        self._operands = operands

    @property
    def name(self):
        """Unique string identifier for operation type."""
        return self._name

    @property
    def num_qubits(self):
        """Number of qubits."""
        return self._num_qubits

    @property
    def num_clbits(self):
        """Number of classical bits."""
        return self._num_clbits

    @property
    def num_operands(self):
        """Number of operands."""
        return len(self._operands)

    @property
    def operands(self):
        """List of operands to specialize a specific Operation instance."""
        return self._operands
