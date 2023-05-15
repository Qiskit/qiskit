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

"""Barrier instruction.

Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
with the :meth:`~qiskit.circuit.QuantumCircuit.barrier` method.
"""

from qiskit.exceptions import QiskitError
from .instruction import Instruction


class Barrier(Instruction):
    """Barrier instruction.

    A barrier is a visual indicator of the grouping of a circuit section.
    It also acts as a directive for circuit compilation to separate pieces
    of a circuit so that any optimizations or re-writes are constrained
    to only act between barriers."""

    _directive = True

    def __init__(self, num_qubits, label=None):
        """Create new barrier instruction.

        Args:
            num_qubits (int): the number of qubits for the barrier type [Default: 0].
            label (str): the barrier label

        Raises:
            TypeError: if barrier label is invalid.
        """
        self._label = label
        super().__init__("barrier", num_qubits, 0, [], label=label)

    def inverse(self):
        """Special case. Return self."""
        return Barrier(self.num_qubits)

    def broadcast_arguments(self, qargs, cargs):
        yield [qarg for sublist in qargs for qarg in sublist], []

    def c_if(self, classical, val):
        raise QiskitError("Barriers are compiler directives and cannot be conditional.")
