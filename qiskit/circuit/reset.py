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

"""
Qubit reset to computational zero.
"""
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.operation import Operation


class Reset(Instruction, Operation):
    """Qubit reset."""

    def __init__(self):
        """Create new reset instruction."""
        self._name = "reset"
        self._num_qubits = 1
        self._num_clbits = 0
        self._params = []
        super().__init__(self._name, self._num_qubits, self._num_clbits, self._params)

    def broadcast_arguments(self, qargs, cargs):
        for qarg in qargs[0]:
            yield [qarg], []

    @property
    def name(self):
        """Return the name."""
        return self._name

    @name.setter
    def name(self, name):
        """Set the name."""
        pass

    @property
    def num_qubits(self):
        """Number of qubits."""
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, num_qubits):
        """Set num_qubits."""
        pass

    @property
    def num_clbits(self):
        """Number of classical bits."""
        return self._num_clbits

    @num_clbits.setter
    def num_clbits(self, num_clbits):
        """Set num_clbits."""
        pass

    @property
    def num_params(self):
        """Number of parameters."""
        return len(self._params)
