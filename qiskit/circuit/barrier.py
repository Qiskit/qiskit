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

"""Barrier instruction."""

from qiskit.exceptions import QiskitError
from .instruction import Instruction
from .operation import Operation


class Barrier(Instruction, Operation):
    """Barrier instruction."""

    _directive = True

    def __init__(self, num_qubits):
        """Create new barrier instruction."""
        self._name = "barrier"
        self._num_qubits = num_qubits
        self._num_clbits = 0
        self._params = []
        super().__init__(self._name, self._num_qubits, self._num_clbits, self._params)

    def inverse(self):
        """Special case. Return self."""
        return Barrier(self.num_qubits)

    def broadcast_arguments(self, qargs, cargs):
        yield [qarg for sublist in qargs for qarg in sublist], []

    def c_if(self, classical, val):
        raise QiskitError("Barriers are compiler directives and cannot be conditional.")

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
