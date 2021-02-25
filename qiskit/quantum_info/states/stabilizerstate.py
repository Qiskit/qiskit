# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Stabilizer state class.
"""

import copy
import re
from numbers import Number

import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info.operators.symplectic import Clifford, Pauli

class StabilizerState(QuantumState):
    """Statevector class"""

    def __init__(self, data, validate=True):
        """Initialize a StabilizerState object."""

        # Initialize from a Clifford
        if isinstance(data, Clifford):
            self._data = data

        # Initialize from a Pauli
        elif isinstance(data, Pauli):
            self._data = data

        # Initialize from another StabilizerState
        elif isinstance(data, StabilizerState):
            self._data = data._data

        # Initialize from a QuantumCircuit or Instruction object
        elif isinstance(data, (QuantumCircuit, Instruction)):
            self._data = Clifford.from_circuit(data)

        # Validate table is a symplectic matrix
        if validate and not self.is_valid():
            raise QiskitError(
                'Invalid StabilizerState. Input is not a valid Clifford.')

        # Initialize
        super().__init__()

    def __eq__(self, other):
        return (self._data == other._data)

    def __repr__(self):
        return 'StabilizerState({})'.format(repr(self._data))

    @property
    def data(self):
        """Return StabilizerState data"""
        return self._data

    def is_valid(self, atol=None, rtol=None):
        """Return True if a valid StabilizerState."""
        return Clifford.is_unitary(self.data)

    def to_operator(self):
        """Convert state to matrix operator class"""
        return Clifford(self.data).to_operator()

    def conjugate(self):
        """Return the conjugate of the operator."""
        return StabilizerState(Clifford.conjugate(self.data))

    def tensor(self, other):
        """Return the tensor product stabilzier state self ⊗ other.

        Args:
            other (StabilizerState): a stabilizer state object.

        Returns:
            StabilizerState: the tensor product operator self ⊗ other.

        Raises:
            QiskitError: if other is not a StabilizerState.
        """
        if not isinstance(other, StabilizerState):
            other = StabilizerState(other)
        return StabilizerState((self.data).tensor(other.data))

    def compose(self, other):
        """Return the composed operator of self and other.

        Args:
            other (StabilizerState): a stabilizer state object.

        Returns:
            StabilizerState: the composed stabilizer state of self and other.

        Raises:
            QiskitError: if other is not a StabilizerState.
        """
        if not isinstance(other, StabilizerState):
            other = StabilizerState(other)
        return StabilizerState((self.data).compose(other.data))
