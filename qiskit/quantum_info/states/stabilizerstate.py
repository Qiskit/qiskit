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

    def __init__(self, data):
        """Initialize a StabilizerState object.
        Args:
            data (StabilizerState or Clifford or Pauli or QuantumCircuit or
                  qiskit.circuit.Instruction):
                Data from which the stabilizer state can be constructed.

        Raises:
            QiskitError: if input data is not a valid stabilizer state (not a valid Clifford).
        """

        # Initialize from a Clifford
        if isinstance(data, Clifford):
            self._data = data

        # Initialize from a Pauli
        elif isinstance(data, Pauli):
            self._data = Clifford(data.to_instruction())

        # Initialize from another StabilizerState
        elif isinstance(data, StabilizerState):
            self._data = data._data

        # Initialize from a QuantumCircuit or Instruction object
        elif isinstance(data, (QuantumCircuit, Instruction)):
            self._data = Clifford.from_circuit(data)

        # Validate that Clifford table is a symplectic matrix
        if not self.is_valid():
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

    def transpose(self):
        """Return the transpose of the operator."""
        return StabilizerState(Clifford.transpose(self.data))

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

    def expand(self, other):
        """Return the tensor product stabilzier state other ⊗ self.

        Args:
            other (StabilizerState): a stabilizer state object.

        Returns:
            StabilizerState: the tensor product operator other ⊗ self.

        Raises:
            QiskitError: if other is not a StabilizerState.
        """
        if not isinstance(other, StabilizerState):
            other = StabilizerState(other)
        return StabilizerState((self.data).expand(other.data))

    def evolve(self, other, qargs=None):
        """Evolve a stabilizer state by a Clifford operator.

        Args:
            other (Clifford or QuantumCircuit or Instruction): The Clifford operator to evolve by.
            qargs (list): a list of stabilizer subsystem positions to apply the operator on.

        Returns:
            StabilizerState: the output stabilizer state.

        Raises:
            QiskitError: if other is not a StabilizerState.
            QiskitError: if the operator dimension does not match the
                         specified StabilizerState subsystem dimensions.
        """
        if qargs is None:
            qargs = getattr(other, 'qargs', None)

        if not isinstance(other, StabilizerState):
            other = StabilizerState(other)
        return StabilizerState((self.data).compose(other.data, qargs))


    def expectation_value(self, oper, qargs=None):
        """Compute the expectation value of an operator.

        Args:
            oper (BaseOperator): an operator to evaluate expval.
            qargs (None or list): subsystems to apply the operator on.

        Returns:
            complex: the expectation value.
        """
        pass

    def probabilities(self, qargs=None, decimals=None):
        """Return the subsystem measurement probability vector.

        Measurement probabilities are with respect to measurement in the
        computation (diagonal) basis.

        Args:
            qargs (None or list): subsystems to return probabilities for,
                if None return for all subsystems (Default: None).
            decimals (None or int): the number of decimal places to round
                values. If None no rounding is done (Default: None).

        Returns:
            np.array: The Numpy vector array of probabilities.
        """
        pass

    def reset(self, qargs=None):
        """Reset state or subsystems to the 0-state.

        Args:
            qargs (list or None): subsystems to reset, if None all
                                  subsystems will be reset to their 0-state
                                  (Default: None).

        Returns:
            StabilizerState: the reset state.

        Additional Information:
            If all subsystems are reset this will return the ground state
            on all subsystems. If only a some subsystems are reset this
            function will perform a measurement on those subsystems and
            evolve the subsystems so that the collapsed post-measurement
            states are rotated to the 0-state. The RNG seed for this
            sampling can be set using the :meth:`seed` method.
        """
        pass


    def measure(self, qargs=None):
        """Measure subsystems and return outcome and post-measure state.

        Note that this function uses the QuantumStates internal random
        number generator for sampling the measurement outcome. The RNG
        seed can be set using the :meth:`seed` method.

        Args:
            qargs (list or None): subsystems to sample measurements for,
                                  if None sample measurement of all
                                  subsystems (Default: None).

        Returns:
            tuple: the pair ``(outcome, state)`` where ``outcome`` is the
                   measurement outcome string label, and ``state`` is the
                   collapsed post-measurement stabilizer state for the
                   corresponding outcome.
        """
        pass
