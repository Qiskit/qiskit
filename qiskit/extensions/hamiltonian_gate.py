# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Gate described by the time evolution of a Hermitian Hamiltonian operator.
"""

from numbers import Number
import numpy
import scipy.linalg

from qiskit.circuit import Gate, ParameterExpression
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.extensions.exceptions import ExtensionError

from .unitary import UnitaryGate


class HamiltonianGate(Gate):
    """Class for representing evolution by a Hermitian Hamiltonian operator as a gate. This gate
    resolves to a UnitaryGate, which can be decomposed into basis gates if it is 2 qubits or
    less, or simulated directly in Aer for more qubits. """

    def __init__(self, data, time, label=None):
        """Create a gate from a hamiltonian operator and evolution time parameter t

        Args:
            data (matrix or Operator): unitary operator.
            time (float or complex): time evolution parameter.
            label (str): unitary name for backend [Default: None].

        Raises:
            ExtensionError: if input data is not an N-qubit unitary operator.
        """
        if hasattr(data, 'to_matrix'):
            # If input is Gate subclass or some other class object that has
            # a to_matrix method this will call that method.
            data = data.to_matrix()
        elif hasattr(data, 'to_operator'):
            # If input is a BaseOperator subclass this attempts to convert
            # the object to an Operator so that we can extract the underlying
            # numpy matrix from `Operator.data`.
            data = data.to_operator().data
        # Convert to numpy array in case not already an array
        data = numpy.array(data, dtype=complex)
        # Check input is unitary
        if not is_hermitian_matrix(data):
            raise ExtensionError("Input matrix is not Hermitian.")
        # Check input is N-qubit matrix
        input_dim, output_dim = data.shape
        num_qubits = int(numpy.log2(input_dim))
        if input_dim != output_dim or 2**num_qubits != input_dim:
            raise ExtensionError(
                "Input matrix is not an N-qubit operator.")

        self._qasm_name = None
        self._qasm_definition = None
        self._qasm_def_written = False
        # Store instruction params
        super().__init__('hamiltonian', num_qubits, [data, time], label=label)

    def __eq__(self, other):
        if not isinstance(other, HamiltonianGate):
            return False
        if self.label != other.label:
            return False
        # Should we match unitaries as equal if they are equal
        # up to global phase?
        operators_eq = matrix_equal(self.params[0], other.params[0], ignore_phase=True)
        times_eq = self.params[1] == other.params[1]
        return operators_eq and times_eq

    def to_matrix(self):
        """Return matrix for the unitary."""
        unbound_t = False
        if isinstance(self.params[1], ParameterExpression):
            unbound_t = len(self.params[1].parameters) > 0
        if isinstance(self.params[1], Number) or not unbound_t:
            # pylint: disable=no-member
            return scipy.linalg.expm(1j * self.params[0] * self.params[1])
        else:
            raise NotImplementedError("Unable to generate Unitary matrix for "
                                      "unbound t parameter {}".format(self.params[1]))

    def inverse(self):
        """Return the adjoint of the unitary."""
        return self.adjoint()

    def conjugate(self):
        """Return the conjugate of the Hamiltonian."""
        return HamiltonianGate(numpy.transpose(self.params[0]), -self.params[1])

    def adjoint(self):
        """Return the adjoint of the unitary."""
        return HamiltonianGate(self.params[0], -self.params[1])

    def transpose(self):
        """Return the transpose of the Hamiltonian."""
        return HamiltonianGate(numpy.transpose(self.params[0]), self.params[1])

    def _define(self):
        """Calculate a subcircuit that implements this unitary."""
        return UnitaryGate(self.to_matrix())

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        r"""Return controlled version of gate, relying on control implementation in Unitary.

        Args:
            num_ctrl_qubits (int): number of controls to add to gate (default=1)
            label (str): optional gate label
            ctrl_state (int or str or None): The control state in decimal or as a
                bit string (e.g. '1011'). If None, use 2**num_ctrl_qubits-1.

        Returns:
            UnitaryGate: controlled version of gate.

        Raises:
            QiskitError: invalid ctrl_state
        """
        return self._define().control(num_ctrl_qubits=num_ctrl_qubits,
                                      label=label,
                                      ctrl_state=ctrl_state)

    def qasm(self):
        """ The qasm for the UnitaryGate defining this HamiltonianGate
        """
        return self._define().qasm()


def hamiltonian(self, matrix, time, qubits, label=None):
    """Apply u2 to q."""
    if isinstance(qubits, QuantumRegister):
        qubits = qubits[:]
    return self.append(HamiltonianGate(data=matrix, time=time, label=label), qubits, [])


QuantumCircuit.hamiltonian = hamiltonian
