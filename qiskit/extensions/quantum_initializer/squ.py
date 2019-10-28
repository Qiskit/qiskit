# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name
# pylint: disable=missing-param-doc
# pylint: disable=missing-type-doc

"""
Decompose an arbitrary 2*2 unitary into three rotation gates: U=R_zR_yR_z.
Note that the decomposition is up to a global phase shift.
(This is a well known decomposition, which can be found for example in Nielsen and Chuang's book
"Quantum computation and quantum information".)
"""

import cmath

import numpy as np

from qiskit.circuit import Gate
from qiskit.circuit.quantumcircuit import QuantumRegister, QuantumCircuit, Qubit
from qiskit.quantum_info.operators.predicates import is_unitary_matrix
from qiskit.exceptions import QiskitError

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class SingleQubitUnitary(Gate):
    """
    u = 2*2 unitary (given as a (complex) numpy.ndarray)

    mode - determines the used decomposition by providing the rotation axes

    up_to_diagonal - the single-qubit unitary is decomposed up to a diagonal matrix,
                     i.e. a unitary u' is implemented such that there exists a 2*2 diagonal
                     gate d with u = d.dot(u').
    """

    def __init__(self, u, mode="ZYZ", up_to_diagonal=False):
        if mode != "ZYZ":
            raise QiskitError("The decomposition mode is not known.")
        # Check if the matrix u has the right dimensions and if it is a unitary
        if not u.shape == (2, 2):
            raise QiskitError("The dimension of the input matrix is not equal to (2,2).")
        if not is_unitary_matrix(u):
            raise QiskitError("The 2*2 matrix is not unitary.")
        self.mode = mode
        self.up_to_diagonal = up_to_diagonal
        # Create new gate
        super().__init__("unitary", 1, [u])

    # Returns the diagonal gate D up to which the single-qubit unitary u is implemented,
    # i.e., u=D.u', where u' is the unitary implemented by the found circuit.
    def _get_diag(self):
        _, diag = self._dec_single_qubit_unitary()
        return diag

    def _define(self):
        squ_circuit, _ = self._dec_single_qubit_unitary()
        gate = squ_circuit.to_instruction()
        q = QuantumRegister(self.num_qubits)
        squ_circuit = QuantumCircuit(q)
        squ_circuit.append(gate, q[:])
        self.definition = squ_circuit.data

    def _dec_single_qubit_unitary(self):
        """
        Call to create a circuit with gates that implement the single qubit unitary u.
        Returns: QuantumCircuit: circuit for implementing u
                 (up to a diagonal if up_to_diagonal=True)
        """
        diag = [1., 1.]
        q = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(q)
        # First, we find the rotation angles (where we can ignore the global phase)
        (a, b, c, _) = self._zyz_dec()
        # Add the gates to o the circuit
        is_identity = True
        if abs(a) > _EPS:
            circuit.rz(a, q[0])
            is_identity = False
        if abs(b) > _EPS:
            circuit.ry(b, q[0])
            is_identity = False
        if abs(c) > _EPS:
            if self.up_to_diagonal:
                diag = [np.exp(-1j * c / 2.), np.exp(1j * c / 2.)]
            else:
                circuit.rz(c, q[0])
                is_identity = False
        if is_identity:
            circuit.iden(q[0])
        return circuit, diag

    def _zyz_dec(self):
        """
        This method finds rotation angles (a,b,c,d) in the decomposition
        u=exp(id)*Rz(c).Ry(b).Rz(a)
        (where "." denotes matrix multiplication)
        """
        u = self.params[0]
        u00 = u.item(0, 0)
        u01 = u.item(0, 1)
        u10 = u.item(1, 0)
        u11 = u.item(1, 1)
        # Handle special case if the entry (0,0) of the unitary is equal to zero
        if np.abs(u00) < _EPS:
            # Note that u10 can't be zero, since u is unitary (and u00 == 0)
            c = cmath.phase(-u01 / u10)
            d = cmath.phase(u01 * np.exp(-1j * c / 2))
            return 0., -np.pi, -c, d
        # Handle special case if the entry (0,1) of the unitary is equal to zero
        if np.abs(u01) < _EPS:
            # Note that u11 can't be zero, since u is unitary (and u01 == 0)
            c = cmath.phase(u00 / u11)
            d = cmath.phase(u00 * np.exp(-1j * c / 2))
            return 0., 0., -c, d
        b = 2 * np.arccos(np.abs(u00))
        if np.sin(b / 2) - np.cos(b / 2) > 0:
            c = cmath.phase(-u00 / u10)
            a = cmath.phase(u00 / u01)
        else:
            c = -cmath.phase(-u10 / u00)
            a = -cmath.phase(u01 / u00)
        d = cmath.phase(u00 * np.exp(-1j * (a + c) / 2))
        # The decomposition works with another convention for the rotation gates
        # (the one using negative angles).
        # Therefore, we have to take the inverse of the angles at the end.
        return -a, -b, -c, d


def squ(self, u, qubit, mode="ZYZ", up_to_diagonal=False):
    """ Decompose an arbitrary 2*2 unitary into three rotation gates :math:`U=R_zR_yR_z`.

    Note that the decomposition is up to a global phase shift.

    (This is a well known decomposition, which can be found for example in Nielsen and Chuang's book
    "Quantum computation and quantum information".)

    Args:
        u (ndarray): 2*2 unitary (given as a (complex) ndarray)
        qubit (QuantumRegister|Qubit): the qubit, on which the gate is acting on
        mode (string): determines the used decomposition by providing the rotation axes.
            The allowed modes are: "ZYZ" (default)
        up_to_diagonal (bool):  if set to True, the single-qubit unitary is decomposed up to
            a diagonal matrix, i.e. a unitary u' is implemented such that there exists a 2*2
            diagonal gate d with u = d.dot(u')
    Returns:
        QuantumCircuit: the single-qubit unitary (up to a diagonal gate if
        up_to_diagonal = True) is attached to the circuit.

    Raises:
        QiskitError: if the format is wrong; if the array u is not unitary
    """

    if isinstance(qubit, QuantumRegister):
        qubit = qubit[:]
        if len(qubit) == 1:
            qubit = qubit[0]
        else:
            raise QiskitError("The target qubit is a QuantumRegister containing more than"
                              " one qubits.")
    # Check if there is one target qubit provided
    if not isinstance(qubit, Qubit):
        raise QiskitError("The target qubit is not a single qubit from a QuantumRegister.")
    return self.append(SingleQubitUnitary(u, mode, up_to_diagonal), [qubit], [])


QuantumCircuit.squ = squ
