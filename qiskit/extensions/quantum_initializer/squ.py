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

"""Decompose an arbitrary 2*2 unitary into three rotation gates: U=R_zR_yR_z.

Note that the decomposition is up to a global phase shift.
(This is a well known decomposition, which can be found for example in Nielsen and Chuang's book
"Quantum computation and quantum information".)
"""

import cmath

import numpy as np

from qiskit.circuit import QuantumRegister, Qubit, QuantumCircuit
from qiskit.circuit.gate import Gate
from qiskit.quantum_info.operators.predicates import is_unitary_matrix
from qiskit.exceptions import QiskitError
from qiskit.util import deprecate_arguments

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class SingleQubitUnitary(Gate):
    """
    u = 2*2 unitary (given as a (complex) numpy.ndarray)

    mode - determines the used decomposition by providing the rotation axes

    up_to_diagonal - the single-qubit unitary is decomposed up to a diagonal matrix,
                     i.e. a unitary u' is implemented such that there exists a 2*2 diagonal
                     gate d with u = d.dot(u').
    """

    # pylint: disable=unused-argument, invalid-name
    @deprecate_arguments({'u': 'unitary'})
    def __init__(self, unitary_matrix, mode='ZYZ', up_to_diagonal=False, u=None):
        """Create a new single qubit gate based on the unitary ``u``."""
        if mode not in ['ZYZ']:
            raise QiskitError("The decomposition mode is not known.")
        # Check if the matrix u has the right dimensions and if it is a unitary
        if unitary_matrix.shape != (2, 2):
            raise QiskitError("The dimension of the input matrix is not equal to (2,2).")
        if not is_unitary_matrix(unitary_matrix):
            raise QiskitError("The 2*2 matrix is not unitary.")

        self.mode = mode
        self.up_to_diagonal = up_to_diagonal
        self._diag = None

        # Create new gate
        super().__init__("unitary", 1, [unitary_matrix])

    @property
    def diag(self):
        """Returns the diagonal gate D up to which the single-qubit unitary u is implemented.

        I.e. u=D.u', where u' is the unitary implemented by the found circuit.
        """
        if self._diag is None:
            self._define()
        return self._diag

    def _define(self):
        """Define the gate using the decomposition."""

        if self.mode == 'ZYZ':
            circuit, diag = self._zyz_circuit()
        else:
            raise QiskitError('The decomposition mode is not known.')

        self._diag = diag

        self.definition = circuit

    def _zyz_circuit(self):
        """Get the circuit for the ZYZ decomposition."""
        q = QuantumRegister(self.num_qubits)
        qc = QuantumCircuit(q, name=self.name)

        diag = [1., 1.]
        alpha, beta, gamma, _ = self._zyz_dec()

        if abs(alpha) > _EPS:
            qc.rz(alpha, q[0])
        if abs(beta) > _EPS:
            qc.ry(beta, q[0])
        if abs(gamma) > _EPS:
            if self.up_to_diagonal:
                diag = [np.exp(-1j * gamma / 2.), np.exp(1j * gamma / 2.)]
            else:
                qc.rz(gamma, q[0])

        return qc, diag

    def _zyz_dec(self):
        """Finds rotation angles (a,b,c,d) in the decomposition u=exp(id)*Rz(c).Ry(b).Rz(a).

        Note that where "." denotes matrix multiplication.
        """
        unitary = self.params[0]
        u00 = unitary.item(0, 0)
        u01 = unitary.item(0, 1)
        u10 = unitary.item(1, 0)
        u11 = unitary.item(1, 1)
        # Handle special case if the entry (0,0) of the unitary is equal to zero
        if np.abs(u00) < _EPS:
            # Note that u10 can't be zero, since u is unitary (and u00 == 0)
            gamma = cmath.phase(-u01 / u10)
            delta = cmath.phase(u01 * np.exp(-1j * gamma / 2))
            return 0., -np.pi, -gamma, delta
        # Handle special case if the entry (0,1) of the unitary is equal to zero
        if np.abs(u01) < _EPS:
            # Note that u11 can't be zero, since u is unitary (and u01 == 0)
            gamma = cmath.phase(u00 / u11)
            delta = cmath.phase(u00 * np.exp(-1j * gamma / 2))
            return 0., 0., -gamma, delta
        beta = 2 * np.arccos(np.abs(u00))
        if np.sin(beta / 2) - np.cos(beta / 2) > 0:
            gamma = cmath.phase(-u00 / u10)
            alpha = cmath.phase(u00 / u01)
        else:
            gamma = -cmath.phase(-u10 / u00)
            alpha = -cmath.phase(u01 / u00)
        delta = cmath.phase(u00 * np.exp(-1j * (alpha + gamma) / 2))
        # The decomposition works with another convention for the rotation gates
        # (the one using negative angles).
        # Therefore, we have to take the inverse of the angles at the end.
        return -alpha, -beta, -gamma, delta


# pylint: disable=unused-argument, invalid-name, missing-type-doc, missing-param-doc
@deprecate_arguments({'u': 'unitary'})
def squ(self, unitary_matrix, qubit, mode='ZYZ', up_to_diagonal=False, *, u=None):
    """Decompose an arbitrary 2*2 unitary into three rotation gates.

    Note that the decomposition is up to a global phase shift.
    (This is a well known decomposition, which can be found for example in Nielsen and Chuang's book
    "Quantum computation and quantum information".)

    Args:
        unitary_matrix (ndarray): 2*2 unitary (given as a (complex) ndarray).
        qubit (QuantumRegister | Qubit): The qubit which the gate is acting on.
        mode (string): determines the used decomposition by providing the rotation axes.
            The allowed modes are: "ZYZ" (default)
        up_to_diagonal (bool):  if set to True, the single-qubit unitary is decomposed up to
            a diagonal matrix, i.e. a unitary u' is implemented such that there exists a 2*2
            diagonal gate d with u = d.dot(u')
        u (ndarray): Deprecated, use ``unitary_matrix`` instead.

    Returns:
        InstructionSet: The single-qubit unitary instruction attached to the circuit.

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
    return self.append(SingleQubitUnitary(unitary_matrix, mode, up_to_diagonal), [qubit], [])


QuantumCircuit.squ = squ
