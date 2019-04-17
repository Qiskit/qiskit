# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.ß

"""
Decompose an arbitrary 2*2 unitary into three rotation gates: U=R_zR_yR_z. Note that the decomposition is up to
a global phase shift.
(This is a well known decomposition, which can be found for example in Nielsen and Chuang's book
"Quantum computation and quantum information".)
# ToDo: Add decomposition mode XYX, which decomposes U into U=R_xR_yR_x
"""

import cmath

import numpy as np

from qiskit.circuit import Gate
from qiskit.circuit.quantumcircuit import QuantumRegister, QuantumCircuit
from qiskit.exceptions import QiskitError

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class SingleQubitUnitary(Gate):
    """
    u = 2*2 unitary (given as a (complex) numpy.ndarray)
    mode - determines the used decomposition by providing the rotation axes
    up_to_diagonal - the single-qubit unitary is decomposed up to a diagonal matrix, i.e. a unitary u' is implemented
                     such that there exists a 2*2 diagonal gate d with u = d.dot(u').
    """

    def __init__(self, u, mode="ZYZ", up_to_diagonal=False):
        if not mode == "ZYZ":
            raise QiskitError("The decomposition mode is not known.")
        # Check if the matrix u has the right dimensions and if it is a unitary
        if not u.shape == (2, 2):
            raise QiskitError("The dimension of the input matrix is not equal to (2,2).")
        if not is_isometry(u):
            raise QiskitError("The 2*2 matrix is not unitary.")
        self.mode = mode
        self.up_to_diagonal = up_to_diagonal
        # Create new gate
        super().__init__("unitary", 1, [u])

    # Returns the diagonal gate D up to which the single-qubit unitary u is implemented, i.e., u=D.u', where u' is
    # the unitary implemented by the found circuit.
    def get_diag(self):
        _, diag = self._dec_single_qubit_unitary()
        return diag

    def _define(self):
        squ_circuit,_ = self._dec_single_qubit_unitary()
        gate = squ_circuit.to_instruction()
        q = QuantumRegister(self.num_qubits)
        squ_circuit = QuantumCircuit(q)
        squ_circuit.append(gate, q[:])
        self.definition = squ_circuit.data

    def _dec_single_qubit_unitary(self):
        """
        Call to create a circuit with gates that implement the single qubit unitary u.
        Returns: QuantumCircuit: circuit for implementing u (up to a diagonal if up_to_diagonal=True)
        """
        diag = [1., 1.]
        q = QuantumRegister(self.num_qubits)
        circuit = QuantumCircuit(q)
        # First, we find the rotation angles (where we can ignore the global phase)
        (a, b, c, _) = self._zyz_dec()
        # Add the gates to o the circuit
        is_identity = True
        if abs(a) > _EPS:
            circuit.rz(a,q[0])
            is_identity = False
        if abs(b) > _EPS:
            circuit.ry(b,q[0])
            is_identity = False
        if abs(c) > _EPS:
            if self.up_to_diagonal:
                diag = [np.exp(-1j * c / 2.), np.exp(1j * c / 2.)]
            else:
                circuit.rz(c,q[0])
                is_identity = False
        if is_identity:
            circuit.iden(q[0])
        return circuit, diag

    def _zyz_dec(self):
        """
        This method finds rotation angles (a,b,c,d) in the decomposition u=exp(id)*Rz(c).Ry(b).Rz(a) (where "." denotes
        matrix multiplication)
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
        if 0 < np.sin(b / 2) - np.cos(b / 2):
            c = cmath.phase(-u00 / u10)
            a = cmath.phase(u00 / u01)
        else:
            c = -cmath.phase(-u10 / u00)
            a = -cmath.phase(u01 / u00)
        d = cmath.phase(u00 * np.exp(-1j * (a + c) / 2))
        # The decomposition works with another convention for the rotation gates (the one using negative angles).
        # Therefore, we have to take the inverse of the angles at the end.
        return -a, -b, -c, d


def ct(m):
    return np.transpose(np.conjugate(m))


def is_isometry(m):
    return np.allclose(ct(m).dot(m), np.eye(m.shape[1], m.shape[1]), atol=_EPS)


def squ(self, params, qubit, mode="ZYZ", up_to_diagonal=False):
    if isinstance(qubit, QuantumRegister):
        qubit = qubit[:]
        if len(qubit) == 1:
            qubit = qubit[0]
        else:
            raise QiskitError("The target qubit is a QuantumRegister containing more than one qubits.")
    # Check if there is one target qubit provided
    if not (type(qubit) == tuple and type(qubit[0]) == QuantumRegister):
        raise QiskitError("The target qubit is not a single qubit from a QuantumRegister.")
    return self.append(SingleQubitUnitary(params, mode, up_to_diagonal), [qubit], [])


QuantumCircuit.squ = squ

