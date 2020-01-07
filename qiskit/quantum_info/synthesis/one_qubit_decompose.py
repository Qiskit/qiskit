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

# pylint: disable=invalid-name
"""
Decompose single-qubit unitary into Euler angles.
"""

import math
import numpy as np
import scipy.linalg as la

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.extensions.standard import HGate, U3Gate, U1Gate, RXGate, RYGate, RZGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.operators.predicates import is_unitary_matrix

DEFAULT_ATOL = 1e-12


class OneQubitEulerDecomposer:
    """A class for decomposing 1-qubit unitaries into Euler angle rotations.

    Allowed basis and their decompositions are:
        U3: U -> exp(1j*phase) * U3(theta, phi, lam)
        U1X: U -> exp(1j*phase) * U1(lam).RX(pi/2).U1(theta+pi).RX(pi/2).U1(phi+pi)
        ZYZ: U -> exp(1j*phase) * RZ(phi).RY(theta).RZ(lam)
        ZXZ: U -> exp(1j*phase) * RZ(phi).RX(theta).RZ(lam)
        XYX: U -> exp(1j*phase) * RX(phi).RY(theta).RX(lam)
    """
    def __init__(self, basis='U3'):
        if basis not in ['U3', 'U1X', 'ZYZ', 'ZXZ', 'XYX']:
            raise QiskitError("OneQubitEulerDecomposer: unsupported basis")
        self._basis = basis

    def __call__(self, unitary_mat, simplify=True, atol=DEFAULT_ATOL):
        """Decompose single qubit gate into a circuit.

        Args:
            unitary_mat (array_like): 1-qubit unitary matrix
            simplify (bool): remove zero-angle rotations [Default: True]
            atol (float): absolute tolerance for checking angles zero.

        Returns:
            QuantumCircuit: the decomposed single-qubit gate circuit

        Raises:
            QiskitError: if input is invalid or synthesis fails.
        """
        if hasattr(unitary_mat, 'to_operator'):
            # If input is a BaseOperator subclass this attempts to convert
            # the object to an Operator so that we can extract the underlying
            # numpy matrix from `Operator.data`.
            unitary_mat = unitary_mat.to_operator().data
        if hasattr(unitary_mat, 'to_matrix'):
            # If input is Gate subclass or some other class object that has
            # a to_matrix method this will call that method.
            unitary_mat = unitary_mat.to_matrix()
        # Convert to numpy array incase not already an array
        unitary_mat = np.asarray(unitary_mat, dtype=complex)

        # Check input is a 2-qubit unitary
        if unitary_mat.shape != (2, 2):
            raise QiskitError("OneQubitEulerDecomposer: "
                              "expected 2x2 input matrix")
        if not is_unitary_matrix(unitary_mat):
            raise QiskitError("OneQubitEulerDecomposer: "
                              "input matrix is not unitary.")
        circuit = self._circuit(unitary_mat, simplify=simplify, atol=atol)
        # Check circuit is correct
        if not Operator(circuit).equiv(Operator(unitary_mat)):
            raise QiskitError("OneQubitEulerDecomposer: "
                              "synthesis failed within required accuracy.")
        return circuit

    def _angles(self, unitary_mat):
        """Return Euler angles for given basis."""
        if self._basis in ['U3', 'U1X', 'ZYZ']:
            return self._angles_zyz(unitary_mat)
        if self._basis == 'ZXZ':
            return self._angles_zxz(unitary_mat)
        if self._basis == 'XYX':
            return self._angles_xyx(unitary_mat)
        raise QiskitError("OneQubitEulerDecomposer: invalid basis")

    def _circuit(self, unitary_mat, simplify=True, atol=DEFAULT_ATOL):
        # NOTE: The 4th variable is phase to be used later
        theta, phi, lam, _ = self._angles(unitary_mat)
        if self._basis == 'U3':
            return self._circuit_u3(theta, phi, lam)
        if self._basis == 'U1X':
            return self._circuit_u1x(theta,
                                     phi,
                                     lam,
                                     simplify=simplify,
                                     atol=atol)
        if self._basis == 'ZYZ':
            return self._circuit_zyz(theta,
                                     phi,
                                     lam,
                                     simplify=simplify,
                                     atol=atol)
        if self._basis == 'ZXZ':
            return self._circuit_zxz(theta,
                                     phi,
                                     lam,
                                     simplify=simplify,
                                     atol=atol)
        if self._basis == 'XYX':
            return self._circuit_xyx(theta,
                                     phi,
                                     lam,
                                     simplify=simplify,
                                     atol=atol)
        raise QiskitError("OneQubitEulerDecomposer: invalid basis")

    @staticmethod
    def _angles_zyz(unitary_mat):
        """Return euler angles for special unitary matrix in ZYZ basis.

        In this representation U = exp(1j * phase) * Rz(phi).Ry(theta).Rz(lam)
        """
        # We rescale the input matrix to be special unitary (det(U) = 1)
        # This ensures that the quaternion representation is real
        coeff = la.det(unitary_mat)**(-0.5)
        phase = -np.angle(coeff)
        U = coeff * unitary_mat  # U in SU(2)
        # OpenQASM SU(2) parameterization:
        # U[0, 0] = exp(-i(phi+lambda)/2) * cos(theta/2)
        # U[0, 1] = -exp(-i(phi-lambda)/2) * sin(theta/2)
        # U[1, 0] = exp(i(phi-lambda)/2) * sin(theta/2)
        # U[1, 1] = exp(i(phi+lambda)/2) * cos(theta/2)
        theta = 2 * math.atan2(abs(U[1, 0]), abs(U[0, 0]))
        phiplambda = 2 * np.angle(U[1, 1])
        phimlambda = 2 * np.angle(U[1, 0])
        phi = (phiplambda + phimlambda) / 2.0
        lam = (phiplambda - phimlambda) / 2.0
        return theta, phi, lam, phase

    @staticmethod
    def _angles_zxz(unitary_mat):
        """Return euler angles for special unitary matrix in ZXZ basis.

        In this representation U = exp(1j * phase) * Rz(phi).Rx(theta).Rz(lam)
        """
        theta, phi, lam, phase = OneQubitEulerDecomposer._angles_zyz(unitary_mat)
        return theta, phi + np.pi / 2, lam - np.pi / 2, phase

    @staticmethod
    def _angles_xyx(unitary_mat):
        """Return euler angles for special unitary matrix in XYX basis.

        In this representation U = exp(1j * phase) * Rx(phi).Ry(theta).Rx(lam)
        """
        # We use the fact that
        # Rx(a).Ry(b).Rx(c) = H.Rz(a).Ry(-b).Rz(c).H
        had = HGate().to_matrix()
        mat_zyz = np.dot(np.dot(had, unitary_mat), had)
        theta, phi, lam, phase = OneQubitEulerDecomposer._angles_zyz(mat_zyz)
        return -theta, phi, lam, phase

    @staticmethod
    def _circuit_u3(theta, phi, lam):
        circuit = QuantumCircuit(1)
        circuit.append(U3Gate(theta, phi, lam), [0])
        return circuit

    @staticmethod
    def _circuit_u1x(theta, phi, lam, simplify=True, atol=DEFAULT_ATOL):
        # Check for U1 and U2 decompositions into minimimal
        # required X90 pulses
        if simplify and np.allclose([theta, phi], [0., 0.], atol=atol):
            # zero X90 gate decomposition
            circuit = QuantumCircuit(1)
            circuit.append(U1Gate(lam), [0])
            return circuit
        if simplify and np.isclose(theta, np.pi / 2, atol=atol):
            # single X90 gate decomposition
            circuit = QuantumCircuit(1)
            circuit.append(U1Gate(lam - np.pi / 2), [0])
            circuit.append(RXGate(np.pi / 2), [0])
            circuit.append(U1Gate(phi + np.pi / 2), [0])
            return circuit
        # General two-X90 gate decomposition
        circuit = QuantumCircuit(1)
        circuit.append(U1Gate(lam), [0])
        circuit.append(RXGate(np.pi / 2), [0])
        circuit.append(U1Gate(theta + np.pi), [0])
        circuit.append(RXGate(np.pi / 2), [0])
        circuit.append(U1Gate(phi + np.pi), [0])
        return circuit

    @staticmethod
    def _circuit_zyz(theta, phi, lam, simplify=True, atol=DEFAULT_ATOL):
        circuit = QuantumCircuit(1)
        if simplify and np.isclose(theta, 0.0, atol=atol):
            circuit.append(RZGate(phi + lam), [0])
            return circuit
        if not simplify or not np.isclose(lam, 0.0, atol=atol):
            circuit.append(RZGate(lam), [0])
        if not simplify or not np.isclose(theta, 0.0, atol=atol):
            circuit.append(RYGate(theta), [0])
        if not simplify or not np.isclose(phi, 0.0, atol=atol):
            circuit.append(RZGate(phi), [0])
        return circuit

    @staticmethod
    def _circuit_zxz(theta, phi, lam, simplify=False, atol=DEFAULT_ATOL):
        if simplify and np.isclose(theta, 0.0, atol=atol):
            circuit = QuantumCircuit(1)
            circuit.append(RZGate(phi + lam), [0])
            return circuit
        circuit = QuantumCircuit(1)
        if not simplify or not np.isclose(lam, 0.0, atol=atol):
            circuit.append(RZGate(lam), [0])
        if not simplify or not np.isclose(theta, 0.0, atol=atol):
            circuit.append(RXGate(theta), [0])
        if not simplify or not np.isclose(phi, 0.0, atol=atol):
            circuit.append(RZGate(phi), [0])
        return circuit

    @staticmethod
    def _circuit_xyx(theta, phi, lam, simplify=True, atol=DEFAULT_ATOL):
        circuit = QuantumCircuit(1)
        if simplify and np.isclose(theta, 0.0, atol=atol):
            circuit.append(RXGate(phi + lam), [0])
            return circuit
        if not simplify or not np.isclose(lam, 0.0, atol=atol):
            circuit.append(RXGate(lam), [0])
        if not simplify or not np.isclose(theta, 0.0, atol=atol):
            circuit.append(RYGate(theta), [0])
        if not simplify or not np.isclose(phi, 0.0, atol=atol):
            circuit.append(RXGate(phi), [0])
        return circuit
