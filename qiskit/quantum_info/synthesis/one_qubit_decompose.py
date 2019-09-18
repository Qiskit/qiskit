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
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.operators.predicates import is_unitary_matrix

DEFAULT_ATOL = 1e-12


class OneQubitEulerDecomposer:
    """A class for decomposing 1-qubit unitaries into Eular angle rotations.

    Allowed basis and their decompositions are:
        U3: U -> phase * U3(theta, phi, lam)
        U1X: U -> phase * U1(lam).RX(pi/2).U1(theta+pi).RX(pi/2).U1(phi+pi)
        ZYZ: U -> phase * RZ(phi).RY(theta).RZ(lam)
        XYX: U -> phase * RX(phi).RY(theta).RX(lam)
    """

    def __init__(self, basis='U3'):
        if basis not in ['U3', 'U1X', 'ZYZ', 'XYX']:
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
        if not Operator(circuit).equiv(unitary_mat):
            raise QiskitError("OneQubitEulerDecomposer: "
                              "synthesis failed within required accuracy.")
        return circuit

    def _angles(self, unitary_mat, atol=DEFAULT_ATOL):
        """Return Euler angles for given basis."""
        if self._basis in ['U3', 'U1X', 'ZYZ']:
            return self._angles_zyz(unitary_mat)
        if self._basis == 'XYX':
            return self._angles_xyx(unitary_mat, atol=atol)
        raise QiskitError("OneQubitEulerDecomposer: invalid basis")

    def _circuit(self, unitary_mat, simplify=True, atol=DEFAULT_ATOL):
        # Add phase to matrix to make it special unitary
        # This ensure that the quaternion representation is real
        angles = self._angles(unitary_mat)
        if self._basis == 'U3':
            return self._circuit_u3(angles)
        if self._basis == 'U1X':
            return self._circuit_u1x(angles, simplify=simplify, atol=atol)
        if self._basis == 'ZYZ':
            return self._circuit_zyz(angles, simplify=simplify, atol=atol)
        if self._basis == 'XYX':
            return self._circuit_xyx(angles, simplify=simplify, atol=atol)
        raise QiskitError("OneQubitEulerDecomposer: invalid basis")

    @staticmethod
    def _angles_zyz(unitary_matrix):
        """Return euler angles for unitary matrix in ZYZ basis.

        In this representation U = Rz(phi).Ry(theta).Rz(lam)
        """
        if unitary_matrix.shape != (2, 2):
            raise QiskitError("euler_angles_1q: expected 2x2 matrix")
        phase = la.det(unitary_matrix)**(-1.0/2.0)
        U = phase * unitary_matrix  # U in SU(2)
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
        return theta, phi, lam

    @staticmethod
    def _quaternions(unitary_matrix):
        """Return quaternions for a special unitary matrix"""
        # Get quaternions (q0, q1, q2, q3)
        # so that su_mat =  q0*I - 1j * (q1*X + q2*Y + q3*Z)
        # We get them from the canonical ZYZ euler angles
        theta, phi, lam = OneQubitEulerDecomposer._angles_zyz(
            unitary_matrix)
        quats = np.zeros(4, dtype=complex)
        quats[0] = math.cos(0.5 * theta) * math.cos(0.5 * (lam + phi))
        quats[1] = math.sin(0.5 * theta) * math.sin(0.5 * (lam - phi))
        quats[2] = math.sin(0.5 * theta) * math.cos(0.5 * (lam - phi))
        quats[3] = math.cos(0.5 * theta) * math.sin(0.5 * (lam + phi))
        return quats

    @staticmethod
    def _angles_xyx(unitary_matrix, atol=DEFAULT_ATOL):
        """Return euler angles for unitary matrix in XYX basis.

        In this representation U = Rx(phi).Ry(theta).Rx(lam)
        """
        # pylint: disable=too-many-return-statements
        quats = OneQubitEulerDecomposer._quaternions(unitary_matrix)
        # Check quaternions for pure Pauli rotations
        if np.allclose(abs(quats), np.array([1., 0., 0., 0.]), atol=atol):
            # Identity
            return np.zeros(3, dtype=float)
        if np.allclose(abs(quats), np.array([0., 1., 0., 0.]), atol=atol):
            # +/- RX180
            return np.array([0., 0, -quats[1] * np.pi])
        if np.allclose(abs(quats), np.array([0., 0., 1., 0.]), atol=atol):
            # +/- RY180
            return np.array([quats[2] * np.pi, 0., 0.])
        if np.allclose(abs(quats), np.array([0., 0., 0., 1.]), atol=atol):
            # +/- RZ180
            return np.array([np.pi, quats[3] * np.pi, 0.])
        if np.allclose(quats[[1, 3]], np.array([0., 0.]), atol=atol):
            # RY rotation
            arg = np.clip(np.real(2 * quats[0] * quats[2]), -1., 1.)
            return np.array([math.asin(arg), 0., 0.])
        if np.allclose(quats[[2, 3]], np.array([0., 0.]), atol=atol):
            # RX rotation
            arg = np.clip(np.real(2 * quats[0] * quats[1]), -1., 1.)
            return np.array([0., 0., math.asin(arg)])
        # General case
        return np.array([
            math.acos(np.clip(np.real(quats[0] * quats[0] + quats[1] * quats[1]
                                      - quats[2] * quats[2] - quats[3] * quats[3]),
                              -1., 1.)),
            math.atan2(np.real(quats[1] * quats[2] + quats[0] * quats[3]),
                       np.real(quats[0] * quats[2] - quats[1] * quats[3])),
            math.atan2(np.real(quats[1] * quats[2] - quats[0] * quats[3]),
                       np.real(quats[0] * quats[2] + quats[1] * quats[3]))])

    @staticmethod
    def _circuit_u3(angles):
        theta, phi, lam = angles
        circuit = QuantumCircuit(1)
        circuit.u3(theta, phi, lam, 0)
        return circuit

    @staticmethod
    def _circuit_u1x(angles, simplify=True, atol=DEFAULT_ATOL):
        # Check for U1 and U2 decompositions into minimimal
        # required X90 pulses
        theta, phi, lam = angles
        if simplify and np.allclose([theta, phi], [0., 0.], atol=atol):
            # zero X90 gate decomposition
            circuit = QuantumCircuit(1)
            circuit.u1(lam, 0)
            return circuit
        if simplify and np.isclose(theta, np.pi / 2, atol=atol):
            # single X90 gate decomposition
            circuit = QuantumCircuit(1)
            circuit.u1(lam - np.pi / 2, 0)
            circuit.rx(np.pi / 2, 0)
            circuit.u1(phi + np.pi / 2, 0)
            return circuit
        # General two-X90 gate decomposition
        circuit = QuantumCircuit(1)
        circuit.u1(lam, 0)
        circuit.rx(np.pi / 2, 0)
        circuit.u1(theta + np.pi, 0)
        circuit.rx(np.pi / 2, 0)
        circuit.u1(phi + np.pi, 0)
        return circuit

    @staticmethod
    def _circuit_zyz(angles, simplify=True, atol=DEFAULT_ATOL):
        theta, phi, lam = angles
        if simplify and np.isclose(theta, 0.0, atol=atol):
            circuit = QuantumCircuit(1)
            circuit.rz(phi + lam, 0)
            return circuit
        circuit = QuantumCircuit(1)
        if not simplify or not np.isclose(lam, 0.0, atol=atol):
            circuit.rz(lam, 0)
        if not simplify or not np.isclose(theta, 0.0, atol=atol):
            circuit.ry(theta, 0)
        if not simplify or not np.isclose(phi, 0.0, atol=atol):
            circuit.rz(phi, 0)
        return circuit

    @staticmethod
    def _circuit_xyx(angles, simplify=True, atol=DEFAULT_ATOL):
        theta, phi, lam = angles
        circuit = QuantumCircuit(1)
        if not simplify or not np.isclose(lam, 0.0, atol=atol):
            circuit.rx(lam, 0)
        if not simplify or not np.isclose(theta, 0.0, atol=atol):
            circuit.ry(theta, 0)
        if not simplify or not np.isclose(phi, 0.0, atol=atol):
            circuit.rx(phi, 0)
        return circuit
