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
from qiskit.quantum_info.operators.predicates import is_unitary_matrix


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

    def __call__(self, unitary_mat, validate=True, simplify=True, atol=1e-12):
        """Decompose single qubit gate into a circuit"""
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
        if validate and not is_unitary_matrix(unitary_mat):
            raise QiskitError("OneQubitEulerDecomposer: "
                              "input matrix is not unitary.")
        return self._circuit(unitary_mat, simplify=simplify, atol=atol)

    def _angles(self, unitary_mat, atol=1e-12):
        """Return Euler angles"""
        # Add phase to matrix to make it special unitary
        # This ensure that the quaternion representation is real
        su_phase = self._special_unitary_phase(unitary_mat)
        quats = self._quaternions(su_phase * unitary_mat)
        if self._basis in ['U3', 'U1X', 'ZYZ']:
            return self._quaternions2euler_zyz(quats, atol=atol)
        if self._basis == 'XYX':
            return self._quaternions2euler_xyx(quats, atol=atol)
        raise QiskitError("OneQubitEulerDecomposer: invalid basis")

    def _circuit(self, unitary_mat, simplify=True, atol=1e-12):
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
    def _special_unitary_phase(unitary_mat):
        """Return phase to make unitary special unitary.

        This means that det(phase * unitary_mat) = 1
        """
        return 1.0 / np.sqrt(la.det(unitary_mat))

    @staticmethod
    def _quaternions(su_mat):
        """Return quaternions for a special unitary matrix"""
        # Get quaternions (q0, q1, q2, q3)
        # so that su_mat =  q0*I - 1j * (q1*X + q2*Y + q3*Z)
        quats = np.zeros(4, dtype=complex)
        quats[0] = 0.5 * (su_mat[0, 0] + su_mat[1, 1])
        quats[1] = 0.5j * (su_mat[0, 1] + su_mat[1, 0])
        quats[2] = 0.5 * (-su_mat[0, 1] + su_mat[1, 0])
        quats[3] = 0.5j * (su_mat[0, 0] - su_mat[1, 1])
        # Discard imaginary part (which should be zero)
        return quats.real

    @staticmethod
    def _quaternions2euler_zyz(quats, atol=1e-12):
        # pylint: too-many-return-statements
        # Check quaternions for pure Pauli rotations
        if np.allclose(abs(quats), np.array([1., 0., 0., 0.]), atol=atol):
            # Identity
            return np.zeros(3, dtype=float)
        if np.allclose(abs(quats), np.array([0., 1., 0., 0.]), atol=atol):
            # +/- RX180
            return np.array([np.pi, 0., -quats[1] * np.pi])
        if np.allclose(abs(quats), np.array([0., 0., 1., 0.]), atol=atol):
            # +/- RY180
            return np.array([quats[2] * np.pi, 0., 0.])
        if np.allclose(abs(quats), np.array([0., 0., 0., 1.]), atol=atol):
            # +/- RZ180
            return np.array([0., 0., quats[3] * np.pi])
        if np.allclose(quats[[1, 3]], np.array([0., 0.]), atol=atol):
            # RY rotation
            arg = np.clip(2 * quats[0] * quats[2], -1., 1.)
            return np.array([math.asin(arg), 0., 0.])
        if np.allclose(quats[[1, 2]], np.array([0., 0.]), atol=atol):
            # RZ rotation
            arg = np.clip(2 * quats[0] * quats[3], -1., 1.)
            return np.array([0., 0., math.asin(arg)])
        # General case
        return np.array([
            math.acos(np.clip(quats[0] * quats[0] - quats[1] * quats[1]
                              - quats[2] * quats[2] + quats[3] * quats[3],
                              -1., 1.)),
            math.atan2(quats[2] * quats[3] - quats[0] * quats[1],
                       quats[0] * quats[2] + quats[1] * quats[3]),
            math.atan2(quats[2] * quats[3] + quats[0] * quats[1],
                       quats[0] * quats[2] - quats[1] * quats[3])])

    @staticmethod
    def _quaternions2euler_xyx(quats, atol=1e-12):
        # pylint: too-many-return-statements
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
            return np.array([math.asin(2 * quats[0] * quats[2]), 0., 0.])
        if np.allclose(quats[[2, 3]], np.array([0., 0.]), atol=atol):
            # RX rotation
            return np.array([0., 0., math.asin(2 * quats[0] * quats[1])])
        # General case
        return np.array([
            math.acos(quats[0] * quats[0] + quats[1] * quats[1]
                      - quats[2] * quats[2] - quats[3] * quats[3]),
            math.atan2(quats[1] * quats[2] + quats[0] * quats[3],
                       quats[0] * quats[2] - quats[1] * quats[3]),
            math.atan2(quats[1] * quats[2] - quats[0] * quats[3],
                       quats[0] * quats[2] + quats[1] * quats[3])])

    @staticmethod
    def _circuit_u3(angles):
        theta, phi, lam = angles
        circuit = QuantumCircuit(1)
        circuit.u3(theta, phi, lam, 0)
        return circuit

    @staticmethod
    def _circuit_u1x(angles, simplify=True, atol=1e-12):
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
    def _circuit_zyz(angles, simplify=True, atol=1e-12):
        theta, phi, lam = angles
        circuit = QuantumCircuit(1)
        if not simplify or not np.isclose(lam, 0.0, atol=atol):
            circuit.rz(lam, 0)
        if not simplify or not np.isclose(theta, 0.0, atol=atol):
            circuit.ry(theta, 0)
        if not simplify or not np.isclose(phi, 0.0, atol=atol):
            circuit.rz(phi, 0)
        return circuit

    @staticmethod
    def _circuit_xyx(angles, simplify=True, atol=1e-12):
        theta, phi, lam = angles
        circuit = QuantumCircuit(1)
        if not simplify or not np.isclose(lam, 0.0, atol=atol):
            circuit.rx(lam, 0)
        if not simplify or not np.isclose(theta, 0.0, atol=atol):
            circuit.ry(theta, 0)
        if not simplify or not np.isclose(phi, 0.0, atol=atol):
            circuit.rx(phi, 0)
        return circuit
