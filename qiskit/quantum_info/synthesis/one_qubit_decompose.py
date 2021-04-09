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
Decompose a single-qubit unitary via Euler angles.
"""

import math
import numpy as np
import scipy.linalg as la

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.library.standard_gates import (UGate, PhaseGate, U3Gate,
                                                   U2Gate, U1Gate, RXGate, RYGate,
                                                   RZGate, RGate, SXGate)
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_unitary_matrix

DEFAULT_ATOL = 1e-12

ONE_QUBIT_EULER_BASIS_GATES = {
    'U3': ['u3'],
    'U321': ['u3', 'u2', 'u1'],
    'U': ['u'],
    'PSX': ['p', 'sx'],
    'U1X': ['u1', 'rx'],
    'RR': ['r'],
    'ZYZ': ['rz', 'ry'],
    'ZXZ': ['rz', 'rx'],
    'XYX': ['rx', 'ry'],
    'ZSX': ['rz', 'sx'],
}


class OneQubitEulerDecomposer:
    r"""A class for decomposing 1-qubit unitaries into Euler angle rotations.

    The resulting decomposition is parameterized by 3 Euler rotation angle
    parameters :math:`(\theta, \phi, \lambda)`, and a phase parameter
    :math:`\gamma`. The value of the parameters for an input unitary depends
    on the decomposition basis. Allowed bases and the resulting circuits are
    shown in the following table. Note that for the non-Euler bases (U3, U1X,
    RR), the ZYZ Euler parameters are used.

    .. list-table:: Supported circuit bases
        :widths: auto
        :header-rows: 1

        * - Basis
          - Euler Angle Basis
          - Decomposition Circuit
        * - 'ZYZ'
          - :math:`Z(\phi) Y(\theta) Z(\lambda)`
          - :math:`e^{i\gamma} R_Z(\phi).R_Y(\theta).R_Z(\lambda)`
        * - 'ZXZ'
          - :math:`Z(\phi) X(\theta) Z(\lambda)`
          - :math:`e^{i\gamma} R_Z(\phi).R_X(\theta).R_Z(\lambda)`
        * - 'XYX'
          - :math:`X(\phi) Y(\theta) X(\lambda)`
          - :math:`e^{i\gamma} R_X(\phi).R_Y(\theta).R_X(\lambda)`
        * - 'U3'
          - :math:`Z(\phi) Y(\theta) Z(\lambda)`
          - :math:`e^{i\gamma} U_3(\theta,\phi,\lambda)`
        * - 'U321'
          - :math:`Z(\phi) Y(\theta) Z(\lambda)`
          - :math:`e^{i\gamma} U_3(\theta,\phi,\lambda)`
        * - 'U'
          - :math:`Z(\phi) Y(\theta) Z(\lambda)`
          - :math:`e^{i\gamma} U_3(\theta,\phi,\lambda)`
        * - 'PSX'
          - :math:`Z(\phi) Y(\theta) Z(\lambda)`
          - :math:`e^{i\gamma} U_1(\phi+\pi).R_X\left(\frac{\pi}{2}\right).`
            :math:`U_1(\theta+\pi).R_X\left(\frac{\pi}{2}\right).U_1(\lambda)`
        * - 'ZSX'
          - :math:`Z(\phi) Y(\theta) Z(\lambda)`
          - :math:`e^{i\gamma} U_1(\phi+\pi).R_X\left(\frac{\pi}{2}\right).`
            :math:`R_Z(\theta+\pi).S_X\left(\frac{\pi}{2}\right).U_1(\lambda)`
        * - 'U1X'
          - :math:`Z(\phi) Y(\theta) Z(\lambda)`
          - :math:`e^{i\gamma} U_1(\phi+\pi).R_X\left(\frac{\pi}{2}\right).`
            :math:`U_1(\theta+\pi).R_X\left(\frac{\pi}{2}\right).U_1(\lambda)`
        * - 'RR'
          - :math:`Z(\phi) Y(\theta) Z(\lambda)`
          - :math:`e^{i\gamma} R\left(-\pi,\frac{\phi-\lambda+\pi}{2}\right).`
            :math:`R\left(\theta+\pi,\frac{\pi}{2}-\lambda\right)`
    """

    def __init__(self, basis='U3'):
        """Initialize decomposer

        Supported bases are: 'U', 'PSX', 'ZSX', 'U321', 'U3', 'U1X', 'RR', 'ZYZ', 'ZXZ', 'XYX'.

        Args:
            basis (str): the decomposition basis [Default: 'U3']

        Raises:
            QiskitError: If input basis is not recognized.
        """
        self.basis = basis  # sets: self._basis, self._params, self._circuit

    def __call__(self,
                 unitary,
                 simplify=True,
                 atol=DEFAULT_ATOL):
        """Decompose single qubit gate into a circuit.

        Args:
            unitary (Operator or Gate or array): 1-qubit unitary matrix
            simplify (bool): reduce gate count in decomposition [Default: True].
            atol (float): absolute tolerance for checking angles when simplifying
                         returned circuit [Default: 1e-12].
        Returns:
            QuantumCircuit: the decomposed single-qubit gate circuit

        Raises:
            QiskitError: if input is invalid or synthesis fails.
        """
        if hasattr(unitary, 'to_operator'):
            # If input is a BaseOperator subclass this attempts to convert
            # the object to an Operator so that we can extract the underlying
            # numpy matrix from `Operator.data`.
            unitary = unitary.to_operator().data
        elif hasattr(unitary, 'to_matrix'):
            # If input is Gate subclass or some other class object that has
            # a to_matrix method this will call that method.
            unitary = unitary.to_matrix()
        # Convert to numpy array in case not already an array
        unitary = np.asarray(unitary, dtype=complex)

        # Check input is a 2-qubit unitary
        if unitary.shape != (2, 2):
            raise QiskitError("OneQubitEulerDecomposer: "
                              "expected 2x2 input matrix")
        if not is_unitary_matrix(unitary):
            raise QiskitError("OneQubitEulerDecomposer: "
                              "input matrix is not unitary.")
        return self._decompose(unitary, simplify=simplify, atol=atol)

    def _decompose(self, unitary, simplify=True, atol=DEFAULT_ATOL):
        theta, phi, lam, phase = self._params(unitary)
        circuit = self._circuit(theta, phi, lam, phase,
                                simplify=simplify,
                                atol=atol)
        return circuit

    @property
    def basis(self):
        """The decomposition basis."""
        return self._basis

    @basis.setter
    def basis(self, basis):
        """Set the decomposition basis."""
        basis_methods = {
            'U321': (self._params_u3, self._circuit_u321),
            'U3': (self._params_u3, self._circuit_u3),
            'U': (self._params_u3, self._circuit_u),
            'PSX': (self._params_u1x, self._circuit_psx),
            'ZSX': (self._params_u1x, self._circuit_zsx),
            'U1X': (self._params_u1x, self._circuit_u1x),
            'RR': (self._params_zyz, self._circuit_rr),
            'ZYZ': (self._params_zyz, self._circuit_zyz),
            'ZXZ': (self._params_zxz, self._circuit_zxz),
            'XYX': (self._params_xyx, self._circuit_xyx)
        }
        if basis not in basis_methods:
            raise QiskitError("OneQubitEulerDecomposer: unsupported basis {}".format(basis))
        self._basis = basis
        self._params, self._circuit = basis_methods[self._basis]

    def angles(self, unitary):
        """Return the Euler angles for input array.

        Args:
            unitary (np.ndarray): 2x2 unitary matrix.

        Returns:
            tuple: (theta, phi, lambda).
        """
        theta, phi, lam, _ = self._params(unitary)
        return theta, phi, lam

    def angles_and_phase(self, unitary):
        """Return the Euler angles and phase for input array.

        Args:
            unitary (np.ndarray): 2x2 unitary matrix.

        Returns:
            tuple: (theta, phi, lambda, phase).
        """
        return self._params(unitary)

    @staticmethod
    def _params_zyz(mat):
        """Return the Euler angles and phase for the ZYZ basis."""
        # We rescale the input matrix to be special unitary (det(U) = 1)
        # This ensures that the quaternion representation is real
        coeff = la.det(mat)**(-0.5)
        phase = -np.angle(coeff)
        su_mat = coeff * mat  # U in SU(2)
        # OpenQASM SU(2) parameterization:
        # U[0, 0] = exp(-i(phi+lambda)/2) * cos(theta/2)
        # U[0, 1] = -exp(-i(phi-lambda)/2) * sin(theta/2)
        # U[1, 0] = exp(i(phi-lambda)/2) * sin(theta/2)
        # U[1, 1] = exp(i(phi+lambda)/2) * cos(theta/2)
        theta = 2 * math.atan2(abs(su_mat[1, 0]), abs(su_mat[0, 0]))
        phiplambda = 2 * np.angle(su_mat[1, 1])
        phimlambda = 2 * np.angle(su_mat[1, 0])
        phi = (phiplambda + phimlambda) / 2.0
        lam = (phiplambda - phimlambda) / 2.0
        return theta, phi, lam, phase

    @staticmethod
    def _params_zxz(mat):
        """Return the Euler angles and phase for the ZXZ basis."""
        theta, phi, lam, phase = OneQubitEulerDecomposer._params_zyz(mat)
        return theta, phi + np.pi / 2, lam - np.pi / 2, phase

    @staticmethod
    def _params_xyx(mat):
        """Return the Euler angles and phase for the XYX basis."""
        # We use the fact that
        # Rx(a).Ry(b).Rx(c) = H.Rz(a).Ry(-b).Rz(c).H
        mat_zyz = 0.5 * np.array(
            [[
                mat[0, 0] + mat[0, 1] + mat[1, 0] + mat[1, 1],
                mat[0, 0] - mat[0, 1] + mat[1, 0] - mat[1, 1]
            ],
             [
                 mat[0, 0] + mat[0, 1] - mat[1, 0] - mat[1, 1],
                 mat[0, 0] - mat[0, 1] - mat[1, 0] + mat[1, 1]
             ]],
            dtype=complex)
        theta, phi, lam, phase = OneQubitEulerDecomposer._params_zyz(mat_zyz)
        return -theta, phi, lam, phase

    @staticmethod
    def _params_u3(mat):
        """Return the Euler angles and phase for the U3 basis."""
        # The determinant of U3 gate depends on its params
        # via det(u3(theta, phi, lam)) = exp(1j*(phi+lam))
        # Since the phase is wrt to a SU matrix we must rescale
        # phase to correct this
        theta, phi, lam, phase = OneQubitEulerDecomposer._params_zyz(mat)
        return theta, phi, lam, phase - 0.5 * (phi + lam)

    @staticmethod
    def _params_u1x(mat):
        """Return the Euler angles and phase for the U1X basis."""
        # The determinant of this decomposition depends on its params
        # Since the phase is wrt to a SU matrix we must rescale
        # phase to correct this
        theta, phi, lam, phase = OneQubitEulerDecomposer._params_zyz(mat)
        return theta, phi, lam, phase - 0.5 * (theta + phi + lam)

    @staticmethod
    def _circuit_zyz(theta,
                     phi,
                     lam,
                     phase,
                     simplify=True,
                     atol=DEFAULT_ATOL):
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr, global_phase=phase)
        if simplify and math.isclose(theta, 0.0, abs_tol=atol):
            circuit._append(RZGate(phi + lam), [qr[0]], [])
            return circuit
        if not simplify or not math.isclose(lam, 0.0, abs_tol=atol):
            circuit._append(RZGate(lam), [qr[0]], [])
        if not simplify or not math.isclose(theta, 0.0, abs_tol=atol):
            circuit._append(RYGate(theta), [qr[0]], [])
        if not simplify or not math.isclose(phi, 0.0, abs_tol=atol):
            circuit._append(RZGate(phi), [qr[0]], [])
        return circuit

    @staticmethod
    def _circuit_zxz(theta,
                     phi,
                     lam,
                     phase,
                     simplify=True,
                     atol=DEFAULT_ATOL):
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr, global_phase=phase)
        if simplify and math.isclose(theta, 0.0, abs_tol=atol):
            circuit._append(RZGate(phi + lam), [qr[0]], [])
            return circuit
        if not simplify or not math.isclose(lam, 0.0, abs_tol=atol):
            circuit._append(RZGate(lam), [qr[0]], [])
        if not simplify or not math.isclose(theta, 0.0, abs_tol=atol):
            circuit._append(RXGate(theta), [qr[0]], [])
        if not simplify or not math.isclose(phi, 0.0, abs_tol=atol):
            circuit._append(RZGate(phi), [qr[0]], [])
        return circuit

    @staticmethod
    def _circuit_xyx(theta,
                     phi,
                     lam,
                     phase,
                     simplify=True,
                     atol=DEFAULT_ATOL):
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr, global_phase=phase)
        if simplify and math.isclose(theta, 0.0, abs_tol=atol):
            circuit._append(RXGate(phi + lam), [qr[0]], [])
            return circuit
        if not simplify or not math.isclose(lam, 0.0, abs_tol=atol):
            circuit._append(RXGate(lam), [qr[0]], [])
        if not simplify or not math.isclose(theta, 0.0, abs_tol=atol):
            circuit._append(RYGate(theta), [qr[0]], [])
        if not simplify or not math.isclose(phi, 0.0, abs_tol=atol):
            circuit._append(RXGate(phi), [qr[0]], [])
        return circuit

    @staticmethod
    def _circuit_u3(theta,
                    phi,
                    lam,
                    phase,
                    simplify=True,
                    atol=DEFAULT_ATOL):
        # pylint: disable=unused-argument
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr, global_phase=phase)
        circuit._append(U3Gate(theta, phi, lam), [qr[0]], [])
        return circuit

    @staticmethod
    def _circuit_u321(theta,
                      phi,
                      lam,
                      phase,
                      simplify=True,
                      atol=DEFAULT_ATOL):
        rtol = 1e-9  # default is 1e-5, too far from atol=1e-12
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr, global_phase=phase)
        if simplify and (math.isclose(theta, 0.0, abs_tol=atol, rel_tol=rtol)):
            phi_lam = phi + lam
            if not (math.isclose(phi_lam, 0.0, abs_tol=atol, rel_tol=rtol) or
                    math.isclose(phi_lam, 2*np.pi, abs_tol=atol, rel_tol=rtol)):
                circuit._append(U1Gate(_mod2pi(phi+lam)), [qr[0]], [])
        elif simplify and math.isclose(theta, np.pi/2, abs_tol=atol, rel_tol=rtol):
            circuit._append(U2Gate(phi, lam), [qr[0]], [])
        else:
            circuit._append(U3Gate(theta, phi, lam), [qr[0]], [])
        return circuit

    @staticmethod
    def _circuit_u(theta,
                   phi,
                   lam,
                   phase,
                   simplify=True,
                   atol=DEFAULT_ATOL):
        # pylint: disable=unused-argument
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr, global_phase=phase)
        circuit._append(UGate(theta, phi, lam), [qr[0]], [])
        return circuit

    @staticmethod
    def _circuit_psx(theta,
                     phi,
                     lam,
                     phase,
                     simplify=True,
                     atol=DEFAULT_ATOL):
        # Shift theta and phi so decomposition is
        # Phase(phi+pi).SX.Phase(theta+pi).SX.Phase(lam)
        theta = _mod2pi(theta + np.pi)
        phi = _mod2pi(phi + np.pi)
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr, global_phase=phase - np.pi / 2)
        # Check for decomposition into minimimal number required SX gates
        abs_theta = abs(theta)
        if simplify and math.isclose(abs_theta, np.pi, abs_tol=atol):
            lam_phi_theta = _mod2pi(lam + phi + theta)
            abs_lam_phi_theta = _mod2pi(abs(lam + phi + theta))
            if not (math.isclose(abs_lam_phi_theta, 0., abs_tol=atol) or
                    math.isclose(abs_lam_phi_theta, 2*np.pi, abs_tol=atol)):
                circuit._append(PhaseGate(lam_phi_theta), [qr[0]], [])
            circuit.global_phase += np.pi / 2
        elif simplify and (math.isclose(abs_theta, np.pi/2, abs_tol=atol) or
                           math.isclose(abs_theta, 3*np.pi/2, abs_tol=atol)):
            lam_theta = _mod2pi(lam + theta)
            abs_lam_theta = _mod2pi(abs(lam + theta))
            if not (math.isclose(abs_lam_theta, 0, abs_tol=atol) or
                    math.isclose(abs_lam_theta, 2*np.pi, abs_tol=atol)):
                circuit._append(PhaseGate(lam_theta), [qr[0]], [])
            circuit._append(SXGate(), [qr[0]], [])
            phi_theta = _mod2pi(phi + theta)
            abs_phi_theta = _mod2pi(abs(phi_theta))
            if not (math.isclose(abs_phi_theta, 0, abs_tol=atol) or
                    math.isclose(abs_phi_theta, 2*np.pi, abs_tol=atol)):
                circuit._append(PhaseGate(_mod2pi(phi + theta)), [qr[0]], [])
            if (math.isclose(theta, -np.pi / 2, abs_tol=atol) or math.isclose(
                    theta, 3 * np.pi / 2, abs_tol=atol)):
                circuit.global_phase += np.pi / 2
        else:
            abs_lam = abs(lam)
            if not (math.isclose(abs_lam, 0., abs_tol=atol) or
                    math.isclose(abs_lam, 2*np.pi, abs_tol=atol)):
                circuit._append(PhaseGate(lam), [qr[0]], [])
            circuit._append(SXGate(), [qr[0]], [])
            if not (math.isclose(abs_theta, 0., abs_tol=atol) or
                    math.isclose(abs_theta, 2*np.pi, abs_tol=atol)):
                circuit._append(PhaseGate(theta), [qr[0]], [])
            circuit._append(SXGate(), [qr[0]], [])
            abs_phi = abs(phi)
            if not (math.isclose(abs_phi, 0., abs_tol=atol) or
                    math.isclose(abs_phi, 2*np.pi, abs_tol=atol)):
                circuit._append(PhaseGate(phi), [qr[0]], [])
        return circuit

    @staticmethod
    def _circuit_zsx(theta,
                     phi,
                     lam,
                     phase,
                     simplify=True,
                     atol=DEFAULT_ATOL):
        # Shift theta and phi so decomposition is
        # RZ(phi+pi).SX.RZ(theta+pi).SX.RZ(lam)
        theta = _mod2pi(theta + np.pi)
        phi = _mod2pi(phi + np.pi)
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr, global_phase=phase - np.pi / 2)
        # Check for decomposition into minimimal number required SX gates
        abs_theta = abs(theta)
        if simplify and math.isclose(abs_theta, np.pi, abs_tol=atol):
            lam_phi_theta = _mod2pi(lam + phi + theta)
            abs_lam_phi_theta = _mod2pi(abs(lam + phi + theta))
            if not (math.isclose(abs_lam_phi_theta, 0., abs_tol=atol) or
                    math.isclose(abs_lam_phi_theta, 2*np.pi, abs_tol=atol)):
                circuit._append(RZGate(lam_phi_theta), [qr[0]], [])
                circuit.global_phase += 0.5 * lam_phi_theta
            circuit.global_phase += np.pi / 2
        elif simplify and (math.isclose(abs_theta, np.pi/2, abs_tol=atol) or
                           math.isclose(abs_theta, 3*np.pi/2, abs_tol=atol)):
            lam_theta = _mod2pi(lam + theta)
            abs_lam_theta = _mod2pi(abs(lam + theta))
            if not (math.isclose(abs_lam_theta, 0, abs_tol=atol) or
                    math.isclose(abs_lam_theta, 2*np.pi, abs_tol=atol)):
                circuit._append(RZGate(lam_theta), [qr[0]], [])
                circuit.global_phase += 0.5 * lam_theta
            circuit._append(SXGate(), [qr[0]], [])
            phi_theta = _mod2pi(phi + theta)
            abs_phi_theta = _mod2pi(abs(phi_theta))
            if not (math.isclose(abs_phi_theta, 0, abs_tol=atol) or
                    math.isclose(abs_phi_theta, 2*np.pi, abs_tol=atol)):
                circuit._append(RZGate(phi_theta), [qr[0]], [])
                circuit.global_phase += 0.5 * phi_theta
            if (math.isclose(theta, -np.pi / 2, abs_tol=atol) or
                    math.isclose(theta, 3 * np.pi / 2, abs_tol=atol)):
                circuit.global_phase += np.pi / 2
        else:
            abs_lam = abs(lam)
            if not (math.isclose(abs_lam, 0., abs_tol=atol) or
                    math.isclose(abs_lam, 2*np.pi, abs_tol=atol)):
                circuit._append(RZGate(lam), [qr[0]], [])
                circuit.global_phase += 0.5 * lam
            circuit._append(SXGate(), [qr[0]], [])
            if not (math.isclose(abs_theta, 0., abs_tol=atol) or
                    math.isclose(abs_theta, 2*np.pi, abs_tol=atol)):
                circuit._append(RZGate(theta), [qr[0]], [])
                circuit.global_phase += 0.5 * theta
            circuit._append(SXGate(), [qr[0]], [])
            abs_phi = abs(phi)
            if not (math.isclose(abs_phi, 0., abs_tol=atol) or
                    math.isclose(abs_phi, 2*np.pi, abs_tol=atol)):
                circuit._append(RZGate(phi), [qr[0]], [])
                circuit.global_phase += 0.5 * phi
        return circuit

    @staticmethod
    def _circuit_u1x(theta,
                     phi,
                     lam,
                     phase,
                     simplify=True,
                     atol=DEFAULT_ATOL):
        # Shift theta and phi so decomposition is
        # U1(phi).X90.U1(theta).X90.U1(lam)
        theta += np.pi
        phi += np.pi
        # Check for decomposition into minimimal number required X90 pulses
        if simplify and math.isclose(abs(theta), np.pi, abs_tol=atol):
            # Zero X90 gate decomposition
            qr = QuantumRegister(1, 'qr')
            circuit = QuantumCircuit(qr, global_phase=phase)
            circuit._append(U1Gate(lam + phi + theta), [qr[0]], [])
            return circuit
        if simplify and math.isclose(abs(theta), np.pi/2, abs_tol=atol):
            # Single X90 gate decomposition
            qr = QuantumRegister(1, 'qr')
            circuit = QuantumCircuit(qr, global_phase=phase)
            circuit._append(U1Gate(lam + theta), [qr[0]], [])
            circuit._append(RXGate(np.pi / 2), [qr[0]], [])
            circuit._append(U1Gate(phi + theta), [qr[0]], [])
            return circuit
        # General two-X90 gate decomposition
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr, global_phase=phase)
        circuit._append(U1Gate(lam), [qr[0]], [])
        circuit._append(RXGate(np.pi / 2), [qr[0]], [])
        circuit._append(U1Gate(theta), [qr[0]], [])
        circuit._append(RXGate(np.pi / 2), [qr[0]], [])
        circuit._append(U1Gate(phi), [qr[0]], [])
        return circuit

    @staticmethod
    def _circuit_rr(theta,
                    phi,
                    lam,
                    phase,
                    simplify=True,
                    atol=DEFAULT_ATOL):
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr, global_phase=phase)
        if not simplify or not math.isclose(theta, -np.pi, abs_tol=atol):
            circuit._append(RGate(theta + np.pi, np.pi / 2 - lam), [qr[0]], [])
        circuit._append(RGate(-np.pi, 0.5 * (phi - lam + np.pi)), [qr[0]], [])
        return circuit


def _mod2pi(angle):
    if angle >= 0:
        return math.fmod(angle, 2*np.pi)
    else:
        return math.fmod(angle, -2*np.pi)
