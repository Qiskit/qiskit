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
import cmath
import numpy as np
import scipy.linalg as la

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.library.standard_gates import (
    UGate,
    PhaseGate,
    U3Gate,
    U2Gate,
    U1Gate,
    RXGate,
    RYGate,
    RZGate,
    RGate,
    SXGate,
    XGate,
)
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_unitary_matrix

DEFAULT_ATOL = 1e-12

ONE_QUBIT_EULER_BASIS_GATES = {
    "U3": ["u3"],
    "U321": ["u3", "u2", "u1"],
    "U": ["u"],
    "PSX": ["p", "sx"],
    "U1X": ["u1", "rx"],
    "RR": ["r"],
    "ZYZ": ["rz", "ry"],
    "ZXZ": ["rz", "rx"],
    "XZX": ["rz", "rx"],
    "XYX": ["rx", "ry"],
    "ZSXX": ["rz", "sx", "x"],
    "ZSX": ["rz", "sx"],
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
        * - 'XZX'
          - :math:`X(\phi) Z(\theta) X(\lambda)`
          - :math:`e^{i\gamma} R_X(\phi).R_Z(\theta).R_X(\lambda)`
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
          - :math:`e^{i\gamma} R_Z(\phi+\pi).\sqrt{X}.`
            :math:`R_Z(\theta+\pi).\sqrt{X}.R_Z(\lambda)`
        * - 'ZSXX'
          - :math:`Z(\phi) Y(\theta) Z(\lambda)`
          - :math:`e^{i\gamma} R_Z(\phi+\pi).\sqrt{X}.R_Z(\theta+\pi).\sqrt{X}.R_Z(\lambda)`
            or
            :math:`e^{i\gamma} R_Z(\phi+\pi).X.R_Z(\lambda)`
        * - 'U1X'
          - :math:`Z(\phi) Y(\theta) Z(\lambda)`
          - :math:`e^{i\gamma} U_1(\phi+\pi).R_X\left(\frac{\pi}{2}\right).`
            :math:`U_1(\theta+\pi).R_X\left(\frac{\pi}{2}\right).U_1(\lambda)`
        * - 'RR'
          - :math:`Z(\phi) Y(\theta) Z(\lambda)`
          - :math:`e^{i\gamma} R\left(-\pi,\frac{\phi-\lambda+\pi}{2}\right).`
            :math:`R\left(\theta+\pi,\frac{\pi}{2}-\lambda\right)`
    """

    def __init__(self, basis="U3"):
        """Initialize decomposer

        Supported bases are: 'U', 'PSX', 'ZSXX', 'ZSX', 'U321', 'U3', 'U1X', 'RR', 'ZYZ', 'ZXZ',
        'XYX', 'XZX'.

        Args:
            basis (str): the decomposition basis [Default: 'U3']

        Raises:
            QiskitError: If input basis is not recognized.
        """
        self.basis = basis  # sets: self._basis, self._params, self._circuit

    def __call__(self, unitary, simplify=True, atol=DEFAULT_ATOL):
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
        if hasattr(unitary, "to_operator"):
            # If input is a BaseOperator subclass this attempts to convert
            # the object to an Operator so that we can extract the underlying
            # numpy matrix from `Operator.data`.
            unitary = unitary.to_operator().data
        elif hasattr(unitary, "to_matrix"):
            # If input is Gate subclass or some other class object that has
            # a to_matrix method this will call that method.
            unitary = unitary.to_matrix()
        # Convert to numpy array in case not already an array
        unitary = np.asarray(unitary, dtype=complex)

        # Check input is a 2-qubit unitary
        if unitary.shape != (2, 2):
            raise QiskitError("OneQubitEulerDecomposer: " "expected 2x2 input matrix")
        if not is_unitary_matrix(unitary):
            raise QiskitError("OneQubitEulerDecomposer: " "input matrix is not unitary.")
        return self._decompose(unitary, simplify=simplify, atol=atol)

    def _decompose(self, unitary, simplify=True, atol=DEFAULT_ATOL):
        theta, phi, lam, phase = self._params(unitary)
        circuit = self._circuit(theta, phi, lam, phase, simplify=simplify, atol=atol)
        return circuit

    @property
    def basis(self):
        """The decomposition basis."""
        return self._basis

    @basis.setter
    def basis(self, basis):
        """Set the decomposition basis."""
        basis_methods = {
            "U321": (self._params_u3, self._circuit_u321),
            "U3": (self._params_u3, self._circuit_u3),
            "U": (self._params_u3, self._circuit_u),
            "PSX": (self._params_u1x, self._circuit_psx),
            "ZSX": (self._params_u1x, self._circuit_zsx),
            "ZSXX": (self._params_u1x, self._circuit_zsxx),
            "U1X": (self._params_u1x, self._circuit_u1x),
            "RR": (self._params_zyz, self._circuit_rr),
            "ZYZ": (self._params_zyz, self._circuit_zyz),
            "ZXZ": (self._params_zxz, self._circuit_zxz),
            "XYX": (self._params_xyx, self._circuit_xyx),
            "XZX": (self._params_xzx, self._circuit_xzx),
        }
        if basis not in basis_methods:
            raise QiskitError(f"OneQubitEulerDecomposer: unsupported basis {basis}")
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
        coeff = la.det(mat) ** (-0.5)
        phase = -cmath.phase(coeff)
        su_mat = coeff * mat  # U in SU(2)
        # OpenQASM SU(2) parameterization:
        # U[0, 0] = exp(-i(phi+lambda)/2) * cos(theta/2)
        # U[0, 1] = -exp(-i(phi-lambda)/2) * sin(theta/2)
        # U[1, 0] = exp(i(phi-lambda)/2) * sin(theta/2)
        # U[1, 1] = exp(i(phi+lambda)/2) * cos(theta/2)
        theta = 2 * math.atan2(abs(su_mat[1, 0]), abs(su_mat[0, 0]))
        phiplambda2 = cmath.phase(su_mat[1, 1])
        phimlambda2 = cmath.phase(su_mat[1, 0])
        phi = phiplambda2 + phimlambda2
        lam = phiplambda2 - phimlambda2
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
            [
                [
                    mat[0, 0] + mat[0, 1] + mat[1, 0] + mat[1, 1],
                    mat[0, 0] - mat[0, 1] + mat[1, 0] - mat[1, 1],
                ],
                [
                    mat[0, 0] + mat[0, 1] - mat[1, 0] - mat[1, 1],
                    mat[0, 0] - mat[0, 1] - mat[1, 0] + mat[1, 1],
                ],
            ],
            dtype=complex,
        )
        theta, phi, lam, phase = OneQubitEulerDecomposer._params_zyz(mat_zyz)
        newphi, newlam = _mod_2pi(phi + np.pi), _mod_2pi(lam + np.pi)
        return theta, newphi, newlam, phase + (newphi + newlam - phi - lam) / 2

    @staticmethod
    def _params_xzx(umat):
        det = np.linalg.det(umat)
        phase = (-1j * np.log(det)).real / 2
        mat = umat / np.sqrt(det)
        mat_zxz = _h_conjugate(mat)
        theta, phi, lam, phase_zxz = OneQubitEulerDecomposer._params_zxz(mat_zxz)
        return theta, phi, lam, phase + phase_zxz

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
    def _circuit_kak(
        theta,
        phi,
        lam,
        phase,
        simplify=True,
        atol=DEFAULT_ATOL,
        allow_non_canonical=True,
        k_gate=RZGate,
        a_gate=RYGate,
    ):
        """
        Installs the angles phi, theta, and lam into a KAK-type decomposition of the form
        K(phi) . A(theta) . K(lam) , where K and A are an orthogonal pair drawn from RZGate, RYGate,
        and RXGate.

        Args:
            theta (float): The middle KAK parameter.  Expected to lie in [0, pi).
            phi (float): The first KAK parameter.
            lam (float): The final KAK parameter.
            phase (float): The input global phase.
            k_gate (Callable): The constructor for the K gate Instruction.
            a_gate (Callable): The constructor for the A gate Instruction.
            simplify (bool): Indicates whether gates should be elided / coalesced where possible.
            allow_non_canonical (bool): Indicates whether we are permitted to reverse the sign of
                the middle parameter, theta, in the output.  When this and `simplify` are both
                enabled, we take the opportunity to commute half-rotations in the outer gates past
                the middle gate, which permits us to coalesce them at the cost of reversing the sign
                of theta.

        Returns:
            QuantumCircuit: The assembled circuit.
        """
        gphase = phase - (phi + lam) / 2
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        if not simplify:
            atol = -1.0
        # Early return for the middle-gate-free case
        if abs(theta) < atol:
            lam, phi = lam + phi, 0
            # NOTE: The following normalization is safe, because the gphase correction below
            #       fixes a particular diagonal entry to 1, which prevents any potential phase
            #       slippage coming from _mod_2pi injecting multiples of 2pi.
            lam = _mod_2pi(lam, atol)
            if abs(lam) > atol:

                circuit._append(k_gate(lam), [qr[0]], [])
                gphase += lam / 2
            circuit.global_phase = gphase
            return circuit
        if abs(theta - np.pi) < atol:
            gphase += phi
            lam, phi = lam - phi, 0
        if allow_non_canonical and (
            abs(_mod_2pi(lam + np.pi)) < atol or abs(_mod_2pi(phi + np.pi)) < atol
        ):
            lam, theta, phi = lam + np.pi, -theta, phi + np.pi
        lam = _mod_2pi(lam, atol)
        if abs(lam) > atol:
            gphase += lam / 2
            circuit._append(k_gate(lam), [qr[0]], [])
        circuit._append(a_gate(theta), [qr[0]], [])
        phi = _mod_2pi(phi, atol)
        if abs(phi) > atol:
            gphase += phi / 2
            circuit._append(k_gate(phi), [qr[0]], [])
        circuit.global_phase = gphase
        return circuit

    def _circuit_zyz(
        self, theta, phi, lam, phase, simplify=True, atol=DEFAULT_ATOL, allow_non_canonical=True
    ):
        return self._circuit_kak(
            theta,
            phi,
            lam,
            phase,
            simplify=simplify,
            atol=atol,
            allow_non_canonical=allow_non_canonical,
            k_gate=RZGate,
            a_gate=RYGate,
        )

    def _circuit_zxz(
        self, theta, phi, lam, phase, simplify=True, atol=DEFAULT_ATOL, allow_non_canonical=True
    ):
        return self._circuit_kak(
            theta,
            phi,
            lam,
            phase,
            simplify=simplify,
            atol=atol,
            allow_non_canonical=allow_non_canonical,
            k_gate=RZGate,
            a_gate=RXGate,
        )

    def _circuit_xzx(
        self, theta, phi, lam, phase, simplify=True, atol=DEFAULT_ATOL, allow_non_canonical=True
    ):
        return self._circuit_kak(
            theta,
            phi,
            lam,
            phase,
            simplify=simplify,
            atol=atol,
            allow_non_canonical=allow_non_canonical,
            k_gate=RXGate,
            a_gate=RZGate,
        )

    def _circuit_xyx(
        self, theta, phi, lam, phase, simplify=True, atol=DEFAULT_ATOL, allow_non_canonical=True
    ):
        return self._circuit_kak(
            theta,
            phi,
            lam,
            phase,
            simplify=simplify,
            atol=atol,
            allow_non_canonical=allow_non_canonical,
            k_gate=RXGate,
            a_gate=RYGate,
        )

    @staticmethod
    def _circuit_u3(theta, phi, lam, phase, simplify=True, atol=DEFAULT_ATOL):
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr, global_phase=phase)
        phi = _mod_2pi(phi, atol)
        lam = _mod_2pi(lam, atol)
        if not simplify or abs(theta) > atol or abs(phi) > atol or abs(lam) > atol:
            circuit._append(U3Gate(theta, phi, lam), [qr[0]], [])
        return circuit

    @staticmethod
    def _circuit_u321(theta, phi, lam, phase, simplify=True, atol=DEFAULT_ATOL):
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr, global_phase=phase)
        if not simplify:
            atol = -1.0
        if abs(theta) < atol:
            tot = _mod_2pi(phi + lam, atol)
            if abs(tot) > atol:
                circuit._append(U1Gate(tot), [qr[0]], [])
        elif abs(theta - np.pi / 2) < atol:
            circuit._append(U2Gate(_mod_2pi(phi, atol), _mod_2pi(lam, atol)), [qr[0]], [])
        else:
            circuit._append(U3Gate(theta, _mod_2pi(phi, atol), _mod_2pi(lam, atol)), [qr[0]], [])
        return circuit

    @staticmethod
    def _circuit_u(theta, phi, lam, phase, simplify=True, atol=DEFAULT_ATOL):
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr, global_phase=phase)
        if not simplify:
            atol = -1.0
        phi = _mod_2pi(phi, atol)
        lam = _mod_2pi(lam, atol)
        if abs(theta) > atol or abs(phi) > atol or abs(lam) > atol:
            circuit._append(UGate(theta, phi, lam), [qr[0]], [])
        return circuit

    @staticmethod
    def _circuit_psx_gen(theta, phi, lam, phase, atol, pfun, xfun, xpifun=None):
        """
        Generic X90, phase decomposition

        NOTE: `pfun` is responsible for eliding gates where appropriate (e.g., at angle value 0).
        """
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr, global_phase=phase)
        # Early return for zero SX decomposition
        if np.abs(theta) < atol:
            pfun(circuit, qr, lam + phi)
            return circuit
        # Early return for single SX decomposition
        if abs(theta - np.pi / 2) < atol:
            pfun(circuit, qr, lam - np.pi / 2)
            xfun(circuit, qr)
            pfun(circuit, qr, phi + np.pi / 2)
            return circuit
        # General double SX decomposition
        if abs(theta - np.pi) < atol:
            circuit.global_phase += lam
            phi, lam = phi - lam, 0
        if abs(_mod_2pi(lam + np.pi)) < atol or abs(_mod_2pi(phi)) < atol:
            lam, theta, phi = lam + np.pi, -theta, phi + np.pi
            circuit.global_phase -= theta
        # Shift theta and phi to turn the decomposition from
        # RZ(phi).RY(theta).RZ(lam) = RZ(phi).RX(-pi/2).RZ(theta).RX(pi/2).RZ(lam)
        # into RZ(phi+pi).SX.RZ(theta+pi).SX.RZ(lam) .
        theta, phi = theta + np.pi, phi + np.pi
        circuit.global_phase -= np.pi / 2
        # Emit circuit
        pfun(circuit, qr, lam)
        if xpifun and abs(_mod_2pi(theta)) < atol:
            xpifun(circuit, qr)
        else:
            xfun(circuit, qr)
            pfun(circuit, qr, theta)
            xfun(circuit, qr)
        pfun(circuit, qr, phi)

        return circuit

    @staticmethod
    def _circuit_psx(theta, phi, lam, phase, simplify=True, atol=DEFAULT_ATOL):
        if not simplify:
            atol = -1.0

        def fnz(circuit, qr, phi):
            phi = _mod_2pi(phi, atol)
            if abs(phi) > atol:
                circuit._append(PhaseGate(phi), [qr[0]], [])

        def fnx(circuit, qr):
            circuit._append(SXGate(), [qr[0]], [])

        return OneQubitEulerDecomposer._circuit_psx_gen(theta, phi, lam, phase, atol, fnz, fnx)

    @staticmethod
    def _circuit_zsx(theta, phi, lam, phase, simplify=True, atol=DEFAULT_ATOL):
        if not simplify:
            atol = -1.0

        def fnz(circuit, qr, phi):
            phi = _mod_2pi(phi, atol)
            if abs(phi) > atol:
                circuit._append(RZGate(phi), [qr[0]], [])
                circuit.global_phase += phi / 2

        def fnx(circuit, qr):
            circuit._append(SXGate(), [qr[0]], [])

        return OneQubitEulerDecomposer._circuit_psx_gen(theta, phi, lam, phase, atol, fnz, fnx)

    @staticmethod
    def _circuit_u1x(theta, phi, lam, phase, simplify=True, atol=DEFAULT_ATOL):
        if not simplify:
            atol = -1.0

        def fnz(circuit, qr, phi):
            phi = _mod_2pi(phi, atol)
            if abs(phi) > atol:
                circuit._append(U1Gate(phi), [qr[0]], [])

        def fnx(circuit, qr):
            circuit.global_phase += np.pi / 4
            circuit._append(RXGate(np.pi / 2), [qr[0]], [])

        return OneQubitEulerDecomposer._circuit_psx_gen(theta, phi, lam, phase, atol, fnz, fnx)

    @staticmethod
    def _circuit_zsxx(theta, phi, lam, phase, simplify=True, atol=DEFAULT_ATOL):
        if not simplify:
            atol = -1.0

        def fnz(circuit, qr, phi):
            phi = _mod_2pi(phi, atol)
            if abs(phi) > atol:
                circuit._append(RZGate(phi), [qr[0]], [])
                circuit.global_phase += phi / 2

        def fnx(circuit, qr):
            circuit._append(SXGate(), [qr[0]], [])

        def fnxpi(circuit, qr):
            circuit._append(XGate(), [qr[0]], [])

        return OneQubitEulerDecomposer._circuit_psx_gen(
            theta, phi, lam, phase, atol, fnz, fnx, fnxpi
        )

    @staticmethod
    def _circuit_rr(theta, phi, lam, phase, simplify=True, atol=DEFAULT_ATOL):
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr, global_phase=phase)
        if not simplify:
            atol = -1.0
        if abs(theta) < atol and abs(phi) < atol and abs(lam) < atol:
            return circuit
        if abs(theta - np.pi) > atol:
            circuit._append(RGate(theta - np.pi, _mod_2pi(np.pi / 2 - lam, atol)), [qr[0]], [])
        circuit._append(RGate(np.pi, _mod_2pi(0.5 * (phi - lam + np.pi), atol)), [qr[0]], [])
        return circuit


def _mod_2pi(angle: float, atol: float = 0):
    """Wrap angle into interval [-π,π). If within atol of the endpoint, clamp to -π"""
    wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
    if abs(wrapped - np.pi) < atol:
        wrapped = -np.pi
    return wrapped


def _h_conjugate(su2):
    """Return su2 conjugated by Hadamard gate. No warning if input matrix is not in su2."""
    return np.array(
        [
            [su2[0, 0].real + 1j * su2[1, 0].imag, 1j * su2[0, 0].imag + su2[1, 0].real],
            [1j * su2[0, 0].imag - su2[1, 0].real, su2[0, 0].real - 1j * su2[1, 0].imag],
        ],
        dtype=complex,
    )
