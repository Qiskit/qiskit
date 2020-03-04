# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Two-pulse single-qubit gate.
"""

import numpy
from qiskit.circuit import ControlledGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.util import deprecate_arguments


# pylint: disable=cyclic-import
class U3Gate(Gate):
    r"""Two-pulse single-qubit gate.

    **Matrix Definition**

    The matrix for this gate is given by:

    .. math::

        U_3(\theta, \phi, \lambda) = \begin{bmatrix}
                \cos(\theta / 2) & -e^{i\lambda}\sin(\theta / 2) \\
                e^{i\phi}\sin(\theta / 2) & e^{i(\phi+\lambda)}\cos(\theta / 2)
            \end{bmatrix}
    """

    def __init__(self, theta, phi, lam, phase=0, label=None):
        """Create new two-pulse single qubit gate."""
        super().__init__('u3', 1, [theta, phi, lam],
                         phase=phase, label=label)

    def inverse(self):
        """Invert this gate.

        u3(theta, phi, lamb)^dagger = u3(-theta, -lam, -phi)
        """
        return U3Gate(-self.params[0], -self.params[2], -self.params[1],
                      phase=-self.phase)

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Controlled version of this gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if ctrl_state is None:
            if num_ctrl_qubits == 1:
                return CU3Gate(*self.params)
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label,
                               ctrl_state=ctrl_state)

    def _matrix_definition(self):
        """Return a Numpy.array for the U3 gate."""
        theta, phi, lam = self.params
        theta, phi, lam = float(theta), float(phi), float(lam)
        return numpy.array([
            [
                numpy.cos(theta / 2),
                -numpy.exp(1j * lam) * numpy.sin(theta / 2)
            ],
            [
                numpy.exp(1j * phi) * numpy.sin(theta / 2),
                numpy.exp(1j * (phi + lam)) * numpy.cos(theta / 2)
            ]
        ], dtype=complex)


@deprecate_arguments({'q': 'qubit'})
def u3(self, theta, phi, lam, qubit, *, q=None):  # pylint: disable=invalid-name,unused-argument
    """Apply U3 gate with angle theta, phi, and lam to a specified qubit (qubit).
    u3(θ, φ, λ) := U(θ, φ, λ) = Rz(φ + 3π)Rx(π/2)Rz(θ + π)Rx(π/2)Rz(λ)

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit.circuit import QuantumCircuit, Parameter

            theta = Parameter('theta')
            phi = Parameter('φ')
            lam = Parameter('λ')
            circuit = QuantumCircuit(1)
            circuit.u3(theta,phi,lam,0)
            circuit.draw()

        Matrix Representation:

        .. jupyter-execute::

            import numpy
            from qiskit.extensions.standard.u3 import U3Gate
            U3Gate(numpy.pi/2,numpy.pi/2,numpy.pi/2).to_matrix()
    """
    return self.append(U3Gate(theta, phi, lam), [qubit], [])


QuantumCircuit.u3 = u3


class CU3Meta(type):
    """A metaclass to ensure that Cu3Gate and CU3Gate are of the same type.

    Can be removed when Cu3Gate gets removed.
    """
    @classmethod
    def __instancecheck__(mcs, inst):
        return type(inst) in {CU3Gate, Cu3Gate}  # pylint: disable=unidiomatic-typecheck


class CU3Gate(ControlledGate, metaclass=CU3Meta):
    """The controlled-u3 gate."""

    def __init__(self, theta, phi, lam, phase=0, label=None):
        """Create new cu3 gate."""
        super().__init__('cu3', 2, [theta, phi, lam], phase=0, label=None,
                         num_ctrl_qubits=1)
        self.base_gate = U3Gate(theta, phi, lam, phase=phase)

    def _define(self):
        """
        gate cu3(theta,phi,lambda) c, t
        { u1((lambda+phi)/2) c;
          u1((lambda-phi)/2) t;
          cx c,t;
          u3(-theta/2,0,-(phi+lambda)/2) t;
          cx c,t;
          u3(theta/2,phi,0) t;
        }
        """
        from qiskit.extensions.standard.u1 import U1Gate
        from qiskit.extensions.standard.x import CXGate
        q = QuantumRegister(2, 'q')
        self.definition = [
            (U1Gate((self.params[2] + self.params[1]) / 2, phase=self.phase), [q[0]], []),
            (U1Gate((self.params[2] - self.params[1]) / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (U3Gate(-self.params[0] / 2, 0, -(self.params[1] + self.params[2]) / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (U3Gate(self.params[0] / 2, self.params[1], 0), [q[1]], [])
        ]

    def inverse(self):
        """Invert this gate."""
        return CU3Gate(-self.params[0], -self.params[2], -self.params[1],
                       phase=-self.phase)

    def _matrix_definition(self):
        """Return a Numpy.array for the Cu3 gate."""
        theta, phi, lam = self.params
        theta, phi, lam = float(theta), float(phi), float(lam)
        half_sine = numpy.sin(theta / 2)
        half_cosine = numpy.cos(theta / 2)
        return numpy.array([
            [1, 0, 0, 0],
            [0, half_cosine, 0, -numpy.exp(1j * lam) * half_sine],
            [0, 0, 1, 0],
            [0, numpy.exp(1j * phi) * half_sine, 0, numpy.exp(1j * (phi + lam)) * half_cosine]
        ])


class Cu3Gate(CU3Gate, metaclass=CU3Meta):
    """The deprecated CU3Gate class."""

    def __init__(self, theta, phi, lam):
        import warnings
        warnings.warn('The class Cu3Gate is deprecated as of 0.14.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the class CU3Gate instead.',
                      DeprecationWarning, stacklevel=2)
        super().__init__(theta, phi, lam)


@deprecate_arguments({'ctl': 'control_qubit',
                      'tgt': 'target_qubit'})
def cu3(self, theta, phi, lam, control_qubit, target_qubit,
        *, ctl=None, tgt=None):  # pylint: disable=unused-argument
    """Apply cU3 gate from a specified control (control_qubit) to target (target_qubit) qubit
    with angle theta, phi, and lam.
    A cU3 gate implements a U3(theta,phi,lam) on the target qubit when the
    control qubit is in state |1>.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit.circuit import QuantumCircuit, Parameter

            theta = Parameter('θ')
            phi = Parameter('φ')
            lam = Parameter('λ')
            circuit = QuantumCircuit(2)
            circuit.cu3(theta,phi,lam,0,1)
            circuit.draw()
    """
    return self.append(CU3Gate(theta, phi, lam), [control_qubit, target_qubit], [])


QuantumCircuit.cu3 = cu3
