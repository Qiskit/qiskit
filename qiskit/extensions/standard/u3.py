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
    """Two-pulse single-qubit gate."""

    def __init__(self, theta, phi, lam, label=None):
        """Create new two-pulse single qubit gate."""
        super().__init__("u3", 1, [theta, phi, lam], label=label)

    def inverse(self):
        """Invert this gate.

        u3(theta, phi, lamb)^dagger = u3(-theta, -lam, -phi)
        """
        return U3Gate(-self.params[0], -self.params[2], -self.params[1])

    def control(self, num_ctrl_qubits=1, label=None):
        """Controlled version of this gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            return Cu3Gate(*self.params)
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label)

    def to_matrix(self):
        """Return a Numpy.array for the U3 gate."""
        theta, phi, lam = self.params
        theta, phi, lam = float(theta), float(phi), float(lam)
        return numpy.array(
            [[
                numpy.cos(theta / 2),
                -numpy.exp(1j * lam) * numpy.sin(theta / 2)
            ],
             [
                 numpy.exp(1j * phi) * numpy.sin(theta / 2),
                 numpy.exp(1j * (phi + lam)) * numpy.cos(theta / 2)
             ]],
            dtype=complex)


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


class Cu3Gate(ControlledGate):
    """controlled-u3 gate."""

    def __init__(self, theta, phi, lam):
        """Create new cu3 gate."""
        super().__init__("cu3", 2, [theta, phi, lam], num_ctrl_qubits=1)
        self.base_gate = U3Gate(theta, phi, lam)

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
        from qiskit.extensions.standard.x import CnotGate
        definition = []
        q = QuantumRegister(2, "q")
        rule = [
            (U1Gate((self.params[2] + self.params[1]) / 2), [q[0]], []),
            (U1Gate((self.params[2] - self.params[1]) / 2), [q[1]], []),
            (CnotGate(), [q[0], q[1]], []),
            (U3Gate(-self.params[0] / 2, 0, -(self.params[1] + self.params[2]) / 2), [q[1]], []),
            (CnotGate(), [q[0], q[1]], []),
            (U3Gate(self.params[0] / 2, self.params[1], 0), [q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return Cu3Gate(-self.params[0], -self.params[2], -self.params[1])


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
    return self.append(Cu3Gate(theta, phi, lam),
                       [control_qubit, target_qubit],
                       [])


QuantumCircuit.cu3 = cu3
