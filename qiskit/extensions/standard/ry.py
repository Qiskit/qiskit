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
Rotation around the y-axis.
"""
import math
import numpy
from qiskit.circuit import ControlledGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.qasm import pi
from qiskit.util import deprecate_arguments


class RYGate(Gate):
    r"""rotation around the y-axis.

    **Matrix Definition**

    The matrix for this gate is given by:

    .. math::

        U_{\text{RY}}(\theta)
            = \exp\left(-i \frac{\theta}{2} \sigma_Y \right)
            = \begin{bmatrix}
                \cos(\theta / 2) & -\sin(\theta / 2) \\
                \sin(\theta / 2) &  \cos(\theta / 2)
            \end{bmatrix}
    """

    def __init__(self, theta, phase=0, label=None):
        """Create new ry single qubit gate."""
        super().__init__("ry", 1, [theta],
                         phase=phase, label=label)

    def _define(self):
        """
        gate ry(theta) a { r(theta, pi/2) a; }
        """
        from qiskit.extensions.standard.r import RGate
        q = QuantumRegister(1, "q")
        self.definition = [
            (RGate(self.params[0], pi/2, phase=self.phase),
             [q[0]], [])
        ]

    def control(self, num_ctrl_qubits=1, label=None):
        """Controlled version of this gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if num_ctrl_qubits == 1 and not self.phase:
            return CryGate(self.params[0], label=label)
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label)

    def inverse(self):
        """Invert this gate.

        ry(theta)^dagger = ry(-theta)
        """
        return RYGate(-self.params[0], phase=-self.phase)

    def _matrix_definition(self):
        """Return a Numpy.array for the RY gate."""
        cos = math.cos(self.params[0] / 2)
        sin = math.sin(self.params[0] / 2)
        return numpy.array([[cos, -sin],
                            [sin, cos]], dtype=complex)


@deprecate_arguments({'q': 'qubit'})
def ry(self, theta, qubit, *, q=None):  # pylint: disable=invalid-name,unused-argument
    """Apply Ry gate with angle theta to a specified qubit (qubit).
    An Ry gate implements a theta radian rotation of the qubit state vector about the
    y axis of the Bloch sphere.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit.circuit import QuantumCircuit, Parameter

            theta = Parameter('θ')
            circuit = QuantumCircuit(1)
            circuit.ry(theta,0)
            circuit.draw()

        Matrix Representation:

        .. jupyter-execute::

            import numpy
            from qiskit.extensions.standard.ry import RYGate
            RYGate(numpy.pi/2).to_matrix()
    """
    return self.append(RYGate(theta), [qubit], [])


QuantumCircuit.ry = ry


class CryGate(ControlledGate):
    r"""Controlled rotation around the z axis.

    **Matrix Definition**

    The matrix for this gate is given by:

    .. math::

        U_{\text{Cry}}(\theta) =
            I \otimes |0 \rangle\!\langle 0| +
            U_{\text{RZ}}(\theta) \otimes |1 \rangle\!\langle 1|
            = \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & \cos\left(\frac{\theta}{2}\right) & 0 & -\sin\left(\frac{\theta}{2}\right) \\
                0 & 0 & 1 & 0 \\
                0 & \sin\left(\frac{\theta}{2}\right) & 0 & \cos\left(\frac{\theta}{2}\right)
            \end{bmatrix}
    """

    def __init__(self, theta, phase=0, label=None):
        """Create new cry gate."""
        super().__init__("cry", 2, [theta], phase=phase, label=label,
                         num_ctrl_qubits=1)
        self.base_gate = RYGate(theta)

    def _define(self):
        """
        gate cry(lambda) a,b
        { u3(lambda/2,0,0) b; cx a,b;
          u3(-lambda/2,0,0) b; cx a,b;
        }

        """
        from qiskit.extensions.standard.x import CnotGate
        from qiskit.extensions.standard.u3 import U3Gate
        q = QuantumRegister(2, "q")
        self.definition = [
            (U3Gate(self.params[0] / 2, 0, 0, phase=self.phase), [q[1]], []),
            (CnotGate(), [q[0], q[1]], []),
            (U3Gate(-self.params[0] / 2, 0, 0), [q[1]], []),
            (CnotGate(), [q[0], q[1]], [])
        ]

    def inverse(self):
        """Invert this gate."""
        return CryGate(-self.params[0], phase=-self.phase)

    def _matrix_definition(self):
        """Return a Numpy.array for the Controlled-Ry gate."""
        theta = float(self.params[0])
        return numpy.array([[1, 0, 0, 0],
                            [0, numpy.cos(theta / 2), 0, -numpy.sin(theta / 2)],
                            [0, 0, 1, 0],
                            [0, numpy.sin(theta / 2), 0, numpy.cos(theta / 2)]],
                           dtype=complex)


@deprecate_arguments({'ctl': 'control_qubit',
                      'tgt': 'target_qubit'})
def cry(self, theta, control_qubit, target_qubit,
        *, ctl=None, tgt=None):  # pylint: disable=unused-argument
    """Apply cry from ctl to tgt with angle theta."""
    return self.append(CryGate(theta), [control_qubit, target_qubit], [])


QuantumCircuit.cry = cry
