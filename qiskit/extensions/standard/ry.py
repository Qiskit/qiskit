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
    """rotation around the y-axis."""

    def __init__(self, theta):
        """Create new ry single qubit gate."""
        super().__init__("ry", 1, [theta])

    def _define(self):
        """
        gate ry(theta) a { r(theta, pi/2) a; }
        """
        from qiskit.extensions.standard.r import RGate
        definition = []
        q = QuantumRegister(1, "q")
        rule = [
            (RGate(self.params[0], pi/2), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def control(self, num_ctrl_qubits=1, label=None):
        """Controlled version of this gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            return CryGate(self.params[0])
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label)

    def inverse(self):
        """Invert this gate.

        ry(theta)^dagger = ry(-theta)
        """
        return RYGate(-self.params[0])

    def to_matrix(self):
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
    """controlled-ry gate."""

    def __init__(self, theta):
        """Create new cry gate."""
        super().__init__("cry", 2, [theta], num_ctrl_qubits=1)
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
        definition = []
        q = QuantumRegister(2, "q")
        rule = [
            (U3Gate(self.params[0] / 2, 0, 0), [q[1]], []),
            (CnotGate(), [q[0], q[1]], []),
            (U3Gate(-self.params[0] / 2, 0, 0), [q[1]], []),
            (CnotGate(), [q[0], q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return CryGate(-self.params[0])


@deprecate_arguments({'ctl': 'control_qubit',
                      'tgt': 'target_qubit'})
def cry(self, theta, control_qubit, target_qubit,
        *, ctl=None, tgt=None):  # pylint: disable=unused-argument
    """Apply cry from ctl to tgt with angle theta."""
    return self.append(CryGate(theta), [control_qubit, target_qubit], [])


QuantumCircuit.cry = cry
