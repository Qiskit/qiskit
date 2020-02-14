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
Rotation around the z-axis.
"""
from qiskit.circuit import Gate
from qiskit.circuit import ControlledGate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.util import deprecate_arguments


class RZGate(Gate):
    """rotation around the z-axis."""

    def __init__(self, phi):
        """Create new rz single qubit gate."""
        super().__init__("rz", 1, [phi])

    def _define(self):
        """
        gate rz(phi) a { u1(phi) a; }
        """
        from qiskit.extensions.standard.u1 import U1Gate
        definition = []
        q = QuantumRegister(1, "q")
        rule = [
            (U1Gate(self.params[0]), [q[0]], [])
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
            return CrzGate(self.params[0])
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label)

    def inverse(self):
        """Invert this gate.

        rz(phi)^dagger = rz(-phi)
        """
        return RZGate(-self.params[0])


@deprecate_arguments({'q': 'qubit'})
def rz(self, phi, qubit, *, q=None):  # pylint: disable=invalid-name,unused-argument
    """Apply Rz gate with angle phi to a specified qubit (qubit).
    An Rz gate implemements a phi radian rotation of the qubit state vector about the
    z axis of the Bloch sphere.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit.circuit import QuantumCircuit, Parameter

            phi = Parameter('φ')
            circuit = QuantumCircuit(1)
            circuit.rz(phi,0)
            circuit.draw()
    """
    return self.append(RZGate(phi), [qubit], [])


QuantumCircuit.rz = rz


class CrzGate(ControlledGate):
    """controlled-rz gate."""

    def __init__(self, theta):
        """Create new crz gate."""
        super().__init__("crz", 2, [theta], num_ctrl_qubits=1)
        self.base_gate = RZGate(theta)

    def _define(self):
        """
        gate crz(lambda) a,b
        { u1(lambda/2) b; cx a,b;
          u1(-lambda/2) b; cx a,b;
        }
        """
        from qiskit.extensions.standard.x import CnotGate
        from qiskit.extensions.standard.u1 import U1Gate
        definition = []
        q = QuantumRegister(2, "q")
        rule = [
            (U1Gate(self.params[0] / 2), [q[1]], []),
            (CnotGate(), [q[0], q[1]], []),
            (U1Gate(-self.params[0] / 2), [q[1]], []),
            (CnotGate(), [q[0], q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return CrzGate(-self.params[0])


@deprecate_arguments({'ctl': 'control_qubit', 'tgt': 'target_qubit'})
def crz(self, theta, control_qubit, target_qubit,
        *, ctl=None, tgt=None):  # pylint: disable=unused-argument
    """Apply cRz gate from a specified control (control_qubit) to target (target_qubit) qubit
    with angle theta. A cRz gate implements a theta radian rotation of the qubit state vector
    about the z axis of the Bloch sphere when the control qubit is in state |1>.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit.circuit import QuantumCircuit, Parameter

            theta = Parameter('θ')
            circuit = QuantumCircuit(2)
            circuit.crz(theta,0,1)
            circuit.draw()
    """
    return self.append(CrzGate(theta), [control_qubit, target_qubit], [])


QuantumCircuit.crz = crz
