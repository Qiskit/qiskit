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
controlled-u1 gate.
"""
from qiskit.circuit import ControlledGate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.cx import CnotGate


class Cu1Gate(ControlledGate):
    """controlled-u1 gate."""

    def __init__(self, theta):
        """Create new cu1 gate."""
        super().__init__("cu1", 2, [theta], num_ctrl_qubits=1)
        self.base_gate = U1Gate
        self.base_gate_name = "u1"

    def _define(self):
        """
        gate cu1(lambda) a,b
        { u1(lambda/2) a; cx a,b;
          u1(-lambda/2) b; cx a,b;
          u1(lambda/2) b;
        }
        """
        definition = []
        q = QuantumRegister(2, "q")
        rule = [
            (U1Gate(self.params[0] / 2), [q[0]], []),
            (CnotGate(), [q[0], q[1]], []),
            (U1Gate(-self.params[0] / 2), [q[1]], []),
            (CnotGate(), [q[0], q[1]], []),
            (U1Gate(self.params[0] / 2), [q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return Cu1Gate(-self.params[0])


def cu1(self, theta, ctl, tgt):
    """Apply cU1 gate from a specified control (ctl) to target (tgt) qubit with angle theta.
    The cU1 gate applies a U1 gate on the target qubit when the control qubit is in state |1>.

    Examples:

        Construct a circuit with cU1 gate.

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(2)
            circuit.x(0) # This brings the quantum state from |0> to |1>
            circuit.ry(numpy.pi/2,1) # This brings the quantum state from |0> to |+>
            theta = numpy.pi/2
            circuit.cu1(theta,0,1)
            circuit.draw()

        Resulting Statevector:
        [ 0+0j, 0.707+0j, 0+0j, 0+0.707j ]
    """
    return self.append(Cu1Gate(theta), [ctl, tgt], [])


QuantumCircuit.cu1 = cu1
