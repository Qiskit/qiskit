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
controlled-u3 gate.
"""
from qiskit.circuit import ControlledGate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.u3 import U3Gate
from qiskit.extensions.standard.cx import CnotGate


class Cu3Gate(ControlledGate):
    """controlled-u3 gate."""

    def __init__(self, theta, phi, lam):
        """Create new cu3 gate."""
        super().__init__("cu3", 2, [theta, phi, lam], num_ctrl_qubits=1)
        self.base_gate = U3Gate
        self.base_gate_name = "u3"

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


def cu3(self, theta, phi, lam, ctl, tgt):
        """Apply cU3 gate from a specified control (ctl) to target (tgt) qubit with angle theta, phi, and lam.
    The cU3 gate applies a U3 gate on the target qubit when the control qubit is in state |1>.

    Example:
    circuit = QuantumCircuit(2)
    circuit.x(0) # This brings the quantum state from |0> to |1>
    theta = numpy.pi/2
    phi = numpy.pi/2
    lam = numpy.pi/2
    circuit.cu3(theta,phi,lam,0,1)
    circuit.draw()
            ┌───┐                      
    q_0: |0>┤ X ├──────────■───────────
            └───┘┌─────────┴──────────┐
    q_1: |0>─────┤ U3(pi/2,pi/2,pi/2) ├
                 └────────────────────┘
    Resulting Statevector:
    [ 0+0j, 0.707+0j, 0+0j, 0+0.707j ]
    """
    return self.append(Cu3Gate(theta, phi, lam), [ctl, tgt], [])


QuantumCircuit.cu3 = cu3
