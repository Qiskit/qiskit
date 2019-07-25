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
controlled-rz gate.
"""
from qiskit.circuit import ControlledGate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
import qiskit.extensions.standard.u1 as u1
import qiskit.extensions.standard.cx as cx
import qiskit.extensions.standard.rz as rz


class CrzGate(ControlledGate):
    """controlled-rz gate."""

    def __init__(self, theta):
        """Create new crz gate."""
        super().__init__("crz", 2, [theta], num_ctrl_qubits=1)

    def _define(self):
        """
        gate crz(lambda) a,b
        { u1(lambda/2) b; cx a,b;
          u1(-lambda/2) b; cx a,b;
        }
        """
        definition = []
        q = QuantumRegister(2, "q")
        rule = [
            (u1.U1Gate(self.params[0] / 2), [q[1]], []),
            (cx.CnotGate(), [q[0], q[1]], []),
            (u1.U1Gate(-self.params[0] / 2), [q[1]], []),
            (cx.CnotGate(), [q[0], q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return CrzGate(-self.params[0])


def crz(self, theta, ctl, tgt):
    """Apply crz from ctl to tgt with angle theta."""
    return self.append(CrzGate(theta), [ctl, tgt], [])


QuantumCircuit.crz = crz
