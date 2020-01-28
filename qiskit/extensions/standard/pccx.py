# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""The 2-control relative-phase Toffoli gate."""

from qiskit.circuit import QuantumCircuit, ControlledGate, QuantumRegister
from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.u2 import U2Gate
from qiskit.extensions.standard.cx import CnotGate
from qiskit.extensions.standard.x import XGate
from qiskit.qasm import pi


class PCCnotGate(ControlledGate):
    """The 2-control relative-phase Toffoli gate."""

    def __init__(self):
        """Create a new PCCX gate."""
        super().__init__('pccx', 3, [], num_ctrl_qubits=2)
        self.base_gate = XGate
        self.base_gate_name = 'x'

    def _define(self):
        """
        gate pccx a,b,c
        { u2(0,pi) c;
          u1(pi/4) c;
          cx b, c;
          u1(-pi/4) c;
          cx a, c;
          u1(pi/4) c;
          cx b, c;
          u1(-pi/4) c;
          u2(0,pi) c;
        }
        """
        definition = []
        q = QuantumRegister(3, 'q')
        rule = [
            (U2Gate(0, pi), [q[2]], []),  # H gate
            (U1Gate(pi / 4), [q[2]], []),  # T gate
            (CnotGate(), [q[1], q[2]], []),
            (U1Gate(-pi / 4), [q[2]], []),  # inverse T gate
            (CnotGate(), [q[0], q[2]], []),
            (U1Gate(pi / 4), [q[2]], []),
            (CnotGate(), [q[1], q[2]], []),
            (U1Gate(-pi / 4), [q[2]], []),  # inverse T gate
            (U2Gate(0, pi), [q[2]], []),  # H gate
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition


def pccx(self, ctl1, ctl2, tgt):
    """Apply the 2-Control Relative-Phase Toffoli gate.

    The implementation is based on https://arxiv.org/pdf/1508.03273.pdf Figure 3.
    """
    return self.append(PCCXGate(), [ctl1, ctl2, tgt], [])


QuantumCircuit.pccx = pccx
