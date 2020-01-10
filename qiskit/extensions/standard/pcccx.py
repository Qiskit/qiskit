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
"""
The 3-control relative-phase Toffoli gate.
"""

from qiskit.circuit import QuantumCircuit, ControlledGate, QuantumRegister
from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.u2 import U2Gate
from qiskit.extensions.standard.cx import CXGate
from qiskit.extensions.standard.x import XGate
from qiskit.qasm import pi


class PCCCXGate(ControlledGate):
    """The 3-control relative-phase Toffoli gate."""

    def __init__(self):
        """Create a new PCCCX gate."""
        super().__init__('pcccx', 3, [], num_ctrl_qubits=3)
        self.base_gate = XGate
        self.base_gate_name = 'x'

    def _define(self):
        """
        gate pcccx a,b,c,d
        { u2(0,pi) d;
          u1(pi/4) d;
          cx c,d;
          u1(-pi/4) d;
          u2(0,pi) d;
          cx a,d;
          u1(pi/4) d;
          cx b,d;
          u1(-pi/4) d;
          cx a,d;
          u1(pi/4) d;
          cx b,d;
          u1(-pi/4) d;
          u2(0,pi) d;
          u1(pi/4) d;
          cx c,d;
          u1(-pi/4) d;
          u2(0,pi) d;
        }
        """
        definition = []
        q = QuantumRegister(4, 'q')

        rule = [
            (U2Gate(0, pi), [q[3]], []),  # H gate
            (U1Gate(pi / 4), [q[3]], []),  # T gate
            (CXGate(), [q[2], q[3]], []),
            (U1Gate(-pi / 4), [q[3]], []),  # inverse T gate
            (CXGate(), [q[0], q[3]], []),
            (U1Gate(pi / 4), [q[3]], []),
            (CXGate(), [q[1], q[3]], []),
            (U1Gate(-pi / 4), [q[3]], []),
            (CXGate(), [q[0], q[3]], []),
            (U1Gate(pi / 4), [q[3]], []),
            (CXGate(), [q[1], q[3]], []),
            (U1Gate(-pi / 4), [q[3]], []),
            (U2Gate(0, pi), [q[3]], []),
            (U1Gate(pi / 4), [q[3]], []),
            (CXGate(), [q[2], q[3]], []),
            (U1Gate(-pi / 4), [q[3]], []),
            (U2Gate(0, pi), [q[3]], []),
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition


def pcccx(self, ctl1, ctl2, ctl3, tgt):
    """
    Apply 3-Control Relative-Phase Toffoli gate.

    The implementation is based on https://arxiv.org/pdf/1508.03273.pdf Figure 4.
    """
    return self.append(PCCCXGate(), [ctl1, ctl2, ctl3, tgt], [])


QuantumCircuit.pcccx = pcccx
