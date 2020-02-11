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

"""The simplified 3-controlled Toffoli gate."""

import numpy

from qiskit.circuit import QuantumCircuit, Gate, QuantumRegister
from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.u2 import U2Gate
from qiskit.extensions.standard.x import CnotGate
from qiskit.qasm import pi


class RCCCXGate(Gate):
    """The simplified 3-controlled Toffoli gate.

    The simplified Toffoli gate implements the Toffoli gate up to relative phases.
    Note, that the simplified Toffoli is not equivalent to the Toffoli. But can be used in places
    where the Toffoli gate is uncomputed again.

    This concrete implementation is from https://arxiv.org/abs/1508.03273, the complete circuit
    of Fig. 4.
    """

    def __init__(self):
        """Create a new PCCCX gate."""
        super().__init__('rcccx', 4, [])

    def _define(self):
        """
        gate rcccx a,b,c,d
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
            (CnotGate(), [q[2], q[3]], []),
            (U1Gate(-pi / 4), [q[3]], []),  # inverse T gate
            (U2Gate(0, pi), [q[3]], []),
            (CnotGate(), [q[0], q[3]], []),
            (U1Gate(pi / 4), [q[3]], []),
            (CnotGate(), [q[1], q[3]], []),
            (U1Gate(-pi / 4), [q[3]], []),
            (CnotGate(), [q[0], q[3]], []),
            (U1Gate(pi / 4), [q[3]], []),
            (CnotGate(), [q[1], q[3]], []),
            (U1Gate(-pi / 4), [q[3]], []),
            (U2Gate(0, pi), [q[3]], []),
            (U1Gate(pi / 4), [q[3]], []),
            (CnotGate(), [q[2], q[3]], []),
            (U1Gate(-pi / 4), [q[3]], []),
            (U2Gate(0, pi), [q[3]], []),
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def to_matrix(self):
        """Return a numpy.array for the PCCCX gate."""
        return numpy.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1j, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=complex)


def rcccx(self, ctl1, ctl2, ctl3, tgt):
    """Apply the simplified, relative-phase 3-control Toffoli gate."""
    return self.append(RCCCXGate(), [ctl1, ctl2, ctl3, tgt], [])


QuantumCircuit.rcccx = rcccx
