# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The simplified Toffoli gate."""

import numpy

from qiskit.circuit import QuantumCircuit, Gate, QuantumRegister
from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.u2 import U2Gate
from qiskit.extensions.standard.x import CnotGate
from qiskit.qasm import pi


class RCCXGate(Gate):
    """The simplified Toffoli gate, also referred to as Margolus gate.

    The simplified Toffoli gate implements the Toffoli gate up to relative phases.
    This implementation requires three CX gates which is the minimal amount possible,
    as shown in https://arxiv.org/abs/quant-ph/0312225.
    Note, that the simplified Toffoli is not equivalent to the Toffoli. But can be used in places
    where the Toffoli gate is uncomputed again.

    This concrete implementation is from https://arxiv.org/abs/1508.03273, the dashed box
    of Fig. 3.
    """

    def __init__(self):
        """Create a new simplified CCX gate."""
        super().__init__('rccx', 3, [])

    def _define(self):
        """
        gate rccx a,b,c
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

    def to_matrix(self):
        """Return a numpy.array for the simplified CCX gate."""
        return numpy.array([[1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, -1j],
                            [0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, -1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 1j, 0, 0, 0, 0]], dtype=complex)


def rccx(self, ctl1, ctl2, tgt):
    """Apply the simplified, relative-phase Toffoli gate."""
    return self.append(RCCXGate(), [ctl1, ctl2, tgt], [])


QuantumCircuit.rccx = rccx
