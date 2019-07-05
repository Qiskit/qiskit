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
Toffoli gate. Controlled-Controlled-X.
"""

import numpy

from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.extensions.standard.h import HGate
from qiskit.extensions.standard.cx import CnotGate
from qiskit.extensions.standard.t import TGate
from qiskit.extensions.standard.t import TdgGate


class ToffoliGate(Gate):
    """Toffoli gate."""

    def __init__(self):
        """Create new Toffoli gate."""
        super().__init__("ccx", 3, [])

    def _define(self):
        """
        gate ccx a,b,c
        {
        h c; cx b,c; tdg c; cx a,c;
        t c; cx b,c; tdg c; cx a,c;
        t b; t c; h c; cx a,b;
        t a; tdg b; cx a,b;}
        """
        definition = []
        q = QuantumRegister(3, "q")
        rule = [
            (HGate(), [q[2]], []),
            (CnotGate(), [q[1], q[2]], []),
            (TdgGate(), [q[2]], []),
            (CnotGate(), [q[0], q[2]], []),
            (TGate(), [q[2]], []),
            (CnotGate(), [q[1], q[2]], []),
            (TdgGate(), [q[2]], []),
            (CnotGate(), [q[0], q[2]], []),
            (TGate(), [q[1]], []),
            (TGate(), [q[2]], []),
            (HGate(), [q[2]], []),
            (CnotGate(), [q[0], q[1]], []),
            (TGate(), [q[0]], []),
            (TdgGate(), [q[1]], []),
            (CnotGate(), [q[0], q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return ToffoliGate()  # self-inverse

    def to_matrix(self):
        """Return a Numpy.array for the Toffoli gate."""
        return numpy.array([[1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0]], dtype=complex)


def ccx(self, ctl1, ctl2, tgt):
    """Apply Toffoli to from ctl1 and ctl2 to tgt."""
    return self.append(ToffoliGate(), [ctl1, ctl2, tgt], [])


QuantumCircuit.ccx = ccx
