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
Fredkin gate. Controlled-SWAP.
"""
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.extensions.standard.cx import CnotGate
from qiskit.extensions.standard.ccx import ToffoliGate


class FredkinGate(Gate):
    """Fredkin gate."""

    def __init__(self):
        """Create new Fredkin gate."""
        super().__init__("cswap", 3, [])

    def _define(self):
        """
        gate cswap a,b,c
        { cx c,b;
          ccx a,b,c;
          cx c,b;
        }
        """
        definition = []
        q = QuantumRegister(3, "q")
        rule = [
            (CnotGate(), [q[2], q[1]], []),
            (ToffoliGate(), [q[0], q[1], q[2]], []),
            (CnotGate(), [q[2], q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return FredkinGate()  # self-inverse


def cswap(self, ctl, tgt1, tgt2):
    """Apply Fredkin to circuit."""
    return self.append(FredkinGate(), [ctl, tgt1, tgt2], [])


QuantumCircuit.cswap = cswap
