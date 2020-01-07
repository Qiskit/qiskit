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
Controlled-Swap gate or Fredkin gate.
"""

from qiskit.circuit import ControlledGate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.extensions.standard.cx import CXGate
from qiskit.extensions.standard.ccx import CCXGate
from qiskit.extensions.standard.swap import SwapGate


class CSwapGate(ControlledGate):
    """The controlled-swap gate, also called Fredkin gate."""

    def __init__(self):
        """Create new CSwap gate."""
        super().__init__('cswap', 3, [], num_ctrl_qubits=1)
        self.base_gate = SwapGate
        self.base_gate_name = 'swap'

    def _define(self):
        """
        gate cswap a,b,c
        { cx c,b;
          ccx a,b,c;
          cx c,b;
        }
        """
        definition = []
        q = QuantumRegister(3, 'q')
        rule = [
            (CXGate(), [q[2], q[1]], []),
            (CCXGate(), [q[0], q[1], q[2]], []),
            (CXGate(), [q[2], q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return CSwapGate()  # self-inverse


def cswap(self, ctl, tgt1, tgt2):
    """Apply CSwap to circuit."""
    return self.append(CSwapGate(), [ctl, tgt1, tgt2], [])


# support both cswap and fredkin as methods of QuantumCircuit
QuantumCircuit.cswap = cswap
QuantumCircuit.fredkin = cswap
