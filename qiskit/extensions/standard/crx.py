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
controlled-rx gate.
"""
from qiskit.circuit import ControlledGate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.extensions.standard.cu3 import Cu3Gate
from qiskit.extensions.standard.rx import RXGate
from qiskit.qasm import pi

class CrxGate(ControlledGate):
    """controlled-rx gate."""

    def __init__(self, theta):
        """Create new crx gate."""
        super().__init__('crx', 2, [theta], num_ctrl_qubits=1)
        self.base_gate = RXGate
        self.base_gate_name = 'rx'

    def _define(self):
        """
        gate crx(lambda) a,b
        { cu3(lambda,-pi/2,pi/2) a,b;
        }

        """
        definition = []
        q = QuantumRegister(2, 'q')
        rule = [
            (Cu3Gate(self.params[0], -pi/2, pi/2), [q[0], q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return CrxGate(-self.params[0])


def crx(self, theta, ctl, tgt):
    """Apply crx from ctl to tgt with angle theta."""
    return self.append(CrxGate(theta), [ctl, tgt], [])

QuantumCircuit.crx = crx
