# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
controlled-rz gate.
"""
from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.decorators import _op_expand
from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.cx import CnotGate


class CrzGate(Gate):
    """controlled-rz gate."""

    def __init__(self, theta, circ=None):
        """Create new crz gate."""
        super().__init__("crz", 2, [theta], circ)

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
            (U1Gate(self.params[0]/2), [q[1]], []),
            (CnotGate(), [q[0], q[1]], []),
            (U1Gate(-self.params[0]/2), [q[1]], []),
            (CnotGate(), [q[0], q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        self.params[0] = -self.params[0]
        self.definition = None
        return self


@_op_expand(2)
def crz(self, theta, ctl, tgt):
    """Apply crz from ctl to tgt with angle theta."""
    return self.append(CrzGate(theta, self), [ctl, tgt], [])


QuantumCircuit.crz = crz
CompositeGate.crz = crz
