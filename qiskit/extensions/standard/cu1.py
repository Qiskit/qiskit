# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
controlled-u1 gate.
"""
from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.decorators import _op_expand
from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.cx import CnotGate


class Cu1Gate(Gate):
    """controlled-u1 gate."""

    def __init__(self, theta):
        """Create new cu1 gate."""
        super().__init__("cu1", 2, [theta])

    def _define(self):
        """
        gate cu1(lambda) a,b
        { u1(lambda/2) a; cx a,b;
          u1(-lambda/2) b; cx a,b;
          u1(lambda/2) b;
        }
        """
        definition = []
        q = QuantumRegister(2, "q")
        rule = [
            (U1Gate(self.params[0]/2), [q[0]], []),
            (CnotGate(), [q[0], q[1]], []),
            (U1Gate(-self.params[0]/2), [q[1]], []),
            (CnotGate(), [q[0], q[1]], []),
            (U1Gate(self.params[0]/2), [q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return Cu1Gate(-self.params[0])


@_op_expand(2)
def cu1(self, theta, ctl, tgt):
    """Apply cu1 from ctl to tgt with angle theta."""
    return self.append(Cu1Gate(theta), [ctl, tgt], [])


QuantumCircuit.cu1 = cu1
CompositeGate.cu1 = cu1
