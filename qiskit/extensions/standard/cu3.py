# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
controlled-u3 gate.
"""
from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.decorators import _op_expand
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.u3 import U3Gate
from qiskit.extensions.standard.cx import CnotGate


class Cu3Gate(Gate):
    """controlled-u3 gate."""

    def __init__(self, theta, phi, lam, circ=None):
        """Create new cu3 gate."""
        super().__init__("cu3", 2, [theta, phi, lam], circ)

    def _define_decompositions(self):
        """
        gate cu3(theta,phi,lambda) c, t
        { u1((lambda-phi)/2) t; cx c,t;
          u3(-theta/2,0,-(phi+lambda)/2) t; cx c,t;
          u3(theta/2,phi,0) t;
        }
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(2, "q")
        decomposition.add_qreg(q)
        rule = [
            (U1Gate((self.params[2] - self.params[1])/2), [q[1]], []),
            (CnotGate(), [q[0], q[1]], []),
            (U3Gate(-self.params[0]/2, 0, -(self.params[1]+self.params[2])/2), [q[1]], []),
            (CnotGate(), [q[0], q[1]], []),
            (U3Gate(self.params[0]/2, self.params[1], 0), [q[1]], [])
        ]
        for inst in rule:
            decomposition.apply_operation_back(*inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        self.params[0] = -self.params[0]
        phi = self.params[1]
        self.params[1] = -self.params[2]
        self.params[2] = -phi
        self._decompositions = None
        return self


@_op_expand(2)
def cu3(self, theta, phi, lam, ctl, tgt):
    """Apply cu3 from ctl to tgt with angle theta, phi, lam."""
    return self.append(Cu3Gate(theta, phi, lam, self), [ctl, tgt], [])


QuantumCircuit.cu3 = cu3
CompositeGate.cu3 = cu3
