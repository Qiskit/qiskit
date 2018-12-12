# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
controlled-u3 gate.
"""
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import InstructionSet
from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard import header  # pylint: disable=unused-import
from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.u3 import U3Gate
from qiskit.extensions.standard.cx import CnotGate


class Cu3Gate(Gate):
    """controlled-u3 gate."""

    def __init__(self, theta, phi, lam, ctl, tgt, circ=None):
        """Create new cu3 gate."""
        super().__init__("cu3", [theta, phi, lam], [ctl, tgt], circ)

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
        decomposition.add_basis_element("u1", 1, 0, 1)
        decomposition.add_basis_element("u3", 1, 0, 3)
        decomposition.add_basis_element("cx", 2, 0, 0)
        rule = [
            U1Gate((self.param[2] - self.param[1])/2, q[1]),
            CnotGate(q[0], q[1]),
            U3Gate(-self.param[0]/2, 0, -(self.param[1]+self.param[2])/2, q[1]),
            CnotGate(q[0], q[1]),
            U3Gate(self.param[0]/2, self.param[1], 0, q[1])
        ]
        for inst in rule:
            decomposition.apply_operation_back(inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        self.param[0] = -self.param[0]
        phi = self.param[1]
        self.param[1] = -self.param[2]
        self.param[2] = -phi
        self._decompositions = None
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.cu3(self.param[0], self.param[1],
                                 self.param[2], self.qargs[0], self.qargs[1]))


def cu3(self, theta, phi, lam, ctl, tgt):
    """Apply cu3 from ctl to tgt with angle theta, phi, lam."""
    if isinstance(ctl, QuantumRegister) and \
       isinstance(tgt, QuantumRegister) and len(ctl) == len(tgt):
        instructions = InstructionSet()
        for i in range(ctl.size):
            instructions.add(self.cu3(theta, phi, lam, (ctl, i), (tgt, i)))
        return instructions

    if isinstance(ctl, QuantumRegister):
        instructions = InstructionSet()
        for j in range(ctl.size):
            instructions.add(self.cu3(theta, phi, lam, (ctl, j), tgt))
        return instructions

    if isinstance(tgt, QuantumRegister):
        instructions = InstructionSet()
        for j in range(tgt.size):
            instructions.add(self.cu3(theta, phi, lam, ctl, (tgt, j)))
        return instructions

    self._check_qubit(ctl)
    self._check_qubit(tgt)
    self._check_dups([ctl, tgt])
    return self._attach(Cu3Gate(theta, phi, lam, ctl, tgt, self))


QuantumCircuit.cu3 = cu3
