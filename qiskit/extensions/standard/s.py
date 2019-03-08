# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
S=diag(1,i) Clifford phase gate or its inverse.
"""
from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.decorators import _op_expand
from qiskit.qasm import pi
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard.u1 import U1Gate


class SGate(Gate):
    """S=diag(1,i) Clifford phase gate."""

    def __init__(self, circ=None):
        """Create new S gate."""
        super().__init__("s", 1, [], circ)

    def _define_decompositions(self):
        """
        gate s a { u1(pi/2) a; }
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(1, "q")
        decomposition.add_qreg(q)
        rule = [
            (U1Gate(pi/2), [q[0]], [])
        ]
        for inst in rule:
            decomposition.apply_operation_back(*inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        inv = SdgGate()
        _, qargs, cargs = self.circuit.data[-1]
        self.circuit.data[-1] = (inv, qargs, cargs)
        return inv


class SdgGate(Gate):
    """Sdg=diag(1,-i) Clifford adjoin phase gate."""

    def __init__(self, circ=None):
        """Create new Sdg gate."""
        super().__init__("sdg", 1, [], circ)

    def _define_decompositions(self):
        """
        gate sdg a { u1(-pi/2) a; }
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(1, "q")
        decomposition.add_qreg(q)
        rule = [
            (U1Gate(-pi/2), q[0], [])
        ]
        for inst in rule:
            decomposition.apply_operation_back(*inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        inv = SGate()
        _, qargs, cargs = self.circuit.data[-1]
        self.circuit.data[-1] = (inv, qargs, cargs)
        return inv


@_op_expand(1)
def s(self, q):
    """Apply S to q."""
    return self.append(SGate(self), [q], [])


@_op_expand(1)
def sdg(self, q):
    """Apply Sdg to q."""
    return self.append(SdgGate(self), [q], [])


QuantumCircuit.s = s
QuantumCircuit.sdg = sdg
CompositeGate.s = s
CompositeGate.sdg = sdg
