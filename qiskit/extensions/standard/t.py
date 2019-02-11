# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
T=sqrt(S) phase gate or its inverse.
"""
from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.decorators import _op_expand
from qiskit.qasm import pi
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard.u1 import U1Gate


class TGate(Gate):
    """T Gate: pi/4 rotation around Z axis."""

    def __init__(self, circ=None):
        """Create new T gate."""
        super().__init__("t", [], circ)

    def _define_decompositions(self):
        """
        gate t a { u1(pi/4) a; }
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(1, "q")
        decomposition.add_qreg(q)
        rule = [
            (U1Gate(pi/4), [q[0]], [])
        ]
        for inst in rule:
            decomposition.apply_operation_back(*inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        print(self.circuit.data[-1])
        inv = TdgGate()
        _, qargs, cargs = self.circuit.data[-1]
        self.circuit.data[-1] = (inv, qargs, cargs)
        return inv


class TdgGate(Gate):
    """T Gate: -pi/4 rotation around Z axis."""

    def __init__(self, circ=None):
        """Create new Tdg gate."""
        super().__init__("tdg", [], circ)

    def _define_decompositions(self):
        """
        gate t a { u1(pi/4) a; }
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(1, "q")
        decomposition.add_qreg(q)
        rule = [
            (U1Gate(-pi/4), q[0], [])
        ]
        for inst in rule:
            decomposition.apply_operation_back(*inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        inv = TGate()
        _, qargs, cargs = self.circuit.data[-1]
        self.circuit.data[-1] = (inv, qargs, cargs)
        return inv


@_op_expand(1)
def t(self, q):
    """Apply T to q."""
    return self._attach(TGate(self), [q], [])


@_op_expand(1)
def tdg(self, q):
    """Apply Tdg to q."""
    return self._attach(TdgGate(self), [q], [])


QuantumCircuit.t = t
QuantumCircuit.tdg = tdg
CompositeGate.t = t
CompositeGate.tdg = tdg
