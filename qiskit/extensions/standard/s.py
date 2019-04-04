# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,arguments-differ

"""
S=diag(1,i) Clifford phase gate or its inverse.
"""
from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.decorators import _op_expand
from qiskit.qasm import pi
from qiskit.extensions.standard.u1 import U1Gate


class SGate(Gate):
    """S=diag(1,i) Clifford phase gate."""

    def __init__(self):
        """Create new S gate."""
        super().__init__("s", 1, [])

    def _define(self):
        """
        gate s a { u1(pi/2) a; }
        """
        definition = []
        q = QuantumRegister(1, "q")
        rule = [
            (U1Gate(pi/2), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return SdgGate()


class SdgGate(Gate):
    """Sdg=diag(1,-i) Clifford adjoint phase gate."""

    def __init__(self):
        """Create new Sdg gate."""
        super().__init__("sdg", 1, [])

    def _define(self):
        """
        gate sdg a { u1(-pi/2) a; }
        """
        definition = []
        q = QuantumRegister(1, "q")
        rule = [
            (U1Gate(-pi/2), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return SGate()


@_op_expand(1)
def s(self, q):
    """Apply S to q."""
    return self.append(SGate(), [q], [])


@_op_expand(1)
def sdg(self, q):
    """Apply Sdg to q."""
    return self.append(SdgGate(), [q], [])


QuantumCircuit.s = s
QuantumCircuit.sdg = sdg
CompositeGate.s = s
CompositeGate.sdg = sdg
