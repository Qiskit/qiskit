# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
controlled-Phase gate.
"""
from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.decorators import _op_expand
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.standard.h import HGate
from qiskit.extensions.standard.cx import CnotGate


class CzGate(Gate):
    """controlled-Z gate."""

    def __init__(self, circ=None):
        """Create new CZ gate."""
        super().__init__("cz", [], circ)

    def _define_decompositions(self):
        """
        gate cz a,b { h b; cx a,b; h b; }
        """
        decomposition = DAGCircuit()
        q = QuantumRegister(2, "q")
        decomposition.add_qreg(q)
        rule = [
            (HGate(), [q[1]], []),
            (CnotGate(), [q[0], q[1]], []),
            (HGate(), [q[1]], [])
        ]
        for inst in rule:
            decomposition.apply_operation_back(*inst)
        self._decompositions = [decomposition]

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse


@_op_expand(2)
def cz(self, ctl, tgt):
    """Apply CZ to circuit."""
    return self._attach(CzGate(self), [ctl, tgt], [])


QuantumCircuit.cz = cz
CompositeGate.cz = cz
