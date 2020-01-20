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
Controlled-Y gate.
"""
import numpy
from qiskit.circuit import ControlledGate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.extensions.standard.y import YGate
from qiskit.extensions.standard.s import SGate
from qiskit.extensions.standard.s import SInvGate
from qiskit.extensions.standard.cx import CXGate


class CYMeta(type):
    """
    Metaclass to ensure that Cy and CY are of the same type.
    Can be removed when CyGate gets removed.
    """
    @classmethod
    def __instancecheck__(mcs, inst):
        return type(inst) in {CYGate, CyGate}  # pylint: disable=unidiomatic-typecheck


class CYGate(ControlledGate, metaclass=CYMeta):
    """The controlled-Y gate."""

    def __init__(self):
        """Create new CY gate."""
        super().__init__('cy', 2, [], num_ctrl_qubits=1)
        self.base_gate = YGate
        self.base_gate_name = 'y'

    def _define(self):
        """
        gate cy a,b { sinv b; cx a,b; s b; }
        """
        definition = []
        q = QuantumRegister(2, "q")
        rule = [
            (SInvGate(), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (SGate(), [q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return CYGate()  # self-inverse

    def to_matrix(self):
        """Return a numpy.array for the CY gate."""
        return numpy.array([[1, 0, 0, 0],
                            [0, 0, 0, -1j],
                            [0, 0, 1, 0],
                            [0, 1j, 0, 0]], dtype=complex)


class CyGate(CYGate, metaclass=CYMeta):
    """
    Deprecated CYGate class.
    """
    def __init__(self):
        import warnings
        warnings.warn('CyGate is deprecated, use CYGate (uppercase) instead!', DeprecationWarning,
                      2)
        super().__init__()


def cy(self, ctl, tgt):  # pylint: disable=invalid-name
    """Apply CY to circuit."""
    return self.append(CYGate(), [ctl, tgt], [])


QuantumCircuit.cy = cy
