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
Controlled-not gate.
"""
import numpy

from qiskit.circuit import ControlledGate
from qiskit.circuit import QuantumCircuit
from qiskit.extensions.standard.x import XGate


class CXMeta(type):
    """
    Metaclass to ensure that Cnot and CX are of the same type.
    Can be removed when CnotGate gets removed.
    """
    @classmethod
    def __instancecheck__(mcs, inst):
        return type(inst) in {CnotGate, CXGate}  # pylint: disable=unidiomatic-typecheck


class CXGate(ControlledGate, metaclass=CXMeta):
    """The controlled-X gate."""

    def __init__(self):
        """Create new cx gate."""
        super().__init__('cx', 2, [], num_ctrl_qubits=1)
        self.base_gate = XGate
        self.base_gate_name = 'x'

    def inverse(self):
        """Invert this gate."""
        return CXGate()  # self-inverse

    def to_matrix(self):
        """Return a numpy.array for the CX gate."""
        return numpy.array([[1, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0]], dtype=complex)


class CnotGate(CXGate, metaclass=CXMeta):
    """
    Deprecated CXGate class.
    """
    def __init__(self):
        import warnings
        warnings.warn('CnotGate is deprecated, use CXGate (uppercase) instead!', DeprecationWarning,
                      2)
        super().__init__()


def cx(self, ctl, tgt):  # pylint: disable=invalid-name
    """Apply CX from ctl to tgt."""
    return self.append(CXGate(), [ctl, tgt], [])


# support both cx and cnot in QuantumCircuits
QuantumCircuit.cx = cx
QuantumCircuit.cnot = cx
