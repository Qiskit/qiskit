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
T=sqrt(S) phase gate or its inverse.
"""
import warnings
import numpy
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.qasm import pi
from qiskit.extensions.standard.u1 import U1Gate


class TGate(Gate):
    """T Gate: pi/4 rotation around Z axis."""

    def __init__(self, label=None):
        """Create new T gate."""
        super().__init__('t', 1, [], label=label)

    def _define(self):
        """
        gate t a { u1(pi/4) a; }
        """
        definition = []
        q = QuantumRegister(1, 'q')
        rule = [
            (U1Gate(pi / 4), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return TInvGate()

    def to_matrix(self):
        """Return a numpy.array for the T gate."""
        return numpy.array([[1, 0],
                            [0, (1 + 1j) / numpy.sqrt(2)]], dtype=complex)


class TinvMeta(type):
    """
    A metaclass to ensure that Tinv and Tdg are of the same type.
    Can be removed when TInvGate gets removed.
    """
    @classmethod
    def __instancecheck__(mcs, inst):
        return type(inst) in {TInvGate, TdgGate}  # pylint: disable=unidiomatic-typecheck


class TInvGate(Gate, metaclass=TinvMeta):
    """T Gate: -pi/4 rotation around Z axis."""

    def __init__(self, label=None):
        """Create a new Tinv gate."""
        super().__init__('tinv', 1, [], label=label)

    def _define(self):
        """
        gate t a { u1(pi/4) a; }
        """
        definition = []
        q = QuantumRegister(1, 'q')
        rule = [
            (U1Gate(-pi / 4), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return TGate()

    def to_matrix(self):
        """Return a numpy.array for the inverse T gate."""
        return numpy.array([[1, 0],
                            [0, (1 - 1j) / numpy.sqrt(2)]], dtype=complex)


class TdgGate(TInvGate, metaclass=TinvMeta):
    """The deprecated Tinv gate."""

    def __init__(self):
        warnings.warn('TdgGate is deprecated, use TInvGate instead!', DeprecationWarning, 2)
        super().__init__()


def t(self, q):  # pylint: disable=invalid-name
    """Apply T to q."""
    return self.append(TGate(), [q], [])


def tinv(self, q):
    """Apply Tdg to q."""
    return self.append(TInvGate(), [q], [])


def tdg(self, q):
    """Apply Tdg (deprecated!) to q."""
    warnings.warn('tdg() is deprecated, use tinv() instead!', DeprecationWarning, 2)
    return self.append(TdgGate(), [q], [])


QuantumCircuit.t = t
QuantumCircuit.tinv = tinv
QuantumCircuit.tdg = tdg  # deprecated, remove once TdgGate is removed
