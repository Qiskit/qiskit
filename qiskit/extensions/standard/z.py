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
Pauli Z (phase-flip) gate.
"""
import numpy
from qiskit.circuit import Gate
from qiskit.circuit import ControlledGate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.qasm import pi


class ZGate(Gate):
    """Pauli Z (phase-flip) gate."""

    def __init__(self, label=None):
        """Create new Z gate."""
        super().__init__('z', 1, [], label=label)

    def _define(self):
        from qiskit.extensions.standard.u1 import U1Gate
        definition = []
        q = QuantumRegister(1, 'q')
        rule = [
            (U1Gate(pi), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def control(self, num_ctrl_qubits=1, label=None):
        """Controlled version of this gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            return CzGate()
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label)

    def inverse(self):
        """Invert this gate."""
        return ZGate()  # self-inverse

    def to_matrix(self):
        """Return a numpy.array for the Z gate."""
        return numpy.array([[1, 0],
                            [0, -1]], dtype=complex)


def z(self, q):
    """Apply Z to q."""
    return self.append(ZGate(), [q], [])


QuantumCircuit.z = z


class CZMeta(type):
    """A metaclass to ensure that CzGate and CZGate are of the same type.

    Can be removed when CzGate gets removed.
    """
    @classmethod
    def __instancecheck__(mcs, inst):
        return type(inst) in {CZGate, CzGate}  # pylint: disable=unidiomatic-typecheck


class CZGate(ControlledGate, metaclass=CZMeta):
    """The controlled-Z gate."""

    def __init__(self, label=None):
        """Create new CZ gate."""
        super().__init__('cz', 2, [], label=label, num_ctrl_qubits=1)
        self.base_gate = ZGate
        self.base_gate_name = 'z'

    def _define(self):
        """
        gate cz a,b { h b; cx a,b; h b; }
        """
        from qiskit.extensions.standard.h import HGate
        from qiskit.extensions.standard.x import CXGate
        definition = []
        q = QuantumRegister(2, 'q')
        rule = [
            (HGate(), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (HGate(), [q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return CZGate()  # self-inverse

    def to_matrix(self):
        """Return a numpy.array for the CZ gate."""
        return numpy.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, -1]], dtype=complex)


class CzGate(CZGate, metaclass=CZMeta):
    """The deprecated CZGate class."""
    def __init__(self):
        import warnings
        warnings.warn('The class CzGate is deprecated as of 0.12.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the class CZGate instead.',
                      DeprecationWarning, stacklevel=2)
        super().__init__()


def cz(self, ctl, tgt):  # pylint: disable=invalid-name
    """Apply CZ to circuit."""
    return self.append(CZGate(), [ctl, tgt], [])


QuantumCircuit.cz = cz
