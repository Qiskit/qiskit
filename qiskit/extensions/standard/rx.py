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
Rotation around the x-axis.
"""
import math
import numpy
from qiskit.circuit import Gate
from qiskit.circuit import ControlledGate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.u3 import U3Gate
from qiskit.extensions.standard.cx import CXGate
from qiskit.qasm import pi


class RXGate(Gate):
    """The rotation around the x-axis."""

    def __init__(self, theta):
        """Create new rx single qubit gate."""
        super().__init__('rx', 1, [theta])

    def _define(self):
        """
        gate rx(theta) a {r(theta, 0) a;}
        """
        from qiskit.extensions.standard.r import RGate
        definition = []
        q = QuantumRegister(1, 'q')
        rule = [
            (RGate(self.params[0], 0), [q[0]], [])
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
            return CRXGate(self.params[0])
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label)

    def inverse(self):
        """Invert this gate.

        rx(theta)^dagger = rx(-theta)
        """
        return RXGate(-self.params[0])

    def to_matrix(self):
        """Return a numpy.array for the RX gate."""
        cos = math.cos(self.params[0] / 2)
        sin = math.sin(self.params[0] / 2)
        return numpy.array([[cos, -1j * sin],
                            [-1j * sin, cos]], dtype=complex)


def rx(self, theta, q):  # pylint: disable=invalid-name
    """Apply RX to q."""
    return self.append(RXGate(theta), [q], [])


QuantumCircuit.rx = rx


class CRXMeta(type):
    """A metaclass to ensure that CrxGate and CRXGate are of the same type.

    Can be removed when CrxGate gets removed.
    """
    @classmethod
    def __instancecheck__(mcs, inst):
        return type(inst) in {CRXGate, CrxGate}  # pylint: disable=unidiomatic-typecheck


class CRXGate(ControlledGate, metaclass=CRXMeta):
    """The controlled-rx gate."""

    def __init__(self, theta):
        """Create new crx gate."""
        super().__init__('crx', 2, [theta], num_ctrl_qubits=1)
        self.base_gate = RXGate
        self.base_gate_name = 'rx'

    def _define(self):
        """
        gate cu3(theta,phi,lambda) c, t
        { u1(pi/2) t;
          cx c,t;
          u3(-theta/2,0,0) t;
          cx c,t;
          u3(theta/2,-pi/2,0) t;
        }
        """
        definition = []
        q = QuantumRegister(2, 'q')
        rule = [
            (U1Gate(pi / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (U3Gate(-self.params[0] / 2, 0, 0), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (U3Gate(self.params[0] / 2, -pi / 2, 0), [q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return CRXGate(-self.params[0])


class CrxGate(CRXGate, metaclass=CRXMeta):
    """The deprecated CRXGate class."""

    def __init__(self, theta):
        import warnings
        warnings.warn('The class CrxGate is deprecated as of 0.12.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the class CRXGate instead.',
                      DeprecationWarning, stacklevel=2)
        super().__init__(theta)


def crx(self, theta, ctl, tgt):
    """Apply crx from ctl to tgt with angle theta."""
    return self.append(CRXGate(theta), [ctl, tgt], [])


QuantumCircuit.crx = crx
