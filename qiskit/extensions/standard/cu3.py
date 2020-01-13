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
Controlled-u3 gate.
"""
from qiskit.circuit import ControlledGate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.u3 import U3Gate
from qiskit.extensions.standard.cx import CXGate


class CU3Meta(type):
    """
    Metaclass to ensure that Cu3 and CU3 are of the same type.
    Can be removed when Cu3Gate gets removed.
    """
    @classmethod
    def __instancecheck__(mcs, inst):
        return type(inst) in {CU3Gate, Cu3Gate}  # pylint: disable=unidiomatic-typecheck


class CU3Gate(ControlledGate, metaclass=CU3Meta):
    """The controlled-u3 gate."""

    def __init__(self, theta, phi, lam):
        """Create new cu3 gate."""
        super().__init__('cu3', 2, [theta, phi, lam], num_ctrl_qubits=1)
        self.base_gate = U3Gate
        self.base_gate_name = 'u3'

    def _define(self):
        """
        gate cu3(theta,phi,lambda) c, t
        { u1((lambda+phi)/2) c;
          u1((lambda-phi)/2) t;
          cx c,t;
          u3(-theta/2,0,-(phi+lambda)/2) t;
          cx c,t;
          u3(theta/2,phi,0) t;
        }
        """
        definition = []
        q = QuantumRegister(2, 'q')
        rule = [
            (U1Gate((self.params[2] + self.params[1]) / 2), [q[0]], []),
            (U1Gate((self.params[2] - self.params[1]) / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (U3Gate(-self.params[0] / 2, 0, -(self.params[1] + self.params[2]) / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (U3Gate(self.params[0] / 2, self.params[1], 0), [q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return CU3Gate(-self.params[0], -self.params[2], -self.params[1])


class Cu3Gate(CU3Gate, metaclass=CU3Meta):
    """
    Deprecated CU3Gate class.
    """
    def __init__(self, theta, phi, lam):
        import warnings
        warnings.warn('Cu3Gate is deprecated, use CU3Gate (uppercase) instead!', DeprecationWarning,
                      2)
        super().__init__(theta, phi, lam)


def cu3(self, theta, phi, lam, ctl, tgt):
    """Apply cu3 from ctl to tgt with angle theta, phi, lam."""
    return self.append(CU3Gate(theta, phi, lam), [ctl, tgt], [])


QuantumCircuit.cu3 = cu3
