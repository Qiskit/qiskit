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

# pylint: disable=invalid-name

"""
controlled-H gate.
"""
import numpy as np

from qiskit.circuit import ControlledGate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
import qiskit.extensions.standard.h as h
import qiskit.extensions.standard.cx as cx
import qiskit.extensions.standard.t as t
import qiskit.extensions.standard.s as s


class CHGate(ControlledGate):
    """controlled-H gate."""

    def __init__(self):
        """Create new CH gate."""
        super().__init__("ch", 2, [], num_ctrl_qubits=1)
        self.base_gate = h.HGate

    def _define(self):
        """
        gate ch a,b {
            s b;
            h b;
            t b;
            cx a, b;
            tdg b;
            h b;
            sdg b;
        }
        """
        definition = []
        q = QuantumRegister(2, "q")
        rule = [
            (s.SGate(), [q[1]], []),
            (h.HGate(), [q[1]], []),
            (t.TGate(), [q[1]], []),
            (cx.CnotGate(), [q[0], q[1]], []),
            (t.TdgGate(), [q[1]], []),
            (h.HGate(), [q[1]], []),
            (s.SdgGate(), [q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return CHGate()  # self-inverse

    def to_matrix(self):
        """Return a Numpy.array for the Ch gate."""
        return np.array([[1, 0, 0, 0],
                         [0, 1/np.sqrt(2), 0, 1/np.sqrt(2)],
                         [0, 0, 1, 0],
                         [0, 1/np.sqrt(2), 0, -1/np.sqrt(2)]], dtype=complex)


def ch(self, ctl, tgt):
    """Apply CH from ctl to tgt."""
    return self.append(CHGate(), [ctl, tgt], [])


QuantumCircuit.ch = ch
