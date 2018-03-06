# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Fundamental controlled-NOT gate.
"""
from qiskit import CompositeGate
from qiskit import Gate
from qiskit import QuantumCircuit
from qiskit.extensions.standard import header  # pylint: disable=unused-import


class CXBase(Gate):
    """Fundamental controlled-NOT gate."""

    def __init__(self, ctl, tgt, circ=None):
        """Create new CX instruction."""
        super().__init__("CX", [], [ctl, tgt], circ)

    def qasm(self):
        """Return OPENQASM string."""
        ctl = self.arg[0]
        tgt = self.arg[1]
        return self._qasmif("CX %s[%d],%s[%d];" % (ctl[0].name, ctl[1],
                                                   tgt[0].name, tgt[1]))

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.cx_base(self.arg[0], self.arg[1]))


def cx_base(self, ctl, tgt):
    """Apply CX ctl, tgt."""
    self._check_qubit(ctl)
    self._check_qubit(tgt)
    self._check_dups([ctl, tgt])
    return self._attach(CXBase(ctl, tgt, self))


QuantumCircuit.cx_base = cx_base
CompositeGate.cx_base = cx_base
