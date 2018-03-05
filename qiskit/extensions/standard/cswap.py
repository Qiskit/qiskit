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
Fredkin gate. Controlled-SWAP.
"""
from qiskit import CompositeGate
from qiskit import QuantumCircuit
from qiskit.extensions.standard import header  # pylint: disable=unused-import


class FredkinGate(CompositeGate):
    """Fredkin gate."""

    def __init__(self, ctl, tgt1, tgt2, circ=None):
        """Create new Fredkin gate."""
        super().__init__("fredkin", [], [ctl, tgt1, tgt2], circ)
        self.cx(tgt2, tgt1)
        self.ccx(ctl, tgt1, tgt2)
        self.cx(tgt2, tgt1)

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.cswap(self.arg[0], self.arg[1], self.arg[2]))


def cswap(self, ctl, tgt1, tgt2):
    """Apply Fredkin to circuit."""
    self._check_qubit(ctl)
    self._check_qubit(tgt1)
    self._check_qubit(tgt2)
    self._check_dups([ctl, tgt1, tgt2])
    return self._attach(FredkinGate(ctl, tgt1, tgt2, self))


QuantumCircuit.cswap = cswap
CompositeGate.cswap = cswap
