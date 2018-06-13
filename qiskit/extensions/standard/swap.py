# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
SWAP gate.
"""
from qiskit import CompositeGate
from qiskit import Gate
from qiskit import QuantumCircuit
from qiskit._instructionset import InstructionSet
from qiskit._quantumregister import QuantumRegister
from qiskit.extensions.standard import header  # pylint: disable=unused-import


class SwapGate(Gate):
    """SWAP gate."""

    def __init__(self, ctl, tgt, circ=None):
        """Create new SWAP gate."""
        super().__init__("swap", [], [ctl, tgt], circ)

    def qasm(self):
        """Return OPENQASM string."""
        ctl = self.arg[0]
        tgt = self.arg[1]
        return self._qasmif("swap %s[%d],%s[%d];" % (ctl[0].name, ctl[1],
                                                     tgt[0].name, tgt[1]))

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.swap(self.arg[0], self.arg[1]))


def swap(self, ctl, tgt):
    """Apply SWAP from ctl to tgt."""
    if isinstance(ctl, QuantumRegister) and \
            isinstance(tgt, QuantumRegister) and len(ctl) == len(tgt):
        instructions = InstructionSet()
        for j in range(ctl.size):
            instructions.add(self.swap((ctl, j), (tgt, j)))
        return instructions

    self._check_qubit(ctl)
    self._check_qubit(tgt)
    self._check_dups([ctl, tgt])
    return self._attach(SwapGate(ctl, tgt, self))


QuantumCircuit.swap = swap
CompositeGate.swap = swap
