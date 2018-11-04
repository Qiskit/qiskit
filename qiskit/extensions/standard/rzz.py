# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
two-qubit ZZ-rotation gate.
"""
from qiskit import Gate
from qiskit import QuantumCircuit
from qiskit._instructionset import InstructionSet
from qiskit._quantumregister import QuantumRegister
from qiskit.extensions.standard import header  # pylint: disable=unused-import


class RZZGate(Gate):
    """Two-qubit ZZ-rotation gate."""

    def __init__(self, theta, ctl, tgt, circ=None):
        """Create new rzz gate."""
        super().__init__("rzz", [theta], [ctl, tgt], circ)

    def inverse(self):
        """Invert this gate."""
        self.param[0] = -self.param[0]
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.rzz(self.param[0], self.qargs[0], self.qargs[1]))


def rzz(self, theta, ctl, tgt):
    """Apply RZZ to circuit."""
    if isinstance(ctl, QuantumRegister) and \
       isinstance(tgt, QuantumRegister) and len(ctl) == len(tgt):
        instructions = InstructionSet()
        for i in range(ctl.size):
            instructions.add(self.rzz(theta, (ctl, i), (tgt, i)))
        return instructions

    self._check_qubit(ctl)
    self._check_qubit(tgt)
    self._check_dups([ctl, tgt])
    return self._attach(RZZGate(theta, ctl, tgt, self))


# Add to QuantumCircuit class
QuantumCircuit.rzz = rzz
