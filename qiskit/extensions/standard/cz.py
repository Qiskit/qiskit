# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
controlled-Phase gate.
"""
from qiskit import CompositeGate
from qiskit import Gate
from qiskit import QuantumCircuit
from qiskit._instructionset import InstructionSet
from qiskit._quantumregister import QuantumRegister
from qiskit.extensions.standard import header  # pylint: disable=unused-import
from qiskit.extensions.standard.x import XGate


class CzGate(Gate):
    """controlled-Z gate."""

    def __init__(self, ctl, tgt, circ=None):
        """Create new CZ gate."""
        super().__init__("cz", [], [ctl, tgt], circ)

    def qasm(self):
        """Return OPENQASM string."""
        ctl = self.arg[0]
        tgt = self.arg[1]
        return self._qasmif("cz %s[%d],%s[%d];" % (ctl[0].name, ctl[1],
                                                   tgt[0].name, tgt[1]))

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.cz(self.arg[0], self.arg[1]))

    def q_if(self, *qregs):
        if not qregs:
            return self

        # Dynamically create the Gate that will perform
        # a multi-controlled Z gate by using a multi-controlled
        # X gate and 2 H gates.
        class _MultiControlledZGateWithCx(CompositeGate):
            def __init__(self, circuit, target, *controls):
                super().__init__(self.__class__.__name__,  # name
                                 [],                       # parameters
                                 [target] + controls,      # qubits
                                 circuit)                  # circuit
                self.h(target)
                self._attach(XGate(target, self).q_if(*controls))
                self.h(target)

        return _MultiControlledZGateWithCx(self._get_circuit(), self.arg[1], [self.arg[1]] + qregs)


def cz(self, ctl, tgt):
    """Apply CZ to circuit."""
    if isinstance(ctl, QuantumRegister) and \
       isinstance(tgt, QuantumRegister) and len(ctl) == len(tgt):
        instructions = InstructionSet()
        for i in range(ctl.size):
            instructions.add(self.cz((ctl, i), (tgt, i)))
        return instructions

    if isinstance(ctl, QuantumRegister):
        instructions = InstructionSet()
        for j in range(ctl.size):
            instructions.add(self.cz((ctl, j), tgt))
        return instructions

    if isinstance(tgt, QuantumRegister):
        instructions = InstructionSet()
        for j in range(tgt.size):
            instructions.add(self.cz(ctl, (tgt, j)))
        return instructions

    self._check_qubit(ctl)
    self._check_qubit(tgt)
    self._check_dups([ctl, tgt])
    return self._attach(CzGate(ctl, tgt, self))


QuantumCircuit.cz = cz
CompositeGate.cz = cz
