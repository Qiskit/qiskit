# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Qubit reset to computational zero.
"""
from ._instruction import Instruction


class Reset(Instruction):
    """Qubit reset."""

    def __init__(self, qubit, circ=None):
        """Create new reset instruction."""
        super().__init__("reset", [], [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        return self._qasmif("reset %s[%d];" % (qubit[0].name, qubit[1]))

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.reset(self.arg[0]))
