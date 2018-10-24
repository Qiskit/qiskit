# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Aer's qasm_simulator single qubit wait gate.
"""
from qiskit import CompositeGate
from qiskit import Gate
from qiskit import QuantumCircuit
from qiskit._instructionset import InstructionSet
from qiskit._quantumregister import QuantumRegister
from qiskit.qasm import _node as node


class WaitGate(Gate):
    """Wait gate."""

    def __init__(self, t, qubit, circ=None):
        """Create new wait gate."""
        super().__init__("wait", [t], [qubit], circ)

    def qasm(self):
        """Return OPENQASM string."""
        qubit = self.arg[0]
        t = self.param[0]
        return self._qasmif("wait(%f) %s[%d];" % (t,
                                                  qubit[0].name,
                                                  qubit[1]))

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.wait(self.param[0], self.arg[0]))


def wait(self, t, q):
    """Apply wait for time t to q."""
    if isinstance(q, QuantumRegister):
        gs = InstructionSet()
        for j in range(q.size):
            gs.add(self.wait(t, (q, j)))
        return gs
    self._check_qubit(q)
    return self._attach(WaitGate(t, q, self))


# Add to QuantumCircuit and CompositeGate classes
QuantumCircuit.wait = wait
CompositeGate.wait = wait


# idle for time t (identity)
QuantumCircuit.definitions["wait"] = {
    "print": False,
    "opaque": False,
    "n_args": 1,
    "n_bits": 1,
    "args": ["t"],
    "bits": ["a"],
    # gate wait(t) a { }
    "body": node.GateBody([])
}
