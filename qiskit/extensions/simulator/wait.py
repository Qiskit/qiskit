# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Aer's qasm_simulator single qubit wait gate.
"""
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit import InstructionSet
from qiskit.qasm import _node as node


class WaitGate(Gate):  # pylint: disable=abstract-method
    """Wait gate."""

    def __init__(self, t, qubit, circ=None):
        """Create new wait gate."""
        super().__init__("wait", [t], [qubit], circ)

    def inverse(self):
        """Invert this gate."""
        return self  # self-inverse

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        self._modifiers(circ.wait(self.param[0], self.qargs[0]))


def wait(self, t, q):
    """Apply wait for time t to q."""
    if isinstance(q, QuantumRegister):
        gs = InstructionSet()
        for j in range(q.size):
            gs.add(self.wait(t, (q, j)))
        return gs
    self._check_qubit(q)
    return self._attach(WaitGate(t, q, self))


# Add to QuantumCircuit class
QuantumCircuit.wait = wait

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
