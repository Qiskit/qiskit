# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""bridge gate."""

from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister


class BRGGate(Gate):
    r"""
    The Bridge gate.

    This implementation is based on Page 2, Figure 2 of [1], with the definition of
    Bridge_gate which is composed of 4 CNOTs gates.

    [1] Toshinari et al., https://arxiv.org/pdf/1907.02686.pdf


    **Circuit symbol:**

    .. parsed-literal::

        q_0: ──■──
               │
        q_1:  ───
             ┌─┴─┐
        q_2: ┤ X ├
             └───┘


    ..Note::

    The bridge gate is used as a transformation rule in mapping.
    """

    def __init__(self, label=None):
        """Create new BrG gate."""
        super().__init__("brg", 3, [])

    def _define(self):
        """gate Bridge-Gate {cx b,c; cx a,b; cx b,c; cx a,b;}"""

        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .x import CXGate

        q = QuantumRegister(3, "q")
        qc = QuantumCircuit(q, name=self.name)

        rules = [
            (CXGate(), [q[1], q[2]], []),
            (CXGate(), [q[0], q[1]], []),
            (CXGate(), [q[1], q[2]], []),
            (CXGate(), [q[0], q[1]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc
