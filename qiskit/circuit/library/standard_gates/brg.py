"""bridge gate."""

import numpy as np

from qiskit.circuit.gate import Gate
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.library.standard_gates import CXGate
from qiskit.circuit import EquivalenceLibrary


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
        q_1: -────
             ┌─┴─┐
        q_2: ┤ X ├
             └───┘


    ..Note::

    The bridge gate is used as a transformation rules in mapping.

    """

    def __init__(self, label=None):
        """Create new BrG gate."""
        super().__init__('brg', 3, [])

    def _define(self):
        """gate Bridge-Gate {cx b,c; cx a,b; cx b,c; cx a,b;} """
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .x import CXGate

        q = QuantumRegister(3, 'q')
        qc = QuantumCircuit(q, name=self.name)

        rules = [
            (CXGate(), [q[1], q[2]], []),
            (CXGate(), [q[0], q[1]], []),
            (CXGate(), [q[1], q[2]], []),
            (CXGate(), [q[0], q[1]], [])
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc