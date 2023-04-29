# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Double-CNOT gate."""

import numpy as np
from qiskit.circuit.singleton_gate import SingletonGate
from qiskit.circuit.quantumregister import QuantumRegister


class DCXGate(SingletonGate):
    r"""Double-CNOT gate.

    A 2-qubit Clifford gate consisting of two back-to-back
    CNOTs with alternate controls.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.dcx` method.

    .. parsed-literal::
                  ┌───┐
        q_0: ──■──┤ X ├
             ┌─┴─┐└─┬─┘
        q_1: ┤ X ├──■──
             └───┘

    This is a classical logic gate, equivalent to a CNOT-SWAP (CNS) sequence,
    and locally equivalent to an iSWAP.

    .. math::

        DCX\ q_0, q_1 =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0
            \end{pmatrix}
    """

    def __init__(self, label=None, duration=None, unit=None, _condition=None):
        """Create new DCX gate."""
        if unit is None:
            unit = "dt"

        super().__init__(
            "dcx", 2, [], label=label, _condition=_condition, duration=duration, unit=unit
        )

    def _define(self):
        """
        gate dcx a, b { cx a, b; cx b, a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .x import CXGate

        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(CXGate(), [q[0], q[1]], []), (CXGate(), [q[1], q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def __array__(self, dtype=None):
        """Return a numpy.array for the DCX gate."""
        return np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=dtype)
