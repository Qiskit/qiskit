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

"""B gate."""

import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister


class BGate(Gate):
    r"""B (Berkeley) gate.

    A 2-qubit Clifford gate introduced in [1] with the special
    property that it can simulate any 2-qubit unitary with just
    two applications (plus local gates).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.b` method.

    .. parsed-literal::

                 ┌────────────┐┌────────────┐
            q_0: ┤0           ├┤0           ├
                 │  Rxx(-π/2) ││  Ryy(-π/4) │
            q_1: ┤1           ├┤1           ├
                 └────────────┘└────────────┘

    This gate is locally equivalent to:

                      ┌──────┐
            q_0: ──■──┤  √X  ├
                 ┌─┴─┐└──┬───┘
            q_1: ┤ X ├───■────
                 └───┘         

    .. math::

        B\ q_0, q_1 =
            \begin{pmatrix}
                cos(\pi/8)      & 0             & 0             & sin(\pi/8)i   \\
                0               & sin(\pi/8)    & cos(\pi/8)i   & 0             \\
                0               & cos(\pi/8)i   & sin(\pi/8)    & 0             \\
                sin(\pi/8)i     & 0             & 0             & cos(\pi/8)
            \end{pmatrix}
    """

    def __init__(self):
        """Create new B gate."""
        super().__init__("b", 2, [])

    def _define(self):
        """
        gate B a, b { rxx(-pi/2) a, b; ryy(-pi/4) a, b; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .rxx import RXXGate
        from .ryy import RYYGate

        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(RXXGate(-pi/2), [q[0], q[1]], []), (RYYGate(-pi/4), [q[1], q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def __array__(self, dtype=None):
        """Return a numpy.array for the DCX gate."""
        return np.array([[np.cos(np.pi/8), 0, 0, np.sin(np.pi/8)*1j],
                         [0, np.sin(np.pi/8), np.cos(np.pi/8)*1j, 0],
                         [0, np.cos(np.pi/8)*1j, np.sin(np.pi/8), 0],
                         [np.sin(np.pi/8)*1j, 0, 0, np.cos(np.pi/8)]], dtype=dtype)
