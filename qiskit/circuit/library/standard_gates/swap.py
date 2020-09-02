# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Swap gate."""

import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister


class SwapGate(Gate):
    r"""The SWAP gate.

    This is a symmetric and Clifford gate.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ─X─
              │
        q_1: ─X─

    **Matrix Representation:**

    .. math::

        SWAP =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}

    The gate is equivalent to a state swap and is a classical logic gate.

    .. math::

        |a, b\rangle \rightarrow |b, a\rangle
    """

    def __init__(self, label=None):
        """Create new SWAP gate."""
        super().__init__('swap', 2, [], label=label)

    def _define(self):
        """
        gate swap a,b { cx a,b; cx b,a; cx a,b; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .x import CXGate
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (CXGate(), [q[0], q[1]], []),
            (CXGate(), [q[1], q[0]], []),
            (CXGate(), [q[0], q[1]], [])
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Return a (multi-)controlled-SWAP gate.

        One control returns a CSWAP (Fredkin) gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            gate = CSwapGate(label=label, ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
            return gate
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)

    def inverse(self):
        """Return inverse Swap gate (itself)."""
        return SwapGate()  # self-inverse

    def to_matrix(self):
        """Return a numpy.array for the SWAP gate."""
        return numpy.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]], dtype=complex)


class CSwapMeta(type):
    """A Metaclass to ensure that CSwapGate and FredkinGate are of the same type.

    Can be removed when FredkinGate gets removed.
    """
    @classmethod
    def __instancecheck__(mcs, inst):
        return type(inst) in {CSwapGate, FredkinGate}  # pylint: disable=unidiomatic-typecheck


class CSwapGate(ControlledGate, metaclass=CSwapMeta):
    r"""Controlled-X gate.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ─X─
              │
        q_1: ─X─
              │
        q_2: ─■─


    **Matrix representation:**

    .. math::

        CSWAP\ q_0, q_1, q_2 =
            |0 \rangle \langle 0| \otimes I \otimes I +
            |1 \rangle \langle 1| \otimes SWAP =
            \begin{pmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_2. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::

            q_0: ─■─
                  │
            q_1: ─X─
                  │
            q_2: ─X─

        .. math::

            CSWAP\ q_2, q_1, q_0 =
                |0 \rangle \langle 0| \otimes I \otimes I +
                |1 \rangle \langle 1| \otimes SWAP =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
                \end{pmatrix}

    In the computational basis, this gate swaps the states of
    the two target qubits if the control qubit is in the
    :math:`|1\rangle` state.

    .. math::
        |0, b, c\rangle \rightarrow |0, b, c\rangle
        |1, b, c\rangle \rightarrow |1, c, b\rangle
    """
    # Define class constants. This saves future allocation time.
    _matrix1 = numpy.array([[1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1]], dtype=complex)
    _matrix0 = numpy.array([[1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1]], dtype=complex)

    def __init__(self, label=None, ctrl_state=None):
        """Create new CSWAP gate."""
        super().__init__('cswap', 3, [], num_ctrl_qubits=1, label=label,
                         ctrl_state=ctrl_state)
        self.base_gate = SwapGate()

    def _define(self):
        """
        gate cswap a,b,c
        { cx c,b;
          ccx a,b,c;
          cx c,b;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .x import CXGate, CCXGate
        q = QuantumRegister(3, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (CXGate(), [q[2], q[1]], []),
            (CCXGate(), [q[0], q[1], q[2]], []),
            (CXGate(), [q[2], q[1]], [])
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        """Return inverse CSwap gate (itself)."""
        return CSwapGate(ctrl_state=self.ctrl_state)  # self-inverse

    def to_matrix(self):
        """Return a numpy.array for the Fredkin (CSWAP) gate."""
        if self.ctrl_state:
            return self._matrix1
        else:
            return self._matrix0


class FredkinGate(CSwapGate, metaclass=CSwapMeta):
    """The deprecated CSwapGate class."""

    def __init__(self):
        import warnings
        warnings.warn('The class FredkinGate is deprecated as of 0.14.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the class CSwapGate instead.',
                      DeprecationWarning, stacklevel=2)
        super().__init__()
