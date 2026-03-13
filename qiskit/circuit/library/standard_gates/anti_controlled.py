# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Anti-controlled gates.

An anti-controlled gate applies the target operation when the control qubit
is in the |0⟩ state, as opposed to the standard controlled gate which
activates on |1⟩. Equivalently, it is a controlled gate with an X gate
applied before and after the control qubit.
"""

from __future__ import annotations

from math import sqrt

import numpy
from qiskit.circuit.gate import Gate
from qiskit.circuit._utils import with_gate_array


_ACH_ARRAY = numpy.array(
    [
        [1 / sqrt(2), 0, 1 / sqrt(2), 0],
        [0, 1, 0, 0],
        [1 / sqrt(2), 0, -1 / sqrt(2), 0],
        [0, 0, 0, 1],
    ],
    dtype=numpy.complex128,
)
_ACX_ARRAY = numpy.array(
    [
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=numpy.complex128,
)


@with_gate_array(_ACH_ARRAY)
class ACHGate(Gate):
    r"""Anti-controlled Hadamard gate.

    Applies a Hadamard on the target qubit if the control is
    in the :math:`|0\rangle` state.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.ach` method.

    Circuit symbol:

    .. code-block:: text

             ┌───┐     ┌───┐
        q_0: ┤ X ├──■──┤ X ├
             └───┘┌─┴─┐└───┘
        q_1: ─────┤ H ├─────
                  └───┘

    This is equivalent to a controlled-H gate with the control state
    set to :math:`|0\rangle`.

    Matrix representation:

    .. math::

        ACH\ q_0, q_1 =
            H \otimes |0\rangle\langle 0| + I \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                \frac{1}{\sqrt{2}} & 0 & \frac{1}{\sqrt{2}} & 0 \\
                0 & 1 & 0 & 0 \\
                \frac{1}{\sqrt{2}} & 0 & -\frac{1}{\sqrt{2}} & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. code-block:: text

                      ┌───┐
            q_0: ─────┤ H ├─────
                 ┌───┐└─┬─┘┌───┐
            q_1: ┤ X ├──■──┤ X ├
                 └───┘     └───┘

        .. math::

            ACH\ q_1, q_0 =
                |0\rangle\langle 0| \otimes H + |1\rangle\langle 1| \otimes I =
                \begin{pmatrix}
                    \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0 & 0 \\
                    \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & 1
                \end{pmatrix}
    """

    def __init__(self, label: str | None = None):
        """Create new ACH gate.

        Args:
            label: An optional label for the gate.
        """
        super().__init__("ach", 2, [], label=label)

    def _define(self):
        """Decomposition: X on control, CH, X on control."""
        from qiskit.circuit import QuantumCircuit

        q = QuantumCircuit(2, name=self.name)
        q.x(0)
        q.ch(0, 1)
        q.x(0)
        self.definition = q

    def inverse(self, annotated: bool = False):
        r"""Return inverted ACH gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            ACHGate: inverse gate (self-inverse).
        """
        return ACHGate()  # self-inverse

    def __eq__(self, other):
        return isinstance(other, ACHGate)

@with_gate_array(_ACX_ARRAY)
class ACXGate(Gate):
    r"""Anti-controlled X gate.

    Applies an X (NOT) gate on the target qubit if the control is
    in the :math:`|0\rangle` state.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.acx` method.

    Circuit symbol:

    .. code-block:: text

             ┌───┐     ┌───┐
        q_0: ┤ X ├──■──┤ X ├
             └───┘┌─┴─┐└───┘
        q_1: ─────┤ X ├─────
                  └───┘

    This is equivalent to a controlled-X gate with the control state
    set to :math:`|0\rangle`.

    Matrix representation:

    .. math::

        ACX\ q_0, q_1 =
            X \otimes |0\rangle\langle 0| + I \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                0 & 0 & 1 & 0 \\
                0 & 1 & 0 & 0 \\
                1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}
            
    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. code-block:: text

                      ┌───┐
            q_0: ─────┤ X ├─────
                 ┌───┐└─┬─┘┌───┐
            q_1: ┤ X ├──■──┤ X ├
                 └───┘     └───┘

        .. math::

            ACX\ q_1, q_0 =
                |0\rangle\langle 0| \otimes X + |1\rangle\langle 1| \otimes I =
                \begin{pmatrix}
                    0 & 1 & 0 & 0 \\
                    1 & 0 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & 1
                \end{pmatrix}
    
    """

    def __init__(self, label: str | None = None):
        """Create new ACX gate.

        Args:
            label: An optional label for the gate.
        """
        super().__init__("acx", 2, [], label=label)

    def _define(self):
        """Decomposition: X on control, CX, X on control."""
        from qiskit.circuit import QuantumCircuit

        q = QuantumCircuit(2, name=self.name)
        q.x(0)
        q.cx(0, 1)
        q.x(0)
        self.definition = q

    def inverse(self, annotated: bool = False):
        r"""Return inverted ACX gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            ACXGate: inverse gate (self-inverse).
        """
        return ACXGate()  # self-inverse

    def __eq__(self, other):
        return isinstance(other, ACXGate)


_ACY_ARRAY = numpy.array(
    [
        [0, 0, -1j, 0],
        [0, 1, 0, 0],
        [1j, 0, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=numpy.complex128,
)
_ACZ_ARRAY = numpy.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ],
    dtype=numpy.complex128,
)


@with_gate_array(_ACY_ARRAY)
class ACYGate(Gate):
    r"""Anti-controlled Y gate.

    Applies a Y (Pauli-Y) gate on the target qubit if the control is
    in the :math:`|0\rangle` state.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.acy` method.

    Circuit symbol:

    .. code-block:: text

             ┌───┐     ┌───┐
        q_0: ┤ X ├──■──┤ X ├
             └───┘┌─┴─┐└───┘
        q_1: ─────┤ Y ├─────
                  └───┘

    This is equivalent to a controlled-Y gate with the control state
    set to :math:`|0\rangle`.

    Matrix representation:

    .. math::

        ACY\ q_0, q_1 =
            Y \otimes |0\rangle\langle 0| + I \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                0 & 0 & -i & 0 \\
                0 & 1 & 0 & 0 \\
                i & 0 & 0 & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. code-block:: text

                      ┌───┐
            q_0: ─────┤ Y ├─────
                 ┌───┐└─┬─┘┌───┐
            q_1: ┤ X ├──■──┤ X ├
                 └───┘     └───┘

        .. math::

            ACY\ q_1, q_0 =
                |0\rangle\langle 0| \otimes Y + |1\rangle\langle 1| \otimes I =
                \begin{pmatrix}
                    0 & -i & 0 & 0 \\
                    i & 0 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & 1
                \end{pmatrix}
    """

    def __init__(self, label: str | None = None):
        """Create new ACY gate.

        Args:
            label: An optional label for the gate.
        """
        super().__init__("acy", 2, [], label=label)

    def _define(self):
        """Decomposition: X on control, CY, X on control."""
        from qiskit.circuit import QuantumCircuit

        q = QuantumCircuit(2, name=self.name)
        q.x(0)
        q.cy(0, 1)
        q.x(0)
        self.definition = q

    def inverse(self, annotated: bool = False):
        r"""Return inverted ACY gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            ACYGate: inverse gate (self-inverse).
        """
        return ACYGate()  # self-inverse

    def __eq__(self, other):
        return isinstance(other, ACYGate)


@with_gate_array(_ACZ_ARRAY)
class ACZGate(Gate):
    r"""Anti-controlled Z gate.

    Applies a Z (Pauli-Z) gate on the target qubit if the control is
    in the :math:`|0\rangle` state.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.acz` method.

    Circuit symbol:

    .. code-block:: text

             ┌───┐     ┌───┐
        q_0: ┤ X ├──■──┤ X ├
             └───┘┌─┴─┐└───┘
        q_1: ─────┤ Z ├─────
                  └───┘

    This is equivalent to a controlled-Z gate with the control state
    set to :math:`|0\rangle`.

    Matrix representation:

    .. math::

        ACZ\ q_0, q_1 =
            Z \otimes |0\rangle\langle 0| + I \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & -1 & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. code-block:: text

                      ┌───┐
            q_0: ─────┤ Z ├─────
                 ┌───┐└─┬─┘┌───┐
            q_1: ┤ X ├──■──┤ X ├
                 └───┘     └───┘

        .. math::

            ACZ\ q_1, q_0 =
                |0\rangle\langle 0| \otimes Z + |1\rangle\langle 1| \otimes I =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & -1 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & 1
                \end{pmatrix}
    """

    def __init__(self, label: str | None = None):
        """Create new ACZ gate.

        Args:
            label: An optional label for the gate.
        """
        super().__init__("acz", 2, [], label=label)

    def _define(self):
        """Decomposition: X on control, CZ, X on control."""
        from qiskit.circuit import QuantumCircuit

        q = QuantumCircuit(2, name=self.name)
        q.x(0)
        q.cz(0, 1)
        q.x(0)
        self.definition = q

    def inverse(self, annotated: bool = False):
        r"""Return inverted ACZ gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            ACZGate: inverse gate (self-inverse).
        """
        return ACZGate()  # self-inverse

    def __eq__(self, other):
        return isinstance(other, ACZGate)