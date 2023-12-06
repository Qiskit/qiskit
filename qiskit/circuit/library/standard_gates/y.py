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

"""Y and CY gates."""

from math import pi
from typing import Optional, Union

# pylint: disable=cyclic-import
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import with_gate_array, with_controlled_gate_array

_Y_ARRAY = [[0, -1j], [1j, 0]]


@with_gate_array(_Y_ARRAY)
class YGate(SingletonGate):
    r"""The single-qubit Pauli-Y gate (:math:`\sigma_y`).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.y` method.

    **Matrix Representation:**

    .. math::

        Y = \begin{pmatrix}
                0 & -i \\
                i & 0
            \end{pmatrix}

    **Circuit symbol:**

    .. parsed-literal::

             ┌───┐
        q_0: ┤ Y ├
             └───┘

    Equivalent to a :math:`\pi` radian rotation about the Y axis.

    .. note::

        A global phase difference exists between the definitions of
        :math:`RY(\pi)` and :math:`Y`.

        .. math::

            RY(\pi) = \begin{pmatrix}
                        0 & -1 \\
                        1 & 0
                      \end{pmatrix}
                    = -i Y

    The gate is equivalent to a bit and phase flip.

    .. math::

        |0\rangle \rightarrow i|1\rangle \\
        |1\rangle \rightarrow -i|0\rangle
    """

    def __init__(self, label: Optional[str] = None, *, duration=None, unit="dt"):
        """Create new Y gate."""
        super().__init__("y", 1, [], label=label, duration=duration, unit=unit)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u3 import U3Gate

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U3Gate(pi, pi / 2, pi / 2), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        """Return a (multi-)controlled-Y gate.

        One control returns a CY gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            gate = CYGate(label=label, ctrl_state=ctrl_state, _base_label=self.label)
            return gate
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)

    def inverse(self):
        r"""Return inverted Y gate (:math:`Y^{\dagger} = Y`)"""
        return YGate()  # self-inverse


@with_controlled_gate_array(_Y_ARRAY, num_ctrl_qubits=1)
class CYGate(SingletonControlledGate):
    r"""Controlled-Y gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.cy` method.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ──■──
             ┌─┴─┐
        q_1: ┤ Y ├
             └───┘

    **Matrix representation:**

    .. math::

        CY\ q_0, q_1 =
        I \otimes |0 \rangle\langle 0| + Y \otimes |1 \rangle\langle 1|  =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & 0 & -i \\
                0 & 0 & 1 & 0 \\
                0 & i & 0 & 0
            \end{pmatrix}


    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::
                 ┌───┐
            q_0: ┤ Y ├
                 └─┬─┘
            q_1: ──■──

        .. math::

            CY\ q_1, q_0 =
                |0 \rangle\langle 0| \otimes I + |1 \rangle\langle 1| \otimes Y =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & -i \\
                    0 & 0 & i & 0
                \end{pmatrix}

    """

    def __init__(
        self,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        duration=None,
        unit="dt",
        _base_label=None,
    ):
        """Create new CY gate."""
        super().__init__(
            "cy",
            2,
            [],
            num_ctrl_qubits=1,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=YGate(label=_base_label),
            duration=duration,
            unit=unit,
        )

    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=1)

    def _define(self):
        """
        gate cy a,b { sdg b; cx a,b; s b; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .s import SGate, SdgGate
        from .x import CXGate

        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(SdgGate(), [q[1]], []), (CXGate(), [q[0], q[1]], []), (SGate(), [q[1]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        """Return inverted CY gate (itself)."""
        return CYGate(ctrl_state=self.ctrl_state)  # self-inverse
