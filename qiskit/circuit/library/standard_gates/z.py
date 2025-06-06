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

"""Z, CZ and CCZ gates."""

from typing import Optional, Union

import numpy

from qiskit.circuit._utils import with_gate_array, with_controlled_gate_array
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit._accelerate.circuit import StandardGate

from .p import PhaseGate

_Z_ARRAY = [[1, 0], [0, -1]]


@with_gate_array(_Z_ARRAY)
class ZGate(SingletonGate):
    r"""The single-qubit Pauli-Z gate (:math:`\sigma_z`).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.z` method.

    **Matrix Representation:**

    .. math::

        Z = \begin{pmatrix}
                1 & 0 \\
                0 & -1
            \end{pmatrix}

    **Circuit symbol:**

    .. code-block:: text

             ┌───┐
        q_0: ┤ Z ├
             └───┘

    Equivalent to a :math:`\pi` radian rotation about the Z axis.

    .. note::

        A global phase difference exists between the definitions of
        :math:`RZ(\pi)` and :math:`Z`.

        .. math::

            RZ(\pi) = \begin{pmatrix}
                        -i & 0 \\
                        0 & i
                      \end{pmatrix}
                    = -i Z

    The gate is equivalent to a phase flip.

    .. math::

        |0\rangle \rightarrow |0\rangle \\
        |1\rangle \rightarrow -|1\rangle
    """

    _standard_gate = StandardGate.Z

    def __init__(self, label: Optional[str] = None):
        """Create new Z gate."""
        super().__init__("z", 1, [], label=label)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """Default definition"""
        # pylint: disable=cyclic-import
        from qiskit.circuit import QuantumCircuit

        #    ┌──────┐
        # q: ┤ P(π) ├
        #    └──────┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.Z._get_definition(self.params), add_regs=True, name=self.name
        )

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        annotated: bool = False,
    ):
        """Return a (multi-)controlled-Z gate.

        One control returns a CZ gate.

        Args:
            num_ctrl_qubits: number of control qubits.
            label: An optional label for the gate [Default: ``None``]
            ctrl_state: control state expressed as integer,
                string (e.g.``'110'``), or ``None``. If ``None``, use all 1s.
            annotated: indicates whether the controlled gate should be implemented
                as an annotated gate.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if not annotated and num_ctrl_qubits == 1:
            gate = CZGate(label=label, ctrl_state=ctrl_state, _base_label=self.label)
        else:
            gate = super().control(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                annotated=annotated,
            )
        return gate

    def inverse(self, annotated: bool = False):
        """Return inverted Z gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            ZGate: inverse gate (self-inverse).
        """
        return ZGate()  # self-inverse

    def power(self, exponent: float, annotated: bool = False):
        return PhaseGate(numpy.pi * exponent)

    def __eq__(self, other):
        return isinstance(other, ZGate)


@with_controlled_gate_array(_Z_ARRAY, num_ctrl_qubits=1)
class CZGate(SingletonControlledGate):
    r"""Controlled-Z gate.

    This is a Clifford and symmetric gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.cz` method.

    **Circuit symbol:**

    .. code-block:: text

        q_0: ─■─
              │
        q_1: ─■─

    **Matrix representation:**

    .. math::

        CZ\ q_0, q_1 =
            I \otimes |0\rangle\langle 0| + Z \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & -1
            \end{pmatrix}

    In the computational basis, this gate flips the phase of
    the target qubit if the control qubit is in the :math:`|1\rangle` state.
    """

    _standard_gate = StandardGate.CZ

    def __init__(
        self,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        _base_label=None,
    ):
        """Create new CZ gate."""
        super().__init__(
            "cz",
            2,
            [],
            label=label,
            num_ctrl_qubits=1,
            ctrl_state=ctrl_state,
            base_gate=ZGate(label=_base_label),
        )

    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=1)

    def _define(self):
        """Default definition"""
        # pylint: disable=cyclic-import
        from qiskit.circuit import QuantumCircuit

        # q_0: ───────■───────
        #      ┌───┐┌─┴─┐┌───┐
        # q_1: ┤ H ├┤ X ├┤ H ├
        #      └───┘└───┘└───┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.CZ._get_definition(self.params), add_regs=True, name=self.name
        )

    def inverse(self, annotated: bool = False):
        """Return inverted CZ gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            CZGate: inverse gate (self-inverse).
        """
        return CZGate(ctrl_state=self.ctrl_state)  # self-inverse

    def __eq__(self, other):
        return isinstance(other, CZGate) and self.ctrl_state == other.ctrl_state


@with_controlled_gate_array(_Z_ARRAY, num_ctrl_qubits=2, cached_states=(3,))
class CCZGate(SingletonControlledGate):
    r"""CCZ gate.

    This is a symmetric gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.ccz` method.

    **Circuit symbol:**

    .. code-block:: text

        q_0: ─■─
              │
        q_1: ─■─
              │
        q_2: ─■─

    **Matrix representation:**

    .. math::

        CCZ\ q_0, q_1, q_2 =
            I \otimes I \otimes |0\rangle\langle 0| + CZ \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & -1
            \end{pmatrix}

    In the computational basis, this gate flips the phase of
    the target qubit if the control qubits are in the :math:`|11\rangle` state.
    """

    _standard_gate = StandardGate.CCZ

    def __init__(
        self,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        _base_label=None,
    ):
        """Create new CCZ gate."""
        super().__init__(
            "ccz",
            3,
            [],
            label=label,
            num_ctrl_qubits=2,
            ctrl_state=ctrl_state,
            base_gate=ZGate(label=_base_label),
        )

    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=2)

    def _define(self):
        """Default definition"""
        # pylint: disable=cyclic-import
        from qiskit.circuit import QuantumCircuit

        # q_0: ───────■───────
        #             │
        # q_1: ───────■───────
        #      ┌───┐┌─┴─┐┌───┐
        # q_2: ┤ H ├┤ X ├┤ H ├
        #      └───┘└───┘└───┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.CCZ._get_definition(self.params), add_regs=True, name=self.name
        )

    def inverse(self, annotated: bool = False):
        """Return inverted CCZ gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            CCZGate: inverse gate (self-inverse).
        """
        return CCZGate(ctrl_state=self.ctrl_state)  # self-inverse

    def __eq__(self, other):
        return isinstance(other, CCZGate) and self.ctrl_state == other.ctrl_state
