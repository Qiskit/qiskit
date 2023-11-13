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

"""The S, Sdg, CS and CSdg gates."""

from math import pi
from typing import Optional, Union

import numpy

from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import with_gate_array, with_controlled_gate_array


_S_ARRAY = numpy.array([[1, 0], [0, 1j]])
_SDG_ARRAY = numpy.array([[1, 0], [0, -1j]])


@with_gate_array(_S_ARRAY)
class SGate(SingletonGate):
    r"""Single qubit S gate (Z**0.5).

    It induces a :math:`\pi/2` phase, and is sometimes called the P gate (phase).

    This is a Clifford gate and a square-root of Pauli-Z.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.s` method.

    **Matrix Representation:**

    .. math::

        S = \begin{pmatrix}
                1 & 0 \\
                0 & i
            \end{pmatrix}

    **Circuit symbol:**

    .. parsed-literal::

             ┌───┐
        q_0: ┤ S ├
             └───┘

    Equivalent to a :math:`\pi/2` radian rotation about the Z axis.
    """

    def __init__(self, label: Optional[str] = None, *, duration=None, unit="dt"):
        """Create new S gate."""
        super().__init__("s", 1, [], label=label, duration=duration, unit=unit)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """
        gate s a { u1(pi/2) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        from .u1 import U1Gate

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U1Gate(pi / 2), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        """Return inverse of S (SdgGate)."""
        return SdgGate()

    def power(self, exponent: float):
        """Raise gate to a power."""
        from .p import PhaseGate

        return PhaseGate(0.5 * numpy.pi * exponent)


@with_gate_array(_SDG_ARRAY)
class SdgGate(SingletonGate):
    r"""Single qubit S-adjoint gate (~Z**0.5).

    It induces a :math:`-\pi/2` phase.

    This is a Clifford gate and a square-root of Pauli-Z.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.sdg` method.

    **Matrix Representation:**

    .. math::

        Sdg = \begin{pmatrix}
                1 & 0 \\
                0 & -i
            \end{pmatrix}

    **Circuit symbol:**

    .. parsed-literal::

             ┌─────┐
        q_0: ┤ Sdg ├
             └─────┘

    Equivalent to a :math:`-\pi/2` radian rotation about the Z axis.
    """

    def __init__(self, label: Optional[str] = None, *, duration=None, unit="dt"):
        """Create new Sdg gate."""
        super().__init__("sdg", 1, [], label=label, duration=duration, unit=unit)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """
        gate sdg a { u1(-pi/2) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        from .u1 import U1Gate

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U1Gate(-pi / 2), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        """Return inverse of Sdg (SGate)."""
        return SGate()

    def power(self, exponent: float):
        """Raise gate to a power."""
        from .p import PhaseGate

        return PhaseGate(-0.5 * numpy.pi * exponent)


@with_controlled_gate_array(_S_ARRAY, num_ctrl_qubits=1)
class CSGate(SingletonControlledGate):
    r"""Controlled-S gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.cs` method.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ──■──
             ┌─┴─┐
        q_1: ┤ S ├
             └───┘

    **Matrix representation:**

    .. math::

        CS \ q_0, q_1 =
        I \otimes |0 \rangle\langle 0| + S \otimes |1 \rangle\langle 1|  =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & i
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
        """Create new CS gate."""
        super().__init__(
            "cs",
            2,
            [],
            label=label,
            num_ctrl_qubits=1,
            ctrl_state=ctrl_state,
            base_gate=SGate(label=_base_label),
            duration=duration,
            _base_label=_base_label,
            unit=unit,
        )

    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=1)

    def _define(self):
        """
        gate cs a,b { h b; cp(pi/2) a,b; h b; }
        """
        from .p import CPhaseGate

        self.definition = CPhaseGate(theta=pi / 2).definition

    def inverse(self):
        """Return inverse of CSGate (CSdgGate)."""
        return CSdgGate(ctrl_state=self.ctrl_state)

    def power(self, exponent: float):
        """Raise gate to a power."""
        from .p import CPhaseGate

        return CPhaseGate(0.5 * numpy.pi * exponent)


@with_controlled_gate_array(_SDG_ARRAY, num_ctrl_qubits=1)
class CSdgGate(SingletonControlledGate):
    r"""Controlled-S^\dagger gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.csdg` method.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ───■───
             ┌──┴──┐
        q_1: ┤ Sdg ├
             └─────┘

    **Matrix representation:**

    .. math::

        CS^\dagger \ q_0, q_1 =
        I \otimes |0 \rangle\langle 0| + S^\dagger \otimes |1 \rangle\langle 1|  =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & -i
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
        """Create new CSdg gate."""
        super().__init__(
            "csdg",
            2,
            [],
            label=label,
            num_ctrl_qubits=1,
            ctrl_state=ctrl_state,
            base_gate=SdgGate(label=_base_label),
            duration=duration,
            unit=unit,
        )

    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=1)

    def _define(self):
        """
        gate csdg a,b { h b; cp(-pi/2) a,b; h b; }
        """
        from .p import CPhaseGate

        self.definition = CPhaseGate(theta=-pi / 2).definition

    def inverse(self):
        """Return inverse of CSdgGate (CSGate)."""
        return CSGate(ctrl_state=self.ctrl_state)

    def power(self, exponent: float):
        """Raise gate to a power."""
        from .p import CPhaseGate

        return CPhaseGate(-0.5 * numpy.pi * exponent)
