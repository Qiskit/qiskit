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

"""Rotation around the Y axis."""

import math
from math import pi
from typing import Optional, Union

import numpy

from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.exceptions import QiskitError

from .rx import _apply_cu, _apply_mcu_graycode, _mcsu2_real_diagonal
from .x import MCXGate


class RYGate(Gate):
    r"""Single-qubit rotation about the Y axis.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.ry` method.

    **Circuit symbol:**

    .. parsed-literal::

             ┌───────┐
        q_0: ┤ Ry(ϴ) ├
             └───────┘

    **Matrix Representation:**

    .. math::

        \newcommand{\rotationangle}{\frac{\theta}{2}}

        RY(\theta) = \exp\left(-i \rotationangle Y\right) =
            \begin{pmatrix}
                \cos\left(\rotationangle\right) & -\sin\left(\rotationangle\right) \\
                \sin\left(\rotationangle\right) & \cos\left(\rotationangle\right)
            \end{pmatrix}
    """

    def __init__(
        self, theta: ParameterValueType, label: Optional[str] = None, *, duration=None, unit="dt"
    ):
        """Create new RY gate."""
        super().__init__("ry", 1, [theta], label=label, duration=duration, unit=unit)

    def _define(self):
        """
        gate ry(theta) a { r(theta, pi/2) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        from .r import RGate

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(RGate(self.params[0], pi / 2), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        annotated: bool = False,
    ):
        """Return a (multi-)controlled-RY gate.

        Args:
            num_ctrl_qubits: number of control qubits.
            label: An optional label for the gate [Default: ``None``]
            ctrl_state: control state expressed as integer,
                string (e.g.``'110'``), or ``None``. If ``None``, use all 1s.
            annotated: indicates whether the controlled gate can be implemented
                as an annotated gate.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if not annotated and num_ctrl_qubits == 1:
            gate = CRYGate(self.params[0], label=label, ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
        else:
            gate = super().control(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                annotated=annotated,
            )
        return gate

    def inverse(self, annotated: bool = False):
        r"""Return inverse RY gate.

        :math:`RY(\lambda)^{\dagger} = RY(-\lambda)`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.RYGate` with an inverted parameter value.

        Returns:
            RYGate: inverse gate.
        """
        return RYGate(-self.params[0])

    def __array__(self, dtype=None):
        """Return a numpy.array for the RY gate."""
        cos = math.cos(self.params[0] / 2)
        sin = math.sin(self.params[0] / 2)
        return numpy.array([[cos, -sin], [sin, cos]], dtype=dtype)

    def power(self, exponent: float):
        """Raise gate to a power."""
        (theta,) = self.params
        return RYGate(exponent * theta)

    def __eq__(self, other):
        if isinstance(other, RYGate):
            return self._compare_parameters(other)
        return False


class CRYGate(ControlledGate):
    r"""Controlled-RY gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.cry` method.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ────■────
             ┌───┴───┐
        q_1: ┤ Ry(ϴ) ├
             └───────┘

    **Matrix representation:**

    .. math::

        \newcommand{\rotationangle}{\frac{\theta}{2}}

        CRY(\theta)\ q_0, q_1 =
            I \otimes |0\rangle\langle 0| + RY(\theta) \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                1 & 0         & 0 & 0 \\
                0 & \cos\left(\rotationangle\right) & 0 & -\sin\left(\rotationangle\right) \\
                0 & 0         & 1 & 0 \\
                0 & \sin\left(\rotationangle\right) & 0 & \cos\left(\rotationangle\right)
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::
                 ┌───────┐
            q_0: ┤ Ry(ϴ) ├
                 └───┬───┘
            q_1: ────■────

        .. math::

            \newcommand{\rotationangle}{\frac{\theta}{2}}

            CRY(\theta)\ q_1, q_0 =
            |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes RY(\theta) =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & \cos\left(\rotationangle\right) & -\sin\left(\rotationangle\right) \\
                    0 & 0 & \sin\left(\rotationangle\right) & \cos\left(\rotationangle\right)
                \end{pmatrix}
    """

    def __init__(
        self,
        theta: ParameterValueType,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        duration=None,
        unit="dt",
        _base_label=None,
    ):
        """Create new CRY gate."""
        super().__init__(
            "cry",
            2,
            [theta],
            num_ctrl_qubits=1,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=RYGate(theta, label=_base_label),
            duration=duration,
            unit=unit,
        )

    def _define(self):
        """
        gate cry(lambda) a,b
        { u3(lambda/2,0,0) b; cx a,b;
          u3(-lambda/2,0,0) b; cx a,b;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        from .x import CXGate

        # q_0: ─────────────■───────────────■──
        #      ┌─────────┐┌─┴─┐┌─────────┐┌─┴─┐
        # q_1: ┤ Ry(λ/2) ├┤ X ├┤ Ry(λ/2) ├┤ X ├
        #      └─────────┘└───┘└─────────┘└───┘
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (RYGate(self.params[0] / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (RYGate(-self.params[0] / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self, annotated: bool = False):
        """Return inverse CRY gate (i.e. with the negative rotation angle)

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.CRYGate` with an inverted parameter value.

        Returns:
            CRYGate: inverse gate.
        ."""
        return CRYGate(-self.params[0], ctrl_state=self.ctrl_state)

    def __array__(self, dtype=None):
        """Return a numpy.array for the CRY gate."""
        half_theta = float(self.params[0]) / 2
        cos = math.cos(half_theta)
        sin = math.sin(half_theta)
        if self.ctrl_state:
            return numpy.array(
                [[1, 0, 0, 0], [0, cos, 0, -sin], [0, 0, 1, 0], [0, sin, 0, cos]], dtype=dtype
            )
        else:
            return numpy.array(
                [[cos, 0, -sin, 0], [0, 1, 0, 0], [sin, 0, cos, 0], [0, 0, 0, 1]], dtype=dtype
            )

    def __eq__(self, other):
        if isinstance(other, CRYGate):
            return self._compare_parameters(other) and self.ctrl_state == other.ctrl_state
        return False


class MCRYGate(ControlledGate):
    r"""The general, multi-controlled X rotation gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.mcrx` method.
    """

    def __init__(
        self,
        theta: ParameterValueType,  # type: ignore
        num_ctrl_qubits: int,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        duration=None,
        unit="dt",
        _name="mcry",
        _base_label=None,
    ):
        """Create new MCRY gate."""
        num_ancilla_qubits = self.__class__.get_num_ancilla_qubits(num_ctrl_qubits)
        super().__init__(
            _name,
            num_ctrl_qubits + 1 + num_ancilla_qubits,
            [theta],
            num_ctrl_qubits=num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=RYGate(theta, label=_base_label),
            duration=duration,
            unit=unit,
        )

    def inverse(self, annotated: bool = False):
        r"""Return inverse MCRY gate (i.e. with the negative rotation angle).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.MCRYGate` with an inverted parameter value.

        Returns:
            MCRYGate: inverse gate.
        """
        # use __class__ so this works for derived classes
        return self.__class__(
            -self.params[0], num_ctrl_qubits=self.num_ctrl_qubits, ctrl_state=self.ctrl_state
        )

    @staticmethod
    def get_num_ancilla_qubits(num_ctrl_qubits: int, mode: str = "noancilla") -> int:
        """Get the number of required ancilla qubits without instantiating the class.

        This staticmethod might be necessary to check the number of ancillas before
        creating the gate, or to use the number of ancillas in the initialization.
        """
        if mode == "noancilla":
            return 0
        if mode == "basic":
            return MCXGate.get_num_ancilla_qubits(num_ctrl_qubits, "v-chain")
        raise QiskitError(f"Unrecognized mode for building MCRY gate: {mode}.")

    def _define(self):
        """Define the MCRY gate without ancillae."""
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        q = QuantumRegister(self.num_qubits, name="q")
        qc = QuantumCircuit(q)
        q_controls = list(range(self.num_ctrl_qubits))
        q_target = self.num_ctrl_qubits
        if self.num_ctrl_qubits == 1:
            _apply_cu(
                qc,
                self.params[0],
                0,
                0,
                q_controls[0],
                q_target,
                use_basis_gates=False,
            )
        elif self.num_ctrl_qubits < 4:
            theta_step = self.params[0] * (1 / (2 ** (self.num_ctrl_qubits - 1)))
            _apply_mcu_graycode(
                qc,
                theta_step,
                0,
                0,
                q_controls,
                q_target,
                use_basis_gates=False,
            )
        else:
            cgate = _mcsu2_real_diagonal(
                RYGate(self.params[0]).to_matrix(),
                num_controls=self.num_ctrl_qubits,
                use_basis_gates=False,
            )
            qc.compose(cgate, q_controls + [q_target], inplace=True)
        self.definition = qc

    @property
    def num_ancilla_qubits(self):
        """The number of ancilla qubits."""
        return self.__class__.get_num_ancilla_qubits(self.num_ctrl_qubits)

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        annotated: bool = False,
    ):
        r"""Return a multi-controlled-RY gate with more control lines.

        Args:
            num_ctrl_qubits: number of control qubits.
            label: An optional label for the gate [Default: ``None``]
            ctrl_state: control state expressed as integer,
                string (e.g.``'110'``), or ``None``. If ``None``, use all 1s.
            annotated: indicates whether the controlled gate can be implemented
                as an annotated gate.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if not annotated and ctrl_state is None:
            # use __class__ so this works for derived classes
            gate = self.__class__(
                self.params[0],
                self.num_ctrl_qubits + num_ctrl_qubits,
                label=label,
                _base_label=self.label,
            )
        else:
            gate = super().control(num_ctrl_qubits, label=label, ctrl_state=ctrl_state)
        return gate


class MCRYPUCXBasis(MCRYGate):
    r"""The general, multi-controlled X rotation gate using p, u, and cx as basis gates.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.mcry` method.
    """

    def __init__(
        self,
        theta: ParameterValueType,  # type: ignore
        num_ctrl_qubits: int,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        duration=None,
        unit="dt",
        _base_label=None,
    ):
        """Create new MCRYPUCXBasis gate."""
        super().__init__(
            theta=theta,
            num_ctrl_qubits=num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state,
            duration=duration,
            unit=unit,
            _base_label=_base_label,
        )

    def _define(self):
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        q = QuantumRegister(self.num_qubits, name="q")
        qc = QuantumCircuit(q)
        q_controls = list(range(self.num_ctrl_qubits))
        q_target = self.num_ctrl_qubits
        if self.num_ctrl_qubits == 1:
            _apply_cu(
                qc,
                self.params[0],
                0,
                0,
                q_controls[0],
                q_target,
                use_basis_gates=True,
            )
        elif self.num_ctrl_qubits < 4:
            theta_step = self.params[0] * (1 / (2 ** (self.num_ctrl_qubits - 1)))
            _apply_mcu_graycode(
                qc,
                theta_step,
                0,
                0,
                q_controls,
                q_target,
                use_basis_gates=True,
            )
        else:
            cgate = _mcsu2_real_diagonal(
                RYGate(self.params[0]).to_matrix(),
                num_controls=self.num_ctrl_qubits,
                use_basis_gates=True,
            )
            qc.compose(cgate, q_controls + [q_target], inplace=True)
        self.definition = qc


class MCRYVChain(MCRYGate):
    """Implement the multi-controlled RX gate using a V-chain of CX gates."""

    def __init__(
        self,
        theta: ParameterValueType,  # type: ignore
        num_ctrl_qubits: int,
        ancilla_qubits: bool = False,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        duration=None,
        unit="dt",
        _base_label=None,
    ):
        super().__init__(
            theta,
            num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state,
            _name="mcry_vchain",
            _base_label=_base_label,
            duration=duration,
            unit=unit,
        )
        self._ancilla_qubits = ancilla_qubits

    def inverse(self, annotated: bool = False):
        """Return inverse MCRY gate (i.e. with the negative rotation angle).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.MCRYVChain` with an inverted parameter value.

        Returns:
            MCRYVChain: inverse gate.
        """
        return MCRYVChain(
            -self.params[0],
            num_ctrl_qubits=self.num_ctrl_qubits,
            ancilla_qubits=self._ancilla_qubits,
            ctrl_state=self.ctrl_state,
        )

    @staticmethod
    def get_num_ancilla_qubits(num_ctrl_qubits: int, mode: str = "basic") -> int:
        """Get the number of required ancilla qubits without instantiating the class.

        This staticmethod might be necessary to check the number of ancillas before
        creating the gate, or to use the number of ancillas in the initialization.
        """
        return MCRYGate.get_num_ancilla_qubits(num_ctrl_qubits, mode)

    def _define(self):
        """Define the MCRY gate using a V-chain of CX gates."""
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        q = QuantumRegister(self.num_qubits, name="q")
        qc = QuantumCircuit(q, name=self.name)
        q_controls = q[: self.num_ctrl_qubits]
        q_target = q[self.num_ctrl_qubits]
        q_ancillas = q[self.num_ctrl_qubits + 1 :]

        qc.ry(self.params[0] / 2, q_target)
        qc.mcx(q_controls, q_target, q_ancillas, mode="v-chain")
        qc.ry(-self.params[0] / 2, q_target)
        qc.mcx(q_controls, q_target, q_ancillas, mode="v-chain")

        self.definition = qc
