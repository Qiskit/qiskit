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

"""Rotation around the Z axis."""
from cmath import exp
from typing import Optional, Union

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.exceptions import QiskitError

from .rx import _apply_cu, _apply_mcu_graycode, _mcsu2_real_diagonal


class RZGate(Gate):
    r"""Single-qubit rotation about the Z axis.

    This is a diagonal gate. It can be implemented virtually in hardware
    via framechanges (i.e. at zero error and duration).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.rz` method.

    **Circuit symbol:**

    .. parsed-literal::

             ┌───────┐
        q_0: ┤ Rz(λ) ├
             └───────┘

    **Matrix Representation:**

    .. math::

        RZ(\lambda) = \exp\left(-i\frac{\lambda}{2}Z\right) =
            \begin{pmatrix}
                e^{-i\frac{\lambda}{2}} & 0 \\
                0 & e^{i\frac{\lambda}{2}}
            \end{pmatrix}

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.U1Gate`
        This gate is equivalent to U1 up to a phase factor.

            .. math::

                U1(\lambda) = e^{i{\lambda}/2}RZ(\lambda)

        Reference for virtual Z gate implementation:
        `1612.00858 <https://arxiv.org/abs/1612.00858>`_
    """

    def __init__(
        self, phi: ParameterValueType, label: Optional[str] = None, *, duration=None, unit="dt"
    ):
        """Create new RZ gate."""
        super().__init__("rz", 1, [phi], label=label, duration=duration, unit=unit)

    def _define(self):
        """
        gate rz(phi) a { u1(phi) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        from .u1 import U1Gate

        q = QuantumRegister(1, "q")
        theta = self.params[0]
        qc = QuantumCircuit(q, name=self.name, global_phase=-theta / 2)
        rules = [(U1Gate(theta), [q[0]], [])]
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
        """Return a (multi-)controlled-RZ gate.

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
            gate = CRZGate(self.params[0], label=label, ctrl_state=ctrl_state)
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
        r"""Return inverted RZ gate

        :math:`RZ(\lambda)^{\dagger} = RZ(-\lambda)`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.RZGate` with an inverted parameter value.

        Returns:
            RZGate: inverse gate.
        """
        return RZGate(-self.params[0])

    def __array__(self, dtype=None):
        """Return a numpy.array for the RZ gate."""
        import numpy as np

        ilam2 = 0.5j * float(self.params[0])
        return np.array([[exp(-ilam2), 0], [0, exp(ilam2)]], dtype=dtype)

    def power(self, exponent: float):
        """Raise gate to a power."""
        (theta,) = self.params
        return RZGate(exponent * theta)

    def __eq__(self, other):
        if isinstance(other, RZGate):
            return self._compare_parameters(other)
        return False


class CRZGate(ControlledGate):
    r"""Controlled-RZ gate.

    This is a diagonal but non-symmetric gate that induces a
    phase on the state of the target qubit, depending on the control state.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.crz` method.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ────■────
             ┌───┴───┐
        q_1: ┤ Rz(λ) ├
             └───────┘

    **Matrix representation:**

    .. math::

        CRZ(\lambda)\ q_0, q_1 =
            I \otimes |0\rangle\langle 0| + RZ(\lambda) \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & e^{-i\frac{\lambda}{2}} & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & e^{i\frac{\lambda}{2}}
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::
                 ┌───────┐
            q_0: ┤ Rz(λ) ├
                 └───┬───┘
            q_1: ────■────

        .. math::

            CRZ(\lambda)\ q_1, q_0 =
                |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes RZ(\lambda) =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & e^{-i\frac{\lambda}{2}} & 0 \\
                    0 & 0 & 0 & e^{i\frac{\lambda}{2}}
                \end{pmatrix}

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.CU1Gate`:
        Due to the global phase difference in the matrix definitions
        of U1 and RZ, CU1 and CRZ are different gates with a relative
        phase difference.
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
        """Create new CRZ gate."""
        super().__init__(
            "crz",
            2,
            [theta],
            num_ctrl_qubits=1,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=RZGate(theta, label=_base_label),
            duration=duration,
            unit=unit,
        )

    def _define(self):
        """
        gate crz(lambda) a,b
        { rz(lambda/2) b; cx a,b;
          rz(-lambda/2) b; cx a,b;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        from .x import CXGate

        # q_0: ─────────────■────────────────■──
        #      ┌─────────┐┌─┴─┐┌──────────┐┌─┴─┐
        # q_1: ┤ Rz(λ/2) ├┤ X ├┤ Rz(-λ/2) ├┤ X ├
        #      └─────────┘└───┘└──────────┘└───┘
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (RZGate(self.params[0] / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (RZGate(-self.params[0] / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self, annotated: bool = False):
        """Return inverse CRZ gate (i.e. with the negative rotation angle).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.CRZGate` with an inverted parameter value.

         Returns:
            CRZGate: inverse gate.
        """
        return CRZGate(-self.params[0], ctrl_state=self.ctrl_state)

    def __array__(self, dtype=None):
        """Return a numpy.array for the CRZ gate."""
        import numpy

        arg = 1j * float(self.params[0]) / 2
        if self.ctrl_state:
            return numpy.array(
                [[1, 0, 0, 0], [0, exp(-arg), 0, 0], [0, 0, 1, 0], [0, 0, 0, exp(arg)]],
                dtype=dtype,
            )
        else:
            return numpy.array(
                [[exp(-arg), 0, 0, 0], [0, 1, 0, 0], [0, 0, exp(arg), 0], [0, 0, 0, 1]],
                dtype=dtype,
            )

    def __eq__(self, other):
        if isinstance(other, CRZGate):
            return self._compare_parameters(other) and self.ctrl_state == other.ctrl_state
        return False


class MCRZGate(ControlledGate):
    r"""The general, multi-controlled X rotation gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.mcrz` method.
    """

    def __init__(
        self,
        lam: ParameterValueType,  # type: ignore
        num_ctrl_qubits: int,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        duration=None,
        unit="dt",
        _base_label=None,
    ):
        """Create new MCRZ gate."""
        super().__init__(
            "mcrz",
            num_ctrl_qubits + 1,
            [lam],
            num_ctrl_qubits=num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=RZGate(lam, label=_base_label),
            duration=duration,
            unit=unit,
        )

    def inverse(self, annotated: bool = False):
        r"""Return inverse MCRZ gate (i.e. with the negative rotation angle).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.MCRZGate` with an inverted parameter value.

        Returns:
            MCRZGate: inverse gate.
        """
        # use __class__ so this works for derived classes
        return self.__class__(
            -self.params[0], num_ctrl_qubits=self.num_ctrl_qubits, ctrl_state=self.ctrl_state
        )

    def _define(self):
        q = QuantumRegister(self.num_qubits, name="q")
        qc = QuantumCircuit(q)
        q_controls = list(range(self.num_ctrl_qubits))
        q_target = self.num_ctrl_qubits
        if self.num_ctrl_qubits == 1:
            qc.append(CRZGate(self.params[0]), q_controls + [q_target])
        else:
            cgate = _mcsu2_real_diagonal(
                RZGate(self.params[0]),
                num_controls=self.num_ctrl_qubits,
                use_basis_gates=False,
            )
            qc.compose(cgate, q_controls + [q_target], inplace=True)
        self.definition = qc

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        annotated: bool = False,
    ):
        r"""Return a multi-controlled-RZ gate with more control lines.

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
                ctrl_state=ctrl_state,
                _base_label=self.label,
            )
        else:
            gate = super().control(num_ctrl_qubits, label=label, ctrl_state=ctrl_state)
        return gate


class MCRZPUCXBasis(MCRZGate):
    r"""The general, multi-controlled X rotation gate using p, u, and cx as basis gates.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.mcrz` method.
    """

    def __init__(
        self,
        lam: ParameterValueType,  # type: ignore
        num_ctrl_qubits: int,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        duration=None,
        unit="dt",
        _base_label=None,
    ):
        """Create new MCRZPUCXBasis gate."""
        super().__init__(
            lam=lam,
            num_ctrl_qubits=num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state,
            duration=duration,
            unit=unit,
            _base_label=_base_label,
        )

    def _define(self):
        q = QuantumRegister(self.num_qubits, name="q")
        qc = QuantumCircuit(q)
        q_controls = list(range(self.num_ctrl_qubits))
        q_target = self.num_ctrl_qubits
        if self.num_ctrl_qubits == 1:
            qc.u(0, 0, self.params[0] / 2, q_target)
            qc.cx(q_controls[0], q_target)
            qc.u(0, 0, -self.params[0] / 2, q_target)
            qc.cx(q_controls[0], q_target)
        else:
            cgate = _mcsu2_real_diagonal(
                RZGate(self.params[0]),
                num_controls=self.num_ctrl_qubits,
                use_basis_gates=True,
            )
            qc.compose(cgate, q_controls + [q_target], inplace=True)
        self.definition = qc
