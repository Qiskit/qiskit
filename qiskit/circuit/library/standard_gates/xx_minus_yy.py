# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Two-qubit XX-YY gate."""

from __future__ import annotations

import math
from cmath import exp
from typing import Optional

import numpy as np

from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType, ParameterExpression
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit._accelerate.circuit import StandardGate


class XXMinusYYGate(Gate):
    r"""XX-YY interaction gate.

    A 2-qubit parameterized XX-YY interaction. Its action is to induce
    a coherent rotation by some angle between :math:`|00\rangle` and :math:`|11\rangle`.

    **Circuit Symbol:**

    .. code-block:: text

             ┌───────────────┐
        q_0: ┤0              ├
             │  (XX-YY)(θ,β) │
        q_1: ┤1              ├
             └───────────────┘

    **Matrix Representation:**

    .. math::

        \newcommand{\rotationangle}{\frac{\theta}{2}}

        R_{XX-YY}(\theta, \beta) q_0, q_1 =
          RZ_1(\beta) \cdot \exp\left(-i \frac{\theta}{2} \frac{XX-YY}{2}\right) \cdot RZ_1(-\beta) =
            \begin{pmatrix}
                \cos\left(\rotationangle\right) & 0 & 0 & -i\sin\left(\rotationangle\right)e^{-i\beta} \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                -i\sin\left(\rotationangle\right)e^{i\beta} & 0 & 0 & \cos\left(\rotationangle\right)
            \end{pmatrix}
    """

    _standard_gate = StandardGate.XXMinusYY

    def __init__(
        self,
        theta: ParameterValueType,
        beta: ParameterValueType = 0,
        label: Optional[str] = "(XX-YY)",
    ):
        """Create new XX-YY gate.

        Args:
            theta: The rotation angle.
            beta: The phase angle.
            label: The label of the gate.
        """
        super().__init__("xx_minus_yy", 2, [theta, beta], label=label)

    def _define(self):
        """Default definition"""

        #       ┌─────┐  ┌────┐┌───┐     ┌─────────┐      ┌─────┐ ┌──────┐┌───┐
        # q_0: ─┤ Sdg ├──┤ √X ├┤ S ├──■──┤ Ry(θ/2) ├───■──┤ Sdg ├─┤ √Xdg ├┤ S ├
        #      ┌┴─────┴─┐├───┬┘└───┘┌─┴─┐├─────────┴┐┌─┴─┐├─────┤┌┴──────┤└───┘
        # q_1: ┤ Rz(-β) ├┤ S ├──────┤ X ├┤ Ry(-θ/2) ├┤ X ├┤ Sdg ├┤ Rz(β) ├─────
        #      └────────┘└───┘      └───┘└──────────┘└───┘└─────┘└───────┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.XXMinusYY._get_definition(self.params), add_regs=True, name=self.name
        )

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: str | int | None = None,
        annotated: bool | None = None,
    ):
        """Return a (multi-)controlled-(XX-YY) gate.

        Args:
            num_ctrl_qubits: number of control qubits.
            label: An optional label for the gate [Default: ``None``]
            ctrl_state: control state expressed as integer,
                string (e.g.``'110'``), or ``None``. If ``None``, use all 1s.
            annotated: indicates whether the controlled gate should be implemented
                as an annotated gate. If ``None``, this is set to ``True`` if
                the gate contains free parameters, in which case it cannot
                yet be synthesized.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if annotated is None:
            annotated = any(isinstance(p, ParameterExpression) for p in self.params)

        gate = super().control(
            num_ctrl_qubits=num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state,
            annotated=annotated,
        )
        return gate

    def inverse(self, annotated: bool = False):
        """Inverse gate.

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.XXMinusYYGate` with inverse
                parameter values.

        Returns:
            XXMinusYYGate: inverse gate.
        """
        theta, beta = self.params
        return XXMinusYYGate(-theta, beta)

    def __array__(self, dtype=None, copy=None):
        """Gate matrix."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        theta, beta = self.params
        cos = math.cos(theta / 2)
        sin = math.sin(theta / 2)
        return np.array(
            [
                [cos, 0, 0, -1j * sin * exp(-1j * beta)],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [-1j * sin * exp(1j * beta), 0, 0, cos],
            ],
            dtype=dtype,
        )

    def power(self, exponent: float, annotated: bool = False):
        theta, beta = self.params
        return XXMinusYYGate(exponent * theta, beta)

    def __eq__(self, other):
        if isinstance(other, XXMinusYYGate):
            return self._compare_parameters(other)
        return False
