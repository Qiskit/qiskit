# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from __future__ import annotations

from math import pi

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.parameterexpression import ParameterValueType


def rzx_zz1(theta: ParameterValueType | None = None):
    """RZX-based template for CX - RZGate - CX.

    .. code-block:: text

                                                                                      »
          q_0: ──■────────────────────────────────────────────■───────────────────────»
               ┌─┴─┐┌───────┐┌────┐┌───────┐┌────┐┌────────┐┌─┴─┐┌────────┐┌─────────┐»
          q_1: ┤ X ├┤ RZ(ϴ) ├┤ √X ├┤ RZ(π) ├┤ √X ├┤ RZ(3π) ├┤ X ├┤ RZ(-ϴ) ├┤ RZ(π/2) ├»
               └───┘└───────┘└────┘└───────┘└────┘└────────┘└───┘└────────┘└─────────┘»
          «                                    ┌──────────┐                      »
          «q_0: ───────────────────────────────┤0         ├──────────────────────»
          «     ┌─────────┐┌─────────┐┌───────┐│  RZX(-ϴ) │┌─────────┐┌─────────┐»
          «q_1: ┤ RX(π/2) ├┤ RZ(π/2) ├┤ RX(ϴ) ├┤1         ├┤ RZ(π/2) ├┤ RX(π/2) ├»
          «     └─────────┘└─────────┘└───────┘└──────────┘└─────────┘└─────────┘»
          «
          «q_0: ───────────
          «     ┌─────────┐
          «q_1: ┤ RZ(π/2) ├
          «     └─────────┘
    """
    if theta is None:
        theta = Parameter("ϴ")

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.rz(theta, 1)
    qc.sx(1)
    qc.rz(np.pi, 1)
    qc.sx(1)
    qc.rz(3 * np.pi, 1)
    qc.cx(0, 1)
    qc.rz(-1 * theta, 1)

    # Hadamard
    qc.rz(np.pi / 2, 1)
    qc.rx(np.pi / 2, 1)
    qc.rz(np.pi / 2, 1)

    qc.rx(theta, 1)
    qc.rzx(-1 * theta, 0, 1)
    # Hadamard
    qc.rz(np.pi / 2, 1)
    qc.rx(np.pi / 2, 1)
    qc.rz(np.pi / 2, 1)
    # The gate content has unitary e^{-i*pi/2} * I; global_phase = pi/2 corrects this
    # so that Operator(rzx_zz1()) == I exactly, as required by TemplateOptimization.
    qc.global_phase = pi / 2

    return qc
