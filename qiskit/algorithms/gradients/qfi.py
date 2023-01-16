# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
A class for the Quantum Fisher Information.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from typing import Literal

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import BaseEstimator
from qiskit.providers import Options

from .lin_comb_estimator_gradient import DerivativeType
from .lin_comb_qgt import LinCombQGT
from .qgt_result import QGTResult

METHOD = Literal["lin_comb"]


class QFI(ABC):
    r"""Computes the Quantum Fisher Information (QFI) given a pure,
    parameterized quantum state. QFI is defined as:

    .. math::

        \mathrm{QFI}_{ij}= 4 \mathrm{Re}[\langle \partial_i \psi | \partial_j \psi \rangle
        - \langle\partial_i \psi | \psi \rangle \langle\psi | \partial_j \psi \rangle].
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        phase_fix: bool = True,
        method: METHOD = "lin_comb",
        options: Options | None = None,
    ):
        r"""
        Args:
            estimator: The estimator used to compute the QFI.
            phase_fix: Whether to calculate the second term (phase fix) of the QFI, which is
                :math:`\langle\partial_i \psi | \psi \rangle \langle\psi | \partial_j \psi \rangle`.
                Default to ``True``.
            options: Backend runtime options used for circuit execution. The order of priority is:
                options in ``run`` method > QFI's default options > primitive's default
                setting. Higher priority setting overrides lower priority setting.
        """
        self._estimator: BaseEstimator = estimator
        self._derivative_type: DerivativeType = DerivativeType.REAL

        if method == "lin_comb":
            self._qfi = LinCombQGT(estimator, phase_fix, DerivativeType.REAL, options=options)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameter_sets: Sequence[set[Parameter]],
        **options,
    ) -> QGTResult:
        """Compute the QFI on the given circuits."""

        result = self._qfi._run(circuits, parameter_values, parameter_sets, **options)
        return QGTResult(
            qgts=[4 * qgt.real for qgt in result.qgts],
            derivative_type=self._derivative_type,
            metadata=result.metadata,
            options=result.options,
        )
