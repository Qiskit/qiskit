# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class for a Variational Principle."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from qiskit import QuantumCircuit

from qiskit.algorithms.gradients import BaseEstimatorGradient, BaseQGT, DerivativeType
from qiskit.circuit import Parameter
from qiskit.quantum_info.operators.base_operator import BaseOperator

from qiskit.algorithms.exceptions import AlgorithmError


class VariationalPrinciple(ABC):
    """A Variational Principle class. It determines the time propagation of parameters in a
    quantum state provided as a parametrized quantum circuit (ansatz)."""

    def __init__(
        self,
        qgt: BaseQGT | None = None,
        gradient: BaseEstimatorGradient | None = None,
    ) -> None:
        """
        Args:
            qgt: Instance of a class used to compute the GQT.
            gradient: Instance of a class used to compute the state gradient.
        """
        self.qgt = qgt
        self.gradient = gradient

    def metric_tensor(self, ansatz: QuantumCircuit, param_values: list[float]) -> np.ndarray:
        """
        Calculates a metric tensor according to the rules of this variational principle.

        Args:
            ansatz: Quantum state in the form of a parametrized quantum circuit.
            param_values: Values of parameters to be bound.

        Returns:
            Metric tensor.

        Raises:
            AlgorithmError: If a QFI job fails.
        """

        self.qgt.derivative_type = DerivativeType.REAL
        try:
            metric_tensor = self.qgt.run([ansatz], [param_values], [None]).result().qgts[0]
        except Exception as exc:

            raise AlgorithmError("The QFI primitive job failed!") from exc
        return metric_tensor

    @abstractmethod
    def evolution_gradient(
        self,
        hamiltonian: BaseOperator,
        ansatz: QuantumCircuit,
        param_values: list[float],
        gradient_params: list[Parameter] | None = None,
    ) -> np.ndarray:
        """
        Calculates an evolution gradient according to the rules of this variational principle.

        Args:
            hamiltonian: Operator used for Variational Quantum Time Evolution.
            ansatz: Quantum state in the form of a parametrized quantum circuit.
            param_values: Values of parameters to be bound.
            gradient_params: List of parameters with respect to which gradients should be computed.
                If ``None`` given, gradients w.r.t. all parameters will be computed.

        Returns:
            An evolution gradient.
        """
        pass
