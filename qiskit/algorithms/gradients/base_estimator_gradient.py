# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Abstract Base class of Gradient for Estimator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence, Mapping
from copy import copy

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .estimator_gradient_result import EstimatorGradientResult


class BaseEstimatorGradient(ABC):
    """Base class for an EstimatorGradient to compute the gradients of the expectation value."""

    def __init__(
        self,
        estimator: BaseEstimator,
        **run_options,
    ):
        """
        Args:
            estimator: The estimator used to compute the gradients.
            run_options: Backend runtime options used for circuit execution. The order of priority is:
                run_options in `run` method > gradient's default run_options > primitive's default
                setting. Higher priority setting overrides lower priority setting.
        """
        self._estimator: BaseEstimator = estimator
        self._circuits: Sequence[QuantumCircuit] | None = None
        if self._circuits is None:
            self._circuits = []
        self._circuit_ids: Mapping[int, int] | None = None
        if self._circuit_ids is None:
            self._circuit_ids = {}
        self._default_run_options = run_options

    def evaluate(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None] | None = None,
        **run_options,
    ) -> EstimatorGradientResult:
        """Run the job of the gradients of expectation values.

        Args:
            circuits: The list of quantum circuits to compute the gradients.
            observables: The list of observables.
            parameter_values: The list of parameter values to be bound to the circuit.
            parameters: The Sequence of Sequence of Parameters to calculate only the gradients of
                the specified parameters. Each Sequence of Parameters corresponds to a circuit in
                `circuits`. Defaults to None, which means that the gradients of all parameters in
                each circuit are calculated.
            run_options: Backend runtime options used for circuit execution. The order of priority is:
                run_options in `run` method > gradient's default run_options > primitive's default
                setting. Higher priority setting overrides lower priority setting.

        Returns:
            The job object of the gradients of the expectation values. The i-th result corresponds to
            ``circuits[i]`` evaluated with parameters bound as ``parameter_values[i]``.

        Raises:
            QiskitError: Invalid arguments are given.
        """
        # Validation
        if len(circuits) != len(parameter_values):
            raise QiskitError(
                f"The number of circuits ({len(circuits)}) does not match "
                f"the number of parameter value sets ({len(parameter_values)})."
            )

        if len(circuits) != len(observables):
            raise QiskitError(
                f"The number of circuits ({len(circuits)}) does not match "
                f"the number of observables ({len(observables)})."
            )

        if parameters is not None:
            if len(circuits) != len(parameters):
                raise QiskitError(
                    f"The number of circuits ({len(circuits)}) does not match "
                    f"the number of the specified parameter sets ({len(parameters)})."
                )

        for i, (circuit, parameter_value) in enumerate(zip(circuits, parameter_values)):
            if len(parameter_value) != circuit.num_parameters:
                raise QiskitError(
                    f"The number of values ({len(parameter_value)}) does not match "
                    f"the number of parameters ({circuit.num_parameters}) for the {i}-th circuit."
                )

        for i, (circuit, observable) in enumerate(zip(circuits, observables)):
            if circuit.num_qubits != observable.num_qubits:
                raise QiskitError(
                    f"The number of qubits of the {i}-th circuit ({circuit.num_qubits}) does "
                    f"not match the number of qubits of the {i}-th observable "
                    f"({observable.num_qubits})."
                )

        # The priority of run option is as follows:
        # run_options in `run` method > gradient's default run_options > primitive's default setting.
        run_opts = copy(self._default_run_options)
        run_opts.update(**run_options)
        return self._evaluate(circuits, observables, parameter_values, parameters, **run_opts)

    @abstractmethod
    def _evaluate(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]] | None = None,
        **run_options,
    ) -> EstimatorGradientResult:
        raise NotImplementedError()
