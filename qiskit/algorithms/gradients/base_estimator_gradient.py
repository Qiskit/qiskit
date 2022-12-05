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
Abstract base class of gradient for ``Estimator``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import copy

import numpy as np

from qiskit import transpile
from qiskit.circuit import Parameter, QuantumCircuit, ParameterExpression
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.primitives.utils import _circuit_key
from qiskit.providers import Options
from qiskit.algorithms import AlgorithmJob
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .estimator_gradient_result import EstimatorGradientResult
from .utils import (
    DerivativeType,
    GradientCircuit,
    _assign_unique_parameters,
    _make_gradient_parameter_values,
    _make_gradient_parameter_set,
    _get_parameter_set,
)


class BaseEstimatorGradient(ABC):
    """Base class for an ``EstimatorGradient`` to compute the gradients of the expectation value."""

    def __init__(
        self,
        estimator: BaseEstimator,
        options: Options | None = None,
    ):
        """
        Args:
            estimator: The estimator used to compute the gradients.
            options: Primitive backend runtime options used for circuit execution.
                The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.
                Higher priority setting overrides lower priority setting
        """
        self._estimator: BaseEstimator = estimator
        self._default_options = Options()
        if options is not None:
            self._default_options.update_options(**options)
        self._gradient_circuit_cache: dict[QuantumCircuit, GradientCircuit] = {}

    def run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None] | None = None,
        **options,
    ) -> AlgorithmJob:
        """Run the job of the estimator gradient on the given circuits.

        Args:
            circuits: The list of quantum circuits to compute the gradients.
            observables: The list of observables.
            parameter_values: The list of parameter values to be bound to the circuit.
            parameters: The sequence of parameters to calculate only the gradients of
                the specified parameters. Each sequence of parameters corresponds to a circuit in
                ``circuits``. Defaults to None, which means that the gradients of all parameters in
                each circuit are calculated.
            options: Primitive backend runtime options used for circuit execution.
                The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.
                Higher priority setting overrides lower priority setting

        Returns:
            The job object of the gradients of the expectation values. The i-th result corresponds to
            ``circuits[i]`` evaluated with parameters bound as ``parameter_values[i]``. The j-th
            element of the i-th result corresponds to the gradient of the i-th circuit with respect
            to the j-th parameter.

        Raises:
            ValueError: Invalid arguments are given.
        """
        if parameters is None:
            # If parameters is None, we calculate the gradients of all parameters in each circuit.
            parameter_sets = [set(circuit.parameters) for circuit in circuits]
        else:
            # If parameters is not None, we calculate the gradients of the specified parameters.
            # None in parameters means that the gradients of all parameters in the corresponding
            # circuit are calculated.
            parameter_sets = [
                set(parameters_) if parameters_ is not None else set(circuits[i].parameters)
                for i, parameters_ in enumerate(parameters)
            ]
        # Validate the arguments.
        self._validate_arguments(circuits, observables, parameter_values, parameter_sets)
        # The priority of run option is as follows:
        # options in ``run`` method > gradient's default options > primitive's default setting.
        opts = copy(self._default_options)
        opts.update_options(**options)
        # Run the job.
        job = AlgorithmJob(
            self._run, circuits, observables, parameter_values, parameter_sets, **opts.__dict__
        )
        job.submit()
        return job

    @abstractmethod
    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameter_sets: Sequence[set[Parameter] | None],
        **options,
    ) -> EstimatorGradientResult:
        """Compute the estimator gradients on the given circuits."""
        raise NotImplementedError()

    def _preprocess(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameter_sets: Sequence[set[Parameter]],
        supported_gates: Sequence[str],
    ) -> tuple[Sequence[QuantumCircuit], Sequence[Sequence[float]], Sequence[set[Parameter]]]:
        """Preprocess the gradient. This makes a gradient circuit for each circuit. The gradient
        circuit is a transpiled circuit by using the supported gates, and has unique parameters.
        ``parameter_values`` and ``parameters`` are also updated to match the gradient circuit.

        Args:
            circuits: The list of quantum circuits to compute the gradients.
            parameter_values: The list of parameter values to be bound to the circuit.
            parameters: The sequence of parameters to calculate only the gradients of the specified
                parameters.
            supported_gates: The supported gates used to transpile the circuit.

        Returns:
            The list of gradient circuits, the list of parameter values, and the list of parameters.
            parameter_values and parameters are updated to match the gradient circuit.
        """
        g_circuits, g_parameter_values, g_parameter_sets = [], [], []
        for circuit, parameter_value_, parameter_set in zip(
            circuits, parameter_values, parameter_sets
        ):
            circuit_key = _circuit_key(circuit)
            if not circuit_key in self._gradient_circuit_cache:
                transpiled_circuit = transpile(
                    circuit, basis_gates=supported_gates, optimization_level=0
                )
                self._gradient_circuit_cache[circuit_key] = _assign_unique_parameters(
                    transpiled_circuit
                )
            gradient_circuit = self._gradient_circuit_cache[circuit_key]
            g_circuits.append(gradient_circuit.gradient_circuit)
            g_parameter_values.append(
                _make_gradient_parameter_values(circuit, gradient_circuit, parameter_value_)
            )
            g_parameter_sets.append(_make_gradient_parameter_set(gradient_circuit, parameter_set))
        return g_circuits, g_parameter_values, g_parameter_sets

    def _postprocess(
        self,
        results: EstimatorGradientResult,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameter_sets: Sequence[set[Parameter] | None],
    ) -> EstimatorGradientResult:
        """Postprocess the gradient. This computes the gradient of the original circuit from the
        gradient of the gradient circuit by using the chain rule.

        Args:
            results: The results of the gradient of the gradient circuits.
            circuits: The list of quantum circuits to compute the gradients.
            parameter_values: The list of parameter values to be bound to the circuit.
            parameters: The sequence of parameters to calculate only the gradients of the specified
                parameters.

        Returns:
            The results of the gradient of the original circuits.
        """
        gradients, metadata = [], []
        for idx, (circuit, parameter_values_, parameter_set) in enumerate(
            zip(circuits, parameter_values, parameter_sets)
        ):
            unique_gradient = np.zeros(len(parameter_set))
            if (
                "derivative_type" in results.metadata[idx]
                and results.metadata[idx]["derivative_type"] == DerivativeType.COMPLEX
            ):
                # If the derivative type is complex, cast the gradient to complex.
                unique_gradient = unique_gradient.astype("complex")
            gradient_circuit = self._gradient_circuit_cache[_circuit_key(circuit)]
            g_parameter_set = _make_gradient_parameter_set(gradient_circuit, parameter_set)
            # Make a map from the gradient parameter to the respective index in the gradient.
            parameter_indices = [param for param in circuit.parameters if param in parameter_set]
            g_parameter_indices = [
                param
                for param in gradient_circuit.gradient_circuit.parameters
                if param in g_parameter_set
            ]
            g_parameter_indices = {param: i for i, param in enumerate(g_parameter_indices)}
            # Compute the original gradient from the gradient of the gradient circuit
            # by using the chain rule.
            for i, parameter in enumerate(parameter_indices):
                for g_parameter, coeff in gradient_circuit.parameter_map[parameter]:
                    # Compute the coefficient
                    if isinstance(coeff, ParameterExpression):
                        local_map = {
                            p: parameter_values_[circuit.parameters.data.index(p)]
                            for p in coeff.parameters
                        }
                        bound_coeff = coeff.bind(local_map)
                    else:
                        bound_coeff = coeff
                    # The original gradient is a sum of the gradients of the parameters in the
                    # gradient circuit multiplied by the coefficients.
                    unique_gradient[i] += (
                        bound_coeff * results.gradients[idx][g_parameter_indices[g_parameter]]
                    )
            gradients.append(unique_gradient)
            metadata.append([{"parameters": parameter_indices}])
        return EstimatorGradientResult(
            gradients=gradients, metadata=metadata, options=results.options
        )

    @staticmethod
    def _validate_arguments(
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameter_sets: Sequence[set[Parameter]],
    ) -> None:
        """Validate the arguments of the ``run`` method.

        Args:
            circuits: The list of quantum circuits to compute the gradients.
            observables: The list of observables.
            parameter_values: The list of parameter values to be bound to the circuit.
            parameters: The Sequence of Sequence of Parameters to calculate only the gradients of
                the specified parameters. Each Sequence of Parameters corresponds to a circuit in
                ``circuits``. Defaults to None, which means that the gradients of all parameters in
                each circuit are calculated.

        Raises:
            ValueError: Invalid arguments are given.
        """
        if len(circuits) != len(parameter_values):
            raise ValueError(
                f"The number of circuits ({len(circuits)}) does not match "
                f"the number of parameter value sets ({len(parameter_values)})."
            )

        if len(circuits) != len(observables):
            raise ValueError(
                f"The number of circuits ({len(circuits)}) does not match "
                f"the number of observables ({len(observables)})."
            )

        if len(circuits) != len(parameter_sets):
            raise ValueError(
                f"The number of circuits ({len(circuits)}) does not match "
                f"the number of the specified parameter sets ({len(parameter_sets)})."
            )

        for i, (circuit, parameter_value) in enumerate(zip(circuits, parameter_values)):
            if not circuit.num_parameters:
                raise ValueError(f"The {i}-th circuit is not parameterised.")
            if len(parameter_value) != circuit.num_parameters:
                raise ValueError(
                    f"The number of values ({len(parameter_value)}) does not match "
                    f"the number of parameters ({circuit.num_parameters}) for the {i}-th circuit."
                )

        for i, (circuit, observable) in enumerate(zip(circuits, observables)):
            if circuit.num_qubits != observable.num_qubits:
                raise ValueError(
                    f"The number of qubits of the {i}-th circuit ({circuit.num_qubits}) does "
                    f"not match the number of qubits of the {i}-th observable "
                    f"({observable.num_qubits})."
                )

        for i, (circuit, parameter_set) in enumerate(zip(circuits, parameter_sets)):
            if not set(parameter_set).issubset(circuit.parameters):
                raise ValueError(
                    f"The {i}-th parameter set contains parameters not present in the "
                    f"{i}-th circuit."
                )

    @property
    def options(self) -> Options:
        """Return the union of estimator options setting and gradient default options,
        where, if the same field is set in both, the gradient's default options override
        the primitive's default setting.

        Returns:
            The gradient default + estimator options.
        """
        return self._get_local_options(self._default_options.__dict__)

    def update_default_options(self, **options):
        """Update the gradient's default options setting.

        Args:
            **options: The fields to update the default options.
        """

        self._default_options.update_options(**options)

    def _get_local_options(self, options: Options) -> Options:
        """Return the union of the primitive's default setting,
        the gradient default options, and the options in the ``run`` method.
        The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.

        Args:
            options: The fields to update the options

        Returns:
            The gradient default + estimator + run options.
        """
        opts = copy(self._estimator.options)
        opts.update_options(**options)
        return opts
