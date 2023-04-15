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
Abstract base class of gradient for ``Sampler``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from copy import copy

from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
from qiskit.primitives import BaseSampler
from qiskit.primitives.utils import _circuit_key
from qiskit.providers import Options
from qiskit.transpiler.passes import TranslateParameterizedGates

from .sampler_gradient_result import SamplerGradientResult
from ..utils import (
    GradientCircuit,
    _assign_unique_parameters,
    _make_gradient_parameters,
    _make_gradient_parameter_values,
)

from ...algorithm_job import AlgorithmJob


class BaseSamplerGradient(ABC):
    """Base class for a ``SamplerGradient`` to compute the gradients of the sampling probability."""

    def __init__(self, sampler: BaseSampler, options: Options | None = None):
        """
        Args:
            sampler: The sampler used to compute the gradients.
            options: Primitive backend runtime options used for circuit execution.
                The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.
                Higher priority setting overrides lower priority setting
        """
        self._sampler: BaseSampler = sampler
        self._default_options = Options()
        if options is not None:
            self._default_options.update_options(**options)
        self._gradient_circuit_cache: dict[tuple, GradientCircuit] = {}

    def run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None] | None = None,
        **options,
    ) -> AlgorithmJob:
        """Run the job of the sampler gradient on the given circuits.

        Args:
            circuits: The list of quantum circuits to compute the gradients.
            parameter_values: The list of parameter values to be bound to the circuit.
            parameters: The sequence of parameters to calculate only the gradients of
                the specified parameters. Each sequence of parameters corresponds to a circuit in
                ``circuits``. Defaults to None, which means that the gradients of all parameters in
                each circuit are calculated. None in the sequence means that the gradients of all
                parameters in the corresponding circuit are calculated.
            options: Primitive backend runtime options used for circuit execution.
                The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.
                Higher priority setting overrides lower priority setting
        Returns:
            The job object of the gradients of the sampling probability. The i-th result
            corresponds to ``circuits[i]`` evaluated with parameters bound as ``parameter_values[i]``.
            The j-th quasi-probability distribution in the i-th result corresponds to the gradients of
            the sampling probability for the j-th parameter in ``circuits[i]``.

        Raises:
            ValueError: Invalid arguments are given.
        """
        if isinstance(circuits, QuantumCircuit):
            # Allow a single circuit to be passed in.
            circuits = (circuits,)
        if parameters is None:
            # If parameters is None, we calculate the gradients of all parameters in each circuit.
            parameters = [circuit.parameters for circuit in circuits]
        else:
            # If parameters is not None, we calculate the gradients of the specified parameters.
            # None in parameters means that the gradients of all parameters in the corresponding
            # circuit are calculated.
            parameters = [
                params if params is not None else circuits[i].parameters
                for i, params in enumerate(parameters)
            ]
        # Validate the arguments.
        self._validate_arguments(circuits, parameter_values, parameters)
        # The priority of run option is as follows:
        # options in `run` method > gradient's default options > primitive's default options.
        opts = copy(self._default_options)
        opts.update_options(**options)
        job = AlgorithmJob(self._run, circuits, parameter_values, parameters, **opts.__dict__)
        job.submit()
        return job

    @abstractmethod
    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
        **options,
    ) -> SamplerGradientResult:
        """Compute the sampler gradients on the given circuits."""
        raise NotImplementedError()

    def _preprocess(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
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
        translator = TranslateParameterizedGates(supported_gates)
        g_circuits, g_parameter_values, g_parameters = [], [], []
        for circuit, parameter_value_, parameters_ in zip(circuits, parameter_values, parameters):
            circuit_key = _circuit_key(circuit)
            if circuit_key not in self._gradient_circuit_cache:
                unrolled = translator(circuit)
                self._gradient_circuit_cache[circuit_key] = _assign_unique_parameters(unrolled)
            gradient_circuit = self._gradient_circuit_cache[circuit_key]
            g_circuits.append(gradient_circuit.gradient_circuit)
            g_parameter_values.append(
                _make_gradient_parameter_values(circuit, gradient_circuit, parameter_value_)
            )
            g_parameters.append(_make_gradient_parameters(gradient_circuit, parameters_))
        return g_circuits, g_parameter_values, g_parameters

    def _postprocess(
        self,
        results: SamplerGradientResult,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None],
    ) -> SamplerGradientResult:
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
        for idx, (circuit, parameter_values_, parameters_) in enumerate(
            zip(circuits, parameter_values, parameters)
        ):
            gradient_circuit = self._gradient_circuit_cache[_circuit_key(circuit)]
            g_parameters = _make_gradient_parameters(gradient_circuit, parameters_)
            # Make a map from the gradient parameter to the respective index in the gradient.
            g_parameter_indices = {param: i for i, param in enumerate(g_parameters)}
            # Compute the original gradient from the gradient of the gradient circuit
            # by using the chain rule.
            gradient = []
            for parameter in parameters_:
                grad_dist: dict[int, float] = defaultdict(float)
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
                    unique_gradient = results.gradients[idx][g_parameter_indices[g_parameter]]
                    for key, value in unique_gradient.items():
                        grad_dist[key] += float(bound_coeff) * value
                gradient.append(dict(grad_dist))
            gradients.append(gradient)
            metadata.append([{"parameters": parameters_}])
        return SamplerGradientResult(
            gradients=gradients, metadata=metadata, options=results.options
        )

    @staticmethod
    def _validate_arguments(
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]],
    ) -> None:
        """Validate the arguments of the ``run`` method.

        Args:
            circuits: The list of quantum circuits to compute the gradients.
            parameter_values: The list of parameter values to be bound to the circuit.
            parameters: The sequence of parameters to calculate only the gradients of the specified
                parameters.

        Raises:
            ValueError: Invalid arguments are given.
        """
        if len(circuits) != len(parameter_values):
            raise ValueError(
                f"The number of circuits ({len(circuits)}) does not match "
                f"the number of parameter value sets ({len(parameter_values)})."
            )

        for i, (circuit, parameter_value) in enumerate(zip(circuits, parameter_values)):
            if not circuit.num_parameters:
                raise ValueError(f"The {i}-th circuit is not parameterised.")

            if len(parameter_value) != circuit.num_parameters:
                raise ValueError(
                    f"The number of values ({len(parameter_value)}) does not match "
                    f"the number of parameters ({circuit.num_parameters}) for the {i}-th circuit."
                )

        if len(circuits) != len(parameters):
            raise ValueError(
                f"The number of circuits ({len(circuits)}) does not match "
                f"the number of the specified parameter sets ({len(parameters)})."
            )

        for i, (circuit, parameters_) in enumerate(zip(circuits, parameters)):
            if not set(parameters_).issubset(circuit.parameters):
                raise ValueError(
                    f"The {i}-th parameter set contains parameters not present in the "
                    f"{i}-th circuit."
                )

    @property
    def options(self) -> Options:
        """Return the union of sampler options setting and gradient default options,
        where, if the same field is set in both, the gradient's default options override
        the primitive's default setting.

        Returns:
            The gradient default + sampler options.
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
            The gradient default + sampler + run options.
        """
        opts = copy(self._sampler.options)
        opts.update_options(**options)
        return opts
