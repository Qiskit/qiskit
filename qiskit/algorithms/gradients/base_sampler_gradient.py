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
Abstract base class of gradient for ``Sampler``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import copy

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.primitives import BaseSampler
from qiskit.providers import Options
from qiskit.algorithms import AlgorithmJob
from .sampler_gradient_result import SamplerGradientResult


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
                each circuit are calculated.
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
        # if ``parameters`` is none, all parameters in each circuit are differentiated.
        if parameters is None:
            parameters = [None for _ in range(len(circuits))]
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
        parameters: Sequence[Sequence[Parameter] | None],
        **options,
    ) -> SamplerGradientResult:
        """Compute the sampler gradients on the given circuits."""
        raise NotImplementedError()

    def _validate_arguments(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None] | None = None,
    ) -> None:
        """Validate the arguments of the ``run`` method.

        Args:
            circuits: The list of quantum circuits to compute the gradients.
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
        if parameters is not None:
            if len(circuits) != len(parameters):
                raise ValueError(
                    f"The number of circuits ({len(circuits)}) does not match "
                    f"the number of the specified parameter sets ({len(parameters)})."
                )

        for i, (circuit, parameter_value) in enumerate(zip(circuits, parameter_values)):
            if not circuit.num_parameters:
                raise ValueError(f"The {i}-th circuit is not parameterised.")

            if len(parameter_value) != circuit.num_parameters:
                raise ValueError(
                    f"The number of values ({len(parameter_value)}) does not match "
                    f"the number of parameters ({circuit.num_parameters}) for the {i}-th circuit."
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
