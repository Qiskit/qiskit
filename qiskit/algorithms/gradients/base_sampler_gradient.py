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
Abstract Base class of Gradient for Sampler.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.primitives import BaseSampler
from .sampler_gradient_job import SamplerGradientJob


class BaseSamplerGradient(ABC):
    """Base class for a SamplerGradient to compute the gradients of the sampling probability."""

    def __init__(self, sampler: BaseSampler, **run_options):
        """
        Args:
            sampler: The sampler used to compute the gradients.
        """
        self._sampler: BaseSampler = sampler
        self._circuits: list[QuantumCircuit] = []
        self._circuit_ids: dict[int, int] = {}
        self._default_run_options = run_options

    def run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        partial: Sequence[Sequence[Parameter]] | None = None,
        **run_options,
    ) -> SamplerGradientJob:
        """Run the job of the gradients of the sampling probability.

        Args:
            circuits: The list of quantum circuits to compute the gradients.
            parameter_values: The list of parameter values to be bound to the circuit.
            partial: The list of Parameters to calculate only the gradients of the specified parameters.
                Defaults to None, which means that the gradients of all parameters will be calculated.
            run_options: Backend runtime options used for circuit execution.

        Returns:
            The job object of the gradients of the sampling probability. The i-th result corresponds to
            ``circuits[i]`` evaluated with parameters bound as ``parameter_values[i]``. The j-th
            quasi-probability distribution in the i-th result corresponds to the gradients of the
            sampling probability for the j-th parameter in ``circuits[i]``.
        """
        # The priority of run option is as follows:
        # run_options in `run` method > gradient's default run_options > primitive's default run_options.
        run_options = run_options or self._default_run_options
        return self._run(circuits, parameter_values, partial, **run_options)

    @abstractmethod
    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        partial: Sequence[Sequence[Parameter]] | None = None,
        **run_options,
    ) -> SamplerGradientJob:
        raise NotImplementedError()
