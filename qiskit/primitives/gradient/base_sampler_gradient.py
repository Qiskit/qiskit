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

from ..base_sampler import BaseSampler
from ..sampler_result import SamplerResult
from .sampler_gradient_result import SamplerGradientResult


class BaseSamplerGradient(ABC):
    def __init__(
        self,
        sampler: BaseSampler,
    ):
        self._sampler = sampler

    @abstractmethod
    def gradient(
        self,
        circuits: Sequence[int | QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> SamplerGradientResult:
        ...

    @abstractmethod
    def sample(
        self,
        circuits: Sequence[int | QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> SamplerResult:
        ...

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._sampler.close()
