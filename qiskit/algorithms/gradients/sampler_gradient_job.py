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
Sampler result class
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit.primitives import SamplerResult
from qiskit.providers import JobStatus


@dataclass(frozen=True)
class SamplerGradientJob:
    """Result of SamplerGradient.

    Args:
        results (list[SamplerResult]): List of SamplerResults. The i-th result corresponds to
            ``circuits[i]`` evaluated with parameters bound as ``parameter_values[i]``. The j-th
            quasi-probability distribution in the i-th result corresponds to the gradients of the
            sampling probability for the j-th parameter in ``circuits[i]``.
        status: List of JobStatus for each SamplerResult.
    """

    results: list[SamplerResult]
    status: list[JobStatus]
