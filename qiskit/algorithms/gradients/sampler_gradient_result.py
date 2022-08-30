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

from typing import Any
from dataclasses import dataclass

from qiskit.result import QuasiDistribution


@dataclass(frozen=True)
class SamplerGradientResult:
    """Result of SamplerGradient.

    Args:
        gradients: The gradients of the quasi distributions.
        metadata: Additional information about the job.
        run_options: run_options for the sampler. Currently, sampler's default run_options is not
        included.
    """

    gradients: list[list[QuasiDistribution]]
    metadata: list[dict[str, Any]]
    run_options: dict[str, Any]
