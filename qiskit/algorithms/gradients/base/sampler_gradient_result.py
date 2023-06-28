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

from qiskit.providers import Options


@dataclass(frozen=True)
class SamplerGradientResult:
    """Result of SamplerGradient."""

    gradients: list[list[dict[int, float]]]
    """The gradients of the sample probabilities."""
    metadata: list[dict[str, Any]]
    """Additional information about the job."""
    options: Options
    """Primitive runtime options for the execution of the job."""
