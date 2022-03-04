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
from typing import Any

from qiskit.result import QuasiDistribution


@dataclass(frozen=True)
class SamplerResult:
    """
    Result of Sampler

    .. code-block:: python

        result = sampler(circuits, params)

    where the i-th elements of `result` correspond to the expectation using the circuit
    given by `circuits[i]` and the parameters bounds by `params[i]`.
    """

    quasi_dists: list[QuasiDistribution]
    metadata: list[dict[str, Any]]
