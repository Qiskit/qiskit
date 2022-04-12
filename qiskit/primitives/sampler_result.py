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
    """Result of Sampler.

    .. code-block:: python

        result = sampler(circuits, params)

    where the i-th elements of ``result`` correspond to the circuit given by ``circuit_indices[i]``,
    and the parameter values bounds by ``params[i]``.
    For example, ``results.quasi_dists[i]`` gives the quasi-probabilities of bitstrings, and
    ``result.metadata[i]`` is a metadata dictionary for this circuit and parameters.

    Args:
        quasi_dists (list[QuasiDistribution]): List of the quasi-probabilities.
        metadata (list[dict]): List of the metadata.
    """

    quasi_dists: list[QuasiDistribution]
    metadata: list[dict[str, Any]]
