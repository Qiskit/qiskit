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
Estimator result class
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit.primitives import EstimatorResult
from qiskit.providers import JobStatus


@dataclass(frozen=True)
class EstimatorGradientJob:
    """Result of EstimatorGradient.

    Args:
        results (list[EstimatorResult]): List of EstimatorResults. The i-th result corresponds to
            ``circuits[i]`` evaluated with parameters bound as ``parameter_values[i]``. The j-th value
            in the i-th result corresponds to the gradients of the j-th parameter in ``circuits[i]``.
        status: List of JobStatus for each EstimatorResult.
    """

    results: list[EstimatorResult]
    status: list[JobStatus]
