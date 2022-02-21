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

from qiskit.result import QuasiDistribution, Result

from .base_result import BaseResult


class SamplerResult(BaseResult):
    """
    Result of Sampler
    """

    quasi_dists: list[QuasiDistribution]
    shots: int
    raw_results: list[Result]
    metadata: list[dict]

    def __getitem__(self, key):
        raw = self.raw_results
        new_result = Result(
            backend_name=raw.backend_name,
            backend_version=raw.backend_version,
            qobj_id=raw.qobj_id,
            job_id=raw.job_id,
            success=raw.success,
            results=raw.results[key],
        )
        return SamplerResult(self.quasi_dists[key], self.shots, new_result, self.metadata[key])
