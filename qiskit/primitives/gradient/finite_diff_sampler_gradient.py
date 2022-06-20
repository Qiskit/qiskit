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

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from qiskit.result import QuasiDistribution

from ..base_sampler import BaseSampler
from ..sampler_result import SamplerResult


class FiniteDiffSamplerGradient:
    def __init__(self, sampler: BaseSampler, epsilon: float = 1e-6):
        self._epsilon = epsilon
        self._sampler = sampler

    def gradient(
        self,
        circuit_index: int,
        parameter_value: Sequence[float],
        **run_options,
    ) -> SamplerResult:
        run_options = run_options.copy()

        dim = len(parameter_value)
        params = [parameter_value]
        for i in range(dim):
            ei = parameter_value.copy()
            ei[i] += self._epsilon
            params.append(ei)
        param_list = np.array(params).tolist()
        circuit_indices = [circuit_index] * (dim + 1)
        results = self._sampler.__call__(circuit_indices, param_list, **run_options)

        quasi_dists = results.quasi_dists
        ret = []
        f_ref = quasi_dists[0]
        for f_i in quasi_dists[1:]:
            ret.append(
                QuasiDistribution({key: (f_i[key] - f_ref[key]) / self._epsilon for key in f_ref})
            )
        return SamplerResult(quasi_dists=ret, metadata=[{}] * len(ret))
