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

from abc import ABC
from collections.abc import Sequence

import numpy as np

from ..estimator_result import EstimatorResult
from .base_gradient import BaseGradientMixin


class FiniteDiffGradientMixin(BaseGradientMixin, ABC):
    def gradient(
        self,
        circuit_index: int,
        observable_index: int,
        parameter_value: Sequence[float],
        **run_options,
    ) -> EstimatorResult:
        run_options = run_options.copy()
        if "epsilon" in run_options:
            epsilon = run_options["epsilon"]
            del run_options["epsilon"]
        else:
            epsilon = 1e-6

        dim = len(parameter_value)
        ret = []
        ret.append(parameter_value)
        for i in range(dim):
            ei = parameter_value.copy()
            ei[i] += epsilon
            ret.append(ei)
        param_array = np.array(ret).tolist()
        circuit_indices = [circuit_index] * (dim + 1)
        observable_indices = [observable_index] * (dim + 1)
        results = self.__call__(circuit_indices, observable_indices, param_array, **run_options)

        values = results.values
        ret = []
        grad = np.zeros(dim)
        f_ref = values[0]
        for i, f_i in enumerate(values[1:]):
            grad[i] = (f_i - f_ref) / epsilon
        return EstimatorResult(values=grad, metadata=[{}] * len(grad))
