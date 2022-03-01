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
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EstimatorResult:
    """
    Result of ExpectationValue
    #TODO doc
    """

    values: "np.ndarray[Any, np.dtype[np.float64]]"
    variances: "np.ndarray[Any, np.dtype[np.float64]]"
    # standard_errors: np.ndarray[Any, np.dtype[np.float64]]
    metadata: list[dict[str, Any]]

    def __add__(self, other: EstimatorResult):
        values = np.concatenate([self.values, other.values])
        variances = np.concatenate([self.variances, other.variances])
        return EstimatorResult(values, variances)
