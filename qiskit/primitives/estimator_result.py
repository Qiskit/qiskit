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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class EstimatorResult:
    """
    Result of ExpectationValue

    .. code-block:: python

        result = estimator(circuits, observables, params)

    where the i-th elements of `result` correspond to the expectation using the circuit and
    observable given by `circuits[i]`, `observables[i]`, and the parameters bounds by `params[i]`.

    Args:
        values (np.ndarray): the array of the expectation values.
        metadata (list[dict]): list of the metadata.
    """

    values: "np.ndarray[Any, np.dtype[np.float64]]"
    metadata: list[dict[str, Any]]
