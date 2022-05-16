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
    """Result of Estimator.

    .. code-block:: python

        result = estimator(circuits, observables, params)

    where the i-th elements of ``result`` correspond to the circuit and observable given by
    ``circuit_indices[i]``, ``observable_indices[i]``, and the parameter values bounds by ``params[i]``.
    For example, ``results.values[i]`` gives the expectation value, and ``result.metadata[i]``
    is a metadata dictionary for this circuit and parameters.

    Args:
        values (np.ndarray): The array of the expectation values.
        metadata (list[dict]): List of the metadata.
    """

    values: "np.ndarray[Any, np.dtype[np.float64]]"
    metadata: list[dict[str, Any]]
