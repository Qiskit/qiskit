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
from warnings import warn

from numpy import array


@dataclass(frozen=True)
class EstimatorResult:
    """Result of Estimator.

    .. code-block:: python

        result = estimator(circuits, observables, params)

    where the i-th elements of ``result`` correspond to the circuit and observable given by
    ``circuit_indices[i]``, ``observable_indices[i]``, and the parameter values bounds by ``params[i]``.
    For example, ``results.expvals[i]`` gives the expectation value, ``results.variances[i]``
    the variance, and ``result.metadata[i]`` is a metadata dictionary for this circuit and
    parameters.

    Args:
        expvals (tuple[float, ...]): Tuple of expectation values.
        variances (tuple[float, ...]): Tuple of variances associated to each expectation value.
        metadata (tuple[dict, ...]): Tuple of metadata.
    """

    expvals: tuple[float, ...]
    variances: tuple[float, ...]
    metadata: tuple[dict[str, Any], ...]

    @property
    def values(self) -> None:
        warn(
            "``EstimatorResult.values`` will be deprecated,"
            "use ``EstimatorResult.expvals`` instead.",
            DeprecationWarning,
        )
        return array(self.expvals)
