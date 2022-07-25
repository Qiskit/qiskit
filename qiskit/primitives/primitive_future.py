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
Future class for primitives.
"""


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from .estimator_result import EstimatorResult
from .sampler_result import SamplerResult

R = TypeVar("R", SamplerResult, EstimatorResult)  # pylint: disable=invalid-name


class PrimitiveFuture(ABC, Generic[R]):
    """TODO: Docs"""

    @abstractmethod
    def result(self) -> R:
        """Return result."""
        ...

    @abstractmethod
    def cancel(self) -> bool:
        """Cancel."""
        ...

    @abstractmethod
    def canceled(self) -> bool:
        """Return True if canceled."""
        ...

    @abstractmethod
    def running(self) -> bool:
        """Return True if running."""
        ...

    @abstractmethod
    def done(self) -> bool:
        """Return True if done."""
        ...
