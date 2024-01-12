# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Primitive job abstract base class
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Union

from ..containers import PrimitiveResult
from .base_result import BasePrimitiveResult

Result = TypeVar("Result", bound=Union[BasePrimitiveResult, PrimitiveResult])
Status = TypeVar("Status")


class BasePrimitiveJob(ABC, Generic[Result, Status]):
    """Primitive job abstract base class."""

    @abstractmethod
    def result(self) -> Result:
        """Return the results of the job."""
        pass

    @abstractmethod
    def status(self) -> Status:
        """Return the status of the job."""
        pass
