# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Options class
"""

from __future__ import annotations

from abc import ABC
from typing import Union

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

primitive_dataclass = dataclass(
    config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True, extra="forbid")
)


@primitive_dataclass
class BasePrimitiveOptions(ABC):
    """Base calss of options for primitives."""

    def update(self, **kwargs):
        """Update the options."""
        for key, val in kwargs.items():
            setattr(self, key, val)


BasePrimitiveOptionsLike = Union[BasePrimitiveOptions, dict]
