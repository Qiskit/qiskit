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
Options class
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping


class BasePrimitiveOptions(ABC):
    """Base class of options for primitives."""

    def update(self, options: BasePrimitiveOptions | Mapping | None = None, **kwargs):
        """Update the options."""
        if options is not None:
            if isinstance(options, Mapping):
                options_dict = options
            elif isinstance(options, BasePrimitiveOptions):
                options_dict = options.__dict__
            else:
                raise TypeError(f"Type {type(options)} is not options nor Mapping class")
            for key, val in options_dict.items():
                setattr(self, key, val)

        for key, val in kwargs.items():
            setattr(self, key, val)
