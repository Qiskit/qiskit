# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
This module implements the abstract base class for algorithm results.
"""

from abc import ABC
import inspect
import pprint


class AlgorithmResult(ABC):
    """Abstract Base Class for algorithm results."""

    def __str__(self) -> str:
        result = {}
        for name, value in inspect.getmembers(self):
            if (
                not name.startswith("_")
                and not inspect.ismethod(value)
                and not inspect.isfunction(value)
                and hasattr(self, name)
            ):

                result[name] = value

        return pprint.pformat(result, indent=4)

    def combine(self, result: "AlgorithmResult") -> None:
        """
        Any property from the argument that exists in the receiver is
        updated.
        Args:
            result: Argument result with properties to be set.
        Raises:
            TypeError: Argument is None
        """
        if result is None:
            raise TypeError("Argument result expected.")
        if result == self:
            return

        # find any result public property that exists in the receiver
        for name, value in inspect.getmembers(result):
            if (
                not name.startswith("_")
                and not inspect.ismethod(value)
                and not inspect.isfunction(value)
                and hasattr(self, name)
            ):
                try:
                    setattr(self, name, value)
                except AttributeError:
                    # some attributes may be read only
                    pass
