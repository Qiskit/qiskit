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
Sampler Pub result class
"""

from __future__ import annotations

from .bit_array import BitArray
from .pub_result import PubResult


class SamplerPubResult(PubResult):
    """Result of Sampler Pub."""

    def _preprocessing(self, name: str | None) -> BitArray:
        if name is None and len(self.data) != 1:
            raise ValueError("Since there is not exactly one data field, a name must be provided.")
        if name is None:
            name = self.data._FIELDS[0]
        data_value = getattr(self.data, name)
        if not isinstance(data_value, BitArray):
            raise TypeError(f"The requested field {name} is not a BitArray.")
        return data_value

    def get_counts(self, loc: tuple[int] | None = None, name: str | None = None):
        """Return a counts dictionary with bitstring keys.

        Args:
            loc: Which entry of this array to return a dictionary for. If None, counts from
                all positions in this array are unioned together.
            name: The field name.

        Returns:
            A dictionary mapping bitstrings to the number of occurrences of that bitstring.

        Raises:
            ValueError: if there are more than one fields and no field name is provided.
            ValueError: if an invalid field name is provided.
        """
        data_value = self._preprocessing(name)
        return data_value.get_counts(loc=loc)

    def get_int_counts(self, loc: tuple[int] | None = None, name: str | None = None):
        r"""Return a counts dictionary, where bitstrings are stored as int\s.

        Args:
            loc: Which entry of this array to return a dictionary for. If None, counts from
                all positions in this array are unioned together.
            name: The field name.

        Returns:
            A dictionary mapping bitstrings to the number of occurrences of that bitstring.

        Raises:
            ValueError: if there are more than one fields and no field name is provided.
            ValueError: if an invalid field name is provided.
        """
        data_value = self._preprocessing(name)
        return data_value.get_int_counts(loc=loc)

    def get_bitstrings(self, loc: tuple[int] | None = None, name: str | None = None):
        """Return a list of bitstrings.

        Args:
            loc: Which entry of this array to return a dictionary for. If None, counts from
                all positions in this array are unioned together.
            name: The field name.

        Returns:
            A dictionary mapping bitstrings to the number of occurrences of that bitstring.

        Raises:
            ValueError: if there are more than one fields and no field name is provided.
            ValueError: if an invalid field name is provided.
        """
        data_value = self._preprocessing(name)
        return data_value.get_bitstrings(loc=loc)
