# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Class for probability distributions."""

import re


# NOTE: A dict subclass should not overload any dunder methods like __getitem__
# this can cause unexpected behavior and issues as the cPython dict
# implementation has many standard methods in C for performance and the dunder
# methods are not always used as expected. For example, update() doesn't call
# __setitem__ so overloading __setitem__ would not always provide the expected
# result
class ProbDistribution(dict):
    """A generic dict-like class for probability distributions."""

    _bitstring_regex = re.compile(r"^[01]+$")

    def __init__(self, data, shots=None):
        """Builds a probability distribution object.

        Args:
            data (dict): Input probability data. Where the keys
                represent a measured classical value and the value is a
                float for the probability of that result.
                The keys can be one of several formats:

                    * A hexadecimal string of the form ``"0x4a"``
                    * A bit string e.g. ``'0b1011'`` or ``"01011"``
                    * An integer

            shots (int): Number of shots the distribution was derived from.

        Raises:
            TypeError: If the input keys are not a string or int
            ValueError: If the string format of the keys is incorrect
        """
        self.shots = shots
        if data:
            first_key = next(iter(data.keys()))
            if isinstance(first_key, int):
                pass
            elif isinstance(first_key, str):
                if first_key.startswith("0x"):
                    hex_raw = data
                    data = {int(key, 0): value for key, value in hex_raw.items()}
                elif first_key.startswith("0b"):
                    bin_raw = data
                    data = {int(key, 0): value for key, value in bin_raw.items()}
                elif self._bitstring_regex.search(first_key):
                    bin_raw = data
                    data = {int("0b" + key, 0): value for key, value in bin_raw.items()}
                else:
                    raise ValueError(
                        "The input keys are not a valid string format, must either "
                        "be a hex string prefixed by '0x' or a binary string "
                        "optionally prefixed with 0b"
                    )
            else:
                raise TypeError("Input data's keys are of invalid type, must be str or int")
        super().__init__(data)

    def binary_probabilities(self, num_bits=None):
        """Build a probabilities dictionary with binary string keys

        Parameters:
            num_bits (int): number of bits in the binary bitstrings (leading
                zeros will be padded). If None, the length will be derived
                from the largest key present.

        Returns:
            dict: A dictionary where the keys are binary strings in the format
                ``"0110"``
        """
        n = len(bin(max(self.keys(), default=0))) - 2 if num_bits is None else num_bits
        return {format(key, "b").zfill(n): value for key, value in self.items()}

    def hex_probabilities(self):
        """Build a probabilities dictionary with hexadecimal string keys

        Returns:
            dict: A dictionary where the keys are hexadecimal strings in the
                format ``"0x1a"``
        """
        return {hex(key): value for key, value in self.items()}
