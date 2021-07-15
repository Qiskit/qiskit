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
        """
        self.shots = shots
        super().__init__(data)
