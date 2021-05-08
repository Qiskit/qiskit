# This code is part of qiskit-runtime.
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

from math import sqrt


class ProbDistribution(dict):
    """A generic dict-like class for probability distributions.
    """
    def __init__(self, data, shots=None):
        """Builds a probability distribution object.

        Parameters:
            data (dict): Input probability data.
            shots (int): Number of shots the distribution was derived from.
        """
        self.shots = shots
        super().__init__(data)
