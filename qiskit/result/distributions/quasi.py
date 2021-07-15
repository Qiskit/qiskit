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
# pylint: disable=no-name-in-module
"""Quasidistribution class"""
import re
from math import sqrt

from qiskit.exceptions import QiskitError
from .mle_prob import quasi_to_probs
from .expval import exp_val
from .probability import ProbDistribution


# NOTE: A dict subclass should not overload any dunder methods like __getitem__
# this can cause unexpected behavior and issues as the cPython dict
# implementation has many standard methods in C for performance and the dunder
# methods are not always used as expected. For example, update() doesn't call
# __setitem__ so overloading __setitem__ would not always provide the expected
# result
class QuasiDistribution(dict):
    """A dict-like class for representing qasi-probabilities."""

    _bitstring_regex = re.compile(r"^[01]+$")

    def __init__(self, data, shots=None, mitigation_overhead=None):
        """Builds a quasiprobability distribution object.

        Parameters:
            data (dict): Input quasiprobability data.
            shots (int): Number of shots the distribution was derived from.
            mitigation_overhead (float): Overhead from performing mitigation.
        """
        self.shots = shots
        self.mitigation_overhead = mitigation_overhead
        super().__init__(data)

    def expval(self):
        """Compute expectation value from distribution.

        Returns:
            float: Expectation value.
        """
        return exp_val(self)

    def expval_and_stddev(self):
        """Compute expectation value and standard deviation estimate from distribution.

        Returns:
            float: Expectation value.
            float: Estimate of standard deviation upper-bound.

        Raises:
            QiskitError: Missing shots or mitigation_overhead information.
        """
        if self.shots is None:
            raise QiskitError("Quasi-dist is missing shots information.")
        if self.mitigation_overhead is None:
            raise QiskitError("Quasi-dist is missing mitigation overhead.")
        return exp_val(self), sqrt(self.mitigation_overhead / self.shots)

    def nearest_probability_distribution(self, return_distance=False):
        """Takes a quasiprobability distribution and maps
        it to the closest probability distribution as defined by
        the L2-norm.

        Parameters:
            return_distance (bool): Return the L2 distance between distributions.

        Returns:
            ProbDistribution: Nearest probability distribution.
            float: Euclidean (L2) distance of distributions.

        Notes:
            Method from Smolin et al., Phys. Rev. Lett. 108, 070502 (2012).
        """
        new_probs, dist = quasi_to_probs(self)
        if return_distance:
            return ProbDistribution(new_probs, self.shots), dist
        return ProbDistribution(new_probs, self.shots)
