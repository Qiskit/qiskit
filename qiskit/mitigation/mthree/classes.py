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
"""mthree classes"""

import math
from mthree.probability import quasi_to_probs
from mthree.expval import exp_val, exp_val_and_stddev
from mthree.exceptions import M3Error


class ProbDistribution(dict):
    """A generic dict-like class for probability distributions.
    """
    def __init__(self, data, shots=None):
        """Initialize a probability distribution.

        Parameters:
            data (dict): Input data.
            shots (int): Number shots taken to form distribution.
        """
        self.shots = shots
        super().__init__(data)

    def expval(self):
        """Compute expectation value from distribution.

        Returns:
            float: Expectation value.
        """
        return exp_val(self)

    def expval_and_stddev(self):
        """Compute expectation value and standard deviation from distribution.

        Returns:
            float: Expectation value.
            float: Standard deviation.
        """
        return exp_val_and_stddev(self)


class QuasiDistribution(dict):
    """A dict-like class for representing quasi-probabilities.
    """
    def __init__(self, data, shots=None, mitigation_overhead=None):
        """Initialize a quasi-distribution.

        Parameters:
            data (dict): Input data.
            shots (int): Number shots taken to form quasi-distribution.
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
            M3Error: Missing shots or mitigation_overhead information.
        """
        if self.shots is None:
            raise M3Error('Quasi-dist is missing shots information.')
        if self.mitigation_overhead is None:
            raise M3Error('Quasi-dist is missing mitigation overhead.')
        return exp_val(self), math.sqrt(self.mitigation_overhead / self.shots)

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
        probs, dist = quasi_to_probs(self)
        if return_distance:
            return ProbDistribution(probs, self.shots), dist
        return ProbDistribution(probs, self.shots)
