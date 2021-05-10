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
"""Quasidistribution class"""

from math import sqrt
from .probability import ProbDistribution


# NOTE: A dict subclass should not overload any dunder methods like __getitem__
# this can cause unexpected behavior and issues as the cPython dict
# implementation has many standard methods in C for performance and the dunder
# methods are not always used as expected. For example, update() doesn't call
# __setitem__ so overloading __setitem__ would not always provide the expected
# result
class QuasiDistribution(dict):
    """A dict-like class for representing qasi-probabilities.

    .. warning::

        This is an unsupported class in the current 0.17.x release series. It
        is present for compatibility with the qiskit-ibmq-provider's beta
        qiskit runtime support, but this interface isn't guaranteed to be
        stable when moving to >=0.18.0 and likely will change.
    """

    def __init__(self, data, shots=None):
        """Builds a quasiprobability distribution object.

        Parameters:
            data (dict): Input quasiprobability data.
            shots (int): Number of shots the distribution was derived from.
        """
        self.shots = shots
        super().__init__(data)

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
        sorted_probs = dict(sorted(self.items(), key=lambda item: item[1]))
        num_elems = len(sorted_probs)
        new_probs = {}
        beta = 0
        diff = 0
        for key, val in sorted_probs.items():
            temp = val + beta / num_elems
            if temp < 0:
                beta += val
                num_elems -= 1
                diff += val * val
            else:
                diff += (beta / num_elems) * (beta / num_elems)
                new_probs[key] = sorted_probs[key] + beta / num_elems
        if return_distance:
            return ProbDistribution(new_probs, self.shots), sqrt(diff)
        return ProbDistribution(new_probs, self.shots)
