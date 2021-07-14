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
"""Utility functions"""
import numpy as np
from qiskit.result.distributions import QuasiDistribution


def counts_to_vector(counts):
    """ Return probability vector from counts dict.

    Parameters:
        counts (dict): Input dict of counts.

    Returns:
        ndarray: 1D array of probabilities.
    """
    num_bitstrings = len(counts)
    shots = sum(counts.values())
    vec = np.zeros(num_bitstrings, dtype=float)
    idx = 0
    for val in counts.values():
        vec[idx] = val / shots
        idx += 1
    return vec


def vector_to_quasiprobs(vec, counts):
    """ Return dict of quasi-probabilities.

    Parameters:
        vec (ndarray): 1d vector of quasi-probabilites.
        counts (dict): Dict of counts

    Returns:
        QuasiDistribution: dict of quasi-probabilities
    """
    out_counts = {}
    idx = 0
    for key in counts:
        out_counts[key] = vec[idx]
        idx += 1
    return QuasiDistribution(out_counts)
