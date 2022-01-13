# This code is part of Mthree.
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
"""
Distributions
-------------

.. autosummary::
   :toctree: ../stubs/

   QuasiDistribution
   ProbDistribution

Distribution collections
------------------------

.. autosummary::
   :toctree: ../stubs/

   QuasiCollection
   ProbCollection
"""

import math
import numpy as np
from qiskit.result import Counts

from .probability import quasi_to_probs
from .expval import exp_val
from .exceptions import M3Error


class ProbDistribution(dict):
    """A generic dict-like class for probability distributions.
    """
    def __init__(self, data, shots=None, mitigation_overhead=None):
        """A generic dict-like class for probability distributions.

        Parameters:
            data (dict or Counts or ProbDistribution or QuasiDistribution): Input data.
            shots (int): Number shots taken to form distribution.

        Raises:
            M3Error: Input not derived from discrete samples.
        """
        # _gen_call means being called from the general utils function
        if isinstance(data, Counts):
            # Convert Counts to probs
            self.shots = int(sum(data.values()))
            self.mitigation_overhead = 1
            _data = {}
            for key, val, in data.items():
                _data[key] = val / self.shots
            data = _data
        else:
            if shots is None:
                self.shots = int(sum(data.values()))
                self.mitigation_overhead = 1
                if self.shots != 1:
                    _data = {}
                    for key, val, in data.items():
                        _data[key] = val / self.shots
                    data = _data
            else:
                self.shots = shots
                self.mitigation_overhead = mitigation_overhead
        super().__init__(data)

    def expval(self, exp_ops=''):
        """Compute expectation value from distribution.

        Parameters:
            exp_ops (str or dict or list): String representation of diagonal
                                           qubit operators
                                           used in computing the expectation value.

        Returns:
            float: Expectation value.

        Raises:
            M3Error: Invalid type passed to exp_ops
        """
        if isinstance(exp_ops, str):
            return exp_val(self, exp_ops=exp_ops)
        elif isinstance(exp_ops, dict):
            return exp_val(self, dict_ops=exp_ops)
        elif isinstance(exp_ops, list):
            return np.array([self.expval(item) for item in exp_ops], dtype=float)
        else:
            raise M3Error('Invalid type passed to exp_ops')

    def stddev(self):
        """Compute standard deviation from distribution.

        Returns:
            float: Standard deviation.

        Raises:
            M3Error: Distribution is missing info.
        """
        if self.shots is None:
            raise M3Error('Prob-dist is missing shots information.')
        if self.mitigation_overhead is None:
            raise M3Error('Prob-dist is missing mitigation overhead.')
        return math.sqrt(self.mitigation_overhead / self.shots)

    def expval_and_stddev(self, exp_ops=''):
        """Compute expectation value and standard deviation from distribution.

        Parameters:
            exp_ops (str or dict): String or dict representation of diagonal qubit operators
                                   used in computing the expectation value.

        Returns:
            float: Expectation value.
            float: Standard deviation.
        """
        return self.expval(exp_ops), self.stddev()


class QuasiDistribution(dict):
    """A dict-like class for representing quasi-probabilities.
    """
    def __init__(self, data, shots=None, mitigation_overhead=None):
        """A dict-like class for representing quasi-probabilities.

        Parameters:
            data (dict): Input data.
            shots (int): Number shots taken to form quasi-distribution.
            mitigation_overhead (float): Overhead from performing mitigation.
        """
        self.shots = shots
        self.mitigation_overhead = mitigation_overhead
        super().__init__(data)

    def expval(self, exp_ops=''):
        """Compute expectation value from distribution.

        Parameters:
            exp_ops (str or dict or list): String or dict representation
                                           of diagonal qubit operators
                                           used in computing the expectation
                                           value.

        Returns:
            float: Expectation value.

        Raises:
            M3Error: Invalid type passed to exp_ops.
        """
        if isinstance(exp_ops, str):
            return exp_val(self, exp_ops=exp_ops)
        elif isinstance(exp_ops, dict):
            return exp_val(self, dict_ops=exp_ops)
        elif isinstance(exp_ops, list):
            return np.array([self.expval(item) for item in exp_ops], dtype=float)
        else:
            raise M3Error('Invalid type passed to exp_ops')

    def stddev(self):
        """Compute standard deviation estimate from distribution.

        Returns:
            float: Estimate of standard deviation upper-bound.

        Raises:
            M3Error: Missing shots or mitigation_overhead information.
        """
        if self.shots is None:
            raise M3Error('Quasi-dist is missing shots information.')
        if self.mitigation_overhead is None:
            raise M3Error('Quasi-dist is missing mitigation overhead.')
        return math.sqrt(self.mitigation_overhead / self.shots)

    def expval_and_stddev(self, exp_ops=''):
        """Compute expectation value and standard deviation estimate from distribution.

        Parameters:
            exp_ops (str or dict): String or dict representation of diagonal qubit operators
                                   used in computing the expectation value.

        Returns:
            float: Expectation value.
            float: Estimate of standard deviation upper-bound.
        """
        return self.expval(exp_ops), self.stddev()

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
            return ProbDistribution(probs, self.shots, self.mitigation_overhead), dist
        return ProbDistribution(probs, self.shots, self.mitigation_overhead)


class QuasiCollection(list):
    """A list subclass that makes handling multiple quasi-distributions easier.
    """
    def __init__(self, data):
        """QuasiCollection constructor.

        Parameters:
            data (list or QuasiCollection): List of QuasiDistribution instances.

        Raises:
            TypeError: Must be list of QuasiDistribution only.
        """
        for dd in data:
            if not isinstance(dd, QuasiDistribution):
                raise TypeError('QuasiCollection requires QuasiDistribution instances.')
        super().__init__(data)

    @property
    def shots(self):
        """Number of shots taken over collection.

        Returns:
            ndarray: Array of shots values.
        """
        return np.array([item.shots for item in self], dtype=int)

    @property
    def mitigation_overhead(self):
        """Mitigation overhead over entire collection.

        Returns:
            ndarray: Array of mitigation overhead values.
        """
        return np.array([item.mitigation_overhead for item in self], dtype=float)

    def expval(self, exp_ops=''):
        """Expectation value over entire collection.

        Parameters:
            exp_ops (str or dict or list): Diagonal operators over which to compute expval.

        Returns:
            ndarray: Array of expectation values.

        Raises:
            M3Error: Length of passes operators does not match container length.
        """
        if isinstance(exp_ops, list):
            if len(exp_ops) != len(self):
                raise M3Error('exp_ops length does not match container length')
            out = []
            for idx, item in enumerate(self):
                out.append(item.expval(exp_ops[idx]))
            return np.array(out, dtype=float)
        return np.array([item.expval(exp_ops) for item in self], dtype=float)

    def expval_and_stddev(self, exp_ops=''):
        """Expectation value and standard deviation over entire collection.

        Parameters:
            exp_ops (str or dict or list): Diagonal operators over which to compute expval.

        Returns:
            list: Tuples of expval and stddev pairs.

        Raises:
            M3Error: Length of passes operators does not match container length.
        """
        if isinstance(exp_ops, list):
            if len(exp_ops) != len(self):
                raise M3Error('exp_ops length does not match container length')
            out = []
            for idx, item in enumerate(self):
                out.append(item.expval_and_stddev(exp_ops[idx]))
            return out
        return [item.expval_and_stddev(exp_ops) for item in self]

    def stddev(self):
        """Standard deviation over entire collection.

        Returns:
            ndarray: Array of standard deviations.
        """
        return np.array([item.stddev() for item in self], dtype=float)

    def nearest_probability_distribution(self):
        """Nearest probability distribution over collection

        Returns:
            ProbCollection: Collection of ProbDistributions.
        """
        return ProbCollection([item.nearest_probability_distribution() for item in self])


class ProbCollection(list):
    """A list subclass that makes handling multiple probability-distributions easier.
    """
    def __init__(self, data):
        """ProbCollection constructor.

        Parameters:
            data (list or ProbCollection): List of ProbDistribution instances.

        Raises:
            TypeError: Must be list of ProbDistribution only.
        """
        for dd in data:
            if not isinstance(dd, ProbDistribution):
                raise TypeError('ProbCollection requires ProbDistribution instances.')
        super().__init__(data)

    @property
    def shots(self):
        """Number of shots taken over collection.

        Returns:
            ndarray: Array of shots values.
        """
        return np.array([item.shots for item in self], dtype=int)

    @property
    def mitigation_overhead(self):
        """Mitigation overhead over entire collection.

        Returns:
            ndarray: Array of mitigation overhead values.
        """
        return np.array([item.mitigation_overhead for item in self], dtype=float)

    def expval(self, exp_ops=''):
        """Expectation value over entire collection.

        Parameters:
            exp_ops (str or dict or list): Diagonal operators over which to compute expval.

        Returns:
            ndarray: Array of expectation values.

        Raises:
            M3Error: Length of passes operators does not match container length.
        """
        if isinstance(exp_ops, list):
            if len(exp_ops) != len(self):
                raise M3Error('exp_ops length does not match container length')
            out = []
            for idx, item in enumerate(self):
                out.append(item.expval(exp_ops[idx]))
            return np.array(out, dtype=float)
        return np.array([item.expval(exp_ops) for item in self], dtype=float)

    def expval_and_stddev(self, exp_ops=''):
        """Expectation value and standard deviation over entire collection.

        Parameters:
            exp_ops (str or dict or list): Diagonal operators over which to compute expval.

        Returns:
            list: Tuples of expval and stddev pairs.

        Raises:
            M3Error: Length of passes operators does not match container length.
        """
        if isinstance(exp_ops, list):
            if len(exp_ops) != len(self):
                raise M3Error('exp_ops length does not match container length')
            out = []
            for idx, item in enumerate(self):
                out.append(item.expval_and_stddev(exp_ops[idx]))
            return out
        return [item.expval_and_stddev(exp_ops) for item in self]

    def stddev(self):
        """Standard deviation over entire collection.

        Returns:
            ndarray: Array of standard deviations.
        """
        return np.array([item.stddev() for item in self], dtype=float)
