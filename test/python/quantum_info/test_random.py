# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test random unitary generation utility."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from qiskit.quantum_info.random import random_unitary, random_density_matrix
from qiskit.test import QiskitTestCase


class TestRandomUtils(QiskitTestCase):
    """Testing qiskit.quantum_info.random.utils"""

    def test_seeded_random_unitary(self):
        """Given the same seed, the output should be expected.
        Following the numpy approach we are going to try to keep the policy of platform-independent
        with randomness.
        https://stackoverflow.com/questions/40676205/cross-platform-numpy-random-seed"""
        seed = 314159
        unitary = random_unitary(4, seed=seed)
        expected = np.array([[0.07749745 + 0.65152329j, 0.40784587 - 0.36593286j,
                              0.03129288 + 0.2530866j, 0.36101855 - 0.27184549j],
                             [0.53016612 - 0.01919937j, 0.48298729 + 0.19460984j,
                              -0.40381565 - 0.53289147j, -0.00508656 + 0.01841975j],
                             [-0.32716712 - 0.11010826j, -0.29569725 - 0.11013661j,
                              -0.23722636 - 0.47383058j, 0.40993924 - 0.57656653j],
                             [0.04033726 - 0.4089958j, 0.39452908 + 0.4163953j,
                              0.37036483 + 0.26451022j, 0.08430501 - 0.53648299j]])
        assert_allclose(unitary.data, expected, atol=1e-8)

    def test_unitary(self):
        """ Test that a random unitary with set seed will not affect later
        results
        """
        seed = 314159
        test_cases = 100
        random_unitary(4, seed=seed)
        rng_before = [np.random.randint(1000) for _ in range(test_cases)]
        random_unitary(4, seed=seed)
        rng_after = [np.random.randint(1000) for _ in range(test_cases)]
        array_equality = all([rng_before[i] == rng_after[i] for i in range(test_cases)])
        self.assertFalse(array_equality)

    def test_density_matrix(self):
        """ Test that a random state with set seed will not affect later
        results.
        """
        seed = 314159
        test_cases = 100
        random_density_matrix(4, seed=seed)
        rng_before = [np.random.randint(1000) for _ in range(test_cases)]
        random_density_matrix(4, seed=seed)
        rng_after = [np.random.randint(1000) for _ in range(test_cases)]
        array_equality = all([rng_before[i] == rng_after[i] for i in range(test_cases)])
        self.assertFalse(array_equality)


if __name__ == '__main__':
    unittest.main()
