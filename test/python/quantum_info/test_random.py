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

from qiskit.quantum_info.random import random_unitary, random_density_matrix
from qiskit.test import QiskitTestCase


class TestRandomUtils(QiskitTestCase):
    """Testing qiskit.quantum_info.random.utils"""

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
