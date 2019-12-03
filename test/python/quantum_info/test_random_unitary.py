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

from qiskit.quantum_info.random import random_unitary
from qiskit.test import QiskitTestCase

class TestRandomUtils(QiskitTestCase):
    """Testing qiskit.quantum_info.random.utils"""

    def test_random_unitary(self):
        """ Test that a random circuit with set seed will not affect later
        results.
        """
        SEED = 314159
        TEST_CASES = 100
        random_unitary(4, seed=SEED)
        rng_before = [np.random.randint(1000) for _ in range(TEST_CASES) ]
        random_unitary(4, seed=SEED)
        rng_after = [np.random.randint(1000) for _ in range(TEST_CASES) ]
        array_equality = all( [ rng_before[i] == rng_after[i] for i in range(TEST_CASES) ])
        self.assertFalse(array_equality)

if __name__ == '__main__':
    unittest.main()
