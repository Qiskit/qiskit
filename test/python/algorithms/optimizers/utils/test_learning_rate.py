# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for LearningRate."""

from test.python.algorithms import QiskitAlgorithmsTestCase
import numpy as np
from qiskit.algorithms.optimizers.optimizer_utils import LearningRate


class TestLearningRate(QiskitAlgorithmsTestCase):
    """Tests for the LearningRate class."""

    def setUp(self):
        super().setUp()
        np.random.seed(12)
        self.initial_point = np.array([1, 1, 1, 1, 0])

    def objective(self, x):
        """Objective Function for the tests"""
        return (np.linalg.norm(x) - 1) ** 2

    def test_learning_rate(self):
        """
        Tests if the learning rate is initialized properly for each kind of input:
        float, list and iterator.
        """
        constant_learning_rate_input = 0.01
        list_learning_rate_input = [0.01 * n for n in range(10)]
        generator_learning_rate_input = lambda: (el for el in list_learning_rate_input)

        with self.subTest("Check constant learning rate."):
            constant_learning_rate = LearningRate(learning_rate=constant_learning_rate_input)
            for _ in range(5):
                self.assertEqual(constant_learning_rate_input, next(constant_learning_rate))

        with self.subTest("Check learning rate list."):
            list_learning_rate = LearningRate(learning_rate=list_learning_rate_input)
            for i in range(5):
                self.assertEqual(list_learning_rate_input[i], next(list_learning_rate))

        with self.subTest("Check learning rate generator."):
            generator_learning_rate = LearningRate(generator_learning_rate_input)
            for i in range(5):
                self.assertEqual(list_learning_rate_input[i], next(generator_learning_rate))
