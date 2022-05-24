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

"""Tests for the UMDA optimizer."""

from test.python.algorithms import QiskitAlgorithmsTestCase
from ddt import ddt

import numpy as np
from qiskit.algorithms.optimizers.umda import UMDA


@ddt
class TestUMDA(QiskitAlgorithmsTestCase):
    """Tests for the UMDA optimizer."""

    def test_truncation(self):
        """Test if truncation is performed as expected"""
        umda = UMDA(maxiter=100, size_gen=20, n_variables=10)

        def objective_function(array):
            return sum(array)

        umda._check_generation(objective_function)
        umda._truncation()

        assert len(umda.evaluations) == int(umda.size_gen * umda.alpha)
        assert umda.generation.shape == (int(umda.size_gen * umda.alpha), umda.n_variables)

    def test_check_generation(self):
        """Test if the solutions are being evaluated as expected"""
        umda = UMDA(maxiter=100, size_gen=20, n_variables=10)

        def objective_function(array):
            return sum(array)

        umda._check_generation(objective_function)
        sols_by_umda = umda.evaluations

        # check by hand
        sols_by_hand = []
        for individual in umda.generation:
            sols_by_hand.append(objective_function(individual))

        assert np.all(sols_by_umda == sols_by_hand)

    def test_initialization(self):
        """Test if the vector of statistics used during runtime is initialized as expected"""
        umda = UMDA(maxiter=100, size_gen=20, n_variables=10)

        assert np.all(umda.vector[0, :] == np.pi)  # mu
        assert np.all(umda.vector[1, :] == 0.5)  # std
        assert np.all(umda.vector[2, :] == 0)  # min
        assert np.all(umda.vector[3, :] == np.pi * 2)  # max

    def test_truncation_sort(self):
        """Test if truncation is performed as expected and solutions are sorted correctly"""
        umda = UMDA(maxiter=100, size_gen=20, n_variables=10)

        def objective_function(array):
            return sum(array)

        umda._check_generation(objective_function)
        umda._truncation()

        for i in range(1, len(umda.evaluations)):
            assert umda.evaluations[i] >= umda.evaluations[i - 1]

    def test_update_vector(self):
        """Test if the vector of statistics is being updated correctly during runtime
        given the generation."""
        umda = UMDA(maxiter=100, size_gen=20, n_variables=4)

        generation = np.array([[1, 1, 1, 1], [2, 3, 2, 3], [4, 5, 1, 4], [2, 2, 2, 2]])

        umda.generation = generation
        umda._update_vector()

        expected_solution = np.zeros((4, 4))

        for i in range(4):
            expected_solution[0, i] = np.mean(generation[:, i])
            expected_solution[1, i] = np.std(generation[:, i])

        expected_solution[2, :] = 0
        expected_solution[3, :] = np.pi * 2

        assert np.all(umda.vector == expected_solution)
