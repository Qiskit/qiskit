# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test library of probability distribution circuits."""

import unittest
from ddt import ddt, data, unpack

import numpy as np
from scipy.stats import multivariate_normal

from qiskit.test.base import QiskitTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UniformDistribution, NormalDistribution, LogNormalDistribution
from qiskit.quantum_info import Statevector


class TestUniformDistribution(QiskitTestCase):
    """Test the uniform distribution circuit."""

    def test_uniform(self):
        """Test the circuit of the uniform distribution is a simple layer of Hadamards."""
        circuit = UniformDistribution(3)
        expected = QuantumCircuit(3)
        expected.h([0, 1, 2])

        self.assertEqual(circuit.decompose(), expected)


@ddt
class TestNormalDistribution(QiskitTestCase):
    """Test the normal distribution circuit."""

    def assertDistributionIsCorrect(self, circuit, num_qubits, mu, sigma, bounds, upto_diag):
        """Assert that ``circuit`` implements the normal distribution correctly.

        This test asserts that the ``circuit`` produces the desired state-vector.
        """
        if not isinstance(num_qubits, (list, np.ndarray)):
            num_qubits = [num_qubits]
        if not isinstance(mu, (list, np.ndarray)):
            mu = [mu]
        if not isinstance(sigma, (list, np.ndarray)):
            sigma = [[sigma]]
        # bit differently to cover the case the users might pass `bounds` as a single list,
        # e.g. [0, 1], instead of a tuple
        if not isinstance(bounds[0], tuple):
            bounds = [bounds]

        # compute the points
        meshgrid = np.meshgrid(
            *(
                np.linspace(bound[0], bound[1], num=2 ** num_qubits[i])
                for i, bound in enumerate(bounds)
            ),
            indexing="ij",
        )
        x = list(zip(*(grid.flatten() for grid in meshgrid)))

        # compute the normalized, truncated probabilities
        probabilities = multivariate_normal.pdf(x, mu, sigma)
        normalized_probabilities = probabilities / np.sum(probabilities)
        expected = np.sqrt(normalized_probabilities)

        # compare to actual statevector from circuit
        actual = Statevector.from_instruction(circuit)
        if upto_diag:
            self.assertTrue(actual.equiv(expected))
        else:
            np.testing.assert_array_almost_equal(expected, actual.data)

    @data(
        [2, None, None, None, False],
        [3, 1.75, 2.5, None, True],
        [2, 1.75, 2.5, (0, 3), False],
        [[1, 2, 2], None, None, None, True],
        [
            [1, 2, 1],
            [0, 1, 1],
            [[1.2, 0, 0], [0, 0.5, 0], [0, 0, 0.1]],
            [(0, 2), (-1, 1), (-3, 3)],
            False,
        ],
    )
    @unpack
    def test_normal(self, num_qubits, mu, sigma, bounds, upto_diag):
        """Test the statevector produced by ``NormalDistribution`` and the default arguments."""

        # construct default values and kwargs dictionary to call the constructor of
        # NormalDistribution. The kwargs dictionary is used to not pass any arguments which are
        # None to test the default values of the class.
        kwargs = {"num_qubits": num_qubits, "upto_diag": upto_diag}

        if mu is None:
            mu = np.zeros(len(num_qubits)) if isinstance(num_qubits, list) else 0
        else:
            kwargs["mu"] = mu

        if sigma is None:
            sigma = np.eye(len(num_qubits)).tolist() if isinstance(num_qubits, list) else 1
        else:
            kwargs["sigma"] = sigma

        if bounds is None:
            bounds = [(-1, 1)] * (len(num_qubits) if isinstance(num_qubits, list) else 1)
        else:
            kwargs["bounds"] = bounds

        normal = NormalDistribution(**kwargs)
        self.assertDistributionIsCorrect(normal, num_qubits, mu, sigma, bounds, upto_diag)

    @data(
        [2, [1, 1], 2, (0, 1)],  # invalid mu
        [2, 1.2, [[1, 0], [0, 1]], (0, 1)],  # invalid sigma
        [2, 1.2, 1, [(0, 1), (0, 1)]],  # invalid bounds
        [[1, 2], 1, [[1, 0], [0, 1]], [(0, 1), (0, 1)]],  # invalid mu
        [[1, 2], [0, 0], [[2]], [(0, 1), (0, 1)]],  # invalid sigma
        [[1, 2], [0, 0], [[1, 0], [0, 1]], [0, 1]],  # invalid bounds
    )
    @unpack
    def test_mismatching_dimensions(self, num_qubits, mu, sigma, bounds):
        """Test passing mismatching dimensions raises an error."""

        with self.assertRaises(ValueError):
            _ = NormalDistribution(num_qubits, mu, sigma, bounds)

    @data([(0, 0), (0, 1)], [(-2, -1), (1, 0)])
    def test_bounds_invalid(self, bounds):
        """Test passing invalid bounds raises."""

        with self.assertRaises(ValueError):
            _ = NormalDistribution([1, 1], [0, 0], [[1, 0], [0, 1]], bounds)


@ddt
class TestLogNormalDistribution(QiskitTestCase):
    """Test the normal distribution circuit."""

    def assertDistributionIsCorrect(self, circuit, num_qubits, mu, sigma, bounds, upto_diag):
        """Assert that ``circuit`` implements the normal distribution correctly.

        This test asserts that the ``circuit`` produces the desired state-vector.
        """
        if not isinstance(num_qubits, (list, np.ndarray)):
            num_qubits = [num_qubits]
        if not isinstance(mu, (list, np.ndarray)):
            mu = [mu]
        if not isinstance(sigma, (list, np.ndarray)):
            sigma = [[sigma]]
        # bit differently to cover the case the users might pass `bounds` as a single list,
        # e.g. [0, 1], instead of a tuple
        if not isinstance(bounds[0], tuple):
            bounds = [bounds]

        # compute the points
        meshgrid = np.meshgrid(
            *(
                np.linspace(bound[0], bound[1], num=2 ** num_qubits[i])
                for i, bound in enumerate(bounds)
            ),
            indexing="ij",
        )
        x = list(zip(*(grid.flatten() for grid in meshgrid)))

        # compute the normalized, truncated probabilities
        probabilities = []
        for x_i in x:
            if np.min(x_i) > 0:
                det = 1 / np.prod(x_i)
                probabilities += [multivariate_normal.pdf(np.log(x_i), mu, sigma) * det]
            else:
                probabilities += [0]
        normalized_probabilities = probabilities / np.sum(probabilities)
        expected = np.sqrt(normalized_probabilities)

        # compare to actual statevector from circuit
        actual = Statevector.from_instruction(circuit)
        if upto_diag:
            self.assertTrue(actual.equiv(expected))
        else:
            np.testing.assert_array_almost_equal(expected, actual.data)

    @data(
        [2, None, None, None, False],
        [3, 1.75, 2.5, None, True],
        [2, 1.75, 2.5, (0, 3), False],
        [[1, 2, 2], None, None, None, True],
        [
            [1, 2, 1],
            [0, 1, 1],
            [[1.2, 0, 0], [0, 0.5, 0], [0, 0, 0.1]],
            [(0, 2), (-1, 1), (-3, 3)],
            False,
        ],
    )
    @unpack
    def test_lognormal(self, num_qubits, mu, sigma, bounds, upto_diag):
        """Test the statevector produced by ``LogNormalDistribution`` and the default arguments."""

        # construct default values and kwargs dictionary to call the constructor of
        # NormalDistribution. The kwargs dictionary is used to not pass any arguments which are
        # None to test the default values of the class.
        kwargs = {"num_qubits": num_qubits, "upto_diag": upto_diag}

        if mu is None:
            mu = np.zeros(len(num_qubits)) if isinstance(num_qubits, list) else 0
        else:
            kwargs["mu"] = mu

        if sigma is None:
            sigma = np.eye(len(num_qubits)).tolist() if isinstance(num_qubits, list) else 1
        else:
            kwargs["sigma"] = sigma

        if bounds is None:
            bounds = [(0, 1)] * (len(num_qubits) if isinstance(num_qubits, list) else 1)
        else:
            kwargs["bounds"] = bounds

        normal = LogNormalDistribution(**kwargs)
        self.assertDistributionIsCorrect(normal, num_qubits, mu, sigma, bounds, upto_diag)


if __name__ == "__main__":
    unittest.main()
