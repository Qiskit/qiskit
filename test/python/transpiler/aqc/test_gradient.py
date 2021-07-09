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
"""
Tests analytical gradient vs the one computed via finite differences.
"""

import unittest

import numpy as np
from scipy.stats import unitary_group

from qiskit.test import QiskitTestCase
from qiskit.transpiler.synthesis.aqc.parametric_circuit import ParametricCircuit


class TestGradientAgainstFiniteDiff(QiskitTestCase):
    """
    Compares analytical gradient vs the one computed via finite difference
    approximation. Also, the test demonstrates that the difference between
    analytical and numerical gradients is up to quadratic term in Taylor
    expansion for small deltas.
    """

    def test_gradient(self):
        """
        Gradient test for specified number of qubits and circuit depth.
        """
        num_qubits = 3
        num_cnots = 14

        circuit = ParametricCircuit(
            num_qubits=num_qubits, layout="spin", connectivity="full", depth=num_cnots
        )

        # Generate random target matrix and random starting point. Repeat until
        # sufficiently large gradient has been encountered.
        # This operation is pretty fast.
        while True:
            target_matrix = self._random_special_unitary(num_qubits=num_qubits)
            thetas = np.random.rand(circuit.num_thetas) * (2.0 * np.pi)
            circuit.set_thetas(thetas)
            fobj0, grad0 = circuit.get_gradient(target_matrix=target_matrix)
            if np.linalg.norm(grad0) > 1e-2:
                break

        grad0_dir = grad0 / np.linalg.norm(grad0)
        numerical_grad = np.zeros(thetas.size)
        thetas_delta = np.zeros(thetas.size)

        # Every angle has a magnitude between 0 and 2*pi. We choose the
        # angle increment (tau) about the same order of magnitude, tau <= 1,
        # and then gradually decrease it towards zero.
        tau = 1.0
        diff_prev = 0.0
        orders = []
        errors = []
        steps = 8
        for step in range(steps):
            # Estimate gradient approximation error.
            for i in range(thetas.size):
                np.copyto(thetas_delta, thetas)
                thetas_delta[i] -= tau
                circuit.set_thetas(thetas_delta)
                fobj1, _ = circuit.get_gradient(target_matrix=target_matrix)
                np.copyto(thetas_delta, thetas)
                thetas_delta[i] += tau
                circuit.set_thetas(thetas_delta)
                fobj2, _ = circuit.get_gradient(target_matrix=target_matrix)
                numerical_grad[i] = (fobj2 - fobj1) / (2.0 * tau)
            errors.append(np.linalg.norm(grad0 - numerical_grad) / np.linalg.norm(grad0))

            # Estimate approximation order (should be quadratic for small tau).
            # Note, we take perturbation in gradient direction. More rigorous
            # approach would take a random direction, although quadratic
            # convergence is less pronounced in this case.
            perturbation = grad0_dir * tau
            circuit.set_thetas(thetas + perturbation)
            fobj, _ = circuit.get_gradient(target_matrix=target_matrix)
            diff = abs(fobj - fobj0 - np.dot(grad0, perturbation))
            orders.append(
                0.0 if step == 0 else float((np.log(diff_prev) - np.log(diff)) / np.log(2.0))
            )

            tau /= 2.0
            diff_prev = diff

        # check errors
        prev_error = errors[0]
        for error in errors[1:]:
            self.assertLess(error, prev_error)
            prev_error = error

        # check orders, skipping first zero
        orders = np.asarray(orders[1:0])
        # pylint:disable=misplaced-comparison-constant
        self.assertTrue(np.all(2 <= orders))
        self.assertTrue(np.all(orders < 3))

    def _random_special_unitary(self, num_qubits: int) -> np.ndarray:
        """
        Generates a random SU matrix.

        Args:
            num_qubits: number of qubits.

        Returns:
            random SU matrix of size 2^n x 2^n.
        """
        d = int(2 ** num_qubits)
        unitary = unitary_group.rvs(d)
        unitary = unitary / (np.linalg.det(unitary) ** (1.0 / float(d)))
        return unitary


if __name__ == "__main__":
    unittest.main()
