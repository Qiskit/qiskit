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
"""Tests GibbsState class."""
import unittest

from ddt import ddt
import numpy as np

from qiskit.algorithms.gibbs_state_preparation.default_ansatz_builder import (
    build_ansatz,
    build_init_ansatz_params_vals,
)
from test.python.algorithms import QiskitAlgorithmsTestCase
from qiskit.circuit import Parameter
from qiskit.algorithms.gibbs_state_preparation.gibbs_state import GibbsState
from qiskit.opflow import Zero, X, SummedOp, Z, I


@ddt
class TestGibbsState(QiskitAlgorithmsTestCase):
    """Tests GibbsState class."""

    def test_gibbs_state_init(self):
        """Initialization test."""
        gibbs_state_function = Zero
        hamiltonian = X
        temperature = 42

        gibbs_state = GibbsState(gibbs_state_function, hamiltonian, temperature)

        np.testing.assert_equal(gibbs_state.gibbs_state_function, Zero)
        np.testing.assert_equal(gibbs_state.hamiltonian, X)
        np.testing.assert_equal(gibbs_state.temperature, 42)

    def test_calc_ansatz_gradients(self):
        """Tests if ansatz gradients are calculated correctly."""
        gibbs_state_function = Zero
        hamiltonian = SummedOp([0.3 * Z ^ Z ^ I ^ I, 0.2 * Z ^ I ^ I ^ I, 0.5 * I ^ Z ^ I ^ I])
        temperature = 42

        depth = 1
        num_qubits = 4

        ansatz = build_ansatz(num_qubits, depth)
        param_values_init = build_init_ansatz_params_vals(num_qubits, depth)

        params_dict = dict(zip(ansatz.ordered_parameters, param_values_init))
        gibbs_state = GibbsState(
            gibbs_state_function, hamiltonian, temperature, ansatz, params_dict
        )

        gradient_method = "param_shift"
        gradients = gibbs_state.calc_ansatz_gradients(gradient_method)

        expected_gradients = [
            [
                (-0.25000000000000006 + 0j),
                0j,
                0j,
                0j,
                0j,
                (0.25000000000000006 + 0j),
                0j,
                0j,
                0j,
                0j,
                (-0.24999999999999994 + 0j),
                0j,
                0j,
                0j,
                0j,
                (0.24999999999999994 + 0j),
            ],
            [
                (0.24999999999999994 + 0j),
                0j,
                0j,
                0j,
                0j,
                (-0.24999999999999994 + 0j),
                0j,
                0j,
                0j,
                0j,
                (-0.24999999999999994 + 0j),
                0j,
                0j,
                0j,
                0j,
                (0.24999999999999994 + 0j),
            ],
            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
            [
                (-0.25000000000000006 + 0j),
                0j,
                0j,
                0j,
                0j,
                (0.25000000000000006 + 0j),
                0j,
                0j,
                0j,
                0j,
                (-0.24999999999999994 + 0j),
                0j,
                0j,
                0j,
                0j,
                (0.24999999999999994 + 0j),
            ],
            [
                (-0.25000000000000006 + 0j),
                0j,
                0j,
                0j,
                0j,
                (-0.24999999999999994 + 0j),
                0j,
                0j,
                0j,
                0j,
                (0.25000000000000006 + 0j),
                0j,
                0j,
                0j,
                0j,
                (0.24999999999999994 + 0j),
            ],
            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
            [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j],
        ]

        np.testing.assert_almost_equal(gradients, expected_gradients)

    def test_calc_ansatz_gradients_missing_ansatz(self):
        """Tests if an expected error is raised when an ansatz is missing when calculating
        ansatz gradients."""
        gibbs_state_function = Zero
        hamiltonian = SummedOp([0.3 * Z ^ Z ^ I ^ I, 0.2 * Z ^ I ^ I ^ I, 0.5 * I ^ Z ^ I ^ I])
        temperature = 42
        param_values_init = np.zeros(2)

        params_dict = dict(zip([Parameter("a"), Parameter("b")], param_values_init))
        gibbs_state = GibbsState(
            gibbs_state_function, hamiltonian, temperature, ansatz_params_dict=params_dict
        )

        gradient_method = "param_shift"
        np.testing.assert_raises(
            ValueError,
            gibbs_state.calc_ansatz_gradients,
            gradient_method,
        )

    def test_calc_hamiltonian_gradients(self):
        """Tests if hamiltonian gradients are calculated correctly."""
        gibbs_state_function = Zero
        param = Parameter("w")
        hamiltonian = SummedOp([param * Z ^ Z ^ I ^ I, 0.2 * Z ^ I ^ I ^ I, 0.5 * I ^ Z ^ I ^ I])
        temperature = 42

        depth = 1
        num_qubits = 4

        ansatz = build_ansatz(num_qubits, depth)
        param_values_init = build_init_ansatz_params_vals(num_qubits, depth)

        params_dict = dict(zip(ansatz.ordered_parameters, param_values_init))

        hamiltonian_gradients = {param: [1.0 / (i + 1) for i in range(len(ansatz.parameters))]}
        # TODO np array
        gibbs_state = GibbsState(
            gibbs_state_function,
            hamiltonian,
            temperature,
            ansatz,
            params_dict,
            hamiltonian_gradients,
        )

        gradient_method = "param_shift"
        final_gradients = gibbs_state.calc_hamiltonian_gradients(gradient_method)

        expected_gradients = [
            -0.21543561 + 0.0j,
            0.20123106 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            -0.21543561 + 0.0j,
            -0.25331439 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
            0.0 + 0.0j,
        ]

        np.testing.assert_almost_equal(final_gradients[param], expected_gradients)
        np.testing.assert_equal(len(final_gradients), 1)


if __name__ == "__main__":
    unittest.main()
