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
from numpy import array

from qiskit import Aer
from qiskit.algorithms.gibbs_state_preparation.default_ansatz_builder import (
    build_ansatz,
    build_init_ansatz_params_vals,
)
from qiskit.utils import QuantumInstance
from test.python.algorithms import QiskitAlgorithmsTestCase
from qiskit.circuit import Parameter
from qiskit.algorithms.gibbs_state_preparation.gibbs_state_sampler import GibbsStateSampler
from qiskit.opflow import Zero, X, SummedOp, Z, I


@ddt
class TestGibbsStateSampler(QiskitAlgorithmsTestCase):
    """Tests GibbsState class."""

    def test_gibbs_state_init(self):
        """Initialization test."""
        gibbs_state_function = Zero
        hamiltonian = X
        temperature = 42

        gibbs_state = GibbsStateSampler(gibbs_state_function, hamiltonian, temperature)

        np.testing.assert_equal(gibbs_state.hamiltonian, X)
        np.testing.assert_equal(gibbs_state.temperature, 42)

    def test_sample(self):
        """Tests if Gibbs state probabilities are sampled correctly.."""
        gibbs_state_function = Zero
        hamiltonian = SummedOp([0.3 * Z ^ Z ^ I ^ I, 0.2 * Z ^ I ^ I ^ I, 0.5 * I ^ Z ^ I ^ I])
        temperature = 42

        depth = 1
        num_qubits = 4

        ansatz = build_ansatz(num_qubits, depth)
        param_values_init = build_init_ansatz_params_vals(num_qubits, depth)

        params_dict = dict(zip(ansatz.ordered_parameters, param_values_init))
        gibbs_state = GibbsStateSampler(
            gibbs_state_function, hamiltonian, temperature, ansatz, params_dict
        )

        backend = Aer.get_backend("qasm_simulator")
        seed = 170
        qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)
        probs = gibbs_state.sample(qi)
        expected_probs = [0.222656, 0.25293, 0.25293, 0.271484]
        np.testing.assert_array_almost_equal(probs, expected_probs)

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
        gibbs_state = GibbsStateSampler(
            gibbs_state_function, hamiltonian, temperature, ansatz, params_dict
        )

        gradient_method = "param_shift"
        backend = Aer.get_backend("qasm_simulator")
        seed = 170
        qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)
        gradients = gibbs_state.calc_ansatz_gradients(qi, gradient_method)

        expected_gradients = [
            [6.37266600e-02, 5.65455000e-02, 6.12852600e-02, 6.87525300e-02],
            [5.60820100e-02, 6.67192900e-02, 5.84185100e-02, 6.92656000e-02],
            [1.74760818e-04, 1.38282776e-04, 4.88758087e-05, 3.81469727e-05],
            [4.18186188e-04, 3.09944153e-05, 4.41074371e-05, 1.07288361e-05],
            [2.92060000e-04, 3.08990000e-04, 1.15390000e-04, 1.05140000e-04],
            [1.93119049e-05, 1.93119049e-05, 9.53674316e-07, 9.53674316e-05],
            [4.02927399e-05, 5.96046448e-06, 2.59637833e-04, 4.00781631e-04],
            [3.81469727e-06, 4.00781631e-04, 3.08990479e-04, 1.93119049e-05],
            [6.00824400e-02, 6.15272500e-02, 6.49652500e-02, 6.34803800e-02],
            [7.15980500e-02, 5.40199300e-02, 6.15272500e-02, 6.34803800e-02],
            [2.94208527e-04, 1.60694122e-04, 1.83343887e-04, 1.54972076e-05],
            [8.10623169e-06, 3.45706940e-05, 2.20537186e-04, 8.05854797e-05],
            [2.88486481e-05, 6.89029694e-05, 1.73807144e-04, 2.38418579e-07],
            [3.81469727e-06, 3.43322754e-05, 2.14576721e-04, 4.67300415e-05],
            [2.14576721e-06, 5.96046448e-06, 7.47680664e-04, 6.95228577e-04],
            [2.14576721e-04, 2.88486481e-05, 2.14576721e-06, 1.15394592e-04],
        ]
        for ind, gradient in enumerate(expected_gradients):
            np.testing.assert_array_almost_equal(gradients[ind], gradient)

    def test_calc_ansatz_gradients_missing_ansatz(self):
        """Tests if an expected error is raised when an ansatz is missing when calculating
        ansatz gradients."""
        gibbs_state_function = Zero
        hamiltonian = SummedOp([0.3 * Z ^ Z ^ I ^ I, 0.2 * Z ^ I ^ I ^ I, 0.5 * I ^ Z ^ I ^ I])
        temperature = 42
        param_values_init = np.zeros(2)

        params_dict = dict(zip([Parameter("a"), Parameter("b")], param_values_init))
        gibbs_state = GibbsStateSampler(
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

        gibbs_state = GibbsStateSampler(
            gibbs_state_function,
            hamiltonian,
            temperature,
            ansatz,
            params_dict,
            hamiltonian_gradients,
        )

        gradient_method = "param_shift"
        backend = Aer.get_backend("qasm_simulator")
        seed = 170
        qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)
        final_gradients = gibbs_state.calc_hamiltonian_gradients(qi, gradient_method)

        expected_gradients = array([0.10587743, 0.10234075, 0.10410559, 0.11696377])

        np.testing.assert_almost_equal(final_gradients[param], expected_gradients)
        np.testing.assert_equal(len(final_gradients), 1)


if __name__ == "__main__":
    unittest.main()
