# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test AerPauliExpectation """

import itertools
import unittest
from test.aqua import QiskitAquaTestCase

import numpy as np

from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import (X, Y, Z, I, CX, H, S,
                                   ListOp, Zero, One, Plus, Minus, StateFn,
                                   AerPauliExpectation, CircuitSampler, CircuitStateFn,
                                   PauliExpectation)
from qiskit.circuit.library import RealAmplitudes


class TestAerPauliExpectation(QiskitAquaTestCase):
    """Pauli Change of Basis Expectation tests."""

    def setUp(self) -> None:
        super().setUp()
        try:
            from qiskit import Aer

            self.seed = 97
            self.backend = Aer.get_backend('qasm_simulator')
            q_instance = QuantumInstance(self.backend, seed_simulator=self.seed,
                                         seed_transpiler=self.seed)
            self.sampler = CircuitSampler(q_instance, attach_results=True)
            self.expect = AerPauliExpectation()
        except Exception as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return

    def test_pauli_expect_pair(self):
        """ pauli expect pair test """
        op = (Z ^ Z)
        # wvf = (Pl^Pl) + (Ze^Ze)
        wvf = CX @ (H ^ I) @ Zero

        converted_meas = self.expect.convert(~StateFn(op) @ wvf)
        sampled = self.sampler.convert(converted_meas)
        self.assertAlmostEqual(sampled.eval(), 0, delta=.1)

    def test_pauli_expect_single(self):
        """ pauli expect single test """
        # TODO bug in Aer with Y measurements
        # paulis = [Z, X, Y, I]
        paulis = [Z, X, I]
        states = [Zero, One, Plus, Minus, S @ Plus, S @ Minus]
        for pauli, state in itertools.product(paulis, states):
            converted_meas = self.expect.convert(~StateFn(pauli) @ state)
            matmulmean = state.adjoint().to_matrix() @ pauli.to_matrix() @ state.to_matrix()
            sampled = self.sampler.convert(converted_meas)
            self.assertAlmostEqual(sampled.eval(), matmulmean, delta=.1)

    def test_pauli_expect_op_vector(self):
        """ pauli expect op vector test """
        paulis_op = ListOp([X, Y, Z, I])
        converted_meas = self.expect.convert(~StateFn(paulis_op))

        plus_mean = (converted_meas @ Plus)
        sampled_plus = self.sampler.convert(plus_mean)
        np.testing.assert_array_almost_equal(sampled_plus.eval(), [1, 0, 0, 1], decimal=1)

        minus_mean = (converted_meas @ Minus)
        sampled_minus = self.sampler.convert(minus_mean)
        np.testing.assert_array_almost_equal(sampled_minus.eval(), [-1, 0, 0, 1], decimal=1)

        zero_mean = (converted_meas @ Zero)
        sampled_zero = self.sampler.convert(zero_mean)
        np.testing.assert_array_almost_equal(sampled_zero.eval(), [0, 0, 1, 1], decimal=1)

        sum_zero = (Plus + Minus) * (.5 ** .5)
        sum_zero_mean = (converted_meas @ sum_zero)
        sampled_zero_mean = self.sampler.convert(sum_zero_mean)
        # !!NOTE!!: Depolarizing channel (Sampling) means interference
        # does not happen between circuits in sum, so expectation does
        # not equal expectation for Zero!!
        np.testing.assert_array_almost_equal(sampled_zero_mean.eval(), [0, 0, 0, 2], decimal=1)

    def test_pauli_expect_state_vector(self):
        """ pauli expect state vector test """
        states_op = ListOp([One, Zero, Plus, Minus])

        paulis_op = X
        converted_meas = self.expect.convert(~StateFn(paulis_op) @ states_op)
        sampled = self.sampler.convert(converted_meas)

        # Small test to see if execution results are accessible
        for composed_op in sampled:
            self.assertIn('counts', composed_op[0].execution_results)

        np.testing.assert_array_almost_equal(sampled.eval(), [0, 0, 1, -1], decimal=1)

    def test_pauli_expect_op_vector_state_vector(self):
        """ pauli expect op vector state vector test """
        # TODO Bug in Aer with Y Measurements!!
        # paulis_op = ListOp([X, Y, Z, I])
        paulis_op = ListOp([X, Z, I])
        states_op = ListOp([One, Zero, Plus, Minus])

        valids = [[+0, 0, 1, -1],
                  # [+0, 0, 0, 0],
                  [-1, 1, 0, -0],
                  [+1, 1, 1, 1]]
        converted_meas = self.expect.convert(~StateFn(paulis_op) @ states_op)
        sampled = self.sampler.convert(converted_meas)
        np.testing.assert_array_almost_equal(sampled.eval(), valids, decimal=1)

    def test_multi_representation_ops(self):
        """ Test observables with mixed representations """
        mixed_ops = ListOp([X.to_matrix_op(),
                            H,
                            H + I,
                            X])
        converted_meas = self.expect.convert(~StateFn(mixed_ops))

        plus_mean = (converted_meas @ Plus)
        sampled_plus = self.sampler.convert(plus_mean)
        np.testing.assert_array_almost_equal(sampled_plus.eval(),
                                             [1, .5 ** .5, (1 + .5 ** .5), 1],
                                             decimal=1)

    def test_parameterized_qobj(self):
        """ grouped pauli expectation test """
        two_qubit_h2 = (-1.052373245772859 * I ^ I) + \
                       (0.39793742484318045 * I ^ Z) + \
                       (-0.39793742484318045 * Z ^ I) + \
                       (-0.01128010425623538 * Z ^ Z) + \
                       (0.18093119978423156 * X ^ X)

        aer_sampler = CircuitSampler(self.sampler.quantum_instance,
                                     param_qobj=True,
                                     attach_results=True)

        var_form = RealAmplitudes()
        var_form.num_qubits = 2

        observable_meas = self.expect.convert(StateFn(two_qubit_h2, is_measurement=True))
        ansatz_circuit_op = CircuitStateFn(var_form)
        expect_op = observable_meas.compose(ansatz_circuit_op).reduce()

        def generate_parameters(num):
            param_bindings = {}
            for param in var_form.parameters:
                values = []
                for _ in range(num):
                    values.append(np.random.rand())
                param_bindings[param] = values
            return param_bindings

        def validate_sampler(ideal, sut, param_bindings):
            expect_sampled = ideal.convert(expect_op, params=param_bindings).eval()
            actual_sampled = sut.convert(expect_op, params=param_bindings).eval()
            self.assertAlmostEqual(actual_sampled, expect_sampled, delta=.1)

        def get_circuit_templates(sampler):
            return sampler._transpiled_circ_templates

        def validate_aer_binding_used(templates):
            self.assertIsNotNone(templates)

        def validate_aer_templates_reused(prev_templates, cur_templates):
            self.assertIs(prev_templates, cur_templates)

        validate_sampler(self.sampler, aer_sampler, generate_parameters(1))
        cur_templates = get_circuit_templates(aer_sampler)

        validate_aer_binding_used(cur_templates)

        prev_templates = cur_templates
        validate_sampler(self.sampler, aer_sampler, generate_parameters(2))
        cur_templates = get_circuit_templates(aer_sampler)

        validate_aer_templates_reused(prev_templates, cur_templates)

        prev_templates = cur_templates
        validate_sampler(self.sampler, aer_sampler, generate_parameters(2))  # same num of params
        cur_templates = get_circuit_templates(aer_sampler)

        validate_aer_templates_reused(prev_templates, cur_templates)

    def test_pauli_expectation_param_qobj(self):
        """ Test PauliExpectation with param_qobj """
        q_instance = QuantumInstance(self.backend, seed_simulator=self.seed,
                                     seed_transpiler=self.seed, shots=20000)
        qubit_op = (1 * I ^ I) + (2 * I ^ Z) + (3 * Z ^ I) + (4 * Z ^ Z) + (5 * X ^ X)
        var_form = RealAmplitudes(qubit_op.num_qubits)
        ansatz_circuit_op = CircuitStateFn(var_form)
        observable = PauliExpectation().convert(~StateFn(qubit_op))
        expect_op = observable.compose(ansatz_circuit_op).reduce()
        params1 = {}
        params2 = {}
        for param in var_form.parameters:
            params1[param] = [0]
            params2[param] = [0, 0]

        sampler1 = CircuitSampler(backend=q_instance, param_qobj=False)
        samples1 = sampler1.convert(expect_op, params=params1)
        val1 = np.real(samples1.eval())[0]
        samples2 = sampler1.convert(expect_op, params=params2)
        val2 = np.real(samples2.eval())
        sampler2 = CircuitSampler(backend=q_instance, param_qobj=True)
        samples3 = sampler2.convert(expect_op, params=params1)
        val3 = np.real(samples3.eval())
        samples4 = sampler2.convert(expect_op, params=params2)
        val4 = np.real(samples4.eval())

        np.testing.assert_array_almost_equal([val1] * 2, val2, decimal=2)
        np.testing.assert_array_almost_equal(val1, val3, decimal=2)
        np.testing.assert_array_almost_equal([val1] * 2, val4, decimal=2)


if __name__ == '__main__':
    unittest.main()
