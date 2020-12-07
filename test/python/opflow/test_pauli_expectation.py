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

""" Test PauliExpectation """

import unittest
from test.aqua import QiskitAquaTestCase

import itertools
import numpy as np

from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.operators import (X, Y, Z, I, CX, H, S,
                                   ListOp, Zero, One, Plus, Minus, StateFn,
                                   PauliExpectation, CircuitSampler)

from qiskit import BasicAer


# pylint: disable=invalid-name


class TestPauliExpectation(QiskitAquaTestCase):
    """Pauli Change of Basis Expectation tests."""

    def setUp(self) -> None:
        super().setUp()
        self.seed = 97
        backend = BasicAer.get_backend('qasm_simulator')
        q_instance = QuantumInstance(backend, seed_simulator=self.seed, seed_transpiler=self.seed)
        self.sampler = CircuitSampler(q_instance, attach_results=True)
        self.expect = PauliExpectation()

    def test_pauli_expect_pair(self):
        """ pauli expect pair test """
        op = (Z ^ Z)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = CX @ (H ^ I) @ Zero

        converted_meas = self.expect.convert(~StateFn(op) @ wf)
        self.assertAlmostEqual(converted_meas.eval(), 0, delta=.1)
        sampled = self.sampler.convert(converted_meas)
        self.assertAlmostEqual(sampled.eval(), 0, delta=.1)

    def test_pauli_expect_single(self):
        """ pauli expect single test """
        paulis = [Z, X, Y, I]
        states = [Zero, One, Plus, Minus, S @ Plus, S @ Minus]
        for pauli, state in itertools.product(paulis, states):
            converted_meas = self.expect.convert(~StateFn(pauli) @ state)
            matmulmean = state.adjoint().to_matrix() @ pauli.to_matrix() @ state.to_matrix()
            self.assertAlmostEqual(converted_meas.eval(), matmulmean, delta=.1)

            sampled = self.sampler.convert(converted_meas)
            self.assertAlmostEqual(sampled.eval(), matmulmean, delta=.1)

    def test_pauli_expect_op_vector(self):
        """ pauli expect op vector test """
        paulis_op = ListOp([X, Y, Z, I])
        converted_meas = self.expect.convert(~StateFn(paulis_op))

        plus_mean = (converted_meas @ Plus)
        np.testing.assert_array_almost_equal(plus_mean.eval(), [1, 0, 0, 1], decimal=1)
        sampled_plus = self.sampler.convert(plus_mean)
        np.testing.assert_array_almost_equal(sampled_plus.eval(), [1, 0, 0, 1], decimal=1)

        minus_mean = (converted_meas @ Minus)
        np.testing.assert_array_almost_equal(minus_mean.eval(), [-1, 0, 0, 1], decimal=1)
        sampled_minus = self.sampler.convert(minus_mean)
        np.testing.assert_array_almost_equal(sampled_minus.eval(), [-1, 0, 0, 1], decimal=1)

        zero_mean = (converted_meas @ Zero)
        np.testing.assert_array_almost_equal(zero_mean.eval(), [0, 0, 1, 1], decimal=1)
        sampled_zero = self.sampler.convert(zero_mean)
        np.testing.assert_array_almost_equal(sampled_zero.eval(), [0, 0, 1, 1], decimal=1)

        sum_zero = (Plus + Minus) * (.5 ** .5)
        sum_zero_mean = (converted_meas @ sum_zero)
        np.testing.assert_array_almost_equal(sum_zero_mean.eval(), [0, 0, 1, 1], decimal=1)
        sampled_zero_mean = self.sampler.convert(sum_zero_mean)
        # !!NOTE!!: Depolarizing channel (Sampling) means interference
        # does not happen between circuits in sum, so expectation does
        # not equal expectation for Zero!!
        np.testing.assert_array_almost_equal(sampled_zero_mean.eval(), [0, 0, 0, 1], decimal=1)

        for i, op in enumerate(paulis_op.oplist):
            mat_op = op.to_matrix()
            np.testing.assert_array_almost_equal(zero_mean.eval()[i],
                                                 Zero.adjoint().to_matrix() @
                                                 mat_op @ Zero.to_matrix(),
                                                 decimal=1)
            np.testing.assert_array_almost_equal(plus_mean.eval()[i],
                                                 Plus.adjoint().to_matrix() @
                                                 mat_op @ Plus.to_matrix(),
                                                 decimal=1)
            np.testing.assert_array_almost_equal(minus_mean.eval()[i],
                                                 Minus.adjoint().to_matrix() @
                                                 mat_op @ Minus.to_matrix(),
                                                 decimal=1)

    def test_pauli_expect_state_vector(self):
        """ pauli expect state vector test """
        states_op = ListOp([One, Zero, Plus, Minus])

        paulis_op = X
        converted_meas = self.expect.convert(~StateFn(paulis_op) @ states_op)
        np.testing.assert_array_almost_equal(converted_meas.eval(), [0, 0, 1, -1], decimal=1)

        sampled = self.sampler.convert(converted_meas)
        np.testing.assert_array_almost_equal(sampled.eval(), [0, 0, 1, -1], decimal=1)

        # Small test to see if execution results are accessible
        for composed_op in sampled:
            self.assertIn('counts', composed_op[1].execution_results)

    def test_pauli_expect_op_vector_state_vector(self):
        """ pauli expect op vector state vector test """
        paulis_op = ListOp([X, Y, Z, I])
        states_op = ListOp([One, Zero, Plus, Minus])

        valids = [[+0, 0, 1, -1],
                  [+0, 0, 0, 0],
                  [-1, 1, 0, -0],
                  [+1, 1, 1, 1]]
        converted_meas = self.expect.convert(~StateFn(paulis_op) @ states_op)
        np.testing.assert_array_almost_equal(converted_meas.eval(), valids, decimal=1)

        sampled = self.sampler.convert(converted_meas)
        np.testing.assert_array_almost_equal(sampled.eval(), valids, decimal=1)

    def test_to_matrix_called(self):
        """ test to matrix called in different situations """
        qs = 45
        states_op = ListOp([Zero ^ qs,
                            One ^ qs,
                            (Zero ^ qs) + (One ^ qs)])
        paulis_op = ListOp([Z ^ qs,
                            (I ^ Z ^ I) ^ int(qs / 3)])

        # 45 qubit calculation - throws exception if to_matrix is called
        # massive is False
        with self.assertRaises(ValueError):
            states_op.to_matrix()
            paulis_op.to_matrix()

        # now set global variable or argument
        try:
            aqua_globals.massive = True
            with self.assertRaises(MemoryError):
                states_op.to_matrix()
                paulis_op.to_matrix()
            aqua_globals.massive = False
            with self.assertRaises(MemoryError):
                states_op.to_matrix(massive=True)
                paulis_op.to_matrix(massive=True)
        finally:
            aqua_globals.massive = False

    def test_not_to_matrix_called(self):
        """ 45 qubit calculation - literally will not work if to_matrix is
            somehow called (in addition to massive=False throwing an error)"""

        qs = 45
        states_op = ListOp([Zero ^ qs,
                            One ^ qs,
                            (Zero ^ qs) + (One ^ qs)])
        paulis_op = ListOp([Z ^ qs,
                            (I ^ Z ^ I) ^ int(qs / 3)])

        converted_meas = self.expect.convert(~StateFn(paulis_op) @ states_op)
        np.testing.assert_array_almost_equal(converted_meas.eval(), [[1, -1, 0], [1, -1, 0]])

    def test_grouped_pauli_expectation(self):
        """ grouped pauli expectation test """
        two_qubit_H2 = (-1.052373245772859 * I ^ I) + \
                       (0.39793742484318045 * I ^ Z) + \
                       (-0.39793742484318045 * Z ^ I) + \
                       (-0.01128010425623538 * Z ^ Z) + \
                       (0.18093119978423156 * X ^ X)
        wf = CX @ (H ^ I) @ Zero
        expect_op = PauliExpectation(group_paulis=False).convert(~StateFn(two_qubit_H2) @ wf)
        self.sampler._extract_circuitstatefns(expect_op)
        num_circuits_ungrouped = len(self.sampler._circuit_ops_cache)
        self.assertEqual(num_circuits_ungrouped, 5)

        expect_op_grouped = PauliExpectation(group_paulis=True).convert(~StateFn(two_qubit_H2) @ wf)
        q_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                     seed_simulator=self.seed, seed_transpiler=self.seed)
        sampler = CircuitSampler(q_instance)
        sampler._extract_circuitstatefns(expect_op_grouped)
        num_circuits_grouped = len(sampler._circuit_ops_cache)
        self.assertEqual(num_circuits_grouped, 2)

    @unittest.skip(reason="IBMQ testing not available in general.")
    def test_ibmq_grouped_pauli_expectation(self):
        """ pauli expect op vector state vector test """
        from qiskit import IBMQ
        p = IBMQ.load_account()
        backend = p.get_backend('ibmq_qasm_simulator')
        q_instance = QuantumInstance(backend, seed_simulator=self.seed, seed_transpiler=self.seed)

        paulis_op = ListOp([X, Y, Z, I])
        states_op = ListOp([One, Zero, Plus, Minus])

        valids = [[+0, 0, 1, -1],
                  [+0, 0, 0, 0],
                  [-1, 1, 0, -0],
                  [+1, 1, 1, 1]]
        converted_meas = self.expect.convert(~StateFn(paulis_op) @ states_op)
        sampled = CircuitSampler(q_instance).convert(converted_meas)
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
                                             [1, .5**.5, (1 + .5**.5), 1],
                                             decimal=1)


if __name__ == '__main__':
    unittest.main()
