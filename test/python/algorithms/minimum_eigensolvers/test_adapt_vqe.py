# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test of the AdaptVQE minimum eigensolver"""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase

from ddt import ddt, data, unpack

import numpy as np

from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.minimum_eigensolvers.adapt_vqe import AdaptVQE, TerminationCriterion
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.opflow import PauliSumOp
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils import algorithm_globals


@ddt
class TestAdaptVQE(QiskitAlgorithmsTestCase):
    """Test of the AdaptVQE minimum eigensolver"""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 42

        with self.assertWarns(DeprecationWarning):
            self.h2_op = PauliSumOp.from_list(
                [
                    ("IIII", -0.8105479805373266),
                    ("ZZII", -0.2257534922240251),
                    ("IIZI", +0.12091263261776641),
                    ("ZIZI", +0.12091263261776641),
                    ("IZZI", +0.17218393261915543),
                    ("IIIZ", +0.17218393261915546),
                    ("IZIZ", +0.1661454325638243),
                    ("ZZIZ", +0.1661454325638243),
                    ("IIZZ", -0.2257534922240251),
                    ("IZZZ", +0.16892753870087926),
                    ("ZZZZ", +0.17464343068300464),
                    ("IXIX", +0.04523279994605788),
                    ("ZXIX", +0.04523279994605788),
                    ("IXZX", -0.04523279994605788),
                    ("ZXZX", -0.04523279994605788),
                ]
            )
            self.excitation_pool = [
                PauliSumOp(
                    SparsePauliOp(["IIIY", "IIZY"], coeffs=[0.5 + 0.0j, -0.5 + 0.0j]), coeff=1.0
                ),
                PauliSumOp(
                    SparsePauliOp(["ZYII", "IYZI"], coeffs=[-0.5 + 0.0j, 0.5 + 0.0j]), coeff=1.0
                ),
                PauliSumOp(
                    SparsePauliOp(
                        ["ZXZY", "IXIY", "IYIX", "ZYZX", "IYZX", "ZYIX", "ZXIY", "IXZY"],
                        coeffs=[
                            -0.125 + 0.0j,
                            0.125 + 0.0j,
                            -0.125 + 0.0j,
                            0.125 + 0.0j,
                            0.125 + 0.0j,
                            -0.125 + 0.0j,
                            0.125 + 0.0j,
                            -0.125 + 0.0j,
                        ],
                    ),
                    coeff=1.0,
                ),
            ]
            self.initial_state = QuantumCircuit(QuantumRegister(4))
            self.initial_state.x(0)
            self.initial_state.x(1)
            self.ansatz = EvolvedOperatorAnsatz(
                self.excitation_pool, initial_state=self.initial_state
            )
            self.optimizer = SLSQP()

    def test_default(self):
        """Default execution"""
        calc = AdaptVQE(VQE(Estimator(), self.ansatz, self.optimizer))

        with self.assertWarns(DeprecationWarning):
            res = calc.compute_minimum_eigenvalue(operator=self.h2_op)

        expected_eigenvalue = -1.85727503

        self.assertAlmostEqual(res.eigenvalue, expected_eigenvalue, places=6)
        np.testing.assert_allclose(res.eigenvalue_history, [expected_eigenvalue], rtol=1e-6)

    def test_with_quantum_info(self):
        """Test behavior with quantum_info-based operators."""
        ansatz = EvolvedOperatorAnsatz(
            [op.primitive for op in self.excitation_pool],
            initial_state=self.initial_state,
        )

        calc = AdaptVQE(VQE(Estimator(), ansatz, self.optimizer))
        res = calc.compute_minimum_eigenvalue(operator=self.h2_op.primitive)

        expected_eigenvalue = -1.85727503

        self.assertAlmostEqual(res.eigenvalue, expected_eigenvalue, places=6)
        np.testing.assert_allclose(res.eigenvalue_history, [expected_eigenvalue], rtol=1e-6)

    def test_converged(self):
        """Test to check termination criteria"""
        calc = AdaptVQE(
            VQE(Estimator(), self.ansatz, self.optimizer),
            gradient_threshold=1e-3,
        )
        with self.assertWarns(DeprecationWarning):
            res = calc.compute_minimum_eigenvalue(operator=self.h2_op)

        self.assertEqual(res.termination_criterion, TerminationCriterion.CONVERGED)

    def test_maximum(self):
        """Test to check termination criteria"""
        calc = AdaptVQE(
            VQE(Estimator(), self.ansatz, self.optimizer),
            max_iterations=1,
        )
        with self.assertWarns(DeprecationWarning):
            res = calc.compute_minimum_eigenvalue(operator=self.h2_op)

        self.assertEqual(res.termination_criterion, TerminationCriterion.MAXIMUM)

    def test_eigenvalue_threshold(self):
        """Test for the eigenvalue_threshold attribute."""
        operator = SparsePauliOp.from_list(
            [
                ("XX", 1.0),
                ("ZX", -0.5),
                ("XZ", -0.5),
            ]
        )
        ansatz = EvolvedOperatorAnsatz(
            [
                SparsePauliOp.from_list([("YZ", 0.4)]),
                SparsePauliOp.from_list([("ZY", 0.5)]),
            ],
            initial_state=QuantumCircuit(2),
        )

        calc = AdaptVQE(
            VQE(Estimator(), ansatz, self.optimizer),
            eigenvalue_threshold=1,
        )
        res = calc.compute_minimum_eigenvalue(operator)

        self.assertEqual(res.termination_criterion, TerminationCriterion.CONVERGED)

    def test_threshold_attribute(self):
        """Test the (pending deprecated) threshold attribute"""
        with self.assertWarns(PendingDeprecationWarning):
            calc = AdaptVQE(
                VQE(Estimator(), self.ansatz, self.optimizer),
                threshold=1e-3,
            )
            with self.assertWarns(DeprecationWarning):
                res = calc.compute_minimum_eigenvalue(operator=self.h2_op)

            self.assertEqual(res.termination_criterion, TerminationCriterion.CONVERGED)

    @data(
        ([1, 1], True),
        ([1, 11], False),
        ([11, 1], False),
        ([1, 12], False),
        ([12, 2], False),
        ([1, 1, 1], True),
        ([1, 2, 1], False),
        ([1, 2, 2], True),
        ([1, 2, 21], False),
        ([1, 12, 2], False),
        ([11, 1, 2], False),
        ([1, 2, 1, 1], True),
        ([1, 2, 1, 2], True),
        ([1, 2, 1, 21], False),
        ([11, 2, 1, 2], False),
        ([1, 11, 1, 111], False),
        ([11, 1, 111, 1], False),
        ([1, 2, 3, 1, 2, 3], True),
        ([1, 2, 3, 4, 1, 2, 3], False),
        ([11, 2, 3, 1, 2, 3], False),
        ([1, 2, 3, 1, 2, 31], False),
        ([1, 2, 3, 4, 1, 2, 3, 4], True),
        ([11, 2, 3, 4, 1, 2, 3, 4], False),
        ([1, 2, 3, 4, 1, 2, 3, 41], False),
        ([1, 2, 3, 4, 5, 1, 2, 3, 4], False),
    )
    @unpack
    def test_cyclicity(self, seq, is_cycle):
        """Test AdaptVQE index cycle detection"""
        self.assertEqual(is_cycle, AdaptVQE._check_cyclicity(seq))

    def test_vqe_solver(self):
        """Test to check if the VQE solver remains the same or not"""
        solver = VQE(Estimator(), self.ansatz, self.optimizer)
        calc = AdaptVQE(solver)
        with self.assertWarns(DeprecationWarning):
            _ = calc.compute_minimum_eigenvalue(operator=self.h2_op)
            self.assertEqual(solver.ansatz, calc.solver.ansatz)

    def test_gradient_calculation(self):
        """Test to check if the gradient calculation"""
        solver = VQE(Estimator(), QuantumCircuit(1), self.optimizer)
        calc = AdaptVQE(solver)
        calc._excitation_pool = [SparsePauliOp("X")]
        res = calc._compute_gradients(operator=SparsePauliOp("Y"), theta=[])
        # compare with manually computed reference value
        self.assertAlmostEqual(res[0][0], 2.0)

    def test_supports_aux_operators(self):
        """Test that auxiliary operators are supported"""
        calc = AdaptVQE(VQE(Estimator(), self.ansatz, self.optimizer))
        with self.assertWarns(DeprecationWarning):
            res = calc.compute_minimum_eigenvalue(operator=self.h2_op, aux_operators=[self.h2_op])

        expected_eigenvalue = -1.85727503

        self.assertAlmostEqual(res.eigenvalue, expected_eigenvalue, places=6)
        self.assertAlmostEqual(res.aux_operators_evaluated[0][0], expected_eigenvalue, places=6)
        np.testing.assert_allclose(res.eigenvalue_history, [expected_eigenvalue], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
