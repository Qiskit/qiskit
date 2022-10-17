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

""" Test of the AdaptVQE minimum eigensolver """

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase

from ddt import ddt, data, unpack

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
        excitation_pool = [
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
        self.ansatz = EvolvedOperatorAnsatz(excitation_pool, initial_state=self.initial_state)
        self.optimizer = SLSQP()

    def test_default(self):
        """Default execution"""
        calc = AdaptVQE(VQE(Estimator(), self.ansatz, self.optimizer))
        res = calc.compute_minimum_eigenvalue(operator=self.h2_op)

        expected_eigenvalue = -1.85727503

        self.assertAlmostEqual(res.eigenvalue, expected_eigenvalue, places=6)

    def test_converged(self):
        """Test to check termination criteria"""
        calc = AdaptVQE(
            VQE(Estimator(), self.ansatz, self.optimizer),
            threshold=1e-3,
        )
        res = calc.compute_minimum_eigenvalue(operator=self.h2_op)

        self.assertEqual(res.termination_criterion, TerminationCriterion.CONVERGED)

    def test_maximum(self):
        """Test to check termination criteria"""
        calc = AdaptVQE(
            VQE(Estimator(), self.ansatz, self.optimizer),
            max_iterations=1,
        )
        res = calc.compute_minimum_eigenvalue(operator=self.h2_op)

        self.assertEqual(res.termination_criterion, TerminationCriterion.MAXIMUM)

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


if __name__ == "__main__":
    unittest.main()
