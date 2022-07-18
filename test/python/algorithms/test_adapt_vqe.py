# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
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
from qiskit.algorithms.minimum_eigen_solvers.vqe import VQE
from qiskit.opflow.gradients.gradient import Gradient
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.opflow import PauliSumOp
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit import BasicAer
from qiskit.algorithms.minimum_eigen_solvers.adapt_vqe import AdaptVQE


class TestAdaptVQE(QiskitAlgorithmsTestCase):
    """Test of the AdaptVQE minimum eigensolver"""

    def setUp(self):
        super().setUp()
        self.seed = 50
        algorithm_globals.random_seed = self.seed
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
        self.ansatz = EvolvedOperatorAnsatz(self.excitation_pool, initial_state=self.initial_state)
        self.quantum_instance = BasicAer.get_backend("statevector_simulator")
        self.qasm_simulator = QuantumInstance(
            BasicAer.get_backend("qasm_simulator"),
            shots=4096,
            seed_simulator=self.seed,
            seed_transpiler=self.seed,
        )

    def test_default(self):
        """Default execution"""
        calc = AdaptVQE(
            solver=VQE(ansatz=self.ansatz, quantum_instance=self.quantum_instance),
            excitation_pool=self.excitation_pool,
        )
        res = calc.compute_minimum_eigenvalue(operator=self.h2_op)

        expected_eigenvalue = -1.85727503

        self.assertAlmostEqual(res.eigenvalue, expected_eigenvalue, places=6)

    def test_finite_diff(self):
        """Test using finite difference gradient"""
        calc = AdaptVQE(
            solver=VQE(ansatz=self.ansatz, quantum_instance=self.quantum_instance),
            excitation_pool=self.excitation_pool,
            adapt_gradient=Gradient(grad_method="fin_diff"),
        )
        res = calc.compute_minimum_eigenvalue(operator=self.h2_op)

        expected_eigenvalue = -1.85727503

        self.assertAlmostEqual(res.eigenvalue, expected_eigenvalue, places=6)

    def test_qasm_simulator(self):
        """Test using qasm simulator"""
        calc = AdaptVQE(
            solver=VQE(ansatz=self.ansatz, quantum_instance=self.qasm_simulator),
            excitation_pool=self.excitation_pool,
        )
        res = calc.compute_minimum_eigenvalue(operator=self.h2_op)

        expected_eigenvalue = -1.8

        self.assertAlmostEqual(res.eigenvalue, expected_eigenvalue, places=1)

    def test_param_shift(self):
        """Test using parameter shift gradient"""
        calc = AdaptVQE(
            solver=VQE(ansatz=self.ansatz, quantum_instance=self.quantum_instance),
            excitation_pool=self.excitation_pool,
            adapt_gradient=Gradient(grad_method="param_shift"),
        )
        res = calc.compute_minimum_eigenvalue(operator=self.h2_op)

        expected_eigenvalue = -1

        self.assertAlmostEqual(res.eigenvalue, expected_eigenvalue, places=0)


if __name__ == "__main__":
    unittest.main()
