""" Test of the AdaptVQE minimum eigensolver """
import unittest
import sys
from test.python.algorithms import QiskitAlgorithmsTestCase
from qiskit.algorithms.minimum_eigen_solvers.vqe import VQE
from qiskit.opflow.gradients.gradient import Gradient
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.opflow import PauliSumOp
from qiskit.utils import algorithm_globals, has_aer
from qiskit import BasicAer
from qiskit.algorithms.minimum_eigen_solvers.adapt_vqe import AdaptVQE

sys.path.append("/Users/freyashah/qiskit-terra/test")


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

    def test_default(self):
        """Default execution"""
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
        initial_state = QuantumCircuit(QuantumRegister(4))
        initial_state.x(0)
        initial_state.x(1)
        ansatz = EvolvedOperatorAnsatz(excitation_pool, initial_state=initial_state)
        quantum_instance = BasicAer.get_backend("statevector_simulator")
        calc = AdaptVQE(
            solver=VQE(quantum_instance=quantum_instance),
            ansatz=ansatz,
            excitation_pool=excitation_pool,
            quantum_instance=quantum_instance,
        )
        res = calc.compute_minimum_eigenvalue(operator=self.h2_op)

        expected_eigenvalue = -1.85727503

        self.assertAlmostEqual(res.eigenvalue, expected_eigenvalue, places=6)

    def test_finite_diff(self):
        """test using finite difference gradient"""
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
        initial_state = QuantumCircuit(QuantumRegister(4))
        initial_state.x(0)
        initial_state.x(1)
        ansatz = EvolvedOperatorAnsatz(excitation_pool, initial_state=initial_state)
        quantum_instance = BasicAer.get_backend("statevector_simulator")
        calc = AdaptVQE(
            solver=VQE(quantum_instance=quantum_instance),
            ansatz=ansatz,
            excitation_pool=excitation_pool,
            adapt_gradient=Gradient(grad_method="fin_diff"),
            quantum_instance=quantum_instance,
        )
        res = calc.compute_minimum_eigenvalue(operator=self.h2_op)

        expected_eigenvalue = -1.85727503

        self.assertAlmostEqual(res.eigenvalue, expected_eigenvalue, places=6)

    def test_param_shift(self):
        """test using parameter shift gradient"""
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
        initial_state = QuantumCircuit(QuantumRegister(4))
        initial_state.x(0)
        initial_state.x(1)
        ansatz = EvolvedOperatorAnsatz(excitation_pool, initial_state=initial_state)
        quantum_instance = BasicAer.get_backend("statevector_simulator")
        calc = AdaptVQE(
            solver=VQE(quantum_instance=quantum_instance),
            ansatz=ansatz,
            excitation_pool=excitation_pool,
            adapt_gradient=Gradient(grad_method="param_shift"),
            quantum_instance=quantum_instance,
        )
        res = calc.compute_minimum_eigenvalue(operator=self.h2_op)

        expected_eigenvalue = -1.85727503

        self.assertAlmostEqual(res.eigenvalue, expected_eigenvalue, places=6)


if __name__ == "__main__":
    unittest.main()
