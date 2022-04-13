""" Test of the Adaptive VQE ground state calculations """
import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase


from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.opflow import PauliSumOp
from qiskit.utils import algorithm_globals, has_aer

if has_aer():
    from qiskit import Aer

from qiskit.algorithms.minimum_eigen_solvers.adaptvqe2 import AdaptVQE2


class TestAdaptVQE2(QiskitAlgorithmsTestCase):
    """Test Adaptive VQE Ground State Calculation"""

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
        self.h2_energy = -1.85727503

        self.expected = -1.85727503

    def test_default(self):
        """TODO."""
        print("qubits:", self.h2_op.num_qubits)
        excitation_pool = [
            PauliSumOp(
                SparsePauliOp(["IIIY", "IIZY"], coeffs=[0.5 + 0.0j, -0.5 + 0.0j]), coeff=1.0
            ),
            PauliSumOp(
                SparsePauliOp(["ZYII", "IYZI"], coeffs=[-0.5 + 0.0j, 0.5 + 0.0j]), coeff=1.0
            ),
            PauliSumOp(
                SparsePauliOp(
                    ["IYIX", "ZYIX", "IYZX", "ZYZX", "IXIY", "ZXIY", "IXZY", "ZXZY"],
                    coeffs=[
                        -0.125 + 0.0j,
                        -0.125 + 0.0j,
                        0.125 + 0.0j,
                        0.125 + 0.0j,
                        0.125 + 0.0j,
                        0.125 + 0.0j,
                        -0.125 + 0.0j,
                        -0.125 + 0.0j,
                    ],
                ),
                coeff=1.0,
            ),
        ]
        ansatz = EvolvedOperatorAnsatz(excitation_pool)
        calc = AdaptVQE2(
            ansatz=ansatz,
            operator=self.h2_op,
            quantum_instance=Aer.get_backend("statevector_simulator"),
        )
        res = calc.solve()
        self.assertAlmostEqual(res.electronic_energies[0], self.expected, places=6)


if __name__ == "__main__":
    unittest.main()
