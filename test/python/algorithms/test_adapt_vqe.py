""" Test of the AdaptVQE minimum eigensolver """
import sys
from qiskit.algorithms.minimum_eigen_solvers.vqe import VQE
sys.path.append("/Users/freyashah/qiskit-terra/test")
import unittest
from qiskit.opflow.gradients.gradient import Gradient
from test.python.algorithms import QiskitAlgorithmsTestCase


from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.opflow import PauliSumOp
from qiskit.utils import algorithm_globals, has_aer
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

    def test_default(self):
        """TODO."""
        excitation_pool = [
            PauliSumOp(
                SparsePauliOp(["IIIY", "IIZY"], coeffs=[0.5 + 0.0j, -0.5 + 0.0j]), coeff=1.0
            ),
            PauliSumOp(
                SparsePauliOp(["ZYII", "IYZI"], coeffs=[-0.5 + 0.0j, 0.5 + 0.0j]), coeff=1.0
            ),
            PauliSumOp(
                SparsePauliOp(
                    ['ZXZY', 'IXIY', 'IYIX', 'ZYZX', 'IYZX', 'ZYIX', 'ZXIY', 'IXZY'],
                    coeffs=[
                        -0.125 + 0.j,
                        0.125 + 0.j,
                        -0.125 + 0.j,
                        0.125 + 0.j,
                        0.125 + 0.j,
                        -0.125 + 0.j,
                        0.125 + 0.j,
                        -0.125 + 0.j
                    ]
                ),
                coeff=1.0
            ),
        ]
        ansatz1 = EvolvedOperatorAnsatz(excitation_pool)
        grad=Gradient(grad_method="param_shift")
        calc = AdaptVQE(
            solver= VQE(ansatz=ansatz1,gradient=grad),
            ansatz= ansatz1,
            excitation_pool=excitation_pool,
            adapt_gradient= grad,
            quantum_instance=BasicAer.get_backend("statevector_simulator")
        )
        res = calc.compute_minimum_eigensolver(operator=self.h2_op)

        expected = -1.85727503

        self.assertAlmostEqual(res.electronic_energies[0], expected, places=6)


if __name__ == "__main__":
    unittest.main()
