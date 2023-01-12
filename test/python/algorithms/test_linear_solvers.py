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

"""Test the quantum linear system solver algorithm."""

import unittest
import warnings
from test.python.algorithms import QiskitAlgorithmsTestCase
from scipy.linalg import expm
import numpy as np
from ddt import ddt, idata, unpack
from qiskit import BasicAer, QuantumCircuit
from qiskit.algorithms.linear_solvers.hhl import HHL
from qiskit.algorithms.linear_solvers.matrices.tridiagonal_toeplitz import TridiagonalToeplitz
from qiskit.algorithms.linear_solvers.matrices.numpy_matrix import NumPyMatrix
from qiskit.algorithms.linear_solvers.observables.absolute_average import AbsoluteAverage
from qiskit.algorithms.linear_solvers.observables.matrix_functional import MatrixFunctional
from qiskit.circuit.library.arithmetic.exact_reciprocal import ExactReciprocal
from qiskit.quantum_info import Operator, partial_trace
from qiskit.opflow import I, Z, StateFn
from qiskit.utils import QuantumInstance
from qiskit import quantum_info
from qiskit.test import slow_test


def _factory_tridiagonal_toeplitz(
    num_state_qubits: int, main_diag: float, off_diag: float, trotter_steps: int = 1
):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        return TridiagonalToeplitz(
            num_state_qubits, main_diag, off_diag, trotter_steps=trotter_steps
        )


def _factory_numpy_matrix(matrix: np.ndarray):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        return NumPyMatrix(matrix)


@ddt
class TestMatrices(QiskitAlgorithmsTestCase):
    """Tests based on the matrices classes.

    This class tests
        * the constructed circuits
    """

    @idata(
        [
            [_factory_tridiagonal_toeplitz(2, 1, -1 / 3)],
            [_factory_tridiagonal_toeplitz(3, 2, 1), 1.1, 3],
            [
                _factory_numpy_matrix(
                    np.array(
                        [
                            [1 / 2, 1 / 6, 0, 0],
                            [1 / 6, 1 / 2, 1 / 6, 0],
                            [0, 1 / 6, 1 / 2, 1 / 6],
                            [0, 0, 1 / 6, 1 / 2],
                        ]
                    )
                )
            ],
        ]
    )
    @unpack
    def test_matrices(self, matrix, time=1.0, power=1):
        """Test the different matrix classes."""
        matrix.evolution_time = time

        num_qubits = matrix.num_state_qubits
        pow_circ = matrix.power(power).control()
        circ_qubits = pow_circ.num_qubits
        qc = QuantumCircuit(circ_qubits)
        qc.append(matrix.power(power).control(), list(range(circ_qubits)))
        # extract the parts of the circuit matrix corresponding to TridiagonalToeplitz
        zero_op = (I + Z) / 2
        one_op = (I - Z) / 2
        proj = Operator((zero_op ^ pow_circ.num_ancillas) ^ (I ^ num_qubits) ^ one_op).data
        circ_matrix = Operator(qc).data
        approx_exp = partial_trace(
            np.dot(proj, circ_matrix), [0] + list(range(num_qubits + 1, circ_qubits))
        ).data

        exact_exp = expm(1j * matrix.evolution_time * power * matrix.matrix)
        np.testing.assert_array_almost_equal(approx_exp, exact_exp, decimal=2)

    @idata(
        [
            [_factory_tridiagonal_toeplitz(2, 1.5, 2.5)],
            [_factory_tridiagonal_toeplitz(4, -1, 1.6)],
        ]
    )
    @unpack
    def test_eigs_bounds(self, matrix):
        """Test the capability of TridiagonalToeplitz matrix class
        to find accurate absolute eigenvalues bounds."""

        matrix_lambda_min, matrix_lambda_max = matrix.eigs_bounds()

        numpy_matrix = matrix.matrix
        eigenvalues, _ = np.linalg.eig(numpy_matrix)
        abs_eigenvalues = np.abs(eigenvalues)
        exact_lambda_min = np.min(abs_eigenvalues)
        exact_lambda_max = np.max(abs_eigenvalues)

        np.testing.assert_almost_equal(matrix_lambda_min, exact_lambda_min, decimal=6)
        np.testing.assert_almost_equal(matrix_lambda_max, exact_lambda_max, decimal=6)


def _factory_absolute_average():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        return AbsoluteAverage()


def _factory_matrix_functional(main_diag: float, off_diag: int):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        return MatrixFunctional(main_diag, off_diag)


@ddt
class TestObservables(QiskitAlgorithmsTestCase):
    """Tests based on the observables classes.

    This class tests
        * the constructed circuits
    """

    @idata(
        [
            [_factory_absolute_average(), [1.0, -2.1, 3.2, -4.3]],
            [_factory_absolute_average(), [-9 / 4, -0.3, 8 / 7, 10, -5, 11.1, 13 / 11, -27 / 12]],
        ]
    )
    @unpack
    def test_absolute_average(self, observable, vector):
        """Test the absolute average observable."""
        init_state = vector / np.linalg.norm(vector)
        num_qubits = int(np.log2(len(vector)))

        qc = QuantumCircuit(num_qubits)
        qc.isometry(init_state, list(range(num_qubits)), None)
        qc.append(observable.observable_circuit(num_qubits), list(range(num_qubits)))

        # Observable operator
        observable_op = observable.observable(num_qubits)
        state_vec = (~StateFn(observable_op) @ StateFn(qc)).eval()

        # Obtain result
        result = observable.post_processing(state_vec, num_qubits)

        # Obtain analytical evaluation
        exact = observable.evaluate_classically(init_state)

        np.testing.assert_almost_equal(result, exact, decimal=2)

    @idata(
        [
            [_factory_matrix_functional(1, -1 / 3), [1.0, -2.1, 3.2, -4.3]],
            [
                _factory_matrix_functional(2 / 3, 11 / 7),
                [-9 / 4, -0.3, 8 / 7, 10, -5, 11.1, 13 / 11, -27 / 12],
            ],
        ]
    )
    @unpack
    def test_matrix_functional(self, observable, vector):
        """Test the matrix functional class."""
        from qiskit.transpiler.passes import RemoveResetInZeroState

        tpass = RemoveResetInZeroState()

        init_state = vector / np.linalg.norm(vector)
        num_qubits = int(np.log2(len(vector)))

        # Get observable circuits
        obs_circuits = observable.observable_circuit(num_qubits)
        qcs = []
        for obs_circ in obs_circuits:
            qc = QuantumCircuit(num_qubits)
            qc.isometry(init_state, list(range(num_qubits)), None)
            qc.append(obs_circ, list(range(num_qubits)))
            qcs.append(tpass(qc.decompose()))

        # Get observables
        observable_ops = observable.observable(num_qubits)
        state_vecs = []
        # First is the norm
        state_vecs.append((~StateFn(observable_ops[0]) @ StateFn(qcs[0])).eval())
        for i in range(1, len(observable_ops), 2):
            state_vecs += [
                (~StateFn(observable_ops[i]) @ StateFn(qcs[i])).eval(),
                (~StateFn(observable_ops[i + 1]) @ StateFn(qcs[i + 1])).eval(),
            ]

        # Obtain result
        result = observable.post_processing(state_vecs, num_qubits)

        # Obtain analytical evaluation
        exact = observable.evaluate_classically(init_state)

        np.testing.assert_almost_equal(result, exact, decimal=2)


@ddt
class TestReciprocal(QiskitAlgorithmsTestCase):
    """Tests based on the reciprocal classes.

    This class tests
        * the constructed circuits
    """

    @idata([[2, 0.1, False], [3, 1 / 9, True]])
    @unpack
    def test_exact_reciprocal(self, num_qubits, scaling, neg_vals):
        """Test the ExactReciprocal class."""
        reciprocal = ExactReciprocal(num_qubits + neg_vals, scaling, neg_vals)

        qc = QuantumCircuit(num_qubits + 1 + neg_vals)
        qc.h(list(range(num_qubits)))
        # If negative eigenvalues, set the sign qubit to 1
        if neg_vals:
            qc.x(num_qubits)
        qc.append(reciprocal, list(range(num_qubits + 1 + neg_vals)))

        # Create the operator 0
        state_vec = quantum_info.Statevector.from_instruction(qc).data[-(2**num_qubits) :]

        # Remove the factor from the hadamards
        state_vec *= np.sqrt(2) ** num_qubits

        # Analytic value
        exact = []
        for i in range(0, 2**num_qubits):
            if i == 0:
                exact.append(0)
            else:
                if neg_vals:
                    exact.append(-scaling / (1 - i / (2**num_qubits)))
                else:
                    exact.append(scaling * (2**num_qubits) / i)

        np.testing.assert_array_almost_equal(state_vec, exact, decimal=2)


@ddt
class TestLinearSolver(QiskitAlgorithmsTestCase):
    """Tests based on the linear solvers classes.

    This class tests
        * the constructed circuits
    """

    @slow_test
    @idata(
        [
            [
                _factory_tridiagonal_toeplitz(2, 1, 1 / 3, trotter_steps=2),
                [1.0, -2.1, 3.2, -4.3],
                _factory_matrix_functional(1, 1 / 2),
            ],
            [
                np.array(
                    [[0, 0, 1.585, 0], [0, 0, -0.585, 1], [1.585, -0.585, 0, 0], [0, 1, 0, 0]]
                ),
                [1.0, 0, 0, 0],
                _factory_matrix_functional(1, 1 / 2),
            ],
            [
                [
                    [1 / 2, 1 / 6, 0, 0],
                    [1 / 6, 1 / 2, 1 / 6, 0],
                    [0, 1 / 6, 1 / 2, 1 / 6],
                    [0, 0, 1 / 6, 1 / 2],
                ],
                [1.0, -2.1, 3.2, -4.3],
                _factory_matrix_functional(1, 1 / 2),
            ],
            [
                np.array([[82, 34], [34, 58]]),
                np.array([[1], [0]]),
                _factory_absolute_average(),
                3,
            ],
            [
                _factory_tridiagonal_toeplitz(3, 1, -1 / 2, trotter_steps=2),
                [-9 / 4, -0.3, 8 / 7, 10, -5, 11.1, 13 / 11, -27 / 12],
                _factory_absolute_average(),
            ],
        ]
    )
    @unpack
    def test_hhl(self, matrix, right_hand_side, observable, decimal=1):
        """Test the HHL class."""
        if isinstance(matrix, QuantumCircuit):
            num_qubits = matrix.num_state_qubits
        elif isinstance(matrix, (np.ndarray)):
            num_qubits = int(np.log2(matrix.shape[0]))
        elif isinstance(matrix, list):
            num_qubits = int(np.log2(len(matrix)))

        rhs = right_hand_side / np.linalg.norm(right_hand_side)

        # Initial state circuit
        qc = QuantumCircuit(num_qubits)
        qc.isometry(rhs, list(range(num_qubits)), None)

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            hhl = HHL()
            self.assertTrue("HHL class is deprecated" in str(caught_warnings[0].message))
            solution = hhl.solve(matrix, qc, observable)
        approx_result = solution.observable

        # Calculate analytical value
        if isinstance(matrix, QuantumCircuit):
            exact_x = np.dot(np.linalg.inv(matrix.matrix), rhs)
        elif isinstance(matrix, (list, np.ndarray)):
            if isinstance(matrix, list):
                matrix = np.array(matrix)
            exact_x = np.dot(np.linalg.inv(matrix), rhs)
        exact_result = observable.evaluate_classically(exact_x)

        np.testing.assert_almost_equal(approx_result, exact_result, decimal=decimal)

    def test_hhl_qi(self):
        """Test the HHL quantum instance getter and setter."""
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            hhl = HHL()
            self.assertTrue("HHL class is deprecated" in str(caught_warnings[0].message))
        self.assertIsNone(hhl.quantum_instance)  # Defaults to None

        # First set a valid quantum instance and check via getter
        qinst = QuantumInstance(backend=BasicAer.get_backend("qasm_simulator"))
        hhl.quantum_instance = qinst
        self.assertEqual(hhl.quantum_instance, qinst)

        # Now set quantum instance back to None and check via getter
        hhl.quantum_instance = None
        self.assertIsNone(hhl.quantum_instance)


if __name__ == "__main__":
    unittest.main()
