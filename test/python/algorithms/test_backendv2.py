# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Providers that support BackendV2 interface"""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import FakeProvider
from qiskit.providers.fake_provider.fake_backend_v2 import FakeBackendSimple
from qiskit.utils import QuantumInstance
from qiskit.algorithms import VQE, Grover, AmplificationProblem
from qiskit.opflow import X, Z, I
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal


class TestBackendV2(QiskitAlgorithmsTestCase):
    """test BackendV2 interface"""

    def setUp(self):
        super().setUp()
        self._provider = FakeProvider()
        self._qasm = FakeBackendSimple()
        self.seed = 50

    def test_vqe_qasm(self):
        """Test the VQE on QASM simulator."""
        optimizer = SPSA(maxiter=300, last_avg=5)
        wavefunction = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")

        with self.assertWarns(DeprecationWarning):
            h2_op = (
                -1.052373245772859 * (I ^ I)
                + 0.39793742484318045 * (I ^ Z)
                - 0.39793742484318045 * (Z ^ I)
                - 0.01128010425623538 * (Z ^ Z)
                + 0.18093119978423156 * (X ^ X)
            )
            qasm_simulator = QuantumInstance(
                self._qasm, shots=1024, seed_simulator=self.seed, seed_transpiler=self.seed
            )

        with self.assertWarns(DeprecationWarning):
            vqe = VQE(
                ansatz=wavefunction,
                optimizer=optimizer,
                max_evals_grouped=1,
                quantum_instance=qasm_simulator,
            )
            result = vqe.compute_minimum_eigenvalue(operator=h2_op)

        self.assertAlmostEqual(result.eigenvalue.real, -1.86, delta=0.05)

    def test_run_circuit_oracle(self):
        """Test execution with a quantum circuit oracle"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        problem = AmplificationProblem(oracle, is_good_state=["11"])

        with self.assertWarns(DeprecationWarning):
            qi = QuantumInstance(
                self._provider.get_backend("fake_yorktown"), seed_simulator=12, seed_transpiler=32
            )

        with self.assertWarns(DeprecationWarning):
            grover = Grover(quantum_instance=qi)
            result = grover.amplify(problem)

        self.assertIn(result.top_measurement, ["11"])

    def test_run_circuit_oracle_single_experiment_backend(self):
        """Test execution with a quantum circuit oracle"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        problem = AmplificationProblem(oracle, is_good_state=["11"])
        backend = self._provider.get_backend("fake_yorktown")
        backend._configuration.max_experiments = 1

        with self.assertWarns(DeprecationWarning):
            qi = QuantumInstance(
                self._provider.get_backend("fake_yorktown"), seed_simulator=12, seed_transpiler=32
            )

        with self.assertWarns(DeprecationWarning):
            grover = Grover(quantum_instance=qi)
            result = grover.amplify(problem)

        self.assertIn(result.top_measurement, ["11"])


if __name__ == "__main__":
    unittest.main()
