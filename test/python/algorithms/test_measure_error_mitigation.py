# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Measurement Error Mitigation """

import unittest

from test.python.algorithms import QiskitAlgorithmsTestCase
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import VQE
from qiskit.opflow import I, X, Z
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import EfficientSU2


class TestMeasurementErrorMitigation(QiskitAlgorithmsTestCase):
    """Test measurement error mitigation."""

    def test_measurement_error_mitigation_with_diff_qubit_order(self):
        """ measurement error mitigation with different qubit order"""
        try:
            from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
            from qiskit import Aer
            from qiskit.providers.aer import noise
        except ImportError as ex:
            self.skipTest("Package doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return

        algorithm_globals.random_seed = 0

        # build noise model
        noise_model = noise.NoiseModel()
        read_err = noise.errors.readout_error.ReadoutError([[0.9, 0.1], [0.25, 0.75]])
        noise_model.add_all_qubit_readout_error(read_err)

        backend = Aer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(backend=backend,
                                           seed_simulator=1679,
                                           seed_transpiler=167,
                                           shots=1000,
                                           noise_model=noise_model,
                                           measurement_error_mitigation_cls=CompleteMeasFitter,
                                           cals_matrix_refresh_period=0)
        # circuit
        qc1 = QuantumCircuit(2, 2)
        qc1.h(0)
        qc1.cx(0, 1)
        qc1.measure(0, 0)
        qc1.measure(1, 1)
        qc2 = QuantumCircuit(2, 2)
        qc2.h(0)
        qc2.cx(0, 1)
        qc2.measure(1, 0)
        qc2.measure(0, 1)

        # this should run smoothly
        quantum_instance.execute([qc1, qc2])
        self.assertGreater(quantum_instance.time_taken, 0.)
        quantum_instance.reset_execution_results()

        # failure case
        qc3 = QuantumCircuit(3, 3)
        qc3.h(2)
        qc3.cx(1, 2)
        qc3.measure(2, 1)
        qc3.measure(1, 2)

        self.assertRaises(QiskitError, quantum_instance.execute, [qc1, qc3])

    def test_measurement_error_mitigation_with_vqe(self):
        """ measurement error mitigation test with vqe """
        try:
            from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
            from qiskit import Aer
            from qiskit.providers.aer import noise
        except ImportError as ex:
            self.skipTest("Package doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return

        algorithm_globals.random_seed = 0

        # build noise model
        noise_model = noise.NoiseModel()
        read_err = noise.errors.readout_error.ReadoutError([[0.9, 0.1], [0.25, 0.75]])
        noise_model.add_all_qubit_readout_error(read_err)

        backend = Aer.get_backend('qasm_simulator')

        quantum_instance = QuantumInstance(
            backend=backend,
            seed_simulator=167,
            seed_transpiler=167,
            noise_model=noise_model,
            measurement_error_mitigation_cls=CompleteMeasFitter
        )

        h2_hamiltonian = -1.052373245772859 * (I ^ I) \
            + 0.39793742484318045 * (I ^ Z) \
            - 0.39793742484318045 * (Z ^ I) \
            - 0.01128010425623538 * (Z ^ Z) \
            + 0.18093119978423156 * (X ^ X)
        optimizer = SPSA(maxiter=200)
        var_form = EfficientSU2(2, reps=1)

        vqe = VQE(
            var_form=var_form,
            optimizer=optimizer,
            quantum_instance=quantum_instance
        )
        result = vqe.compute_minimum_eigenvalue(operator=h2_hamiltonian)
        self.assertGreater(quantum_instance.time_taken, 0.)
        quantum_instance.reset_execution_results()
        self.assertAlmostEqual(result.eigenvalue.real, -1.86, delta=0.05)


if __name__ == '__main__':
    unittest.main()
