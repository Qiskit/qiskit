# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
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
import time

from test.aqua import QiskitAquaTestCase
import numpy as np
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit import QuantumCircuit

from qiskit.aqua.components.oracles import LogicalExpressionOracle
from qiskit.aqua import QuantumInstance, aqua_globals, AquaError
from qiskit.aqua.algorithms import Grover, VQE
from qiskit.aqua.operators import I, X, Z
from qiskit.aqua.components.optimizers import SPSA
from qiskit.circuit.library import EfficientSU2


class TestMeasurementErrorMitigation(QiskitAquaTestCase):
    """Test measurement error mitigation."""

    def setUp(self):
        super().setUp()
        try:
            from qiskit import Aer  # pylint: disable=unused-import,import-outside-toplevel
        except ImportError as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return

    def test_measurement_error_mitigation(self):
        """ measurement error mitigation test """
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit import Aer
            from qiskit.providers.aer import noise
        except ImportError as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return

        aqua_globals.random_seed = 0

        # build noise model
        noise_model = noise.NoiseModel()
        read_err = noise.errors.readout_error.ReadoutError([[0.9, 0.1], [0.25, 0.75]])
        noise_model.add_all_qubit_readout_error(read_err)

        backend = Aer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(backend=backend, seed_simulator=167, seed_transpiler=167,
                                           noise_model=noise_model)

        qi_with_mitigation = \
            QuantumInstance(backend=backend,
                            seed_simulator=167,
                            seed_transpiler=167,
                            noise_model=noise_model,
                            measurement_error_mitigation_cls=CompleteMeasFitter)
        oracle = LogicalExpressionOracle('a & b & c')
        grover = Grover(oracle)

        result_wo_mitigation = grover.run(quantum_instance)
        self.assertGreater(quantum_instance.time_taken, 0.)
        quantum_instance.reset_execution_results()
        prob_top_meas_wo_mitigation = result_wo_mitigation.measurement[
            result_wo_mitigation.top_measurement]

        result_w_mitigation = grover.run(qi_with_mitigation)
        prob_top_meas_w_mitigation = \
            result_w_mitigation.measurement[result_w_mitigation.top_measurement]

        self.assertGreaterEqual(prob_top_meas_w_mitigation, prob_top_meas_wo_mitigation)

    def test_measurement_error_mitigation_auto_refresh(self):
        """ measurement error mitigation auto refresh test """
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit import Aer
            from qiskit.providers.aer import noise
        except ImportError as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return

        aqua_globals.random_seed = 0

        # build noise model
        noise_model = noise.NoiseModel()
        read_err = noise.errors.readout_error.ReadoutError([[0.9, 0.1], [0.25, 0.75]])
        noise_model.add_all_qubit_readout_error(read_err)

        backend = Aer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(backend=backend,
                                           seed_simulator=1679,
                                           seed_transpiler=167,
                                           noise_model=noise_model,
                                           measurement_error_mitigation_cls=CompleteMeasFitter,
                                           cals_matrix_refresh_period=0)
        oracle = LogicalExpressionOracle('a & b & c')
        grover = Grover(oracle)
        _ = grover.run(quantum_instance)
        self.assertGreater(quantum_instance.time_taken, 0.)
        quantum_instance.reset_execution_results()
        cals_matrix_1, timestamp_1 = quantum_instance.cals_matrix(qubit_index=[0, 1, 2])

        time.sleep(15)
        aqua_globals.random_seed = 2
        quantum_instance.set_config(seed_simulator=111)
        _ = grover.run(quantum_instance)
        cals_matrix_2, timestamp_2 = quantum_instance.cals_matrix(qubit_index=[0, 1, 2])

        diff = cals_matrix_1 - cals_matrix_2
        total_diff = np.sum(np.abs(diff))

        self.assertGreater(total_diff, 0.0)
        self.assertGreater(timestamp_2, timestamp_1)

    def test_measurement_error_mitigation_with_dedicated_shots(self):
        """ measurement error mitigation with dedicated shots test """
        # pylint: disable=import-outside-toplevel
        try:
            from qiskit import Aer
            from qiskit.providers.aer import noise
        except ImportError as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return

        aqua_globals.random_seed = 0

        # build noise model
        noise_model = noise.NoiseModel()
        read_err = noise.errors.readout_error.ReadoutError([[0.9, 0.1], [0.25, 0.75]])
        noise_model.add_all_qubit_readout_error(read_err)

        backend = Aer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(backend=backend,
                                           seed_simulator=1679,
                                           seed_transpiler=167,
                                           shots=100,
                                           noise_model=noise_model,
                                           measurement_error_mitigation_cls=CompleteMeasFitter,
                                           cals_matrix_refresh_period=0)
        oracle = LogicalExpressionOracle('a & b & c')
        grover = Grover(oracle)
        _ = grover.run(quantum_instance)
        self.assertGreater(quantum_instance.time_taken, 0.)
        quantum_instance.reset_execution_results()
        cals_matrix_1, timestamp_1 = quantum_instance.cals_matrix(qubit_index=[0, 1, 2])

        quantum_instance.measurement_error_mitigation_shots = 1000
        _ = grover.run(quantum_instance)
        cals_matrix_2, timestamp_2 = quantum_instance.cals_matrix(qubit_index=[0, 1, 2])

        diff = cals_matrix_1 - cals_matrix_2
        total_diff = np.sum(np.abs(diff))

        self.assertGreater(total_diff, 0.0)
        self.assertGreater(timestamp_2, timestamp_1)

    def test_measurement_error_mitigation_with_diff_qubit_order(self):
        """ measurement error mitigation with different qubit order"""
        # pylint: disable=import-outside-toplevel
        try:
            from qiskit import Aer
            from qiskit.providers.aer import noise
        except ImportError as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return

        aqua_globals.random_seed = 0

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

        self.assertRaises(AquaError, quantum_instance.execute, [qc1, qc3])

    def test_measurement_error_mitigation_with_vqe(self):
        """ measurement error mitigation test with vqe """
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit import Aer
            from qiskit.providers.aer import noise
        except ImportError as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return

        aqua_globals.random_seed = 0

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
            operator=h2_hamiltonian,
            quantum_instance=quantum_instance,
            optimizer=optimizer,
        )
        result = vqe.compute_minimum_eigenvalue()
        self.assertGreater(quantum_instance.time_taken, 0.)
        quantum_instance.reset_execution_results()
        self.assertAlmostEqual(result.eigenvalue.real, -1.86, places=2)


if __name__ == '__main__':
    unittest.main()
