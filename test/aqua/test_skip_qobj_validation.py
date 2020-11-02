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

""" Test Skip Qobj Validation """

import unittest
from test.aqua import QiskitAquaTestCase
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance, AquaError


def _compare_dict(dict1, dict2):
    equal = True
    for key1, value1 in dict1.items():
        if key1 not in dict2:
            equal = False
            break
        if value1 != dict2[key1]:
            equal = False
            break
    return equal


class TestSkipQobjValidation(QiskitAquaTestCase):
    """ Test Skip Qobj Validation """
    def setUp(self):
        super().setUp()
        self.random_seed = 10598

        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        # Ensure qubit 0 is measured before qubit 1
        qc.barrier(qr)
        qc.measure(qr[0], cr[0])
        qc.barrier(qr)
        qc.measure(qr[1], cr[1])

        self.qc = qc
        self.backend = BasicAer.get_backend('qasm_simulator')

    def test_wo_backend_options(self):
        """ without backend options test """
        quantum_instance = QuantumInstance(self.backend,
                                           seed_transpiler=self.random_seed,
                                           seed_simulator=self.random_seed,
                                           shots=1024)
        # run without backend_options and without noise
        res_wo_bo = quantum_instance.execute(self.qc).get_counts(self.qc)
        self.assertGreater(quantum_instance.time_taken, 0.)
        quantum_instance.reset_execution_results()
        quantum_instance.skip_qobj_validation = True
        res_wo_bo_skip_validation = quantum_instance.execute(self.qc).get_counts(self.qc)
        self.assertGreater(quantum_instance.time_taken, 0.)
        quantum_instance.reset_execution_results()
        self.assertTrue(_compare_dict(res_wo_bo, res_wo_bo_skip_validation))

    def test_w_backend_options(self):
        """ with backend options test """
        # run with backend_options
        quantum_instance = QuantumInstance(self.backend, seed_transpiler=self.random_seed,
                                           seed_simulator=self.random_seed, shots=1024,
                                           backend_options={
                                               'initial_statevector': [.5, .5, .5, .5]})
        res_w_bo = quantum_instance.execute(self.qc).get_counts(self.qc)
        self.assertGreater(quantum_instance.time_taken, 0.)
        quantum_instance.reset_execution_results()
        quantum_instance.skip_qobj_validation = True
        res_w_bo_skip_validation = quantum_instance.execute(self.qc).get_counts(self.qc)
        self.assertGreater(quantum_instance.time_taken, 0.)
        quantum_instance.reset_execution_results()
        self.assertTrue(_compare_dict(res_w_bo, res_w_bo_skip_validation))

    def test_w_noise(self):
        """ with noise test """
        # build noise model
        # Asymmetric readout error on qubit-0 only
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit.providers.aer.noise import NoiseModel
            from qiskit import Aer
            self.backend = Aer.get_backend('qasm_simulator')
        except ImportError as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return

        probs_given0 = [0.9, 0.1]
        probs_given1 = [0.3, 0.7]
        noise_model = NoiseModel()
        noise_model.add_readout_error([probs_given0, probs_given1], [0])

        quantum_instance = QuantumInstance(self.backend, seed_transpiler=self.random_seed,
                                           seed_simulator=self.random_seed, shots=1024,
                                           noise_model=noise_model)
        res_w_noise = quantum_instance.execute(self.qc).get_counts(self.qc)

        quantum_instance.skip_qobj_validation = True
        res_w_noise_skip_validation = quantum_instance.execute(self.qc).get_counts(self.qc)
        self.assertTrue(_compare_dict(res_w_noise, res_w_noise_skip_validation))

        # BasicAer should fail:
        with self.assertRaises(AquaError):
            _ = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                noise_model=noise_model)

        with self.assertRaises(AquaError):
            quantum_instance = QuantumInstance(BasicAer.get_backend('qasm_simulator'))
            quantum_instance.set_config(noise_model=noise_model)


if __name__ == '__main__':
    unittest.main()
