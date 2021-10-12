# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""Tests for error mitigation routines."""

import unittest
from typing import List
from ddt import ddt

from qiskit import execute
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.mitigation import CompleteReadoutMitigator

# For simulation
import qiskit.utils.mitigation as mit
from qiskit.result import Result
from qiskit.test import QiskitTestCase

try:
    from qiskit import Aer
    from qiskit.providers.aer import noise

    HAS_AER = True
except ImportError:
    HAS_AER = False


class NoisySimulationTest(QiskitTestCase):
    """Base class that contains methods and attributes
    for doing tests of readout error noise with flexible
    readout errors.
    """

    # Example max qubit number
    num_qubits = 4

    if HAS_AER:
        sim = Aer.get_backend("aer_simulator")

        # Create readout errors
        readout_errors = []
        for i in range(num_qubits):
            p_error1 = (i + 1) * 0.002
            p_error0 = 2 * p_error1
            ro_error = noise.ReadoutError([[1 - p_error0, p_error0], [p_error1, 1 - p_error1]])
            readout_errors.append(ro_error)

        # Readout Error only
        noise_model = noise.NoiseModel()
        for i in range(num_qubits):
            noise_model.add_readout_error(readout_errors[i], [i])

    seed_simulator = 100
    shots = 10000
    tolerance = 0.05

    def execute_circs(self, qc_list: List[QuantumCircuit], noise_model=None) -> Result:
        """Run circuits with the readout noise defined in this class"""
        backend = self.sim
        return backend.run(
            qc_list,
            shots=self.shots,
            seed_simulator=self.seed_simulator,
            noise_model=None if noise_model is None else self.noise_model,
            method="density_matrix",
        ).result()

    def compare_results(self, res1, res2):
        """Compare the results between two runs"""
        res1_total_shots = sum(res1.values())
        res2_total_shots = sum(res2.values())
        keys = set(res1.keys()).union(set(res2.keys()))
        total = 0
        for key in keys:
            val1 = res1.get(key, 0) / res1_total_shots
            val2 = res2.get(key, 0) / res2_total_shots
            total += abs(val1 - val2) ** 2
        return total


@ddt
class TestReadoutMitigation(NoisySimulationTest):
    """Testing standard readout mitigation"""

    def setUp(self):
        qc = QuantumCircuit(self.num_qubits)
        qc.h(0)
        for i in range(1, self.num_qubits):
            qc.cx(i - 1, i)
        qc.measure_all()
        result_targ = self.execute_circs(qc)
        result_noise = self.execute_circs(qc, noise_model=self.noise_model)
        self.counts_ideal = result_targ.get_counts(0)
        self.counts_noise = result_noise.get_counts(0)
        self.set_mitigation_matrix()

    def _set_mitigation_matrix(self):
        qr = QuantumRegister(self.num_qubits)
        qubit_list = range(self.num_qubits)
        meas_calibs, state_labels = mit.complete_meas_cal(
            qubit_list=qubit_list, qr=qr, circlabel="mcal"
        )
        cal_res = execute(
            meas_calibs,
            self.sim,
            shots=self.shots,
            seed_simulator=self.seed_simulator,
            basis_gates=self.noise_model.basis_gates,
            noise_model=self.noise_model,
        ).result()

        meas_fitter = mit.CompleteMeasFitter(
            cal_res, state_labels, qubit_list=qubit_list, circlabel="mcal"
        )
        self.mat = meas_fitter.cal_matrix

    @unittest.skipUnless(HAS_AER, "qiskit-aer is required for this test")
    def test_mitigation_improvement(self):
        """Test whether readout mitigation led to more accurate results"""
        unmitigated_error = self.compare_results(self.counts_ideal, self.counts_noise)
        CRM = CompleteReadoutMitigator(self.mat)
        mitigated_counts = (
            CRM.quasi_probabilities(self.counts_noise)[0]
            .nearest_probability_distribution()
            .binary_probabilities()
        )
        mitigated_error = self.compare_results(self.counts_ideal, mitigated_counts)
        self.assertTrue(mitigated_error < unmitigated_error * 0.1)


if __name__ == "__main__":
    unittest.main()
