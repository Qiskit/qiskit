# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""Test M3 quasi attributes"""
from qiskit import QuantumCircuit, transpile
from qiskit.test import QiskitTestCase
from qiskit.mitigation.mthree import M3Mitigation
from qiskit.test.mock import FakeCasablanca


class TestQuasi(QiskitTestCase):
    """Test quasiprobs"""

    def test_quasi_attr_set(self):
        """Test quasi-probs attributes are set"""
        backend = FakeCasablanca()

        N = 6
        qc = QuantumCircuit(N)
        qc.x(range(0, N))
        qc.h(range(0, N))
        for kk in range(N // 2, 0, -1):
            qc.ch(kk, kk - 1)
        for kk in range(N // 2, N - 1):
            qc.ch(kk, kk + 1)
        qc.measure_all()

        mit = M3Mitigation(backend)
        mit.tensored_cals_from_system(qubits)

        trans_qc = transpile(qc, backend)

        for shots in [1000, 2000, 1234]:
            raw_counts = backend.run(trans_qc, shots=shots).result().get_counts()

            quasi1 = mit.apply_correction(
                raw_counts, qubits, return_mitigation_overhead=True, method="direct"
            )
            quasi2 = mit.apply_correction(
                raw_counts, qubits, return_mitigation_overhead=True, method="iterative"
            )

            quasi3 = mit.apply_correction(
                raw_counts, qubits, return_mitigation_overhead=False, method="direct"
            )
            quasi4 = mit.apply_correction(
                raw_counts, qubits, return_mitigation_overhead=False, method="iterative"
            )

            self.assertTrue(quasi1.shots == shots)
            self.assertTrue(quasi2.shots == shots)
            self.assertTrue(quasi3.shots == shots)
            self.assertTrue(quasi4.shots == shots)

            self.assertTrue(quasi1.mitigation_overhead is not None)
            self.assertTrue(quasi2.mitigation_overhead is not None)
            self.assertTrue(quasi3.mitigation_overhead is None)
            self.assertTrue(quasi4.mitigation_overhead is None)
