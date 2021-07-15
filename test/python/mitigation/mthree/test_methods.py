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

"""Test M3 methods"""
from qiskit import QuantumCircuit, transpile
from qiskit.test import QiskitTestCase
from qiskit.mitigation.mthree import M3Mitigation
from qiskit.test.mock import FakeAthens


class TestM3Methods(QiskitTestCase):
    """Test methods"""

    def test_me_equality(self):
        """Make sure direct and iterative solvers agree with each other."""
        qc = QuantumCircuit(5)
        qc.h(2)
        qc.cx(2, 1)
        qc.cx(1, 0)
        qc.cx(2, 3)
        qc.cx(3, 4)
        qc.measure_all()

        backend = FakeAthens()
        trans_qc = transpile(qc, backend)
        raw_counts = backend.run(trans_qc, shots=2048).result().get_counts()

        mit = M3Mitigation(backend)
        mit.tensored_cals_from_system()

        iter_q = mit.apply_correction(raw_counts, range(5), method="iterative")
        direct_q = mit.apply_correction(raw_counts, range(5), method="direct")

        for key, val in direct_q.items():
            self.assertTrue(key in iter_q.keys())
            self.assertTrue(abs(val - iter_q[key]) < 1e-5)

    def test_set_iterative(self):
        """Make sure can overload auto setting"""
        qc = QuantumCircuit(5)
        qc.h(2)
        qc.cx(2, 1)
        qc.cx(1, 0)
        qc.cx(2, 3)
        qc.cx(3, 4)
        qc.measure_all()

        backend = FakeAthens()
        trans_qc = transpile(qc, backend)
        raw_counts = backend.run(trans_qc, shots=4096).result().get_counts()

        mit = M3Mitigation(backend)
        mit.tensored_cals_from_system(shots=4096)

        _, details = mit.apply_correction(raw_counts, range(5), method="iterative", details=True)
        self.assertTrue(details["method"] == "iterative")

        _, details = mit.apply_correction(raw_counts, range(5), details=True)
        self.assertTrue(details["method"] == "direct")
