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

"""Test full pipeline"""
from qiskit import QuantumCircuit, transpile
from qiskit.test import QiskitTestCase
from qiskit.mitigation.mthree import M3Mitigation
from qiskit.test.mock import FakeAthens


class TestFullPipeline(QiskitTestCase):
    """Test pipeline"""

    def test_sim_pipeline(self):
        """Test that the full pipeline does not break"""

        backend = FakeAthens()

        qc = QuantumCircuit(5)
        qc.h(2)
        qc.cx(2, 1)
        qc.cx(2, 3)
        qc.cx(1, 0)
        qc.cx(3, 4)
        qc.measure_all()

        trans_qc = transpile(qc, backend)
        job = backend.run(trans_qc)
        raw_counts = job.result().get_counts()
        mit = mthree.M3Mitigation(backend)
        mit.tensored_cals_from_system()
        mit_counts = mit.apply_correction(raw_counts, qubits=range(5))

        assert mit_counts is not None
