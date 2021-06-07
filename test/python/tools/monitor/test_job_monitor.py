# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the wrapper functionality."""

import io
import unittest
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import BasicAer
from qiskit import execute
from qiskit.tools.monitor import job_monitor
from qiskit.test import QiskitTestCase


class TestJobMonitor(QiskitTestCase):
    """Tools test case."""

    def test_job_monitor(self):
        """Test job_monitor"""
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(2)
        qc = QuantumCircuit(qreg, creg)
        qc.h(qreg[0])
        qc.cx(qreg[0], qreg[1])
        qc.measure(qreg, creg)
        backend = BasicAer.get_backend("qasm_simulator")
        job_sim = execute([qc] * 10, backend)
        output = io.StringIO()
        job_monitor(job_sim, output=output)
        self.assertEqual(job_sim.status().name, "DONE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
