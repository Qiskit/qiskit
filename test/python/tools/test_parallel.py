# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for qiskit/tools/parallel"""
import os
import time

from qiskit.tools.parallel import parallel_map
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.pulse import Schedule
from qiskit.test import QiskitTestCase


def _parfunc(x):
    """Function for testing parallel_map"""
    time.sleep(1)
    return x


def _build_simple_circuit(_):
    qreg = QuantumRegister(2)
    creg = ClassicalRegister(2)
    qc = QuantumCircuit(qreg, creg)
    return qc


def _build_simple_schedule(_):
    return Schedule()


class TestParallel(QiskitTestCase):
    """A class for testing parallel_map functionality."""

    def test_parallel_env_flag(self):
        """Verify parallel env flag is set"""
        self.assertEqual(os.getenv("QISKIT_IN_PARALLEL", None), "FALSE")

    def test_parallel(self):
        """Test parallel_map"""
        ans = parallel_map(_parfunc, list(range(10)))
        self.assertEqual(ans, list(range(10)))

    def test_parallel_circuit_names(self):
        """Verify unique circuit names in parallel"""
        out_circs = parallel_map(_build_simple_circuit, list(range(10)))
        names = [circ.name for circ in out_circs]
        self.assertEqual(len(names), len(set(names)))

    def test_parallel_schedule_names(self):
        """Verify unique schedule names in parallel"""
        out_schedules = parallel_map(_build_simple_schedule, list(range(10)))
        names = [schedule.name for schedule in out_schedules]
        self.assertEqual(len(names), len(set(names)))
