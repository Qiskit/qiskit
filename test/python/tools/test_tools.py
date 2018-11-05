# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin

"""Tests for the wrapper functionality."""

import unittest

import qiskit.tools
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer
from qiskit import execute
from ..common import QiskitTestCase


class TestTools(QiskitTestCase):
    """Tools test case."""
    def test_job_monitor(self):
        """Test job_monitor"""
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(2)
        qc = QuantumCircuit(qreg, creg)
        qc.h(qreg[0])
        qc.cx(qreg[0], qreg[1])
        qc.measure(qreg, creg)
        backend = Aer.get_backend('qasm_simulator')
        job_sim = execute([qc]*10, backend)
        qiskit.tools.job_monitor(job_sim)
        self.assertEqual(job_sim.status().name, 'DONE')


if __name__ == '__main__':
    unittest.main(verbosity=2)
