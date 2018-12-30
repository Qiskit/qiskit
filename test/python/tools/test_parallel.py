# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for qiskit/tools/parallel"""
import os
import time

from qiskit.tools.parallel import parallel_map
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.test import QiskitTestCase


def _parfunc(x):
    """Function for testing parallel_map
    """
    time.sleep(1)
    return x


def _build_simple(_):
    qreg = QuantumRegister(2)
    creg = ClassicalRegister(2)
    qc = QuantumCircuit(qreg, creg)
    return qc


class TestParallel(QiskitTestCase):
    """A class for testing parallel_map functionality.
    """

    def test_parallel_env_flag(self):
        """Verify parallel env flag is set """
        self.assertEqual(os.getenv('QISKIT_IN_PARALLEL', None), 'FALSE')

    def test_parallel(self):
        """Test parallel_map """
        ans = parallel_map(_parfunc, list(range(10)))
        self.assertEqual(ans, list(range(10)))

    def test_parallel_circuit_names(self):
        """Verify unique circuit names in parallel"""
        out_circs = parallel_map(_build_simple, list(range(10)))
        names = [circ.name for circ in out_circs]
        self.assertEqual(len(names), len(set(names)))
