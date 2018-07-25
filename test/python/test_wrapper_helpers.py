# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""Tests for the wrapper functionality."""

import unittest

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.wrapper import execute, available_backends
from .common import QiskitTestCase


class TestWrapperHelpers(QiskitTestCase):
    """Suite for testing `compile`, `execute` and other execution helpers."""

    def setUp(self):
        q = QuantumRegister(3)
        c = ClassicalRegister(3)
        self.circuit = QuantumCircuit(q, c)
        self.circuit.ccx(q[0], q[1], q[2])

    def test_local_execute_and_get_ran_qasm(self):
        """Check if the local backend return the ran qasm."""

        for backend_name in available_backends({'local': True}):
            with self.subTest(backend_name=backend_name):
                result = execute(self.circuit, 'local_qasm_simulator').result()
                self.assertIsNotNone(result.get_ran_qasm(self.circuit.name))


if __name__ == '__main__':
    unittest.main(verbosity=2)
