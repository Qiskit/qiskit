# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring,invalid-name

"""Quick program to test json backend
"""
import unittest
import qiskit
from qiskit import unroll
from .common import QiskitTestCase, Path


class TestJsonOutput(QiskitTestCase):
    """Test Json output.
    This is mostly covered in test_quantumprogram.py but will leave
    here for convenience.
    """
    def setUp(self):
        self.qasm_file_path = self._get_resource_path(
            'qasm/entangled_registers.qasm', Path.EXAMPLES)

    def test_json_output(self):
        circuit = qiskit.load_qasm_file(self.qasm_file_path)

        basis_gates = []  # unroll to base gates, change to test
        unroller = unroll.Unroller(circuit.qasm().parse(),
                                   unroll.JsonBackend(basis_gates))
        json_circuit = unroller.execute()
        self.log.info('test_json_ouptut: %s', json_circuit)


if __name__ == '__main__':
    unittest.main()
