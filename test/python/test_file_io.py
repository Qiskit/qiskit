# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for qiskit.Result"""

import os
import unittest
import qiskit
from qiskit.wrapper import execute
from qiskit.tools import file_io
from .common import QiskitTestCase


class TestFileIO(QiskitTestCase):
    """Test file_io utilities."""

    def test_results_save_load(self):
        """Test saving and loading the results of a circuit.

        Test for the 'unitary_simulator' and 'qasm_simulator'
        """
        metadata = {'testval': 5}
        qr = qiskit.QuantumRegister(2)
        cr = qiskit.QuantumRegister(2)
        qc1 = qiskit.QuantumCircuit(qr, cr, name='qc1')
        qc2 = qiskit.QuantumCircuit(qr, cr, name='qc2')
        qc1.h(qr)
        qc2.cx(qr[0], qr[1])
        circuits = [qc1, qc2]

        result1 = execute(circuits, backend='unitary_simulator').result()
        result2 = execute(circuits, backend='qasm_simulator').result()

        test_1_path = self._get_resource_path('test_save_load1.json')
        test_2_path = self._get_resource_path('test_save_load2.json')

        # delete these files if they exist
        if os.path.exists(test_1_path):
            os.remove(test_1_path)

        if os.path.exists(test_2_path):
            os.remove(test_2_path)

        file1 = file_io.save_result_to_file(result1, test_1_path, metadata=metadata)
        file2 = file_io.save_result_to_file(result2, test_2_path, metadata=metadata)

        _, metadata_loaded1 = file_io.load_result_from_file(file1)
        _, metadata_loaded2 = file_io.load_result_from_file(file1)

        self.assertAlmostEqual(metadata_loaded1['testval'], 5)
        self.assertAlmostEqual(metadata_loaded2['testval'], 5)

        # remove files to keep directory clean
        os.remove(file1)
        os.remove(file2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
