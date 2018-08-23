# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring

"""Quick program to test json backend
"""
import unittest
import qiskit
from qiskit.dagcircuit._dagcircuit import DAGCircuit
from qiskit.unroll import DagUnroller, JsonBackend


class TestJsonOutput(QiskitTestCase):
    """Test Json output.
    """
    def setUp(self):
        self.qasm_file_path = self._get_resource_path(
            'qasm/entangled_registers.qasm', Path.EXAMPLES)

    def test_json_output(self):
        circ = qiskit.load_qasm_file(self.QASM_FILE_PATH)
        dag_circuit = DAGCircuit.fromQuantumCircuit(circ)
        json_circuit = DagUnroller(dag_circuit,
                                   JsonBackend(dag_circuit.basis)).execute()

    self.log.info('test_json_ouptut: %s', json_circuit)


if __name__ == '__main__':
    unittest.main()
