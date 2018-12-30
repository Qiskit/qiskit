# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test Qiskit Unroller class."""

import unittest

from qiskit import qasm
from qiskit.unroll import DagUnroller, JsonBackend
from qiskit.converters import ast_to_dag
from qiskit.test import QiskitTestCase, Path


class UnrollerTest(QiskitTestCase):
    """Test the Unroller."""

    # We need to change the way we create clbit_labels and qubit_labels in order to
    # enable this test, as they are lists but the order is not important so comparing
    # them usually fails.
    @unittest.skip("Temporary skipping")
    def test_dag_to_json(self):
        """Test DagUnroller with JSON backend."""
        ast = qasm.Qasm(filename=self._get_resource_path('example.qasm', Path.QASMS)).parse()
        dag_circuit = ast_to_dag(ast)
        dag_unroller = DagUnroller(dag_circuit, JsonBackend())
        json_circuit = dag_unroller.execute()
        expected_result = {
            'operations':
                [
                    {'qubits': [5], 'texparams': ['0.5 \\pi', '0', '\\pi'],
                     'name': 'U', 'params': [1.5707963267948966, 0.0, 3.141592653589793]},
                    {'name': 'CX', 'qubits': [5, 2]},
                    {'clbits': [2], 'name': 'measure', 'qubits': [2]},
                    {'qubits': [4], 'texparams': ['0.5 \\pi', '0', '\\pi'], 'name': 'U',
                     'params': [1.5707963267948966, 0.0, 3.141592653589793]},
                    {'name': 'CX', 'qubits': [4, 1]},
                    {'clbits': [1], 'name': 'measure', 'qubits': [1]},
                    {'qubits': [3], 'texparams': ['0.5 \\pi', '0', '\\pi'], 'name': 'U',
                     'params': [1.5707963267948966, 0.0, 3.141592653589793]},
                    {'name': 'CX', 'qubits': [3, 0]},
                    {'name': 'barrier', 'qubits': [3, 4, 5]},
                    {'clbits': [5], 'name': 'measure', 'qubits': [5]},
                    {'clbits': [4], 'name': 'measure', 'qubits': [4]},
                    {'clbits': [3], 'name': 'measure', 'qubits': [3]},
                    {'clbits': [0], 'name': 'measure', 'qubits': [0]}
                ],
            'header':
                {
                    'memory_slots': 6,
                    'qubit_labels': [['r', 0], ['r', 1], ['r', 2], ['q', 0], ['q', 1], ['q', 2]],
                    'n_qubits': 6, 'clbit_labels': [['d', 3], ['c', 3]]
                }
        }

        self.assertEqual(json_circuit, expected_result)
