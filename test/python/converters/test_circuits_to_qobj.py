# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

from qiskit.test import QiskitTestCase

from qiskit.converters import circuits_to_qobj
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


class TestCircuitsToQobj(QiskitTestCase):
    """Test Circuits to Qobj."""

    def test_no_run_config(self):
        """Test circuits_to_qobj_without_run_config_set."""
        q = QuantumRegister(2, 'q')
        c = ClassicalRegister(2, 'c')
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.cx(q[0], q[1])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[1])
        qobj = circuits_to_qobj(qc)
        expected = [{'name': 'h', 'qubits': [0]},
                    {'name': 'cx', 'qubits': [0, 1]},
                    {'name': 'measure', 'memory': [0], 'qubits': [0]},
                    {'name': 'measure', 'memory': [1], 'qubits': [1]}]
        self.assertEqual(qobj.to_dict()['experiments'][0]['instructions'],
                         expected)
