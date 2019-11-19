# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Longest Path pass testing"""

import unittest

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.compiler import transpile
from qiskit.transpiler.passes import CountOpsLongestPath
from qiskit.test import QiskitTestCase


class TestCountOpsDeepestPathPass(QiskitTestCase):
    """ Tests for CountOpsLongestPath analysis methods. """

    def test_empty_dag(self):
        """ Empty DAG has empty counts."""
        circuit = QuantumCircuit()

        passmanager = PassManager()
        passmanager.append(CountOpsLongestPath())
        _ = transpile(circuit, pass_manager=passmanager)

        self.assertDictEqual(passmanager.property_set['count_ops_longest_path'], {})

    def test_op_times(self):
        """ A dag with different length operations, where longest path depends
        on operation times dictionary"""
        op_times1 = {
            'h': 1,
            'cx': 1
        }
        op_times2 = {
            'h': 1,
            'cx': 4
        }

        qr = QuantumRegister(4)
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[1])
        circuit.h(qr[1])
        circuit.h(qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[2], qr[3])

        passmanager = PassManager()
        passmanager.append(CountOpsLongestPath(op_times1))
        _ = transpile(circuit, pass_manager=passmanager)

        count_ops1 = passmanager.property_set['count_ops_longest_path']
        self.assertEqual(count_ops1, {'cx': 3, 'h': 6})

        passmanager = PassManager()
        passmanager.append(CountOpsLongestPath(op_times2))
        _ = transpile(circuit, pass_manager=passmanager)

        count_ops2 = passmanager.property_set['count_ops_longest_path']
        self.assertEqual(count_ops2, {'cx': 5})


if __name__ == '__main__':
    unittest.main()
