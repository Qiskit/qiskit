# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the passmanager logic"""

import copy

import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler import PropertySet
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import CommutativeCancellation
from qiskit.transpiler.passes import Optimize1qGates, Unroller
from qiskit.test import QiskitTestCase


class TestPassManager(QiskitTestCase):
    """Test Pass manager logic."""

    def test_callback(self):
        """Test the callback parameter."""
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr, name='MyCircuit')
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.h(qr[0])
        expected_start = QuantumCircuit(qr)
        expected_start.u2(0, np.pi, qr[0])
        expected_start.u2(0, np.pi, qr[0])
        expected_start.u2(0, np.pi, qr[0])
        expected_start_dag = circuit_to_dag(expected_start)

        expected_end = QuantumCircuit(qr)
        expected_end.u2(0, np.pi, qr[0])
        expected_end_dag = circuit_to_dag(expected_end)

        calls = []

        def callback(**kwargs):
            out_dict = kwargs
            out_dict['dag'] = copy.deepcopy(kwargs['dag'])
            calls.append(out_dict)

        passmanager = PassManager()
        passmanager.append(Unroller(['u2']))
        passmanager.append(Optimize1qGates())
        passmanager.run(circuit, callback=callback)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(calls[0]), 5)
        self.assertEqual(calls[0]['count'], 0)
        self.assertEqual(calls[0]['pass_'].name(), 'Unroller')
        self.assertEqual(expected_start_dag, calls[0]['dag'])
        self.assertIsInstance(calls[0]['time'], float)
        self.assertEqual(calls[0]['property_set'], PropertySet())
        self.assertEqual('MyCircuit', calls[0]['dag'].name)
        self.assertEqual(len(calls[1]), 5)
        self.assertEqual(calls[1]['count'], 1)
        self.assertEqual(calls[1]['pass_'].name(), 'Optimize1qGates')
        self.assertEqual(expected_end_dag, calls[1]['dag'])
        self.assertIsInstance(calls[0]['time'], float)
        self.assertEqual(calls[0]['property_set'], PropertySet())
        self.assertEqual('MyCircuit', calls[1]['dag'].name)

    def test_callback_with_pass_requires(self):
        """Test the callback with a pass with another pass requirement."""
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr, name='MyCircuit')
        circuit.z(qr[0])
        circuit.cx(qr[0], qr[2])
        circuit.z(qr[0])
        expected_start = QuantumCircuit(qr)
        expected_start.z(qr[0])
        expected_start.cx(qr[0], qr[2])
        expected_start.z(qr[0])
        expected_start_dag = circuit_to_dag(expected_start)

        expected_end = QuantumCircuit(qr)
        expected_end.cx(qr[0], qr[2])
        expected_end_dag = circuit_to_dag(expected_end)

        calls = []

        def callback(**kwargs):
            out_dict = kwargs
            out_dict['dag'] = copy.deepcopy(kwargs['dag'])
            calls.append(out_dict)

        passmanager = PassManager()
        passmanager.append(CommutativeCancellation())
        passmanager.run(circuit, callback=callback)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(calls[0]), 5)
        self.assertEqual(calls[0]['count'], 0)
        self.assertEqual(calls[0]['pass_'].name(), 'CommutationAnalysis')
        self.assertEqual(expected_start_dag, calls[0]['dag'])
        self.assertIsInstance(calls[0]['time'], float)
        self.assertIsInstance(calls[0]['property_set'], PropertySet)
        self.assertEqual('MyCircuit', calls[0]['dag'].name)
        self.assertEqual(len(calls[1]), 5)
        self.assertEqual(calls[1]['count'], 1)
        self.assertEqual(calls[1]['pass_'].name(), 'CommutativeCancellation')
        self.assertEqual(expected_end_dag, calls[1]['dag'])
        self.assertIsInstance(calls[0]['time'], float)
        self.assertIsInstance(calls[0]['property_set'], PropertySet)
        self.assertEqual('MyCircuit', calls[1]['dag'].name)
