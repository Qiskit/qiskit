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

"""Runtime pass testing"""

import unittest

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.compiler import transpile
from qiskit.transpiler.passes import Runtime
from qiskit.test import QiskitTestCase


class TestRuntimePass(QiskitTestCase):
    """ Tests for Runtime analysis methods. """

    def test_empty_dag(self):
        """ Empty DAG has 0 runtime"""
        circuit = QuantumCircuit()

        passmanager = PassManager()
        passmanager.append(Runtime())
        _ = transpile(circuit, pass_manager=passmanager)

        self.assertEqual(passmanager.property_set['runtime'], 0)

    def test_only_measure(self):
        """ A dag with only measurements"""
        op_times = {
            'barrier': 0,
            'measure': 1
        }

        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.measure_all()

        passmanager = PassManager()
        passmanager.append(Runtime(op_times))
        _ = transpile(circuit, pass_manager=passmanager)

        self.assertEqual(passmanager.property_set['runtime'], 1)

    def test_no_optimes(self):
        """ A dag with 2 parallel operations and no classic bits"""

        qr = QuantumRegister(1)
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])

        passmanager = PassManager()
        passmanager.append(Runtime())
        _ = transpile(circuit, pass_manager=passmanager)

        self.assertEqual(passmanager.property_set['runtime'], 1)

    def test_key_error(self):
        """ KeyError test, if operation is not in op_times dictionary"""
        op_times = {
            'h': 1
        }

        qr = QuantumRegister(1)
        circuit = QuantumCircuit(qr)
        circuit.x(qr[0])

        passmanager = PassManager()
        passmanager.append(Runtime(op_times))

        self.assertRaises(KeyError, transpile, circuit, pass_manager=passmanager)

    def test_parallel_ops(self):
        """ A dag with 2 parallel operations and no classic bits"""
        op_times = {
            'h': 1
        }

        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[1])

        passmanager = PassManager()
        passmanager.append(Runtime(op_times))
        _ = transpile(circuit, pass_manager=passmanager)

        self.assertEqual(passmanager.property_set['runtime'], 1)

    def test_longest_path(self):
        """ A dag with operations on alternating qubits, to test finding
        of longest path"""
        op_times = {
            'h': 1,
            'cx': 1
        }

        qr = QuantumRegister(2)
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

        passmanager = PassManager()
        passmanager.append(Runtime(op_times))
        _ = transpile(circuit, pass_manager=passmanager)

        self.assertEqual(passmanager.property_set['runtime'], 9)

    def test_op_times(self):
        """ A dag with different length operations, where longest path is
        different from path with maximum number of operations"""
        op_times = {
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
        passmanager.append(Runtime(op_times))
        _ = transpile(circuit, pass_manager=passmanager)

        self.assertEqual(passmanager.property_set['runtime'], 20)


if __name__ == '__main__':
    unittest.main()
