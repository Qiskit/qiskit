# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin
# pylint: disable=unused-import

"""Tests for the converters."""

import unittest

from qiskit.converters import qobj_to_circuits
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import BasicAer
from qiskit import compile

from qiskit.qobj import Qobj
from qiskit.transpiler import PassManager
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
import qiskit.extensions.simulator


class TestQobjToCircuits(QiskitTestCase):
    """Test Qobj to Circuits."""

    def setUp(self):
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        self.circuit = QuantumCircuit(qr, cr)
        self.circuit.ccx(qr[0], qr[1], qr[2])
        self.circuit.measure(qr, cr)
        self.dag = circuit_to_dag(self.circuit)

    def test_qobj_to_circuits_single(self):
        """Check that qobj_to_circuits's result matches the qobj ini."""
        backend = BasicAer.get_backend('qasm_simulator')
        qobj_in = compile(self.circuit, backend, pass_manager=PassManager())
        out_circuit = qobj_to_circuits(qobj_in)
        self.assertEqual(circuit_to_dag(out_circuit[0]), self.dag)

    def test_qobj_to_circuits_multiple(self):
        """Check that qobj_to_circuits's result with multiple circuits"""
        backend = BasicAer.get_backend('qasm_simulator')
        qreg1 = QuantumRegister(2)
        qreg2 = QuantumRegister(3)
        creg1 = ClassicalRegister(2)
        creg2 = ClassicalRegister(2)
        circuit_b = QuantumCircuit(qreg1, qreg2, creg1, creg2)
        circuit_b.x(qreg1)
        circuit_b.h(qreg2)
        circuit_b.measure(qreg1, creg1)
        circuit_b.measure(qreg2[0], creg2[1])
        qobj = compile([self.circuit, circuit_b], backend, pass_manager=PassManager())
        dag_list = [circuit_to_dag(x) for x in qobj_to_circuits(qobj)]
        self.assertEqual(dag_list, [self.dag, circuit_to_dag(circuit_b)])

    def test_qobj_to_circuit_with_parameters(self):
        """Check qobj_to_circuit result with a gate that uses parameters."""
        backend = BasicAer.get_backend('qasm_simulator')
        qreg1 = QuantumRegister(2)
        qreg2 = QuantumRegister(3)
        creg1 = ClassicalRegister(2)
        creg2 = ClassicalRegister(2)
        circuit_b = QuantumCircuit(qreg1, qreg2, creg1, creg2)
        circuit_b.x(qreg1)
        circuit_b.h(qreg2)
        circuit_b.u2(0.2, 0.57, qreg2[1])
        circuit_b.measure(qreg1, creg1)
        circuit_b.measure(qreg2[0], creg2[1])
        qobj = compile(circuit_b, backend, pass_manager=PassManager())
        out_circuit = qobj_to_circuits(qobj)
        self.assertEqual(circuit_to_dag(out_circuit[0]),
                         circuit_to_dag(circuit_b))

    def test_qobj_to_circuit_with_sim_instructions(self):
        """Check qobj_to_circuit result with asimulator instruction."""
        backend = BasicAer.get_backend('qasm_simulator')
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        circuit = QuantumCircuit(qr, cr)
        circuit.ccx(qr[0], qr[1], qr[2])
        circuit.snapshot(1)
        circuit.measure(qr, cr)
        dag = circuit_to_dag(circuit)
        qobj_in = compile(circuit, backend, pass_manager=PassManager())
        out_circuit = qobj_to_circuits(qobj_in)
        self.assertEqual(circuit_to_dag(out_circuit[0]), dag)

    def test_qobj_to_circuits_with_nothing(self):
        """Verify that qobj_to_circuits returns None without any data."""
        qobj = Qobj('abc123', {}, {}, {})
        self.assertIsNone(qobj_to_circuits(qobj))

    def test_qobj_to_circuits_single_no_qasm(self):
        """Check that qobj_to_circuits's result matches the qobj ini."""
        backend = BasicAer.get_backend('qasm_simulator')
        qobj_in = compile(self.circuit, backend, pass_manager=PassManager())
        out_circuit = qobj_to_circuits(qobj_in)
        self.assertEqual(circuit_to_dag(out_circuit[0]), self.dag)

    def test_qobj_to_circuits_multiple_no_qasm(self):
        """Check that qobj_to_circuits's result with multiple circuits"""
        backend = BasicAer.get_backend('qasm_simulator')
        qreg1 = QuantumRegister(2)
        qreg2 = QuantumRegister(3)
        creg1 = ClassicalRegister(2)
        creg2 = ClassicalRegister(2)
        circuit_b = QuantumCircuit(qreg1, qreg2, creg1, creg2)
        circuit_b.x(qreg1)
        circuit_b.h(qreg2)
        circuit_b.measure(qreg1, creg1)
        circuit_b.measure(qreg2[0], creg2[1])
        qobj = compile([self.circuit, circuit_b], backend,
                       pass_manager=PassManager())

        dag_list = [circuit_to_dag(x) for x in qobj_to_circuits(qobj)]
        self.assertEqual(dag_list, [self.dag, circuit_to_dag(circuit_b)])


if __name__ == '__main__':
    unittest.main(verbosity=2)
