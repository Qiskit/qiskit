# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin

"""Tests for the wrapper functionality."""

import unittest

from qiskit.tools import qobj_to_circuits
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer
from qiskit import compile

from qiskit.qobj import Qobj
from qiskit.dagcircuit import DAGCircuit
from ..common import QiskitTestCase


class TestQobj2Circuits(QiskitTestCase):
    """Wrapper test case."""

    def setUp(self):
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        self.circuit = QuantumCircuit(qr, cr)
        self.circuit.ccx(qr[0], qr[1], qr[2])
        self.circuit.measure(qr, cr)
        self.dag = DAGCircuit.fromQuantumCircuit(self.circuit)

    def test_qobj_to_circuits_single(self):
        """Check that qobj_to_circuits's result matches the qobj ini."""
        backend = Aer.get_backend('qasm_simulator_py')
        qobj_in = compile(self.circuit, backend, skip_transpiler=True)
        out_circuit = qobj_to_circuits(qobj_in)
        self.assertEqual(DAGCircuit.fromQuantumCircuit(out_circuit[0]), self.dag)

    def test_qobj_to_circuits_multiple(self):
        """Check that qobj_to_circuits's result with multiple circuits"""
        backend = Aer.get_backend('qasm_simulator_py')
        qreg1 = QuantumRegister(2)
        qreg2 = QuantumRegister(3)
        creg1 = ClassicalRegister(2)
        creg2 = ClassicalRegister(2)
        circuit_b = QuantumCircuit(qreg1, qreg2, creg1, creg2)
        circuit_b.x(qreg1)
        circuit_b.h(qreg2)
        circuit_b.measure(qreg1, creg1)
        circuit_b.measure(qreg2[0], creg2[1])
        qobj = compile([self.circuit, circuit_b], backend, skip_transpiler=True)
        dag_list = [DAGCircuit.fromQuantumCircuit(x) for x in qobj_to_circuits(qobj)]
        self.assertEqual(dag_list, [self.dag, DAGCircuit.fromQuantumCircuit(circuit_b)])

    def test_qobj_to_circuits_with_qobj_no_qasm(self):
        """Verify that qobj_to_circuits returns None without QASM."""
        qobj = Qobj('abc123', {}, {}, {})
        self.assertIsNone(qobj_to_circuits(qobj))


if __name__ == '__main__':
    unittest.main(verbosity=2)
