# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin

"""Tests for the wrapper functionality."""

import logging

import fixtures
import testtools

import qiskit.wrapper
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer
from qiskit import compile

from qiskit.qobj import Qobj

LOG = logging.getLogger(__name__)


class TestWrapper(testtools.TestCase):
    """Wrapper test case."""
    def setUp(self):
        super(TestWrapper, self).setUp()
        self.useFixture(fixtures.LoggerFixture())
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        self.circuit = QuantumCircuit(qr, cr)
        self.circuit.ccx(qr[0], qr[1], qr[2])
        self.circuit.measure(qr, cr)

    def test_qobj_to_circuits_single(self):
        """Check that qobj_to_circuits's result matches the qobj ini."""
        backend = Aer.get_backend('qasm_simulator')
        qobj_in = compile(self.circuit, backend, skip_transpiler=True)
        out_circuit = qiskit.wrapper.qobj_to_circuits(qobj_in)
        self.assertEqual(out_circuit[0].qasm(), self.circuit.qasm())

    def test_qobj_to_circuits_multiple(self):
        """Check that qobj_to_circuits's result with multiple circuits"""
        backend = Aer.get_backend('qasm_simulator')
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
        qasm_list = [x.qasm() for x in qiskit.wrapper.qobj_to_circuits(qobj)]
        LOG.warning(qasm_list[1])
        qobj_exp = qobj.experiments[1]
        LOG.warning(str(qobj_exp.header.qubit_labels))
        LOG.warning(str(qobj_exp.header.compiled_circuit_qasm))
        LOG.warning(str(qobj_exp.header.clbit_labels))
        for i in qobj_exp.instructions:
            LOG.warning(str(i))
        LOG.warning(circuit_b.qasm())
        self.assertEqual(qasm_list, [self.circuit.qasm(), circuit_b.qasm()])

    def test_qobj_to_circuits_with_qobj_no_qasm(self):
        """Verify that qobj_to_circuits returns None without QASM."""
        qobj = Qobj('abc123', {}, {}, {})
        self.assertIsNone(qiskit.wrapper.qobj_to_circuits(qobj))
