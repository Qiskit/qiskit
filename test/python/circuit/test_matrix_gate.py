# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=unused-import

"""Test matrix gates"""

import os
import tempfile
import unittest
import numpy

import qiskit.extensions.simulator
from qiskit import BasicAer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute
from qiskit import QiskitError
from qiskit.circuit import Gate
from qiskit.test import QiskitTestCase
from qiskit.transpiler import transpile, PassManager
from qiskit.transpiler.passes import BasicSwap, CXCancellation, Optimize1qGates
from qiskit.converters import circuit_to_dag
from qiskit.converters import circuits_to_qobj
from qiskit.extensions.standard.unitary_matrix import UnitaryMatrixGate


class TestMatrixGate(QiskitTestCase):
    """Matrix gate tests."""

    def test_1q_unitary(self):
        """test 1 qubit unitary matrix"""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)
        matrix = numpy.array([[1, 0], [0, 1]])
        qc.x(qr[0])
        qc.unitary(matrix, qr[0])
        # test of qasm output
        self.log.info(qc.qasm())
        # test of text drawer
        self.log.info(qc)
        dag = circuit_to_dag(qc)
        node_ids = dag.named_nodes('unitary')
        self.assertTrue(len(node_ids) == 1)
        dnode = dag.multi_graph.node[node_ids[0]]
        self.assertIsInstance(dnode['op'], UnitaryMatrixGate)
        for qubit in dnode['qargs']:
            self.assertTrue(qubit[1] in [0, 1])
        self.assertTrue(numpy.allclose(dnode['op'].matrix_rep,
                                       matrix))

    def test_2q_unitary(self):
        """test 2 qubit unitary matrix"""
        backend = BasicAer.get_backend('qasm_simulator')
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr)
        sigmax = numpy.array([[0, 1], [1, 0]])
        sigmay = numpy.array([[0, -1j], [1j, 0]])
        matrix = numpy.kron(sigmax, sigmay)
        qc.x(qr[0])
        qc.unitary(matrix, qr[0], qr[1])
        passman = PassManager()
        passman.append(CXCancellation())
        qc2 = transpile(qc, backend, pass_manager=passman)
        # test of qasm output
        self.log.info(qc2.qasm())
        # test of text drawer
        self.log.info(qc2)
        dag = circuit_to_dag(qc)
        nodes = dag.twoQ_nodes()
        self.assertTrue(len(nodes) == 1)
        dnode = nodes[0]
        self.assertIsInstance(dnode['op'], UnitaryMatrixGate)
        for qubit in dnode['qargs']:
            self.assertTrue(qubit[1] in [0, 1])
        self.assertTrue(numpy.allclose(dnode['op'].matrix_rep,
                                       matrix))

    def test_3q_unitary(self):
        """test 3 qubit unitary matrix on non-consecutive bits"""
        qr = QuantumRegister(4)
        qc = QuantumCircuit(qr)
        sigmax = numpy.array([[0, 1], [1, 0]])
        sigmay = numpy.array([[0, -1j], [1j, 0]])
        matrix = numpy.kron(sigmay, numpy.kron(sigmax, sigmay))
        qc.x(qr[0])
        qc.unitary(matrix, qr[0], qr[1], qr[3])
        qc.cx(qr[3], qr[2])
        # test of qasm output
        self.log.info(qc.qasm())
        # test of text drawer
        self.log.info(qc)
        dag = circuit_to_dag(qc)
        nodes = dag.threeQ_or_more_nodes()
        self.assertTrue(len(nodes) == 1)
        dnode = nodes[0][1]
        self.assertIsInstance(dnode['op'], UnitaryMatrixGate)
        for qubit in dnode['qargs']:
            self.assertTrue(qubit[1] in [0, 1, 3])
        self.assertTrue(numpy.allclose(dnode['op'].matrix_rep,
                                       matrix))

    def test_qobj_with_unitary_matrix(self):
        """test qobj output with unitary matrix"""
        qr = QuantumRegister(4)
        qc = QuantumCircuit(qr)
        sigmax = numpy.array([[0, 1], [1, 0]])
        sigmay = numpy.array([[0, -1j], [1j, 0]])
        matrix = numpy.kron(sigmay, numpy.kron(sigmax, sigmay))
        qc.x(qr[0])
        qc.unitary(matrix, qr[0], qr[1], qr[3])
        qc.cx(qr[3], qr[2])
        qobj = circuits_to_qobj(qc)
        instr = qobj.experiments[0].instructions[1]
        self.assertEqual(instr.name, 'unitary')
        self.assertTrue(numpy.allclose(
            numpy.array(instr.params).astype(numpy.complex64),
            matrix))
