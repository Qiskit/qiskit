# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""Quick program to test the qi tools modules."""

import json
import numpy

import qiskit
from qiskit.extensions.unitary import UnitaryGate
from qiskit.test import QiskitTestCase
from qiskit import BasicAer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.compiler import transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.passes import CXCancellation


class TestUnitaryGate(QiskitTestCase):
    """Tests for the Unitary class."""

    def setUp(self):
        """Setup."""
        pass

    def test_set_matrix(self):
        """Test instantiation"""
        try:
            UnitaryGate([[0, 1], [1, 0]])
        # pylint: disable=broad-except
        except Exception as err:
            self.fail('unexpected exception in init of Unitary: {0}'.format(err))

    def test_set_matrix_raises(self):
        """test non-unitary"""
        try:
            UnitaryGate([[1, 1], [1, 0]])
        # pylint: disable=broad-except
        except Exception:
            pass
        else:
            self.fail('setting Unitary with non-unitary did not raise')

    def test_set_init_with_unitary(self):
        """test instantiation of new unitary with another one (copy)"""
        uni1 = UnitaryGate([[0, 1], [1, 0]])
        uni2 = UnitaryGate(uni1)
        self.assertTrue(uni1 == uni2)
        self.assertFalse(uni1 is uni2)

    def test_conjugate(self):
        """test conjugate"""
        ymat = numpy.array([[0, -1j], [1j, 0]])
        uni = UnitaryGate([[0, 1j], [-1j, 0]])
        self.assertTrue(numpy.array_equal(uni.conjugate().to_matrix(), ymat))

    def test_adjoint(self):
        """test adjoint operation"""
        uni = UnitaryGate([[0, 1j], [-1j, 0]])
        self.assertTrue(numpy.array_equal(uni.adjoint().to_matrix(),
                                          uni.to_matrix()))


class TestUnitaryCircuit(QiskitTestCase):
    """Matrix gate circuit tests."""

    def test_1q_unitary(self):
        """test 1 qubit unitary matrix"""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)
        matrix = numpy.array([[1, 0], [0, 1]])
        qc.x(qr[0])
        qc.append(UnitaryGate(matrix), [qr[0]])
        # test of qasm output
        self.log.info(qc.qasm())
        # test of text drawer
        self.log.info(qc)
        dag = circuit_to_dag(qc)
        dag_nodes = dag.named_nodes('unitary')
        self.assertTrue(len(dag_nodes) == 1)
        dnode = dag_nodes[0]
        self.assertIsInstance(dnode.op, UnitaryGate)
        for qubit in dnode.qargs:
            self.assertTrue(qubit[1] in [0, 1])
        self.assertTrue(numpy.allclose(dnode.op.to_matrix(),
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
        uni2q = UnitaryGate(matrix)
        qc.append(uni2q, [qr[0], qr[1]])
        passman = PassManager()
        passman.append(CXCancellation())
        qc2 = transpile(qc, backend, pass_manager=passman)
        # test of qasm output
        self.log.info(qc2.qasm())
        # test of text drawer
        self.log.info(qc2)
        dag = circuit_to_dag(qc)
        nodes = dag.twoQ_gates()
        self.assertTrue(len(nodes) == 1)
        dnode = nodes[0]
        self.assertIsInstance(dnode.op, UnitaryGate)
        for qubit in dnode.qargs:
            self.assertTrue(qubit[1] in [0, 1])
        self.assertTrue(numpy.allclose(dnode.op.to_matrix(),
                                       matrix))
        qc3 = dag_to_circuit(dag)
        self.assertEqual(qc2, qc3)

    def test_3q_unitary(self):
        """test 3 qubit unitary matrix on non-consecutive bits"""
        qr = QuantumRegister(4)
        qc = QuantumCircuit(qr)
        sigmax = numpy.array([[0, 1], [1, 0]])
        sigmay = numpy.array([[0, -1j], [1j, 0]])
        matrix = numpy.kron(sigmay, numpy.kron(sigmax, sigmay))
        qc.x(qr[0])
        uni3q = UnitaryGate(matrix)
        qc.append(uni3q, [qr[0], qr[1], qr[3]])
        qc.cx(qr[3], qr[2])
        # test of text drawer
        self.log.info(qc)
        dag = circuit_to_dag(qc)
        nodes = dag.threeQ_or_more_gates()
        self.assertTrue(len(nodes) == 1)
        dnode = nodes[0]
        self.assertIsInstance(dnode.op, UnitaryGate)
        for qubit in dnode.qargs:
            self.assertTrue(qubit[1] in [0, 1, 3])
        self.assertTrue(numpy.allclose(dnode.op.to_matrix(),
                                       matrix))

    def test_qobj_with_unitary_matrix(self):
        """test qobj output with unitary matrix"""
        qr = QuantumRegister(4)
        qc = QuantumCircuit(qr)
        sigmax = numpy.array([[0, 1], [1, 0]])
        sigmay = numpy.array([[0, -1j], [1j, 0]])
        matrix = numpy.kron(sigmay, numpy.kron(sigmax, sigmay))
        qc.rx(numpy.pi/4, qr[0])
        uni = UnitaryGate(matrix)
        qc.append(uni, [qr[0], qr[1], qr[3]])
        qc.cx(qr[3], qr[2])
        qobj = qiskit.compiler.assemble(qc)
        instr = qobj.experiments[0].instructions[1]
        self.assertEqual(instr.name, 'unitary')
        self.assertTrue(numpy.allclose(
            numpy.array(instr.params).astype(numpy.complex64),
            matrix))
        # check conversion to dict
        qobj_dict = qobj.as_dict()
        # check json serialization
        self.assertTrue(isinstance(json.dumps(qobj_dict), str))

    def test_labeled_unitary(self):
        """test qobj output with unitary matrix"""
        qr = QuantumRegister(4)
        qc = QuantumCircuit(qr)
        sigmax = numpy.array([[0, 1], [1, 0]])
        sigmay = numpy.array([[0, -1j], [1j, 0]])
        matrix = numpy.kron(sigmax, sigmay)
        uni = UnitaryGate(matrix, label='xy')
        qc.append(uni, [qr[0], qr[1]])
        qobj = qiskit.compiler.assemble(qc)
        instr = qobj.experiments[0].instructions[0]
        self.assertEqual(instr.name, 'unitary')
        self.assertEqual(instr.label, 'xy')
