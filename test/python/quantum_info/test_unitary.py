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
from qiskit.quantum_info.operators.unitary import Unitary
from qiskit.test import QiskitTestCase
from qiskit import BasicAer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.transpiler import transpile, PassManager
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.passes import CXCancellation


class TestUnitary(QiskitTestCase):
    """Tests for the Unitary class."""

    def setUp(self):
        """Setup."""
        pass

    def test_set_matrix(self):
        """Test instantiation"""
        try:
            Unitary([[0, 1], [1, 0]])
        # pylint: disable=broad-except
        except Exception as err:
            self.fail('unexpected exception in init of Unitary: {0}'.format(err))

    def test_set_matrix_raises(self):
        """test non-unitary"""
        try:
            Unitary([[1, 1], [1, 0]])
        # pylint: disable=broad-except
        except Exception:
            pass
        else:
            self.fail('setting Unitary with non-unitary did not raise')

    def test_conjugate(self):
        """test conjugate"""
        ymat = numpy.array([[0, -1j], [1j, 0]])
        uni = Unitary([[0, 1j], [-1j, 0]])
        self.assertTrue(numpy.array_equal(uni.conjugate().matrix, ymat))

    def test_conjugate_inplace(self):
        """test inplace conjugate"""
        ymat = numpy.array([[0, -1j], [1j, 0]])
        uni = Unitary([[0, 1j], [-1j, 0]])
        uni_conj = uni.conjugate(inplace=True)
        self.assertTrue(numpy.array_equal(uni.matrix, ymat))
        self.assertTrue(uni.matrix is uni_conj.matrix)

    def test_adjoint(self):
        """test adjoint operation"""
        uni = Unitary([[0, 1j], [-1j, 0]])
        self.assertTrue(numpy.array_equal(uni.adjoint().matrix, uni.matrix))

    def test_tensor(self):
        """test tensor product of unitaries"""
        sx = [[0, 1], [1, 0]]
        sy = [[0, -1j], [1j, 0]]
        ymat = numpy.kron(sx, numpy.kron(sy, sy))
        ux = Unitary(sx)
        uy = Unitary(sy)
        result = ux.tensor(uy.tensor(uy))
        xmat = result.matrix
        self.assertTrue(numpy.array_equal(xmat, ymat))

    def test_expand(self):
        """test tensor product of unitaries with reverse order"""
        sx = [[0, 1], [1, 0]]
        sy = [[0, -1j], [1j, 0]]
        ymat = numpy.kron(sy, numpy.kron(sy, sx))
        ux = Unitary(sx)
        uy = Unitary(sy)
        result = ux.expand(uy.expand(uy))
        xmat = result.matrix
        self.assertTrue(numpy.array_equal(xmat, ymat))

    def test_compose(self):
        """test unitary composition"""
        sx = numpy.array([[0, 1], [1, 0]])
        sy = numpy.array([[0, -1j], [1j, 0]])
        ymat = sx @ sy
        ux = Unitary(sx)
        uy = Unitary(sy)
        result = (ux.compose(uy))
        xmat = result.matrix
        self.assertTrue(numpy.array_equal(xmat, ymat))

    def test_power(self):
        """test unitary power"""
        uy = Unitary(numpy.array([[0, -1j], [1j, 0]]))
        self.assertTrue(numpy.array_equal(uy.power(0).matrix, numpy.identity(2)))
        self.assertTrue(numpy.array_equal(uy.power(1).matrix, uy.matrix))
        self.assertTrue(numpy.array_equal(uy.power(2).matrix, numpy.identity(2)))
        self.assertTrue(numpy.array_equal(uy.power(-1).matrix,
                                          numpy.linalg.matrix_power(uy.matrix, -1)))


class TestUnitaryCircuit(QiskitTestCase):
    """Matrix gate circuit tests."""

    def test_1q_unitary(self):
        """test 1 qubit unitary matrix"""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)
        matrix = numpy.array([[1, 0], [0, 1]])
        qc.x(qr[0])
        uni1q = Unitary(matrix, qr[0])
        qc += uni1q
        # test of qasm output
        self.log.info(qc.qasm())
        # test of text drawer
        self.log.info(qc)
        dag = circuit_to_dag(qc)
        node_ids = dag.named_nodes('unitary')
        self.assertTrue(len(node_ids) == 1)
        dnode = dag.multi_graph.node[node_ids[0]]
        self.assertIsInstance(dnode['op'], Unitary)
        for qubit in dnode['qargs']:
            self.assertTrue(qubit[1] in [0, 1])
        self.assertTrue(numpy.allclose(dnode['op'].matrix,
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
        uni2q = Unitary(matrix, qr[0], qr[1])
        qc += uni2q
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
        self.assertIsInstance(dnode['op'], Unitary)
        for qubit in dnode['qargs']:
            self.assertTrue(qubit[1] in [0, 1])
        self.assertTrue(numpy.allclose(dnode['op'].matrix,
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
        uni3q = Unitary(matrix, qr[0], qr[1], qr[3])
        qc += uni3q
        qc.cx(qr[3], qr[2])
        # test of text drawer
        self.log.info(qc)
        dag = circuit_to_dag(qc)
        nodes = dag.threeQ_or_more_nodes()
        self.assertTrue(len(nodes) == 1)
        dnode = nodes[0][1]
        self.assertIsInstance(dnode['op'], Unitary)
        for qubit in dnode['qargs']:
            self.assertTrue(qubit[1] in [0, 1, 3])
        self.assertTrue(numpy.allclose(dnode['op'].matrix,
                                       matrix))

    def test_qobj_with_unitary_matrix(self):
        """test qobj output with unitary matrix"""
        qr = QuantumRegister(4)
        qc = QuantumCircuit(qr)
        sigmax = numpy.array([[0, 1], [1, 0]])
        sigmay = numpy.array([[0, -1j], [1j, 0]])
        matrix = numpy.kron(sigmay, numpy.kron(sigmax, sigmay))
        qc.rx(numpy.pi/4, qr[0])
        uni = Unitary(matrix, qr[0], qr[1], qr[3])
        qc += uni
        qc.cx(qr[3], qr[2])
        qobj = qiskit.compiler.assemble_circuits(qc)
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
        uni = Unitary(matrix, qr[0], qr[1], label='xy')
        qc += uni
        qobj = qiskit.compiler.assemble_circuits(qc)
        instr = qobj.experiments[0].instructions[0]
        self.assertEqual(instr.name, 'unitary')
        self.assertEqual(instr.label, 'xy')
