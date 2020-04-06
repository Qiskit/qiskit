# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=arguments-differ,method-hidden

"""Quick program to test the qi tools modules."""

import json
import numpy as np
from numpy.testing import assert_allclose

import qiskit
from qiskit.extensions.hamiltonian_gate import HamiltonianGate
from qiskit.test import QiskitTestCase
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager
from qiskit.compiler import transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.passes import CXCancellation


class TestHamiltonianGate(QiskitTestCase):
    """Tests for the HamiltonianGate class."""

    def setUp(self):
        """Setup."""
        pass

    def test_set_matrix(self):
        """Test instantiation"""
        try:
            HamiltonianGate([[0, 1], [1, 0]], 1)
        # pylint: disable=broad-except
        except Exception as err:
            self.fail('unexpected exception in init of HamiltonianGate: {0}'.format(err))

    def test_set_matrix_raises(self):
        """test non-unitary"""
        try:
            HamiltonianGate([[1, 0], [1, 1]], 1)
        # pylint: disable=broad-except
        except Exception:
            pass
        else:
            self.fail('setting Unitary with non-hermitian matrix did not raise')

    def test_conjugate(self):
        """test conjugate"""
        ymat = np.conj(np.array([[0, -1j], [1j, 0]]))
        ham = HamiltonianGate([[1, 1j], [-1j, 1]], np.pi / 2)
        np.testing.assert_array_almost_equal(ham.conjugate().to_matrix(),
                                             ymat)

    def test_adjoint(self):
        """test adjoint operation"""
        uni = HamiltonianGate([[0, 1j], [-1j, 0]], np.pi)
        np.testing.assert_array_almost_equal(uni.adjoint().to_matrix(),
                                             uni.to_matrix())


class TestHamiltonianCircuit(QiskitTestCase):
    """Hamiltonian gate circuit tests."""

    def test_1q_hamiltonian(self):
        """test 1 qubit hamiltonian"""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)
        matrix = np.zeros((2, 2))
        qc.x(qr[0])
        theta = Parameter('theta')
        qc.append(HamiltonianGate(matrix, theta), [qr[0]])
        qc = qc.bind_parameters({theta: 1})
        # test of qasm output
        self.log.info(qc.qasm())
        # test of text drawer
        self.log.info(qc)
        dag = circuit_to_dag(qc)
        dag_nodes = dag.named_nodes('hamiltonian')
        self.assertTrue(len(dag_nodes) == 1)
        dnode = dag_nodes[0]
        self.assertIsInstance(dnode.op, HamiltonianGate)
        for qubit in dnode.qargs:
            self.assertIn(qubit.index, [0, 1])
        assert_allclose(dnode.op.to_matrix(), np.eye(2))

    def test_2q_hamiltonian(self):
        """test 2 qubit hamiltonian """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr)
        matrix = Operator.from_label('XY')
        qc.x(qr[0])
        theta = Parameter('theta')
        uni2q = HamiltonianGate(matrix, theta)
        qc.append(uni2q, [qr[0], qr[1]])
        passman = PassManager()
        passman.append(CXCancellation())
        qc2 = transpile(qc, pass_manager=passman)

        # Testing bind after transpile
        qc2 = qc2.bind_parameters({theta: np.pi / 2})
        # test of qasm output
        self.log.info(qc2.qasm())
        # test of text drawer
        self.log.info(qc2)
        dag = circuit_to_dag(qc2)
        nodes = dag.two_qubit_ops()
        self.assertEqual(len(nodes), 1)
        dnode = nodes[0]
        self.assertIsInstance(dnode.op, HamiltonianGate)
        for qubit in dnode.qargs:
            self.assertIn(qubit.index, [0, 1])
        # Equality based on Pauli exponential identity
        np.testing.assert_array_almost_equal(dnode.op.to_matrix(), 1j*matrix.data)
        qc3 = dag_to_circuit(dag)
        self.assertEqual(qc2, qc3)

    def test_3q_hamiltonian(self):
        """test 3 qubit hamiltonian on non-consecutive bits"""
        qr = QuantumRegister(4)
        qc = QuantumCircuit(qr)
        qc.x(qr[0])
        matrix = Operator.from_label('XZY')
        theta = Parameter('theta')
        uni3q = HamiltonianGate(matrix, theta)
        qc.append(uni3q, [qr[0], qr[1], qr[3]])
        qc.cx(qr[3], qr[2])
        # test of text drawer
        self.log.info(qc)
        qc = qc.bind_parameters({theta: np.pi / 2})
        dag = circuit_to_dag(qc)
        nodes = dag.multi_qubit_ops()
        self.assertEqual(len(nodes), 1)
        dnode = nodes[0]
        self.assertIsInstance(dnode.op, HamiltonianGate)
        for qubit in dnode.qargs:
            self.assertIn(qubit.index, [0, 1, 3])
        np.testing.assert_almost_equal(dnode.op.to_matrix(), 1j*matrix.data)

    def test_qobj_with_hamiltonian(self):
        """test qobj output with hamiltonian"""
        qr = QuantumRegister(4)
        qc = QuantumCircuit(qr)
        qc.rx(np.pi / 4, qr[0])
        matrix = Operator.from_label('XIZ')
        theta = Parameter('theta')
        uni = HamiltonianGate(matrix, theta, label='XIZ')
        qc.append(uni, [qr[0], qr[1], qr[3]])
        qc.cx(qr[3], qr[2])
        qc = qc.bind_parameters({theta: np.pi / 2})
        qobj = qiskit.compiler.assemble(qc)
        instr = qobj.experiments[0].instructions[1]
        self.assertEqual(instr.name, 'hamiltonian')
        # Also test label
        self.assertEqual(instr.label, 'XIZ')
        np.testing.assert_array_almost_equal(np.array(instr.params[0]).astype(np.complex64),
                                             matrix.data)
        # check conversion to dict
        qobj_dict = qobj.to_dict(validate=True)

        class NumpyEncoder(json.JSONEncoder):
            """Class for encoding json str with complex and np arrays."""
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, complex):
                    return (obj.real, obj.imag)
                return json.JSONEncoder.default(self, obj)

        # check json serialization
        self.assertTrue(isinstance(json.dumps(qobj_dict, cls=NumpyEncoder),
                                   str))
