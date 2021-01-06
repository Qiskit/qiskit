# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test cases for the circuit qasm_file and qasm_string method."""

import io

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.test import QiskitTestCase
from qiskit.circuit.qpy_serialization import dump, load


class TestLoadFromQPY(QiskitTestCase):
    """Test circuit.from_qasm_* set of methods."""

    def test_qpy_full_path(self):
        """
        Test qasm_file and get_circuit.

        If all is correct we should get the qasm file loaded in _qasm_file_path
        """
        qr_a = QuantumRegister(4, 'a')
        qr_b = QuantumRegister(4, 'b')
        cr_c = ClassicalRegister(4, 'c')
        cr_d = ClassicalRegister(4, 'd')
        q_circuit = QuantumCircuit(qr_a, qr_b, cr_c, cr_d, name='MyCircuit',
                                   metadata={'test': 1, 'a': 2},
                                   global_phase=3.14159)
        q_circuit.h(qr_a)
        q_circuit.cx(qr_a, qr_b)
        q_circuit.barrier(qr_a)
        q_circuit.barrier(qr_b)
        q_circuit.measure(qr_a, cr_c)
        q_circuit.measure(qr_b, cr_d)
        qpy_file = io.BytesIO()
        dump(qpy_file, q_circuit)
        qpy_file.seek(0)
        new_circ = load(qpy_file)
        self.assertEqual(q_circuit, new_circ)
        self.assertEqual(q_circuit.global_phase, new_circ.global_phase)
        self.assertEqual(q_circuit.metadata, new_circ.metadata)
        self.assertEqual(q_circuit.name, new_circ.name)

    def test_int_parameter(self):
        qc = QuantumCircuit(1)
        qc.rx(3, 0)
        qpy_file = io.BytesIO()
        dump(qpy_file, qc)
        qpy_file.seek(0)
        new_circ = load(qpy_file)
        self.assertEqual(qc, new_circ)

    def test_float_parameter(self):
        qc = QuantumCircuit(1)
        qc.rx(3.14, 0)
        qpy_file = io.BytesIO()
        dump(qpy_file, qc)
        qpy_file.seek(0)
        new_circ = load(qpy_file)
        self.assertEqual(qc, new_circ)

    def test_numpy_float_parameter(self):
        qc = QuantumCircuit(1)
        qc.rx(np.float32(3.14), 0)
        qpy_file = io.BytesIO()
        dump(qpy_file, qc)
        qpy_file.seek(0)
        new_circ = load(qpy_file)
        self.assertEqual(qc, new_circ)

    def test_numpy_int_parameter(self):
        qc = QuantumCircuit(1)
        qc.rx(np.int16(3), 0)
        qpy_file = io.BytesIO()
        dump(qpy_file, qc)
        qpy_file.seek(0)
        new_circ = load(qpy_file)
        self.assertEqual(qc, new_circ)

    def test_unitary_gate(self):
        qc = QuantumCircuit(1)
        unitary = np.array([[0, 1], [1, 0]])
        qc.unitary(unitary, 0)
        qpy_file = io.BytesIO()
        dump(qpy_file, qc)
        qpy_file.seek(0)
        new_circ = load(qpy_file)
        self.assertEqual(qc, new_circ)
