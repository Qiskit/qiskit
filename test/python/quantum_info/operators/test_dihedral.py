# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Tests for CNOTDihedral functions.
"""

import unittest
import numpy as np
from qiskit.circuit import QuantumCircuit, Gate
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.operators.pauli import Pauli
from qiskit.quantum_info.operators.dihedral import CNOTDihedral
from qiskit.quantum_info.random import random_cnotdihedral


class TestCNOTDihedral(unittest.TestCase):
    """
        Test CNOT-dihedral functions
    """
    def test_1_qubit_identities(self):
        """Tests identities for 1-qubit gates"""
        # T*X*T = X
        elem1 = CNOTDihedral(1)
        elem1.phase(1, 0)
        elem1.flip(0)
        elem1.phase(1, 0)
        elem2 = CNOTDihedral(1)
        elem2.flip(0)
        self.assertEqual(elem1, elem2,
                         'Error: 1-qubit identity does not hold')

        # X*T*X = Tdg
        elem1 = CNOTDihedral(1)
        elem1.flip(0)
        elem1.phase(1, 0)
        elem1.flip(0)
        elem2 = CNOTDihedral(1)
        elem2.phase(7, 0)
        self.assertEqual(elem1, elem2,
                         'Error: 1-qubit identity does not hold')

        # T*X*Tdg = S*X
        elem1 = CNOTDihedral(1)
        elem1.phase(1, 0)
        elem1.flip(0)
        elem1.phase(7, 0)
        elem2 = CNOTDihedral(1)
        elem2.phase(2, 0)
        elem2.flip(0)
        self.assertEqual(elem1, elem2,
                         'Error: 1-qubit identity does not hold')

    def test_2_qubit_identities(self):
        """Tests identities for 2-qubit gates"""
        # CX01 * CX10 * CX01 = CX10 * CX01 * CX10
        elem1 = CNOTDihedral(2)
        elem1.cnot(0, 1)
        elem1.cnot(1, 0)
        elem1.cnot(0, 1)
        elem2 = CNOTDihedral(2)
        elem2.cnot(1, 0)
        elem2.cnot(0, 1)
        elem2.cnot(1, 0)
        self.assertEqual(elem1, elem2,
                         'Error: 2-qubit SWAP identity does not hold')

        # CS01 = CS10 (symmetric)
        elem1 = CNOTDihedral(2)
        elem1.phase(1, 0)
        elem1.phase(1, 1)
        elem1.cnot(0, 1)
        elem1.phase(7, 1)
        elem1.cnot(0, 1)
        elem2 = CNOTDihedral(2)
        elem2.phase(1, 1)
        elem2.phase(1, 0)
        elem2.cnot(1, 0)
        elem2.phase(7, 0)
        elem2.cnot(1, 0)
        self.assertEqual(elem1, elem2,
                         'Error: 2-qubit CS identity does not hold')

        # TI*CS*TdgI = CS"
        elem3 = CNOTDihedral(2)
        elem3.phase(1, 0)
        elem3.phase(1, 0)
        elem3.phase(1, 1)
        elem3.cnot(0, 1)
        elem3.phase(7, 1)
        elem3.cnot(0, 1)
        elem3.phase(7, 0)
        self.assertEqual(elem1, elem3,
                         'Error: 2-qubit CS identity does not hold')

        # IT*CS*ITdg = CS
        elem4 = CNOTDihedral(2)
        elem4.phase(1, 1)
        elem4.phase(1, 0)
        elem4.phase(1, 1)
        elem4.cnot(0, 1)
        elem4.phase(7, 1)
        elem4.cnot(0, 1)
        elem4.phase(7, 1)
        self.assertEqual(elem1, elem4,
                         'Error: 2-qubit CS identity does not hold')

        # XX*CS*XX*SS = CS
        elem5 = CNOTDihedral(2)
        elem5.flip(0)
        elem5.flip(1)
        elem5.phase(1, 0)
        elem5.phase(1, 1)
        elem5.cnot(0, 1)
        elem5.phase(7, 1)
        elem5.cnot(0, 1)
        elem5.flip(0)
        elem5.flip(1)
        elem5.phase(2, 0)
        elem5.phase(2, 1)
        self.assertEqual(elem1, elem5,
                         'Error: 2-qubit CS identity does not hold')

        # CSdg01 = CSdg10 (symmetric)
        elem1 = CNOTDihedral(2)
        elem1.phase(7, 0)
        elem1.phase(7, 1)
        elem1.cnot(0, 1)
        elem1.phase(1, 1)
        elem1.cnot(0, 1)
        elem2 = CNOTDihedral(2)
        elem2.phase(7, 1)
        elem2.phase(7, 0)
        elem2.cnot(1, 0)
        elem2.phase(1, 0)
        elem2.cnot(1, 0)
        self.assertEqual(elem1, elem2,
                         'Error: 2-qubit CSdg identity does not hold')

        # XI*CS*XI*ISdg = CSdg
        elem3 = CNOTDihedral(2)
        elem3.flip(0)
        elem3.phase(1, 0)
        elem3.phase(1, 1)
        elem3.cnot(0, 1)
        elem3.phase(7, 1)
        elem3.cnot(0, 1)
        elem3.flip(0)
        elem3.phase(6, 1)
        self.assertEqual(elem1, elem3,
                         'Error: 2-qubit CSdg identity does not hold')

        # IX*CS*IX*SdgI = CSdg
        elem4 = CNOTDihedral(2)
        elem4.flip(1)
        elem4.phase(1, 0)
        elem4.phase(1, 1)
        elem4.cnot(0, 1)
        elem4.phase(7, 1)
        elem4.cnot(0, 1)
        elem4.flip(1)
        elem4.phase(6, 0)
        self.assertEqual(elem1, elem4,
                         'Error: 2-qubit CSdg identity does not hold')

        # relations for CZ
        elem1 = CNOTDihedral(2)
        elem1.phase(1, 0)
        elem1.phase(1, 1)
        elem1.cnot(0, 1)
        elem1.phase(7, 1)
        elem1.cnot(0, 1)
        elem1.phase(1, 0)
        elem1.phase(1, 1)
        elem1.cnot(0, 1)
        elem1.phase(7, 1)
        elem1.cnot(0, 1)

        elem2 = CNOTDihedral(2)
        elem2.phase(7, 0)
        elem2.phase(7, 1)
        elem2.cnot(0, 1)
        elem2.phase(1, 1)
        elem2.cnot(0, 1)
        elem2.phase(7, 0)
        elem2.phase(7, 1)
        elem2.cnot(0, 1)
        elem2.phase(1, 1)
        elem2.cnot(0, 1)

        elem3 = CNOTDihedral(2)
        elem3.phase(1, 1)
        elem3.phase(1, 0)
        elem3.cnot(1, 0)
        elem3.phase(7, 0)
        elem3.cnot(1, 0)
        elem3.phase(1, 1)
        elem3.phase(1, 0)
        elem3.cnot(1, 0)
        elem3.phase(7, 0)
        elem3.cnot(1, 0)

        elem4 = CNOTDihedral(2)
        elem4.phase(7, 1)
        elem4.phase(7, 0)
        elem4.cnot(1, 0)
        elem4.phase(1, 0)
        elem4.cnot(1, 0)
        elem4.phase(7, 1)
        elem4.phase(7, 0)
        elem4.cnot(1, 0)
        elem4.phase(1, 0)
        elem4.cnot(1, 0)

        # CZ = TdgTdg * CX * T^2I * CX * TdgTdg
        elem5 = CNOTDihedral(2)
        elem5.phase(7, 1)
        elem5.phase(7, 0)
        elem5.cnot(1, 0)
        elem5.phase(2, 0)
        elem5.cnot(1, 0)
        elem5.phase(7, 1)
        elem5.phase(7, 0)

        self.assertEqual(elem1, elem2,
                         'Error: 2-qubit CZ identity does not hold')
        self.assertEqual(elem1, elem3,
                         'Error: 2-qubit CZ identity does not hold')
        self.assertEqual(elem1, elem4,
                         'Error: 2-qubit CZ identity does not hold')
        self.assertEqual(elem1, elem5,
                         'Error: 2-qubit CZ identity does not hold')

        # relations for CX
        elem1 = CNOTDihedral(2)
        elem1.cnot(0, 1)

        # TI*CX*TdgI = CX
        elem2 = CNOTDihedral(2)
        elem2.phase(1, 0)
        elem2.cnot(0, 1)
        elem2.phase(7, 0)

        # IZ*CX*ZZ = CX
        elem3 = CNOTDihedral(2)
        elem3.phase(4, 1)
        elem3.cnot(0, 1)
        elem3.phase(4, 0)
        elem3.phase(4, 1)

        # IX*CX*IX = CX
        elem4 = CNOTDihedral(2)
        elem4.flip(1)
        elem4.cnot(0, 1)
        elem4.flip(1)

        # XI*CX*XX = CX
        elem5 = CNOTDihedral(2)
        elem5.flip(0)
        elem5.cnot(0, 1)
        elem5.flip(0)
        elem5.flip(1)

        self.assertEqual(elem1, elem2,
                         'Error: 2-qubit CX identity does not hold')
        self.assertEqual(elem1, elem3,
                         'Error: 2-qubit CX identity does not hold')
        self.assertEqual(elem1, elem4,
                         'Error: 2-qubit CX identity does not hold')
        self.assertEqual(elem1, elem5,
                         'Error: 2-qubit CX identity does not hold')

        # IT*CX01*CX10*TdgI = CX01*CX10
        elem1 = CNOTDihedral(2)
        elem1.cnot(0, 1)
        elem1.cnot(1, 0)

        elem2 = CNOTDihedral(2)
        elem2.phase(1, 1)
        elem2.cnot(0, 1)
        elem2.cnot(1, 0)
        elem2.phase(7, 0)
        self.assertEqual(elem1, elem2,
                         'Error: 2-qubit CX01*CX10 identity does not hold')

    def test_dihedral_random_decompose(self):
        """
        Test that random elements are CNOTDihedral
        and to_circuit, to_instruction, from_circuit, is_cnotdihedral methods
        """
        for qubit_num in range(1, 9):
            for nseed in range(20):
                elem = random_cnotdihedral(qubit_num, seed=nseed)
                self.assertIsInstance(elem, CNOTDihedral,
                                      'Error: random element is not CNOTDihedral')
                self.assertTrue(elem.is_cnotdihedral(),
                                'Error: random element is not CNOTDihedral')

                test_circ = elem.to_circuit()
                self.assertTrue(test_circ,
                                'Error: cannot decompose a random '
                                'CNOTDihedral element to a circuit')
                test_elem = CNOTDihedral(test_circ)
                # Test of to_circuit and from_circuit methods
                self.assertEqual(elem, test_elem,
                                 'Error: decomposed circuit is not equal '
                                 'to the original circuit')
                # Test that is_cnotdihedral fails if linear part is wrong
                test_elem.linear = np.zeros((qubit_num, qubit_num))
                value = test_elem.is_cnotdihedral()
                self.assertFalse(value,
                                 'Error: is_cnotdihedral is not correct.')

                test_gates = elem.to_instruction()
                self.assertIsInstance(test_gates, Gate,
                                      'Error: cannot decompose a random '
                                      'CNOTDihedral element to a Gate')
                self.assertEqual(test_gates.num_qubits, test_circ.num_qubits,
                                 'Error: wrong num_qubits in decomposed gates')
                test_elem1 = CNOTDihedral(test_gates)
                # Test of to_instruction and from_circuit methods
                self.assertEqual(elem, test_elem1,
                                 'Error: decomposed gates are not equal '
                                 'to the original gates')

    def test_compose_method(self):
        """Test compose method"""
        samples = 10
        nseed = 111
        for qubit_num in range(1, 6):
            for i in range(samples):
                elem1 = random_cnotdihedral(qubit_num, seed=nseed + i)
                elem2 = random_cnotdihedral(qubit_num, seed=nseed + samples + i)
                circ1 = elem1.to_circuit()
                circ2 = elem2.to_circuit()
                value = elem1.compose(elem2)
                target = CNOTDihedral(circ1.extend(circ2))
                self.assertEqual(target, value,
                                 'Error: composed circuit is not the same')

    def test_dot_method(self):
        """Test dot method"""
        samples = 10
        nseed = 222
        for qubit_num in range(1, 6):
            for i in range(samples):
                elem1 = random_cnotdihedral(qubit_num, seed=nseed + i)
                elem2 = random_cnotdihedral(qubit_num, seed=nseed + samples + i)
                circ1 = elem1.to_circuit()
                circ2 = elem2.to_circuit()
                value = elem1.dot(elem2)
                target = CNOTDihedral(circ2.extend(circ1))
                self.assertEqual(target, value,
                                 'Error: composed circuit is not the same')

    def test_tensor_method(self):
        """Test tensor method"""
        samples = 10
        nseed = 333
        for num_qubits_1 in range(1, 5):
            for num_qubits_2 in range(1, 5):
                for i in range(samples):
                    elem1 = random_cnotdihedral(num_qubits_1, seed=nseed + i)
                    elem2 = random_cnotdihedral(num_qubits_2, seed=nseed + samples + i)
                    circ1 = elem1.to_instruction()
                    circ2 = elem2.to_instruction()
                    value = elem1.tensor(elem2)
                    circ = QuantumCircuit(num_qubits_1 + num_qubits_2)
                    qargs = list(range(num_qubits_1))
                    for instr, qregs, _ in circ1.definition:
                        new_qubits = [qargs[tup.index] for tup in qregs]
                        circ.append(instr, new_qubits)
                    qargs = list(range(num_qubits_1, num_qubits_1 + num_qubits_2))
                    for instr, qregs, _ in circ2.definition:
                        new_qubits = [qargs[tup.index] for tup in qregs]
                        circ.append(instr, new_qubits)
                    target = CNOTDihedral(circ)

                    self.assertEqual(target, value,
                                     'Error: tensor circuit is not the same')

    def test_expand_method(self):
        """Test tensor method"""
        samples = 10
        nseed = 444
        for num_qubits_1 in range(1, 5):
            for num_qubits_2 in range(1, 5):
                for i in range(samples):
                    elem1 = random_cnotdihedral(num_qubits_1, seed=nseed + i)
                    elem2 = random_cnotdihedral(num_qubits_2, seed=nseed + samples + i)
                    circ1 = elem1.to_instruction()
                    circ2 = elem2.to_instruction()
                    value = elem2.expand(elem1)
                    circ = QuantumCircuit(num_qubits_1 + num_qubits_2)
                    qargs = list(range(num_qubits_1))
                    for instr, qregs, _ in circ1.definition:
                        new_qubits = [qargs[tup.index] for tup in qregs]
                        circ.append(instr, new_qubits)
                    qargs = list(range(num_qubits_1, num_qubits_1 + num_qubits_2))
                    for instr, qregs, _ in circ2.definition:
                        new_qubits = [qargs[tup.index] for tup in qregs]
                        circ.append(instr, new_qubits)
                    target = CNOTDihedral(circ)

                    self.assertEqual(target, value,
                                     'Error: expand circuit is not the same')

    def test_adjoint(self):
        """Test transpose method"""
        samples = 10
        nseed = 555
        for qubit_num in range(1, 5):
            for i in range(samples):
                elem = random_cnotdihedral(qubit_num, seed=nseed + i)
                circ = elem.to_circuit()
                value = elem.adjoint().to_operator()
                target = Operator(circ).adjoint()
                self.assertTrue(target.equiv(value),
                                'Error: adjoint circuit is not the same')

    def test_transpose(self):
        """Test transpose method"""
        samples = 10
        nseed = 666
        for qubit_num in range(1, 5):
            for i in range(samples):
                elem = random_cnotdihedral(qubit_num, seed=nseed + i)
                circ = elem.to_circuit()
                value = elem.transpose().to_operator()
                target = Operator(circ).transpose()
                self.assertTrue(target.equiv(value),
                                'Error: transpose circuit is not the same')

    def test_conjugate(self):
        """Test transpose method"""
        samples = 10
        nseed = 777
        for qubit_num in range(1, 5):
            for i in range(samples):
                elem = random_cnotdihedral(qubit_num, seed=nseed + i)
                circ = elem.to_circuit()
                value = elem.conjugate().to_operator()
                target = Operator(circ).conjugate()
                self.assertTrue(target.equiv(value),
                                'Error: conjugate circuit is not the same')

    def test_to_matrix(self):
        """Test to_matrix method"""
        samples = 10
        nseed = 888
        for qubit_num in range(1, 5):
            for i in range(samples):
                elem = random_cnotdihedral(qubit_num, seed=nseed + i)
                circ = elem.to_circuit()
                mat = elem.to_matrix()
                self.assertIsInstance(mat, np.ndarray)
                self.assertEqual(mat.shape, 2 * (2 ** qubit_num,))
                value = Operator(mat)
                target = Operator(circ)
                self.assertTrue(value.equiv(target),
                                'Error: matrix of the circuit is not the same')

    def test_init_from_pauli(self):
        """Test initialization from Pauli"""
        samples = 10
        nseed = 999
        for qubit_num in range(1, 5):
            for i in range(samples):
                pauli = Pauli.random(qubit_num, seed=nseed + i)
                elem = CNOTDihedral(pauli)
                value = Operator(pauli)
                target = Operator(elem)
                self.assertTrue(value.equiv(target),
                                'Error: Pauli operator is not the same.')


if __name__ == '__main__':
    unittest.main()
