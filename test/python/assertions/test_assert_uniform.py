# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Qiskit's AssertUniform class."""

import unittest
from qiskit import BasicAer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute
from qiskit.test import QiskitTestCase


class TestAssertUniform(QiskitTestCase):
    """AssertUniform tests."""

    def test_assert_uniform(self):
        """Test AssertUniform
        """
        qc = QuantumCircuit(2, 2)
        qc.h([0, 1])
        bkpt = qc.get_breakpoint_uniform([0, 1], [0, 1], 0.00001)
        BasicAer.backends()
        job = execute(bkpt, BasicAer.get_backend('qasm_simulator'))
        self.assertTrue(job.result().get_assertion_passed(bkpt))

    def test_assert_not_uniform(self):
        """Test AssertUniform with negate True
        """
        qc = QuantumCircuit(2, 2)
        qc.h(1)
        bkpt = qc.get_breakpoint_not_uniform([0, 1], [0, 1], 0.00001)
        BasicAer.backends()
        job = execute(bkpt, BasicAer.get_backend('qasm_simulator'))
        self.assertTrue(job.result().get_assertion_passed(bkpt))

    def test_with_bits(self):
        """Test AssertUniform with bit syntax
        """
        q = QuantumRegister(1)
        q_2 = QuantumRegister(2)
        c = ClassicalRegister(1)
        c_2 = ClassicalRegister(2)
        qc = QuantumCircuit(q, q_2, c, c_2)
        qc.h([0, 2])
        bkpt = qc.get_breakpoint_uniform([q[0], q_2[1]], [c[0], c_2[1]], 0.00001)
        BasicAer.backends()
        job = execute(bkpt, BasicAer.get_backend('qasm_simulator'))
        self.assertTrue(job.result().get_assertion_passed(bkpt))

    def test_with_registers(self):
        """Test AssertUniform with register syntax
        """
        q = QuantumRegister(1)
        q_2 = QuantumRegister(2)
        c = ClassicalRegister(1)
        c_2 = ClassicalRegister(2)
        qc = QuantumCircuit(q, q_2, c, c_2)
        qc.h([0, 1, 2])
        bkpt = qc.get_breakpoint_uniform(q_2, c_2, 0.00001)
        BasicAer.backends()
        job = execute(bkpt, BasicAer.get_backend('qasm_simulator'))
        self.assertTrue(job.result().get_assertion_passed(bkpt))


if __name__ == '__main__':
    unittest.main()
