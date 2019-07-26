# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Qiskit's AssertClassical class."""

import unittest
from qiskit import BasicAer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute
from qiskit.test import QiskitTestCase


class TestAssertClassical(QiskitTestCase):
    """AssertClassical tests."""

    def test_assert_classical(self):
        """Test AssertClassical
        """
        qc = QuantumCircuit(2, 2)
        qc.x(1)
        bkpt = qc.get_breakpoint_classical([0, 1], [0, 1], 0.01, 2)
        BasicAer.backends()
        job = execute(bkpt, BasicAer.get_backend('qasm_simulator'))
        self.assertTrue(job.result().get_assertion_passed(bkpt))

    def test_assert_not_classical(self):
        """Test AssertClassical with negate True
        """
        qc = QuantumCircuit(2, 2)
        qc.h(1)
        bkpt = qc.get_breakpoint_not_classical([0, 1], [0, 1], 0.01, "10")
        BasicAer.backends()
        job = execute(bkpt, BasicAer.get_backend('qasm_simulator'))
        self.assertTrue(job.result().get_assertion_passed(bkpt))

    def test_with_bits(self):
        """Test AssertClassical with bit syntax
        """
        q = QuantumRegister(1)
        qq = QuantumRegister(2)
        c = ClassicalRegister(1)
        cc = ClassicalRegister(2)
        qc = QuantumCircuit(q, qq, c, cc)
        qc.x(0)
        bkpt = qc.get_breakpoint_classical([q[0], qq[0], qq[1]], [c[0], cc[0], cc[1]],
                                           0.01, 1)
        BasicAer.backends()
        job = execute(bkpt, BasicAer.get_backend('qasm_simulator'))
        self.assertTrue(job.result().get_assertion_passed(bkpt))

    def test_with_registers(self):
        """Test AssertProduct with register syntax
        """
        q = QuantumRegister(1)
        qq = QuantumRegister(2)
        c = ClassicalRegister(1)
        cc = ClassicalRegister(2)
        qc = QuantumCircuit(q, qq, c, cc)
        qc.h(0)
        qc.x(2)
        bkpt = qc.get_breakpoint_classical(qq, cc, 0.01)
        BasicAer.backends()
        job = execute(bkpt, BasicAer.get_backend('qasm_simulator'))
        self.assertTrue(job.result().get_assertion_passed(bkpt))


if __name__ == '__main__':
    unittest.main()
