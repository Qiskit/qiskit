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

# pylint: disable=invalid-name

"""Test Qiskit's AssertClassical class."""

import unittest
import qiskit.extensions.simulator
from qiskit import BasicAer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute
from qiskit import QiskitError
from qiskit.quantum_info import state_fidelity
from qiskit.test import QiskitTestCase


class TestAssertClassical(QiskitTestCase):
    """AssertClassical tests."""

    def test_assert_classical(self):
        """Test AssertClassical
        """
        qc1 = QuantumCircuit(2, 2)
        qc1.x(1)
        breakpoint = qc1.get_breakpoint_classical([0, 1], [0, 1], 0.01, 2)
        BasicAer.backends()
        job = execute(breakpoint, BasicAer.get_backend('qasm_simulator'))
        self.assertTrue(job.result().get_assertion_passed(breakpoint))

    def test_assert_not_classical(self):
        """Test AssertClassical with negate True
        """
        qc1 = QuantumCircuit(2, 2)
        qc1.h(1)
        breakpoint = qc1.get_breakpoint_not_classical([0, 1], [0, 1], 0.01, "10")
        BasicAer.backends()
        job = execute(breakpoint, BasicAer.get_backend('qasm_simulator'))
        self.assertTrue(job.result().get_assertion_passed(breakpoint))

    def test_with_bits(self):
        """Test AssertClassical with bit syntax
        """
        q0 = QuantumRegister(1)
        q1 = QuantumRegister(2)
        c0 = ClassicalRegister(1)
        c1 = ClassicalRegister(2)
        qc1 = QuantumCircuit(q0, q1, c0, c1)
        qc1.x(0)
        breakpoint = qc1.get_breakpoint_classical([q0[0], q1[0], q1[1]], [c0[0], c1[0], c1[1]], 0.01, 1)
        BasicAer.backends()
        job = execute(breakpoint, BasicAer.get_backend('qasm_simulator'))
        self.assertTrue(job.result().get_assertion_passed(breakpoint))

    def test_with_registers(self):
        """Test AssertProduct with register syntax
        """
        q0 = QuantumRegister(1)
        q1 = QuantumRegister(2)
        c0 = ClassicalRegister(1)
        c1 = ClassicalRegister(2)
        qc1 = QuantumCircuit(q0, q1, c0, c1)
        qc1.h(0)
        qc1.x(2)
        breakpoint = qc1.get_breakpoint_classical(q1, c1, 0.01)
        BasicAer.backends()
        job = execute(breakpoint, BasicAer.get_backend('qasm_simulator'))
        self.assertTrue(job.result().get_assertion_passed(breakpoint))

if __name__ == '__main__':
    unittest.main()
