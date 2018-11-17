# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=unused-import

"""Test Qiskit's QuantumCircuit class for multiple registers."""

import os
import tempfile
import unittest

from qiskit import Aer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute
from qiskit import QISKitError
from qiskit.quantum_info import state_fidelity, process_fidelity, Pauli, basis_state
from ..common import QiskitTestCase


class TestCircuitMultiRegs(QiskitTestCase):
    """QuantumCircuit Qasm tests."""

    def test_circuit_multi(self):
        """Test circuit multi regs declared at start.
        """
        q0 = QuantumRegister(2, 'q0')
        c0 = ClassicalRegister(2, 'c0')
        q1 = QuantumRegister(2, 'q1')
        c1 = ClassicalRegister(2, 'c1')
        circ = QuantumCircuit(q0, q1)
        circ.x(q0[1])
        circ.x(q1[0])

        meas = QuantumCircuit(q0, q1, c0, c1)
        meas.measure(q0, c0)
        meas.measure(q1, c1)

        qc = circ + meas
        backend_sim = Aer.get_backend('qasm_simulator')
        result = execute(qc, backend_sim).result()
        counts = result.get_counts(qc)
        target = {'01 10': 1024}

        backend_sim = Aer.get_backend('statevector_simulator')
        result = execute(circ, backend_sim).result()
        state = result.get_statevector(circ)

        backend_sim = Aer.get_backend('unitary_simulator')
        result = execute(circ, backend_sim).result()
        unitary = result.get_unitary(circ)
        testdraw = """
q1_1: |0>──────────
         ┌───┐
q1_0: |0>┤ X ├─────
         └───┘┌───┐
q0_1: |0>─────┤ X ├
              └───┘
q0_0: |0>──────────
"""
        self.assertEqual(counts, target)
        self.assertAlmostEqual(state_fidelity(basis_state('0110', 4), state), 1.0, places=7)
        self.assertAlmostEqual(process_fidelity(Pauli(label='IXXI').to_matrix(), unitary),
                                                1.0, places=7)

    def test_circuit_multi_case2(self):
        """Test circuit multi regs declared at start.
        """
        q0 = QuantumRegister(2, 'q0')
        c0 = ClassicalRegister(2, 'c0')
        q1 = QuantumRegister(2, 'q1')
        c1 = ClassicalRegister(2, 'c1')
        circ2 = QuantumCircuit()
        circ2.add_register(q0)
        circ2.add_register(q1)
        circ2.x(q0[1])
        circ2.x(q1[0])

        meas2 = QuantumCircuit()
        meas2.add_register(q0)
        meas2.add_register(q1)
        meas2.add_register(c0)
        meas2.add_register(c1)
        meas2.measure(q0, c0)
        meas2.measure(q1, c1)

        qc2 = circ2 + meas2

        backend_sim = Aer.get_backend('statevector_simulator')
        result = execute(circ2, backend_sim).result()
        state = result.get_statevector(circ2)

        backend_sim = Aer.get_backend('qasm_simulator')
        result = execute(qc2, backend_sim).result()
        counts = result.get_counts(qc2)

        backend_sim = Aer.get_backend('unitary_simulator')
        result = execute(circ2, backend_sim).result()
        unitary = result.get_unitary(circ2)

        target = {'01 10': 1024}
        testdraw = """
q1_1: |0>──────────
         ┌───┐
q1_0: |0>┤ X ├─────
         └───┘┌───┐
q0_1: |0>─────┤ X ├
              └───┘
q0_0: |0>──────────
"""

        #self.assertEqual(testdraw, circ2.draw())
        self.assertEqual(counts, target)
        self.assertAlmostEqual(state_fidelity(basis_state('0110', 4), state), 1.0, places=7)
        self.assertAlmostEqual(process_fidelity(Pauli(label='IXXI').to_matrix(), unitary),
                                                1.0, places=7)
