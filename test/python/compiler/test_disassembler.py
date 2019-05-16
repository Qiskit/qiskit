# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Assembler Test."""

import unittest

import numpy as np

from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.compiler.assemble import assemble
from qiskit.assembler.disassemble import disassemble
from qiskit.assembler.run_config import RunConfig
from qiskit.test import QiskitTestCase


class TestAssembler(QiskitTestCase):
    """Tests for assembling circuits to qobj."""

    def test_disassemble_single_circuit(self):
        """Test assembling a single circuit.
        """
        qr = QuantumRegister(2, name='q')
        cr = ClassicalRegister(2, name='c')
        circ = QuantumCircuit(qr, cr, name='circ')
        circ.h(qr[0])
        circ.cx(qr[0], qr[1])
        circ.measure(qr, cr)

        qobj = assemble(circ, shots=2000, memory=True)
        circuits, run_config_out, headers = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 2)
        self.assertEqual(run_config_out.memory_slots, 2)
        self.assertEqual(run_config_out.shots, 2000)
        self.assertEqual(run_config_out.memory, True)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], circ)
        self.assertEqual({}, headers)

    def test_disssemble_multiple_circuits(self):
        """Test assembling multiple circuits, all should have the same config.
        """
        qr0 = QuantumRegister(2, name='q0')
        qc0 = ClassicalRegister(2, name='c0')
        circ0 = QuantumCircuit(qr0, qc0, name='circ0')
        circ0.h(qr0[0])
        circ0.cx(qr0[0], qr0[1])
        circ0.measure(qr0, qc0)

        qr1 = QuantumRegister(3, name='q1')
        qc1 = ClassicalRegister(3, name='c1')
        circ1 = QuantumCircuit(qr1, qc1, name='circ0')
        circ1.h(qr1[0])
        circ1.cx(qr1[0], qr1[1])
        circ1.cx(qr1[0], qr1[2])
        circ1.measure(qr1, qc1)

        qobj = assemble([circ0, circ1], shots=100, memory=False, seed=6)
        circuits, run_config_out, headers = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 3)
        self.assertEqual(run_config_out.memory_slots, 3)
        self.assertEqual(run_config_out.shots, 100)
        self.assertEqual(run_config_out.memory, False)
        self.assertEqual(run_config_out.seed, 6)
        self.assertEqual(len(circuits), 2)
        for circuit in circuits:
            self.assertIn(circuit, [circ0, circ1])
        self.assertEqual({}, headers)

    def test_assemble_no_run_config(self):
        """Test assembling with no run_config, relying on default.
        """
        qr = QuantumRegister(2, name='q')
        qc = ClassicalRegister(2, name='c')
        circ = QuantumCircuit(qr, qc, name='circ')
        circ.h(qr[0])
        circ.cx(qr[0], qr[1])
        circ.measure(qr, qc)

        qobj = assemble(circ)
        circuits, run_config_out, headers = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 2)
        self.assertEqual(run_config_out.memory_slots, 2)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], circ)
        self.assertEqual({}, headers)

    def test_assemble_initialize(self):
        """Test assembling a circuit with an initialize.
        """
        q = QuantumRegister(2, name='q')
        circ = QuantumCircuit(q, name='circ')
        circ.initialize([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], q[:])

        qobj = assemble(circ)
        circuits, run_config_out, header = disassemble(qobj)
        run_config_out = RunConfig(**run_config_out)
        self.assertEqual(run_config_out.n_qubits, 2)
        self.assertEqual(run_config_out.memory_slots, 0)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], circ)
        self.assertEqual({}, header)


if __name__ == '__main__':
    unittest.main(verbosity=2)
