# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Assembler Test."""

import unittest

import numpy as np

from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.compiler import assemble_circuits
from qiskit.compiler import disassemble
from qiskit.compiler import RunConfig
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

        run_config = RunConfig(shots=2000, memory=True)
        qobj = assemble_circuits(circ, run_config=run_config)
        circuits, run_config_out, headers = disassemble(qobj)
        self.assertEqual({'shots': 2000, 'memory': True, 'memory_slots': 2,
                          'n_qubits': 2}, run_config_out)
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

        run_config = RunConfig(shots=100, memory=False, seed=6)
        qobj = assemble_circuits([circ0, circ1], run_config=run_config)
        circuits, out_run_config, headers = disassemble(qobj)
        self.assertEqual({'shots': 100, 'memory': False, 'memory_slots': 3,
                          'n_qubits': 3, 'seed': 6}, out_run_config)
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

        qobj = assemble_circuits(circ)
        circuits, run_config_out, headers = disassemble(qobj)
        self.assertEqual({'memory_slots': 2, 'n_qubits': 2}, run_config_out)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], circ)
        self.assertEqual({}, headers)

    def test_assemble_initialize(self):
        """Test assembling a circuit with an initialize.
        """
        q = QuantumRegister(2, name='q')
        circ = QuantumCircuit(q, name='circ')
        circ.initialize([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], q[:])

        qobj = assemble_circuits(circ)
        circuits, run_config_out, header = disassemble(qobj)
        self.assertEqual({'memory_slots': 0, 'n_qubits': 2}, run_config_out)
        self.assertEqual(len(circuits), 1)
        self.assertEqual(circuits[0], circ)
        self.assertEqual({}, header)


if __name__ == '__main__':
    unittest.main(verbosity=2)
