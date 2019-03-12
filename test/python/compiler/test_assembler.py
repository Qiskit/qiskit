# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Assembler Test."""

import numpy as np
import unittest

from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.compiler import assemble_circuits
from qiskit.compiler import RunConfig
from qiskit.qobj import QASMQobj
from qiskit.test import QiskitTestCase


class TestAssembler(QiskitTestCase):
    """Tests for assembling circuits to qobj."""

    def test_assemble_single_circuit(self):
        """Test assembling a single circuit.
        """
        q = QuantumRegister(2, name='q')
        c = ClassicalRegister(2, name='c')
        circ = QuantumCircuit(q, c, name='circ')
        circ.h(q[0])
        circ.cx(q[0], q[1])
        circ.measure(q, c)

        run_config = RunConfig(shots=2000, memory=True)
        qobj = assemble_circuits(circ, run_config=run_config)
        self.assertIsInstance(qobj, QASMQobj)
        self.assertEqual(qobj.config.shots, 2000)
        self.assertEqual(qobj.config.memory, True)
        self.assertEqual(len(qobj.experiments), 1)
        self.assertEqual(qobj.experiments[0].instructions[1].name, 'cx')

    def test_assemble_multiple_circuits(self):
        """Test assembling multiple circuits, all should have the same config.
        """
        q0 = QuantumRegister(2, name='q0')
        c0 = ClassicalRegister(2, name='c0')
        circ0 = QuantumCircuit(q0, c0, name='circ0')
        circ0.h(q0[0])
        circ0.cx(q0[0], q0[1])
        circ0.measure(q0, c0)

        q1 = QuantumRegister(3, name='q1')
        c1 = ClassicalRegister(3, name='c1')
        circ1 = QuantumCircuit(q1, c1, name='circ0')
        circ1.h(q1[0])
        circ1.cx(q1[0], q1[1])
        circ1.cx(q1[0], q1[2])
        circ1.measure(q1, c1)

        run_config = RunConfig(shots=100, memory=False, seed=6)
        qobj = assemble_circuits([circ0, circ1], run_config=run_config)
        self.assertIsInstance(qobj, QASMQobj)
        self.assertEqual(qobj.config.seed, 6)
        self.assertEqual(len(qobj.experiments), 2)
        self.assertEqual(qobj.experiments[1].config.n_qubits, 3)
        self.assertEqual(len(qobj.experiments), 2)
        self.assertEqual(len(qobj.experiments[1].instructions), 6)

    def test_assemble_no_run_config(self):
        """Test assembling with no run_config, relying on default.
        """
        q = QuantumRegister(2, name='q')
        c = ClassicalRegister(2, name='c')
        circ = QuantumCircuit(q, c, name='circ')
        circ.h(q[0])
        circ.cx(q[0], q[1])
        circ.measure(q, c)

        qobj = assemble_circuits(circ)
        self.assertIsInstance(qobj, QASMQobj)
        self.assertIsNone(getattr(qobj.config, 'shots', None))

    def test_assemble_initialize(self):
        """Test assembling a circuit with an initialize.
        """
        q = QuantumRegister(2, name='q')
        circ = QuantumCircuit(q, name='circ')
        circ.initialize([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], q[:])

        qobj = assemble_circuits(circ)
        self.assertIsInstance(qobj, QASMQobj)
        self.assertEqual(qobj.experiments[0].instructions[0].name, 'init')
        np.testing.assert_almost_equal(qobj.experiments[0].instructions[0].params,
                                       [0.7071067811865, 0, 0, 0.707106781186])


if __name__ == '__main__':
    unittest.main(verbosity=2)
