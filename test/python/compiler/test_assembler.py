# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Assembler Test."""

import unittest

import numpy as np

from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.compiler import assemble_circuits
from qiskit.compiler import RunConfig
from qiskit.qobj import QasmQobj
from qiskit.test import QiskitTestCase


class TestAssembler(QiskitTestCase):
    """Tests for assembling circuits to qobj."""

    def test_assemble_single_circuit(self):
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
        self.assertIsInstance(qobj, QasmQobj)
        self.assertEqual(qobj.config.shots, 2000)
        self.assertEqual(qobj.config.memory, True)
        self.assertEqual(len(qobj.experiments), 1)
        self.assertEqual(qobj.experiments[0].instructions[1].name, 'cx')

    def test_assemble_multiple_circuits(self):
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
        self.assertIsInstance(qobj, QasmQobj)
        self.assertEqual(qobj.config.seed, 6)
        self.assertEqual(len(qobj.experiments), 2)
        self.assertEqual(qobj.experiments[1].config.n_qubits, 3)
        self.assertEqual(len(qobj.experiments), 2)
        self.assertEqual(len(qobj.experiments[1].instructions), 6)

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
        self.assertIsInstance(qobj, QasmQobj)
        self.assertIsNone(getattr(qobj.config, 'shots', None))

    def test_assemble_initialize(self):
        """Test assembling a circuit with an initialize.
        """
        q = QuantumRegister(2, name='q')
        circ = QuantumCircuit(q, name='circ')
        circ.initialize([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], q[:])

        qobj = assemble_circuits(circ)
        self.assertIsInstance(qobj, QasmQobj)
        self.assertEqual(qobj.experiments[0].instructions[0].name, 'initialize')
        np.testing.assert_almost_equal(qobj.experiments[0].instructions[0].params,
                                       [0.7071067811865, 0, 0, 0.707106781186])

    def test_assemble_opaque_inst(self):
        """Test opaque instruction is assembled as-is"""
        opaque_inst = Instruction(name='my_inst', num_qubits=4,
                                  num_clbits=2, params=[0.5, 0.4])
        q = QuantumRegister(6, name='q')
        c = ClassicalRegister(4, name='c')
        circ = QuantumCircuit(q, c, name='circ')
        circ.append(opaque_inst, [q[0], q[2], q[5], q[3]], [c[3], c[0]])
        qobj = assemble_circuits(circ)
        self.assertIsInstance(qobj, QasmQobj)
        self.assertEqual(len(qobj.experiments[0].instructions), 1)
        self.assertEqual(qobj.experiments[0].instructions[0].name, 'my_inst')
        self.assertEqual(qobj.experiments[0].instructions[0].qubits, [0, 2, 5, 3])
        self.assertEqual(qobj.experiments[0].instructions[0].memory, [3, 0])
        self.assertEqual(qobj.experiments[0].instructions[0].params, [0.5, 0.4])

    def test_measure_to_registers_when_conditionals(self):
        """Verify assemble_circuits maps all measure ops on to a register slot
        for a circuit containing conditionals."""
        qr = QuantumRegister(2)
        cr1 = ClassicalRegister(1)
        cr2 = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr1, cr2)

        qc.measure(qr[0], cr1)  # Measure not required for a later conditional
        qc.measure(qr[1], cr2[1])  # Measure required for a later conditional
        qc.h(qr[1]).c_if(cr2, 3)

        qobj = assemble_circuits(qc)

        first_measure, second_measure = [op for op in qobj.experiments[0].instructions
                                         if op.name == 'measure']

        self.assertTrue(hasattr(first_measure, 'register'))
        self.assertEqual(first_measure.register, first_measure.memory)
        self.assertTrue(hasattr(second_measure, 'register'))
        self.assertEqual(second_measure.register, second_measure.memory)

    def test_convert_to_bfunc_plus_conditional(self):
        """Verify assemble_circuits converts conditionals from QASM to Qobj."""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)

        qc.h(qr[0]).c_if(cr, 1)

        qobj = assemble_circuits(qc)

        bfunc_op, h_op = qobj.experiments[0].instructions

        self.assertEqual(bfunc_op.name, 'bfunc')
        self.assertEqual(bfunc_op.mask, '0x1')
        self.assertEqual(bfunc_op.val, '0x1')
        self.assertEqual(bfunc_op.relation, '==')

        self.assertTrue(hasattr(h_op, 'conditional'))
        self.assertEqual(bfunc_op.register, h_op.conditional)

    def test_resize_value_to_register(self):
        """Verify assemble_circuits converts the value provided on the classical
        creg to its mapped location on the device register."""
        qr = QuantumRegister(1)
        cr1 = ClassicalRegister(2)
        cr2 = ClassicalRegister(2)
        cr3 = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr1, cr2, cr3)

        qc.h(qr[0]).c_if(cr2, 2)

        qobj = assemble_circuits(qc)

        bfunc_op, h_op = qobj.experiments[0].instructions

        self.assertEqual(bfunc_op.name, 'bfunc')
        self.assertEqual(bfunc_op.mask, '0xC')
        self.assertEqual(bfunc_op.val, '0x8')
        self.assertEqual(bfunc_op.relation, '==')

        self.assertTrue(hasattr(h_op, 'conditional'))
        self.assertEqual(bfunc_op.register, h_op.conditional)


if __name__ == '__main__':
    unittest.main(verbosity=2)
