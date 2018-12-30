# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for visualization tools."""

import os
import random
import unittest
from inspect import signature

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.tools.visualization import _utils, circuit_drawer
from qiskit.test import QiskitTestCase


class TestLatexSourceGenerator(QiskitTestCase):
    """Qiskit latex source generator tests."""

    def random_circuit(self, width=3, depth=3, max_operands=3):
        """Generate random circuit of arbitrary size.
        Note: the depth is the layers of independent operation. true depth
        in the image may be more for visualization purposes, if gates overlap.

        Args:
            width (int): number of quantum wires
            depth (int): layers of operations
            max_operands (int): maximum operands of each gate

        Returns:
            QuantumCircuit: constructed circuit
        """
        qr = QuantumRegister(width, "q")
        qc = QuantumCircuit(qr)

        one_q_ops = "iden,u0,u1,u2,u3,x,y,z,h,s,sdg,t,tdg,rx,ry,rz"
        two_q_ops = "cx,cy,cz,ch,crz,cu1,cu3,swap"
        three_q_ops = "ccx"

        # apply arbitrary random operations at every depth
        for _ in range(depth):
            # choose either 1, 2, or 3 qubits for the operation
            remaining_qubits = list(range(width))
            while remaining_qubits:
                max_possible_operands = min(len(remaining_qubits), max_operands)
                num_operands = random.choice(range(max_possible_operands)) + 1
                operands = random.sample(remaining_qubits, num_operands)
                remaining_qubits = [q for q in remaining_qubits if q not in operands]
                if num_operands == 1:
                    operation = random.choice(one_q_ops.split(','))
                elif num_operands == 2:
                    operation = random.choice(two_q_ops.split(','))
                elif num_operands == 3:
                    operation = random.choice(three_q_ops.split(','))
                # every gate is defined as a method of the QuantumCircuit class
                # the code below is so we can call a gate by its name
                gate = getattr(QuantumCircuit, operation)
                op_args = list(signature(gate).parameters.keys())
                num_angles = len(op_args) - num_operands - 1  # -1 for the 'self' arg
                angles = [random.uniform(0, 3.14) for x in range(num_angles)]
                register_operands = [qr[i] for i in operands]
                gate(qc, *angles, *register_operands)

        return qc

    def test_tiny_circuit(self):
        """Test draw tiny circuit."""
        filename = self._get_resource_path('test_tiny.tex')
        qc = self.random_circuit(1, 1, 1)
        try:
            circuit_drawer(qc, filename=filename, output='latex_source')
            self.assertNotEqual(os.path.exists(filename), False)
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_normal_circuit(self):
        """Test draw normal size circuit."""
        filename = self._get_resource_path('test_normal.tex')
        qc = self.random_circuit(5, 5, 3)
        try:
            circuit_drawer(qc, filename=filename, output='latex_source')
            self.assertNotEqual(os.path.exists(filename), False)
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_wide_circuit(self):
        """Test draw wide circuit."""
        filename = self._get_resource_path('test_wide.tex')
        qc = self.random_circuit(100, 1, 1)
        try:
            circuit_drawer(qc, filename=filename, output='latex_source')
            self.assertNotEqual(os.path.exists(filename), False)
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_deep_circuit(self):
        """Test draw deep circuit."""
        filename = self._get_resource_path('test_deep.tex')
        qc = self.random_circuit(1, 100, 1)
        try:
            circuit_drawer(qc, filename=filename, output='latex_source')
            self.assertNotEqual(os.path.exists(filename), False)
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_huge_circuit(self):
        """Test draw huge circuit."""
        filename = self._get_resource_path('test_huge.tex')
        qc = self.random_circuit(40, 40, 1)
        try:
            circuit_drawer(qc, filename=filename, output='latex_source')
            self.assertNotEqual(os.path.exists(filename), False)
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    def test_teleport(self):
        """Test draw teleport circuit."""
        filename = self._get_resource_path('test_teleport.tex')
        qr = QuantumRegister(3, 'q')
        cr = ClassicalRegister(3, 'c')
        qc = QuantumCircuit(qr, cr)
        # Prepare an initial state
        qc.u3(0.3, 0.2, 0.1, qr[0])
        # Prepare a Bell pair
        qc.h(qr[1])
        qc.cx(qr[1], qr[2])
        # Barrier following state preparation
        qc.barrier(qr)
        # Measure in the Bell basis
        qc.cx(qr[0], qr[1])
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        # Apply a correction
        qc.z(qr[2]).c_if(cr, 1)
        qc.x(qr[2]).c_if(cr, 2)
        qc.measure(qr[2], cr[2])
        try:
            circuit_drawer(qc, filename=filename, output='latex_source')
            self.assertNotEqual(os.path.exists(filename), False)
        finally:
            if os.path.exists(filename):
                os.remove(filename)


class TestVisualizationUtils(QiskitTestCase):
    """ Tests for visualizer utilities.
    Since the utilities in qiskit/tools/visualization/_utils.py are used by several visualizers
    the need to be check if the interface or their result changes."""

    def setUp(self):
        self.qr1 = QuantumRegister(2, 'qr1')
        self.qr2 = QuantumRegister(2, 'qr2')
        self.cr1 = ClassicalRegister(2, 'cr1')
        self.cr2 = ClassicalRegister(2, 'cr2')

        self.circuit = QuantumCircuit(self.qr1, self.qr2, self.cr1, self.cr2)
        self.circuit.cx(self.qr2[0], self.qr2[1])
        self.circuit.measure(self.qr2[0], self.cr2[0])
        self.circuit.cx(self.qr2[1], self.qr2[0])
        self.circuit.measure(self.qr2[1], self.cr2[1])
        self.circuit.cx(self.qr1[0], self.qr1[1])
        self.circuit.measure(self.qr1[0], self.cr1[0])
        self.circuit.cx(self.qr1[1], self.qr1[0])
        self.circuit.measure(self.qr1[1], self.cr1[1])

    def test_get_instructions(self):
        """ _get_instructions without reversebits """
        (qregs, cregs, ops) = _utils._get_instructions(self.circuit)

        self.assertEqual([(self.qr1, 0), (self.qr1, 1), (self.qr2, 0), (self.qr2, 1)], qregs)
        self.assertEqual([(self.cr1, 0), (self.cr1, 1), (self.cr2, 0), (self.cr2, 1)], cregs)
        self.assertEqual([op['name'] for op in ops],
                         ['cx', 'measure', 'cx', 'measure', 'cx', 'measure', 'cx', 'measure'])
        self.assertEqual([op['qargs'] for op in ops], [[(self.qr2, 0), (self.qr2, 1)],
                                                       [(self.qr2, 0)],
                                                       [(self.qr2, 1), (self.qr2, 0)],
                                                       [(self.qr2, 1)],
                                                       [(self.qr1, 0), (self.qr1, 1)],
                                                       [(self.qr1, 0)],
                                                       [(self.qr1, 1), (self.qr1, 0)],
                                                       [(self.qr1, 1)]])
        self.assertEqual([op['cargs'] for op in ops], [[],
                                                       [(self.cr2, 0)],
                                                       [],
                                                       [(self.cr2, 1)],
                                                       [],
                                                       [(self.cr1, 0)],
                                                       [],
                                                       [(self.cr1, 1)]])

    def test_get_instructions_reversebits(self):
        """ _get_instructions with reversebits=True """
        (qregs, cregs, ops) = _utils._get_instructions(self.circuit, reversebits=True)

        self.assertEqual([(self.qr2, 1), (self.qr2, 0), (self.qr1, 1), (self.qr1, 0)], qregs)
        self.assertEqual([(self.cr2, 1), (self.cr2, 0), (self.cr1, 1), (self.cr1, 0)], cregs)
        self.assertEqual(['cx', 'measure', 'cx', 'measure', 'cx', 'measure', 'cx', 'measure'],
                         [op['name'] for op in ops])
        self.assertEqual([op['qargs'] for op in ops], [[(self.qr2, 0), (self.qr2, 1)],
                                                       [(self.qr2, 0)],
                                                       [(self.qr2, 1), (self.qr2, 0)],
                                                       [(self.qr2, 1)],
                                                       [(self.qr1, 0), (self.qr1, 1)],
                                                       [(self.qr1, 0)],
                                                       [(self.qr1, 1), (self.qr1, 0)],
                                                       [(self.qr1, 1)]])
        self.assertEqual([op['cargs'] for op in ops], [[],
                                                       [(self.cr2, 0)],
                                                       [],
                                                       [(self.cr2, 1)],
                                                       [],
                                                       [(self.cr1, 0)],
                                                       [],
                                                       [(self.cr1, 1)]])


if __name__ == '__main__':
    unittest.main(verbosity=2)
