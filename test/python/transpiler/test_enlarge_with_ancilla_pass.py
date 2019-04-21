# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test the EnlargeWithAncilla pass"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.mapper import Layout
from qiskit.transpiler.passes import EnlargeWithAncilla
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase


class TestEnlargeWithAncilla(QiskitTestCase):
    """Tests the EnlargeWithAncilla pass."""

    def setUp(self):
        self.qr3 = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(self.qr3)
        circuit.h(self.qr3)
        self.dag = circuit_to_dag(circuit)

    def test_no_extension(self):
        """There are no idle physical bits to extend."""
        layout = Layout([(self.qr3, 0),
                         (self.qr3, 1),
                         (self.qr3, 2)])

        pass_ = EnlargeWithAncilla(layout)
        after = pass_.run(self.dag)

        qregs = list(after.qregs.values())
        self.assertEqual(1, len(qregs))
        self.assertEqual(self.qr3, qregs[0])

    def test_with_extension(self):
        """There are 2 idle physical bits to extend."""
        layout = Layout([(self.qr3, 0),
                         None,
                         (self.qr3, 1),
                         None,
                         (self.qr3, 2)])

        pass_ = EnlargeWithAncilla(layout)
        after = pass_.run(self.dag)

        final_layout = {0: (QuantumRegister(3, 'qr'), 0), 1: (QuantumRegister(2, 'ancilla'), 0),
                        2: (QuantumRegister(3, 'qr'), 1), 3: (QuantumRegister(2, 'ancilla'), 1),
                        4: (QuantumRegister(3, 'qr'), 2)}

        qregs = list(after.qregs.values())
        self.assertEqual(2, len(qregs))
        self.assertEqual(self.qr3, qregs[0])
        self.assertEqual(QuantumRegister(2, name='ancilla'), qregs[1])
        self.assertEqual(final_layout, layout.get_physical_bits())

    def test_name_collision(self):
        """Name collision during ancilla extension."""
        qr_ancilla = QuantumRegister(3, 'ancilla')
        circuit = QuantumCircuit(qr_ancilla)
        circuit.h(qr_ancilla)
        dag = circuit_to_dag(circuit)

        layout = Layout([(qr_ancilla, 0),
                         None,
                         (qr_ancilla, 1),
                         None,
                         (qr_ancilla, 2)])

        pass_ = EnlargeWithAncilla(layout)
        after = pass_.run(dag)

        qregs = list(after.qregs.values())
        self.assertEqual(2, len(qregs))
        self.assertEqual(qr_ancilla, qregs[0])
        self.assertEqual(2, qregs[1].size)
        self.assertRegex(qregs[1].name, r'^ancilla\d+$')


if __name__ == '__main__':
    unittest.main()
