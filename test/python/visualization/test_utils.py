# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for visualization tools."""

import unittest

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Qubit, Clbit
from qiskit.visualization import utils
from qiskit.test import QiskitTestCase


class TestVisualizationUtils(QiskitTestCase):
    """ Tests for visualizer utilities.
    Since the utilities in qiskit/tools/visualization/_utils.py are used by several visualizers
    the need to be check if the interface or their result changes."""

    def setUp(self):
        super().setUp()
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

    def test_get_layered_instructions(self):
        """ _get_layered_instructions without reverse_bits """
        (qregs, cregs, layered_ops) = utils._get_layered_instructions(self.circuit)

        exp = [[('cx', [self.qr2[0], self.qr2[1]], []),
                ('cx', [self.qr1[0], self.qr1[1]], [])],
               [('measure', [self.qr2[0]], [self.cr2[0]])],
               [('measure', [self.qr1[0]], [self.cr1[0]])],
               [('cx', [self.qr2[1], self.qr2[0]], []),
                ('cx', [self.qr1[1], self.qr1[0]], [])],
               [('measure', [self.qr2[1]], [self.cr2[1]])],
               [('measure', [self.qr1[1]], [self.cr1[1]])]
               ]

        self.assertEqual([self.qr1[0], self.qr1[1], self.qr2[0], self.qr2[1]], qregs)
        self.assertEqual([self.cr1[0], self.cr1[1], self.cr2[0], self.cr2[1]], cregs)
        self.assertEqual(exp,
                         [[(op.name, op.qargs, op.cargs) for op in ops] for ops in layered_ops])

    def test_get_layered_instructions_reverse_bits(self):
        """ _get_layered_instructions with reverse_bits=True """
        (qregs, cregs, layered_ops) = utils._get_layered_instructions(self.circuit,
                                                                      reverse_bits=True)

        exp = [[('cx', [self.qr2[0], self.qr2[1]], []),
                ('cx', [self.qr1[0], self.qr1[1]], [])],
               [('measure', [self.qr2[0]], [self.cr2[0]])],
               [('measure', [self.qr1[0]], [self.cr1[0]])],
               [('cx', [self.qr2[1], self.qr2[0]], []),
                ('cx', [self.qr1[1], self.qr1[0]], [])],
               [('measure', [self.qr2[1]], [self.cr2[1]])],
               [('measure', [self.qr1[1]], [self.cr1[1]])]
               ]

        self.assertEqual([self.qr2[1], self.qr2[0], self.qr1[1], self.qr1[0]], qregs)
        self.assertEqual([self.cr2[1], self.cr2[0], self.cr1[1], self.cr1[0]], cregs)
        self.assertEqual(exp,
                         [[(op.name, op.qargs, op.cargs) for op in ops] for ops in layered_ops])

    def test_get_layered_instructions_remove_idle_wires(self):
        """ _get_layered_instructions with idle_wires=False """
        qr1 = QuantumRegister(3, 'qr1')
        qr2 = QuantumRegister(3, 'qr2')
        cr1 = ClassicalRegister(3, 'cr1')
        cr2 = ClassicalRegister(3, 'cr2')

        circuit = QuantumCircuit(qr1, qr2, cr1, cr2)
        circuit.cx(qr2[0], qr2[1])
        circuit.measure(qr2[0], cr2[0])
        circuit.cx(qr2[1], qr2[0])
        circuit.measure(qr2[1], cr2[1])
        circuit.cx(qr1[0], qr1[1])
        circuit.measure(qr1[0], cr1[0])
        circuit.cx(qr1[1], qr1[0])
        circuit.measure(qr1[1], cr1[1])

        (qregs, cregs, layered_ops) = utils._get_layered_instructions(circuit, idle_wires=False)

        exp = [[('cx', [qr2[0], qr2[1]], []),
                ('cx', [qr1[0], qr1[1]], [])],
               [('measure', [qr2[0]], [cr2[0]])],
               [('measure', [qr1[0]], [cr1[0]])],
               [('cx', [qr2[1], qr2[0]], []),
                ('cx', [qr1[1], qr1[0]], [])],
               [('measure', [qr2[1]], [cr2[1]])],
               [('measure', [qr1[1]], [cr1[1]])]
               ]

        self.assertEqual([qr1[0], qr1[1], qr2[0], qr2[1]], qregs)
        self.assertEqual([cr1[0], cr1[1], cr2[0], cr2[1]], cregs)
        self.assertEqual(exp,
                         [[(op.name, op.qargs, op.cargs) for op in ops] for ops in layered_ops])

    def test_get_layered_instructions_left_justification_simple(self):
        """ Test _get_layered_instructions left justification simple since #2802
        q_0: |0>───────■──
                ┌───┐  │
        q_1: |0>┤ H ├──┼──
                ├───┤  │
        q_2: |0>┤ H ├──┼──
                └───┘┌─┴─┐
        q_3: |0>─────┤ X ├
                     └───┘
        """
        qc = QuantumCircuit(4)
        qc.h(1)
        qc.h(2)
        qc.cx(0, 3)

        (_, _, layered_ops) = utils._get_layered_instructions(qc, justify='left')

        l_exp = [[('h', [Qubit(QuantumRegister(4, 'q'), 1)], []),
                  ('h', [Qubit(QuantumRegister(4, 'q'), 2)], [])],
                 [('cx', [Qubit(QuantumRegister(4, 'q'), 0),
                          Qubit(QuantumRegister(4, 'q'), 3)], [])
                  ]
                 ]

        self.assertEqual(l_exp,
                         [[(op.name, op.qargs, op.cargs) for op in ops] for ops in layered_ops])

    def test_get_layered_instructions_right_justification_simple(self):
        """ Test _get_layered_instructions right justification simple since #2802
        q_0: |0>──■───────
                  │  ┌───┐
        q_1: |0>──┼──┤ H ├
                  │  ├───┤
        q_2: |0>──┼──┤ H ├
                ┌─┴─┐└───┘
        q_3: |0>┤ X ├─────
                └───┘
        """
        qc = QuantumCircuit(4)
        qc.h(1)
        qc.h(2)
        qc.cx(0, 3)

        (_, _, layered_ops) = utils._get_layered_instructions(qc, justify='right')

        r_exp = [[('cx', [Qubit(QuantumRegister(4, 'q'), 0),
                          Qubit(QuantumRegister(4, 'q'), 3)], [])],
                 [('h', [Qubit(QuantumRegister(4, 'q'), 1)], []),
                  ('h', [Qubit(QuantumRegister(4, 'q'), 2)], [])
                  ]
                 ]

        self.assertEqual(r_exp,
                         [[(op.name, op.qargs, op.cargs) for op in ops] for ops in layered_ops])

    def test_get_layered_instructions_left_justification_less_simple(self):
        """ Test _get_layered_instructions left justification
        less simple example since #2802
                ┌────────────┐┌───┐┌────────────┐              ┌─┐┌────────────┐┌───┐┌────────────┐
        q_0: |0>┤ U2(0,pi/1) ├┤ X ├┤ U2(0,pi/1) ├──────────────┤M├┤ U2(0,pi/1) ├┤ X ├┤ U2(0,pi/1) ├
                ├────────────┤└─┬─┘├────────────┤┌────────────┐└╥┘└────────────┘└─┬─┘├────────────┤
        q_1: |0>┤ U2(0,pi/1) ├──■──┤ U2(0,pi/1) ├┤ U2(0,pi/1) ├─╫─────────────────■──┤ U2(0,pi/1) ├
                └────────────┘     └────────────┘└────────────┘ ║                    └────────────┘
        q_2: |0>────────────────────────────────────────────────╫──────────────────────────────────
                                                                ║
        q_3: |0>────────────────────────────────────────────────╫──────────────────────────────────
                                                                ║
        q_4: |0>────────────────────────────────────────────────╫──────────────────────────────────
                                                                ║
        c1_0: 0 ════════════════════════════════════════════════╩══════════════════════════════════
        """
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[5];
        creg c1[1];
        u2(0,3.14159265358979) q[0];
        u2(0,3.14159265358979) q[1];
        cx q[1],q[0];
        u2(0,3.14159265358979) q[0];
        u2(0,3.14159265358979) q[1];
        u2(0,3.14159265358979) q[1];
        measure q[0] -> c1[0];
        u2(0,3.14159265358979) q[0];
        cx q[1],q[0];
        u2(0,3.14159265358979) q[0];
        u2(0,3.14159265358979) q[1];
        """
        qc = QuantumCircuit.from_qasm_str(qasm)

        (_, _, layered_ops) = utils._get_layered_instructions(qc, justify='left')

        l_exp = [[('u2', [Qubit(QuantumRegister(5, 'q'), 0)], []),
                  ('u2', [Qubit(QuantumRegister(5, 'q'), 1)], [])],
                 [('cx',
                   [Qubit(QuantumRegister(5, 'q'), 1), Qubit(QuantumRegister(5, 'q'), 0)],
                   [])],
                 [('u2', [Qubit(QuantumRegister(5, 'q'), 0)], []),
                  ('u2', [Qubit(QuantumRegister(5, 'q'), 1)], [])],
                 [('u2', [Qubit(QuantumRegister(5, 'q'), 1)], [])],
                 [('measure',
                   [Qubit(QuantumRegister(5, 'q'), 0)],
                   [Clbit(ClassicalRegister(1, 'c1'), 0)])],
                 [('u2', [Qubit(QuantumRegister(5, 'q'), 0)], [])],
                 [('cx',
                   [Qubit(QuantumRegister(5, 'q'), 1), Qubit(QuantumRegister(5, 'q'), 0)],
                   [])],
                 [('u2', [Qubit(QuantumRegister(5, 'q'), 0)], []),
                  ('u2', [Qubit(QuantumRegister(5, 'q'), 1)], [])]]

        self.assertEqual(l_exp,
                         [[(op.name, op.qargs, op.cargs) for op in ops] for ops in layered_ops])

    def test_get_layered_instructions_right_justification_less_simple(self):
        """ Test _get_layered_instructions right justification
        less simple example since #2802
                ┌────────────┐┌───┐┌────────────┐┌─┐┌────────────┐┌───┐┌────────────┐
        q_0: |0>┤ U2(0,pi/1) ├┤ X ├┤ U2(0,pi/1) ├┤M├┤ U2(0,pi/1) ├┤ X ├┤ U2(0,pi/1) ├
                ├────────────┤└─┬─┘├────────────┤└╥┘├────────────┤└─┬─┘├────────────┤
        q_1: |0>┤ U2(0,pi/1) ├──■──┤ U2(0,pi/1) ├─╫─┤ U2(0,pi/1) ├──■──┤ U2(0,pi/1) ├
                └────────────┘     └────────────┘ ║ └────────────┘     └────────────┘
        q_2: |0>──────────────────────────────────╫──────────────────────────────────
                                                  ║
        q_3: |0>──────────────────────────────────╫──────────────────────────────────
                                                  ║
        q_4: |0>──────────────────────────────────╫──────────────────────────────────
                                                  ║
        c1_0: 0 ══════════════════════════════════╩══════════════════════════════════
        """
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[5];
        creg c1[1];
        u2(0,3.14159265358979) q[0];
        u2(0,3.14159265358979) q[1];
        cx q[1],q[0];
        u2(0,3.14159265358979) q[0];
        u2(0,3.14159265358979) q[1];
        u2(0,3.14159265358979) q[1];
        measure q[0] -> c1[0];
        u2(0,3.14159265358979) q[0];
        cx q[1],q[0];
        u2(0,3.14159265358979) q[0];
        u2(0,3.14159265358979) q[1];
        """
        qc = QuantumCircuit.from_qasm_str(qasm)

        (_, _, layered_ops) = utils._get_layered_instructions(qc, justify='right')

        r_exp = [[('u2', [Qubit(QuantumRegister(5, 'q'), 0)], []),
                  ('u2', [Qubit(QuantumRegister(5, 'q'), 1)], [])],
                 [('cx',
                   [Qubit(QuantumRegister(5, 'q'), 1), Qubit(QuantumRegister(5, 'q'), 0)],
                   [])],
                 [('u2', [Qubit(QuantumRegister(5, 'q'), 0)], []),
                  ('u2', [Qubit(QuantumRegister(5, 'q'), 1)], [])],
                 [('measure',
                   [Qubit(QuantumRegister(5, 'q'), 0)],
                   [Clbit(ClassicalRegister(1, 'c1'), 0)])],
                 [('u2', [Qubit(QuantumRegister(5, 'q'), 0)], []),
                  ('u2', [Qubit(QuantumRegister(5, 'q'), 1)], [])],
                 [('cx',
                   [Qubit(QuantumRegister(5, 'q'), 1), Qubit(QuantumRegister(5, 'q'), 0)],
                   [])],
                 [('u2', [Qubit(QuantumRegister(5, 'q'), 0)], []),
                  ('u2', [Qubit(QuantumRegister(5, 'q'), 1)], [])]]

        self.assertEqual(r_exp,
                         [[(op.name, op.qargs, op.cargs) for op in ops] for ops in layered_ops])

    def test_generate_latex_label_nomathmode(self):
        """Test generate latex label default."""
        self.assertEqual('abc', utils.generate_latex_label('abc'))

    def test_generate_latex_label_nomathmode_utf8char(self):
        """Test generate latex label utf8 characters."""
        self.assertEqual('{\\ensuremath{\\iiint}}X{\\ensuremath{\\forall}}Y',
                         utils.generate_latex_label('∭X∀Y'))

    def test_generate_latex_label_mathmode_utf8char(self):
        """Test generate latex label mathtext with utf8."""
        self.assertEqual(
            'abc_{\\ensuremath{\\iiint}}X{\\ensuremath{\\forall}}Y',
            utils.generate_latex_label('$abc_$∭X∀Y'))

    def test_generate_latex_label_mathmode_underscore_outside(self):
        """Test generate latex label with underscore outside mathmode."""
        self.assertEqual(
            'abc{\\_}{\\ensuremath{\\iiint}}X{\\ensuremath{\\forall}}Y',
            utils.generate_latex_label('$abc$_∭X∀Y'))

    def test_generate_latex_label_escaped_dollar_signs(self):
        """Test generate latex label with escaped dollarsign."""
        self.assertEqual(
            '{\\$}{\\ensuremath{\\forall}}{\\$}',
            utils.generate_latex_label(r'\$∀\$'))

    def test_generate_latex_label_escaped_dollar_sign_in_mathmode(self):
        """Test generate latex label with escaped dollar sign in mathmode."""
        self.assertEqual(
            'a$bc{\\_}{\\ensuremath{\\iiint}}X{\\ensuremath{\\forall}}Y',
            utils.generate_latex_label(r'$a$bc$_∭X∀Y'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
