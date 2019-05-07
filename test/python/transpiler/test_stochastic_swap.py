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

"""Test the Stochastic Swap pass"""

import unittest
from qiskit.transpiler.passes import StochasticSwap
from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.converters import circuit_to_dag
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.test import QiskitTestCase


class TestStochasticSwap(QiskitTestCase):
    """
    Tests the StochasticSwap pass.
    All of the tests use a fixed seed since the results
    may depend on it.
    """

    def test_multiple_registers_with_layout_adjust(self):
        """
        Test two registers + measurements using a layout.
        The mapper will adjust the initial layout so that
        all of the gates can be done without swaps.
        """
        coupling = CouplingMap([[0, 1], [1, 2]])

        qr_q = QuantumRegister(2, 'q')
        qr_a = QuantumRegister(1, 'a')
        cr_c = ClassicalRegister(3, 'c')
        circ = QuantumCircuit(qr_q, qr_a, cr_c)
        circ.cx(qr_q[0], qr_a[0])
        circ.cx(qr_q[1], qr_a[0])
        circ.measure(qr_q[0], cr_c[0])
        circ.measure(qr_q[1], cr_c[1])
        circ.measure(qr_a[0], cr_c[2])
        dag = circuit_to_dag(circ)

        layout = Layout({qr_q[0]: 0, qr_q[1]: 1, qr_a[0]: 2})

        pass_ = StochasticSwap(coupling, layout, 20, 13)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_multiple_registers_with_good_layout(self):
        """
        Test two registers + measurements using a layout.
        The layout makes all gates nearest neighbor.
        """
        coupling = CouplingMap([[0, 1], [1, 2]])

        qr_q = QuantumRegister(2, 'q')
        qr_a = QuantumRegister(1, 'a')
        cr_c = ClassicalRegister(3, 'c')
        circ = QuantumCircuit(qr_q, qr_a, cr_c)
        circ.cx(qr_q[0], qr_a[0])
        circ.cx(qr_q[1], qr_a[0])
        circ.measure(qr_q[0], cr_c[0])
        circ.measure(qr_q[1], cr_c[1])
        circ.measure(qr_a[0], cr_c[2])
        dag = circuit_to_dag(circ)

        layout = Layout({qr_q[0]: 0, qr_a[0]: 1, qr_q[1]: 2})

        pass_ = StochasticSwap(coupling, layout, 20, 13)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_multiple_registers_with_default_layout(self):
        """
        Test two registers + measurements using no layout.
        The default layout will be adjusted to all gates
        become nearest neighbor. The pass has the layout
        in pass_.initial_layout.
        """
        coupling = CouplingMap([[0, 1], [1, 2]])

        qr_q = QuantumRegister(2, 'q')
        qr_a = QuantumRegister(1, 'a')
        cr_c = ClassicalRegister(3, 'c')
        circ = QuantumCircuit(qr_q, qr_a, cr_c)
        circ.cx(qr_q[0], qr_a[0])
        circ.cx(qr_q[1], qr_a[0])
        circ.measure(qr_q[0], cr_c[0])
        circ.measure(qr_q[1], cr_c[1])
        circ.measure(qr_a[0], cr_c[2])
        dag = circuit_to_dag(circ)

        layout = None

        pass_ = StochasticSwap(coupling, layout, 20, 13)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_trivial_case(self):
        """
         q0:--(+)-[H]-(+)-
               |       |
         q1:---.-------|--
                       |
         q2:-----------.--
         Coupling map: [1]--[0]--[2]
        """
        coupling = CouplingMap([[0, 1], [0, 2]])

        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[2])

        dag = circuit_to_dag(circuit)
        pass_ = StochasticSwap(coupling, None, 20, 13)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_trivial_in_same_layer(self):
        """
         q0:--(+)--
               |
         q1:---.---
         q2:--(+)--
               |
         q3:---.---
         Coupling map: [0]--[1]--[2]--[3]
        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[0], qr[1])

        dag = circuit_to_dag(circuit)
        pass_ = StochasticSwap(coupling, None, 20, 13)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_permute_wires_1(self):
        """All of the test_permute_wires tests are derived
        from the basic mapper tests. In this case, the
        stochastic mapper handles a single
        layer by qubit label permutations so as not to
        introduce additional swap gates. The new
        initial layout is found in pass_.initial_layout.
         q0:-------
         q1:--(+)--
               |
         q2:---.---
         Coupling map: [1]--[0]--[2]
        """
        coupling = CouplingMap([[0, 1], [0, 2]])

        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[2])
        dag = circuit_to_dag(circuit)

        pass_ = StochasticSwap(coupling, None, 20, 13)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_permute_wires_2(self):
        """
         qr0:---.---[H]--
                |
         qr1:---|--------
                |
         qr2:--(+)-------
         Coupling map: [0]--[1]--[2]
        """
        coupling = CouplingMap([[1, 0], [1, 2]])

        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        circuit.h(qr[0])
        dag = circuit_to_dag(circuit)

        pass_ = StochasticSwap(coupling, None, 20, 13)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_permute_wires_3(self):
        """
         qr0:--(+)---.--
                |    |
         qr1:---|----|--
                |    |
         qr2:---|----|--
                |    |
         qr3:---.---(+)-
         Coupling map: [0]--[1]--[2]--[3]
        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[3], qr[0])
        dag = circuit_to_dag(circuit)

        pass_ = StochasticSwap(coupling, None, 20, 13)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_permute_wires_4(self):
        """No qubit label permutation occurs if the first
        layer has only single-qubit gates. This is suboptimal
        but seems to be the current behavior.
         qr0:------(+)--
                    |
         qr1:-------|---
                    |
         qr2:-------|---
                    |
         qr3:--[H]--.---
         Coupling map: [0]--[1]--[2]--[3]
         qr0:------X---------
                   |
         qr1:------X-(+)-----
                      |
         qr2:------X--.------
                   |
         qr3:-[H]--X---------
        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, 'q')
        circuit = QuantumCircuit(qr)
        circuit.h(qr[3])
        circuit.cx(qr[3], qr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.h(qr[3])
        expected.swap(qr[2], qr[3])
        expected.swap(qr[0], qr[1])
        expected.cx(qr[2], qr[1])

        pass_ = StochasticSwap(coupling, None, 20, 13)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_permute_wires_5(self):
        """This is the same case as permute_wires_4
        except the single qubit gate is after the two-qubit
        gate, so the layout is adjusted.
         qr0:--(+)------
                |
         qr1:---|-------
                |
         qr2:---|-------
                |
         qr3:---.--[H]--
         Coupling map: [0]--[1]--[2]--[3]
        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[3], qr[0])
        circuit.h(qr[3])
        dag = circuit_to_dag(circuit)

        pass_ = StochasticSwap(coupling, None, 20, 13)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_permute_wires_6(self):
        """
         qr0:--(+)-------.--
                |        |
         qr1:---|--------|--
                |
         qr2:---|--------|--
                |        |
         qr3:---.--[H]--(+)-
         Coupling map: [0]--[1]--[2]--[3]
        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[3], qr[0])
        circuit.h(qr[3])
        circuit.cx(qr[0], qr[3])
        dag = circuit_to_dag(circuit)

        pass_ = StochasticSwap(coupling, None, 20, 13)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_overoptimization_case(self):
        """Check mapper overoptimization.
        The mapper should not change the semantics of the input.
        An overoptimization introduced issue #81:
        https://github.com/Qiskit/qiskit-terra/issues/81
        """
        coupling = CouplingMap([[0, 2], [1, 2], [2, 3]])
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr[0])
        circuit.y(qr[1])
        circuit.z(qr[2])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        circuit.s(qr[1])
        circuit.t(qr[2])
        circuit.h(qr[3])
        circuit.cx(qr[1], qr[2])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[1])
        circuit.measure(qr[2], cr[2])
        circuit.measure(qr[3], cr[3])
        dag = circuit_to_dag(circuit)
        #           ┌───┐                                        ┌─┐
        # q_0: | 0 >┤ X ├────────────■───────────────────────────┤M├─────────
        #           └───┘┌───┐     ┌─┴─┐     ┌───┐               └╥┘┌─┐
        # q_1: | 0 >─────┤ Y ├─────┤ X ├─────┤ S ├────────────■───╫─┤M├──────
        #                └───┘┌───┐└───┘     └───┘┌───┐     ┌─┴─┐ ║ └╥┘┌─┐
        # q_2: | 0 >──────────┤ Z ├───────■───────┤ T ├─────┤ X ├─╫──╫─┤M├───
        #                     └───┘     ┌─┴─┐     └───┘┌───┐└───┘ ║  ║ └╥┘┌─┐
        # q_3: | 0 >────────────────────┤ X ├──────────┤ H ├──────╫──╫──╫─┤M├
        #                               └───┘          └───┘      ║  ║  ║ └╥┘
        # c_0: 0    ══════════════════════════════════════════════╩══╬══╬══╬═
        #                                                            ║  ║  ║
        #  c_1: 0   ═════════════════════════════════════════════════╩══╬══╬═
        #                                                               ║  ║
        #  c_2: 0   ════════════════════════════════════════════════════╩══╬═
        #                                                                  ║
        #  c_3: 0   ═══════════════════════════════════════════════════════╩═
        #
        expected = QuantumCircuit(qr, cr)
        expected.z(qr[2])
        expected.y(qr[1])
        expected.x(qr[0])
        expected.swap(qr[1], qr[2])
        expected.cx(qr[0], qr[2])
        expected.swap(qr[2], qr[3])
        expected.cx(qr[1], qr[2])
        expected.s(qr[3])
        expected.t(qr[1])
        expected.h(qr[2])
        expected.measure(qr[0], cr[0])
        expected.swap(qr[1], qr[2])
        expected.cx(qr[3], qr[2])
        expected.measure(qr[1], cr[3])
        expected.measure(qr[3], cr[1])
        expected.measure(qr[2], cr[2])
        expected_dag = circuit_to_dag(expected)
        #                      ┌───┐     ┌─┐
        # q_0: |0>─────────────┤ X ├──■──┤M├────────────────────────────────────────
        #              ┌───┐   └───┘  │  └╥┘             ┌───┐        ┌───┐┌─┐
        # q_1: |0>─────┤ Y ├─X────────┼───╫───────────■──┤ T ├────────┤ X ├┤M├──────
        #         ┌───┐└───┘ │      ┌─┴─┐ ║         ┌─┴─┐└───┘┌───┐   └─┬─┘└╥┘┌─┐
        # q_2: |0>┤ Z ├──────X──────┤ X ├─╫──X──────┤ X ├─────┤ H ├─X───■───╫─┤M├───
        #         └───┘             └───┘ ║  │ ┌───┐└───┘     └───┘ │       ║ └╥┘┌─┐
        # q_3: |0>────────────────────────╫──X─┤ S ├────────────────X───────╫──╫─┤M├
        #                                 ║    └───┘                        ║  ║ └╥┘
        #  c_0: 0 ════════════════════════╩═════════════════════════════════╬══╬══╬═
        #                                                                   ║  ║  ║
        #  c_1: 0 ══════════════════════════════════════════════════════════╬══╩══╬═
        #                                                                   ║     ║
        #  c_2: 0 ══════════════════════════════════════════════════════════╩═════╬═
        #                                                                         ║
        #  c_3: 0 ════════════════════════════════════════════════════════════════╩═

        #
        # Layout --
        #  {qr[0]: 0,
        #  qr[1]: 1,
        #  qr[2]: 2,
        #  qr[3]: 3}
        pass_ = StochasticSwap(coupling, None, 20, 13)
        after = pass_.run(dag)

        self.assertEqual(expected_dag, after)

    def test_already_mapped(self):
        """Circuit not remapped if matches topology.
        See: https://github.com/Qiskit/qiskit-terra/issues/342
        """
        coupling = CouplingMap(
            [[1, 0], [1, 2], [2, 3], [3, 4], [3, 14], [5, 4], [6, 5],
             [6, 7], [6, 11], [7, 10], [8, 7], [9, 8], [9, 10],
             [11, 10], [12, 5], [12, 11], [12, 13], [13, 4], [13, 14],
             [15, 0], [15, 0], [15, 2], [15, 14]])
        qr = QuantumRegister(16, 'q')
        cr = ClassicalRegister(16, 'c')
        circ = QuantumCircuit(qr, cr)
        circ.cx(qr[3], qr[14])
        circ.cx(qr[5], qr[4])
        circ.h(qr[9])
        circ.cx(qr[9], qr[8])
        circ.x(qr[11])
        circ.cx(qr[3], qr[4])
        circ.cx(qr[12], qr[11])
        circ.cx(qr[13], qr[4])
        for j in range(16):
            circ.measure(qr[j], cr[j])

        dag = circuit_to_dag(circ)

        pass_ = StochasticSwap(coupling, None, 20, 13)
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(circ), after)

    def test_map_with_layout(self):
        """Test using an initial layout."""
        coupling = CouplingMap([[0, 1], [1, 2]])
        qra = QuantumRegister(2, 'qa')
        qrb = QuantumRegister(1, 'qb')
        cr = ClassicalRegister(3, 'r')
        circ = QuantumCircuit(qra, qrb, cr)
        circ.cx(qra[0], qrb[0])
        circ.measure(qra[0], cr[0])
        circ.measure(qra[1], cr[1])
        circ.measure(qrb[0], cr[2])
        dag = circuit_to_dag(circ)

        layout = Layout({qra[0]: 0, qra[1]: 1, qrb[0]: 2})

        pass_ = StochasticSwap(coupling, layout, 20, 13)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_congestion(self):
        """Test code path that falls back to serial layers."""
        coupling = CouplingMap([[0, 1], [1, 2], [1, 3]])
        qr = QuantumRegister(2, 'q')
        ar = QuantumRegister(2, 'a')
        cr = ClassicalRegister(4, 'c')
        circ = QuantumCircuit(qr, ar, cr)
        circ.cx(qr[1], ar[0])
        circ.cx(qr[0], ar[1])
        circ.measure(qr[0], cr[0])
        circ.h(qr)
        circ.h(ar)
        circ.cx(qr[0], qr[1])
        circ.cx(ar[0], ar[1])
        circ.measure(qr[0], cr[0])
        circ.measure(qr[1], cr[1])
        circ.measure(ar[0], cr[2])
        circ.measure(ar[1], cr[3])
        dag = circuit_to_dag(circ)
        #                                             ┌─┐┌───┐        ┌─┐
        # q_0: |0>─────────────────■──────────────────┤M├┤ H ├──■─────┤M├
        #                   ┌───┐  │                  └╥┘└───┘┌─┴─┐┌─┐└╥┘
        # q_1: |0>──■───────┤ H ├──┼───────────────────╫──────┤ X ├┤M├─╫─
        #         ┌─┴─┐┌───┐└───┘  │               ┌─┐ ║      └───┘└╥┘ ║
        # a_0: |0>┤ X ├┤ H ├───────┼─────────■─────┤M├─╫────────────╫──╫─
        #         └───┘└───┘     ┌─┴─┐┌───┐┌─┴─┐┌─┐└╥┘ ║            ║  ║
        # a_1: |0>───────────────┤ X ├┤ H ├┤ X ├┤M├─╫──╫────────────╫──╫─
        #                        └───┘└───┘└───┘└╥┘ ║  ║            ║  ║
        #  c_0: 0 ═══════════════════════════════╬══╬══╩════════════╬══╩═
        #                                        ║  ║               ║
        #  c_1: 0 ═══════════════════════════════╬══╬═══════════════╩════
        #                                        ║  ║
        #  c_2: 0 ═══════════════════════════════╬══╩════════════════════
        #                                        ║
        #  c_3: 0 ═══════════════════════════════╩═══════════════════════
        #
        #                                ┌─┐┌───┐                     ┌─┐
        # q_0: |0>────────────────────■──┤M├┤ H ├──────────────────■──┤M├──────
        #                           ┌─┴─┐└╥┘└───┘┌───┐┌───┐      ┌─┴─┐└╥┘┌─┐
        # q_1: |0>──■───X───────────┤ X ├─╫──────┤ H ├┤ X ├─X────┤ X ├─╫─┤M├───
        #         ┌─┴─┐ │      ┌───┐└───┘ ║      └───┘└─┬─┘ │    └───┘ ║ └╥┘┌─┐
        # a_0: |0>┤ X ├─┼──────┤ H ├──────╫─────────────■───┼──────────╫──╫─┤M├
        #         └───┘ │ ┌───┐└───┘      ║                 │ ┌─┐      ║  ║ └╥┘
        # a_1: |0>──────X─┤ H ├───────────╫─────────────────X─┤M├──────╫──╫──╫─
        #                 └───┘           ║                   └╥┘      ║  ║  ║
        #  c_0: 0 ════════════════════════╩════════════════════╬═══════╩══╬══╬═
        #                                                      ║          ║  ║
        #  c_1: 0 ═════════════════════════════════════════════╬══════════╩══╬═
        #                                                      ║             ║
        #  c_2: 0 ═════════════════════════════════════════════╬═════════════╩═
        #                                                      ║
        #  c_3: 0 ═════════════════════════════════════════════╩═══════════════
        #
        # Layout from mapper:
        # {qr[0]: 0,
        #  qr[1]: 1,
        #  ar[0]: 2,
        #  ar[1]: 3}
        #
        #     2
        #     |
        # 0 - 1 - 3
        expected = QuantumCircuit(qr, ar, cr)
        expected.cx(qr[1], ar[0])
        expected.swap(qr[0], qr[1])
        expected.cx(qr[1], ar[1])
        expected.h(ar[1])
        expected.h(ar[0])
        expected.measure(qr[1], cr[0])
        expected.h(qr[0])
        expected.swap(qr[1], ar[1])
        expected.h(ar[1])
        expected.cx(ar[0], qr[1])
        expected.measure(ar[0], cr[2])
        expected.swap(qr[1], ar[1])
        expected.measure(ar[1], cr[3])
        expected.cx(qr[1], qr[0])
        expected.measure(qr[1], cr[0])
        expected.measure(qr[0], cr[1])
        expected_dag = circuit_to_dag(expected)

        layout = Layout({qr[0]: 0, qr[1]: 1, ar[0]: 2, ar[1]: 3})

        pass_ = StochasticSwap(coupling, layout, 20, 13)
        after = pass_.run(dag)
        self.assertEqual(expected_dag, after)

    def test_all_single_qubit(self):
        """Test all trivial layers."""
        coupling = CouplingMap([[0, 1], [1, 2], [1, 3]])
        qr = QuantumRegister(2, 'q')
        ar = QuantumRegister(2, 'a')
        cr = ClassicalRegister(4, 'c')
        circ = QuantumCircuit(qr, ar, cr)
        circ.h(qr)
        circ.h(ar)
        circ.s(qr)
        circ.s(ar)
        circ.t(qr)
        circ.t(ar)
        circ.measure(qr[0], cr[0])  # intentional duplicate
        circ.measure(qr[0], cr[0])
        circ.measure(qr[1], cr[1])
        circ.measure(ar[0], cr[2])
        circ.measure(ar[1], cr[3])
        dag = circuit_to_dag(circ)

        layout = Layout({qr[0]: 0, qr[1]: 1, ar[0]: 2, ar[1]: 3})

        pass_ = StochasticSwap(coupling, layout, 20, 13)
        after = pass_.run(dag)
        self.assertEqual(dag, after)

    def test_only_output_cx_and_swaps_in_coupling_map(self):
        """Test that output DAG contains only 2q gates from the the coupling map."""

        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[0], qr[3])
        circuit.measure(qr, cr)
        dag = circuit_to_dag(circuit)

        layout = Layout({qr[0]: 0, qr[1]: 1, qr[2]: 2, qr[3]: 3})

        pass_ = StochasticSwap(coupling, layout, 20, 5)
        after = pass_.run(dag)

        valid_couplings = [set([layout[a], layout[b]])
                           for (a, b) in coupling.get_edges()]

        for _2q_gate in after.twoQ_gates():
            self.assertIn(set(_2q_gate.qargs), valid_couplings)

    def test_len_coupling_vs_dag(self):
        """Test error if coupling map and dag are not the same size."""

        coupling = CouplingMap([[0, 1], [1, 2], [2, 3], [3, 4]])
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[0], qr[3])
        circuit.measure(qr, cr)
        dag = circuit_to_dag(circuit)

        pass_ = StochasticSwap(coupling)
        with self.assertRaises(TranspilerError):
            _ = pass_.run(dag)

    def test_len_layout_vs_dag(self):
        """Test error if the layout and dag are not the same size."""

        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[0], qr[3])
        circuit.measure(qr, cr)
        dag = circuit_to_dag(circuit)

        layout = Layout({qr[0]: 0, qr[1]: 1, qr[2]: 2})

        pass_ = StochasticSwap(coupling, layout)
        with self.assertRaises(TranspilerError):
            _ = pass_.run(dag)


if __name__ == '__main__':
    unittest.main()
