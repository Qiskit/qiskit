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
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.test import QiskitTestCase


class TestStochasticSwap(QiskitTestCase):
    """
    Tests the StochasticSwap pass.
    All of the tests use a fixed seed since the results
    may depend on it.
    """

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

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[2])

        dag = circuit_to_dag(circuit)
        pass_ = StochasticSwap(coupling, 20, 13)
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

        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[0], qr[1])

        dag = circuit_to_dag(circuit)
        pass_ = StochasticSwap(coupling, 20, 13)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_permute_wires_1(self):
        """
        q0:--------

        q1:---.----
              |
        q2:--(+)---
        Coupling map: [1]--[0]--[2]
        q0:--x-(+)-
             |  |
        q1:--|--.--
             |
        q2:--x-----
        """
        coupling = CouplingMap([[0, 1], [0, 2]])

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[2])
        dag = circuit_to_dag(circuit)

        pass_ = StochasticSwap(coupling, 20, 11)
        after = pass_.run(dag)

        expected = QuantumCircuit(qr)
        expected.swap(qr[0], qr[2])
        expected.cx(qr[1], qr[0])

        self.assertEqual(circuit_to_dag(expected), after)

    def test_permute_wires_2(self):
        """
        qr0:---.---[H]--
               |
        qr1:---|--------
               |
        qr2:--(+)-------
        Coupling map: [0]--[1]--[2]
        qr0:----.---[H]-
                |
        qr1:-x-(+)------
             |
        qr2:-x----------
        """
        coupling = CouplingMap([[1, 0], [1, 2]])

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        circuit.h(qr[0])
        dag = circuit_to_dag(circuit)

        pass_ = StochasticSwap(coupling, 20, 11)
        after = pass_.run(dag)

        expected = QuantumCircuit(qr)
        expected.swap(qr[1], qr[2])
        expected.cx(qr[0], qr[1])
        expected.h(qr[0])

        self.assertEqual(expected, dag_to_circuit(after))

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
        qr0:-x------------
             |
        qr1:-x--(+)---.---
                 |    |
        qr2:-x---.---(+)--
             |
        qr3:-x------------
        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[3], qr[0])
        dag = circuit_to_dag(circuit)

        pass_ = StochasticSwap(coupling, 20, 13)
        after = pass_.run(dag)

        expected = QuantumCircuit(qr)
        expected.swap(qr[0], qr[1])
        expected.swap(qr[2], qr[3])
        expected.cx(qr[1], qr[2])
        expected.cx(qr[2], qr[1])

        self.assertEqual(circuit_to_dag(expected), after)

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

        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[3])
        circuit.cx(qr[3], qr[0])
        dag = circuit_to_dag(circuit)

        pass_ = StochasticSwap(coupling, 20, 13)
        after = pass_.run(dag)

        expected = QuantumCircuit(qr)
        expected.h(qr[3])
        expected.swap(qr[2], qr[3])
        expected.swap(qr[0], qr[1])
        expected.cx(qr[2], qr[1])

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
         qr0:-x-----------
              |
         qr1:-x--(+)------
                  |
         qr2:-x---.--[H]--
              |
         qr3:-x-----------
        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[3], qr[0])
        circuit.h(qr[3])
        dag = circuit_to_dag(circuit)

        pass_ = StochasticSwap(coupling, 20, 13)
        after = pass_.run(dag)

        expected = QuantumCircuit(qr)
        expected.swap(qr[0], qr[1])
        expected.swap(qr[2], qr[3])
        expected.cx(qr[2], qr[1])
        expected.h(qr[2])

        self.assertEqual(circuit_to_dag(expected), after)

    def test_all_single_qubit(self):
        """Test all trivial layers."""
        coupling = CouplingMap([[0, 1], [1, 2], [1, 3]])
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(4, "c")
        circ = QuantumCircuit(qr, cr)
        circ.h(qr)
        circ.z(qr)
        circ.s(qr)
        circ.t(qr)
        circ.tdg(qr)
        circ.measure(qr[0], cr[0])  # intentional duplicate
        circ.measure(qr[0], cr[0])
        circ.measure(qr[1], cr[1])
        circ.measure(qr[2], cr[2])
        circ.measure(qr[3], cr[3])
        dag = circuit_to_dag(circ)

        pass_ = StochasticSwap(coupling, 20, 13)
        after = pass_.run(dag)
        self.assertEqual(dag, after)

    def test_overoptimization_case(self):
        """Check mapper overoptimization.
        The mapper should not change the semantics of the input.
        An overoptimization introduced issue #81:
        https://github.com/Qiskit/qiskit-terra/issues/81
        """
        coupling = CouplingMap([[0, 2], [1, 2], [2, 3]])
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(4, "c")
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
        pass_ = StochasticSwap(coupling, 20, 19)
        after = pass_.run(dag)

        self.assertEqual(expected_dag, after)

    def test_already_mapped(self):
        """Circuit not remapped if matches topology.
        See: https://github.com/Qiskit/qiskit-terra/issues/342
        """
        coupling = CouplingMap(
            [
                [1, 0],
                [1, 2],
                [2, 3],
                [3, 4],
                [3, 14],
                [5, 4],
                [6, 5],
                [6, 7],
                [6, 11],
                [7, 10],
                [8, 7],
                [9, 8],
                [9, 10],
                [11, 10],
                [12, 5],
                [12, 11],
                [12, 13],
                [13, 4],
                [13, 14],
                [15, 0],
                [15, 0],
                [15, 2],
                [15, 14],
            ]
        )
        qr = QuantumRegister(16, "q")
        cr = ClassicalRegister(16, "c")
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

        pass_ = StochasticSwap(coupling, 20, 13)
        after = pass_.run(dag)
        self.assertEqual(circuit_to_dag(circ), after)

    def test_congestion(self):
        """Test code path that falls back to serial layers."""
        coupling = CouplingMap([[0, 1], [1, 2], [1, 3]])
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(4, "c")
        circ = QuantumCircuit(qr, cr)
        circ.cx(qr[1], qr[2])
        circ.cx(qr[0], qr[3])
        circ.measure(qr[0], cr[0])
        circ.h(qr)
        circ.cx(qr[0], qr[1])
        circ.cx(qr[2], qr[3])
        circ.measure(qr[0], cr[0])
        circ.measure(qr[1], cr[1])
        circ.measure(qr[2], cr[2])
        circ.measure(qr[3], cr[3])
        dag = circuit_to_dag(circ)
        #                                             ┌─┐┌───┐        ┌─┐
        # q_0: |0>─────────────────■──────────────────┤M├┤ H ├──■─────┤M├
        #                   ┌───┐  │                  └╥┘└───┘┌─┴─┐┌─┐└╥┘
        # q_1: |0>──■───────┤ H ├──┼───────────────────╫──────┤ X ├┤M├─╫─
        #         ┌─┴─┐┌───┐└───┘  │               ┌─┐ ║      └───┘└╥┘ ║
        # q_2: |0>┤ X ├┤ H ├───────┼─────────■─────┤M├─╫────────────╫──╫─
        #         └───┘└───┘     ┌─┴─┐┌───┐┌─┴─┐┌─┐└╥┘ ║            ║  ║
        # q_3: |0>───────────────┤ X ├┤ H ├┤ X ├┤M├─╫──╫────────────╫──╫─
        #                        └───┘└───┘└───┘└╥┘ ║  ║            ║  ║
        #  c_0: 0 ═══════════════════════════════╬══╬══╩════════════╬══╩═
        #                                        ║  ║               ║
        #  c_1: 0 ═══════════════════════════════╬══╬═══════════════╩════
        #                                        ║  ║
        #  c_2: 0 ═══════════════════════════════╬══╩════════════════════
        #                                        ║
        #  c_3: 0 ═══════════════════════════════╩═══════════════════════
        #
        #                    ┌───┐                      ┌───┐   ┌─┐
        #  q_0: |0>───────X──┤ H ├──────────────────────┤ X ├───┤M├
        #                 │  └───┘┌─┐        ┌───┐      └─┬─┘┌─┐└╥┘
        #  q_1: |0>──■────X────■──┤M├──────X─┤ X ├─X──────■──┤M├─╫─
        #          ┌─┴─┐┌───┐  │  └╥┘      │ └─┬─┘ │ ┌─┐     └╥┘ ║
        #  q_2: |0>┤ X ├┤ H ├──┼───╫───────┼───■───┼─┤M├──────╫──╫─
        #          └───┘└───┘┌─┴─┐ ║ ┌───┐ │ ┌───┐ │ └╥┘ ┌─┐  ║  ║
        #  q_3: |0>──────────┤ X ├─╫─┤ H ├─X─┤ H ├─X──╫──┤M├──╫──╫─
        #                    └───┘ ║ └───┘   └───┘    ║  └╥┘  ║  ║
        #   c_0: 0 ════════════════╩══════════════════╬═══╬═══╩══╬═
        #                                             ║   ║      ║
        #   c_1: 0 ═══════════════════════════════════╬═══╬══════╩═
        #                                             ║   ║
        #   c_2: 0 ═══════════════════════════════════╩═══╬════════
        #                                                 ║
        #   c_3: 0 ═══════════════════════════════════════╩════════
        #
        #     2
        #     |
        # 0 - 1 - 3

        expected = QuantumCircuit(qr, cr)
        expected.cx(qr[1], qr[2])
        expected.h(qr[2])
        expected.swap(qr[0], qr[1])
        expected.h(qr[0])
        expected.cx(qr[1], qr[3])
        expected.measure(qr[1], cr[0])
        expected.h(qr[3])
        expected.swap(qr[1], qr[3])
        expected.cx(qr[2], qr[1])
        expected.h(qr[3])
        expected.swap(qr[1], qr[3])
        expected.measure(qr[2], cr[2])
        expected.measure(qr[3], cr[3])
        expected.cx(qr[1], qr[0])
        expected.measure(qr[1], cr[0])
        expected.measure(qr[0], cr[1])
        expected_dag = circuit_to_dag(expected)

        pass_ = StochasticSwap(coupling, 20, 999)
        after = pass_.run(dag)

        self.assertEqual(expected_dag, after)

    def test_only_output_cx_and_swaps_in_coupling_map(self):
        """Test that output DAG contains only 2q gates from the the coupling map."""

        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(4, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[0], qr[3])
        circuit.measure(qr, cr)
        dag = circuit_to_dag(circuit)

        pass_ = StochasticSwap(coupling, 20, 5)
        after = pass_.run(dag)

        valid_couplings = [{qr[a], qr[b]} for (a, b) in coupling.get_edges()]

        for _2q_gate in after.two_qubit_ops():
            self.assertIn(set(_2q_gate.qargs), valid_couplings)

    def test_len_cm_vs_dag(self):
        """Test error if the coupling map is smaller than the dag."""

        coupling = CouplingMap([[0, 1], [1, 2]])
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(4, "c")
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

    def test_single_gates_omitted(self):
        """Test if single qubit gates are omitted."""

        coupling_map = [[0, 1], [1, 0], [1, 2], [1, 3], [2, 1], [3, 1], [3, 4], [4, 3]]
        qr = QuantumRegister(5, "q")
        cr = ClassicalRegister(5, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[0], qr[4])
        circuit.cx(qr[1], qr[2])
        circuit.u(1, 1.5, 0.7, qr[3])

        expected = QuantumCircuit(qr, cr)
        expected.cx(qr[1], qr[2])
        expected.u(1, 1.5, 0.7, qr[3])
        expected.swap(qr[0], qr[1])
        expected.swap(qr[3], qr[4])
        expected.cx(qr[1], qr[3])

        expected_dag = circuit_to_dag(expected)

        stochastic = StochasticSwap(CouplingMap(coupling_map), seed=0)
        after = PassManager(stochastic).run(circuit)
        after = circuit_to_dag(after)
        self.assertEqual(expected_dag, after)


if __name__ == "__main__":
    unittest.main()
