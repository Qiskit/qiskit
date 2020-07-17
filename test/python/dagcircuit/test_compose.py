# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for the DAGCircuit object"""

import unittest

from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.test import QiskitTestCase


class TestDagCompose(QiskitTestCase):
    """Test composition of two dags"""

    def setUp(self):
        qreg1 = QuantumRegister(3, 'lqr_1')
        qreg2 = QuantumRegister(2, 'lqr_2')
        creg = ClassicalRegister(2, 'lcr')

        self.circuit_left = QuantumCircuit(qreg1, qreg2, creg)
        self.circuit_left.h(qreg1[0])
        self.circuit_left.x(qreg1[1])
        self.circuit_left.u1(0.1, qreg1[2])
        self.circuit_left.cx(qreg2[0], qreg2[1])

        self.left_qubit0 = qreg1[0]
        self.left_qubit1 = qreg1[1]
        self.left_qubit2 = qreg1[2]
        self.left_qubit3 = qreg2[0]
        self.left_qubit4 = qreg2[1]
        self.left_clbit0 = creg[0]
        self.left_clbit1 = creg[1]
        self.condition1 = (creg, 1)
        self.condition2 = (creg, 2)

    def test_compose_inorder(self):
        """Composing two dags of the same width, default order.

                       ┌───┐
        lqr_1_0: |0>───┤ H ├───     rqr_0: |0>──■───────
                       ├───┤                    │  ┌───┐
        lqr_1_1: |0>───┤ X ├───     rqr_1: |0>──┼──┤ X ├
                    ┌──┴───┴──┐                 │  ├───┤
        lqr_1_2: |0>┤ U1(0.1) ├  +  rqr_2: |0>──┼──┤ Y ├  =
                    └─────────┘               ┌─┴─┐└───┘
        lqr_2_0: |0>─────■─────     rqr_3: |0>┤ X ├─────
                       ┌─┴─┐                  └───┘┌───┐
        lqr_2_1: |0>───┤ X ├───     rqr_4: |0>─────┤ Z ├
                       └───┘                       └───┘
        lcr_0: 0 ═══════════

        lcr_1: 0 ═══════════


                        ┌───┐
         lqr_1_0: |0>───┤ H ├─────■───────
                        ├───┤     │  ┌───┐
         lqr_1_1: |0>───┤ X ├─────┼──┤ X ├
                     ┌──┴───┴──┐  │  ├───┤
         lqr_1_2: |0>┤ U1(0.1) ├──┼──┤ Y ├
                     └─────────┘┌─┴─┐└───┘
         lqr_2_0: |0>─────■─────┤ X ├─────
                        ┌─┴─┐   └───┘┌───┐
         lqr_2_1: |0>───┤ X ├────────┤ Z ├
                        └───┘        └───┘
         lcr_0: 0 ════════════════════════

         lcr_1: 0 ════════════════════════

        """
        qreg = QuantumRegister(5, 'rqr')

        circuit_right = QuantumCircuit(qreg)
        circuit_right.cx(qreg[0], qreg[3])
        circuit_right.x(qreg[1])
        circuit_right.y(qreg[2])
        circuit_right.z(qreg[4])

        dag_left = circuit_to_dag(self.circuit_left)
        dag_right = circuit_to_dag(circuit_right)

        # default wiring: i <- i
        dag_left.compose(dag_right)
        circuit_composed = dag_to_circuit(dag_left)

        circuit_expected = self.circuit_left.copy()
        circuit_expected.cx(self.left_qubit0, self.left_qubit3)
        circuit_expected.x(self.left_qubit1)
        circuit_expected.y(self.left_qubit2)
        circuit_expected.z(self.left_qubit4)

        self.assertEqual(circuit_composed, circuit_expected)

    def test_compose_inorder_smaller(self):
        """Composing with a smaller RHS dag, default order.

                       ┌───┐                       ┌─────┐
        lqr_1_0: |0>───┤ H ├───     rqr_0: |0>──■──┤ Tdg ├
                       ├───┤                  ┌─┴─┐└─────┘
        lqr_1_1: |0>───┤ X ├───     rqr_1: |0>┤ X ├───────
                    ┌──┴───┴──┐               └───┘
        lqr_1_2: |0>┤ U1(0.1) ├  +                          =
                    └─────────┘
        lqr_2_0: |0>─────■─────
                       ┌─┴─┐
        lqr_2_1: |0>───┤ X ├───
                       └───┘
        lcr_0: 0 ══════════════

        lcr_1: 0 ══════════════

                        ┌───┐        ┌─────┐
         lqr_1_0: |0>───┤ H ├─────■──┤ Tdg ├
                        ├───┤   ┌─┴─┐└─────┘
         lqr_1_1: |0>───┤ X ├───┤ X ├───────
                     ┌──┴───┴──┐└───┘
         lqr_1_2: |0>┤ U1(0.1) ├────────────
                     └─────────┘
         lqr_2_0: |0>─────■─────────────────
                        ┌─┴─┐
         lqr_2_1: |0>───┤ X ├───────────────
                        └───┘
         lcr_0: 0 ══════════════════════════

         lcr_1: 0 ══════════════════════════

        """
        qreg = QuantumRegister(2, 'rqr')

        circuit_right = QuantumCircuit(qreg)
        circuit_right.cx(qreg[0], qreg[1])
        circuit_right.tdg(qreg[0])

        dag_left = circuit_to_dag(self.circuit_left)
        dag_right = circuit_to_dag(circuit_right)

        # default wiring: i <- i
        dag_left.compose(dag_right)
        circuit_composed = dag_to_circuit(dag_left)

        circuit_expected = self.circuit_left.copy()
        circuit_expected.cx(self.left_qubit0, self.left_qubit1)
        circuit_expected.tdg(self.left_qubit0)

        self.assertEqual(circuit_composed, circuit_expected)

    def test_compose_permuted(self):
        """Composing two dags of the same width, permuted wires.
                        ┌───┐
         lqr_1_0: |0>───┤ H ├───      rqr_0: |0>──■───────
                        ├───┤                     │  ┌───┐
         lqr_1_1: |0>───┤ X ├───      rqr_1: |0>──┼──┤ X ├
                     ┌──┴───┴──┐                  │  ├───┤
         lqr_1_2: |0>┤ U1(0.1) ├      rqr_2: |0>──┼──┤ Y ├
                     └─────────┘                ┌─┴─┐└───┘
         lqr_2_0: |0>─────■─────  +   rqr_3: |0>┤ X ├─────   =
                        ┌─┴─┐                   └───┘┌───┐
         lqr_2_1: |0>───┤ X ├───      rqr_4: |0>─────┤ Z ├
                        └───┘                        └───┘
         lcr_0: 0 ══════════════

         lcr_1: 0 ══════════════

                        ┌───┐   ┌───┐
         lqr_1_0: |0>───┤ H ├───┤ Z ├
                        ├───┤   ├───┤
         lqr_1_1: |0>───┤ X ├───┤ X ├
                     ┌──┴───┴──┐├───┤
         lqr_1_2: |0>┤ U1(0.1) ├┤ Y ├
                     └─────────┘└───┘
         lqr_2_0: |0>─────■───────■──
                        ┌─┴─┐   ┌─┴─┐
         lqr_2_1: |0>───┤ X ├───┤ X ├
                        └───┘   └───┘
         lcr_0: 0 ═══════════════════

         lcr_1: 0 ═══════════════════
        """
        qreg = QuantumRegister(5, 'rqr')
        circuit_right = QuantumCircuit(qreg)
        circuit_right.cx(qreg[0], qreg[3])
        circuit_right.x(qreg[1])
        circuit_right.y(qreg[2])
        circuit_right.z(qreg[4])

        dag_left = circuit_to_dag(self.circuit_left)
        dag_right = circuit_to_dag(circuit_right)

        # permuted wiring
        dag_left.compose(dag_right, qubits=[self.left_qubit3,
                                            self.left_qubit1,
                                            self.left_qubit2,
                                            self.left_qubit4,
                                            self.left_qubit0])
        circuit_composed = dag_to_circuit(dag_left)

        circuit_expected = self.circuit_left.copy()
        circuit_expected.z(self.left_qubit0)
        circuit_expected.x(self.left_qubit1)
        circuit_expected.y(self.left_qubit2)
        circuit_expected.cx(self.left_qubit3, self.left_qubit4)

        self.assertEqual(circuit_composed, circuit_expected)

    def test_compose_permuted_smaller(self):
        """Composing with a smaller RHS dag, and permuted wires.

                       ┌───┐                       ┌─────┐
        lqr_1_0: |0>───┤ H ├───     rqr_0: |0>──■──┤ Tdg ├
                       ├───┤                  ┌─┴─┐└─────┘
        lqr_1_1: |0>───┤ X ├───     rqr_1: |0>┤ X ├───────
                    ┌──┴───┴──┐               └───┘
        lqr_1_2: |0>┤ U1(0.1) ├  +                          =
                    └─────────┘
        lqr_2_0: |0>─────■─────
                       ┌─┴─┐
        lqr_2_1: |0>───┤ X ├───
                       └───┘
        lcr_0: 0 ══════════════

        lcr_1: 0 ══════════════

                        ┌───┐
         lqr_1_0: |0>───┤ H ├───────────────
                        ├───┤
         lqr_1_1: |0>───┤ X ├───────────────
                     ┌──┴───┴──┐┌───┐
         lqr_1_2: |0>┤ U1(0.1) ├┤ X ├───────
                     └─────────┘└─┬─┘┌─────┐
         lqr_2_0: |0>─────■───────■──┤ Tdg ├
                        ┌─┴─┐        └─────┘
         lqr_2_1: |0>───┤ X ├───────────────
                        └───┘
         lcr_0: 0 ══════════════════════════

         lcr_1: 0 ══════════════════════════
        """
        qreg = QuantumRegister(2, 'rqr')
        circuit_right = QuantumCircuit(qreg)
        circuit_right.cx(qreg[0], qreg[1])
        circuit_right.tdg(qreg[0])

        dag_left = circuit_to_dag(self.circuit_left)
        dag_right = circuit_to_dag(circuit_right)

        # permuted wiring of subset
        dag_left.compose(dag_right, qubits=[self.left_qubit3, self.left_qubit2])
        circuit_composed = dag_to_circuit(dag_left)

        circuit_expected = self.circuit_left.copy()
        circuit_expected.cx(self.left_qubit3, self.left_qubit2)
        circuit_expected.tdg(self.left_qubit3)

        self.assertEqual(circuit_composed, circuit_expected)

    def test_compose_conditional(self):
        """Composing on classical bits.

                       ┌───┐                       ┌───┐ ┌─┐
        lqr_1_0: |0>───┤ H ├───     rqr_0: ────────┤ H ├─┤M├───
                       ├───┤                ┌───┐  └─┬─┘ └╥┘┌─┐
        lqr_1_1: |0>───┤ X ├───     rqr_1: ─┤ X ├────┼────╫─┤M├
                    ┌──┴───┴──┐             └─┬─┘    │    ║ └╥┘
        lqr_1_2: |0>┤ U1(0.1) ├  +         ┌──┴──┐┌──┴──┐ ║  ║
                    └─────────┘     rcr_0: ╡     ╞╡     ╞═╩══╬═
        lqr_2_0: |0>─────■─────            │ = 2 ││ = 1 │    ║
                       ┌─┴─┐        rcr_1: ╡     ╞╡     ╞════╩═
        lqr_2_1: |0>───┤ X ├───            └─────┘└─────┘
                       └───┘
        lcr_0: 0 ══════════════

        lcr_1: 0 ══════════════

                    ┌───┐
        lqr_1_0: ───┤ H ├───────────────────────
                    ├───┤           ┌───┐    ┌─┐
        lqr_1_1: ───┤ X ├───────────┤ H ├────┤M├
                 ┌──┴───┴──┐        └─┬─┘    └╥┘
        lqr_1_2: ┤ U1(0.1) ├──────────┼───────╫─
                 └─────────┘          │       ║
        lqr_2_0: ─────■───────────────┼───────╫─
                    ┌─┴─┐    ┌───┐    │   ┌─┐ ║
        lqr_2_1: ───┤ X ├────┤ X ├────┼───┤M├─╫─
                    └───┘    └─┬─┘    │   └╥┘ ║
                            ┌──┴──┐┌──┴──┐ ║  ║
        lcr_0: ═════════════╡     ╞╡     ╞═╩══╬═
                            │ = 1 ││ = 2 │    ║
        lcr_1: ═════════════╡     ╞╡     ╞════╩═
                            └─────┘└─────┘
        """
        qreg = QuantumRegister(2, 'rqr')
        creg = ClassicalRegister(2, 'rcr')

        circuit_right = QuantumCircuit(qreg, creg)
        circuit_right.x(qreg[1]).c_if(creg, 2)
        circuit_right.h(qreg[0]).c_if(creg, 1)
        circuit_right.measure(qreg, creg)

        # permuted subset of qubits and clbits
        dag_left = circuit_to_dag(self.circuit_left)
        dag_right = circuit_to_dag(circuit_right)

        # permuted subset of qubits and clbits
        dag_left.compose(dag_right, qubits=[self.left_qubit1, self.left_qubit4],
                         clbits=[self.left_clbit1, self.left_clbit0])
        circuit_composed = dag_to_circuit(dag_left)

        circuit_expected = self.circuit_left.copy()
        circuit_expected.x(self.left_qubit4).c_if(*self.condition1)
        circuit_expected.h(self.left_qubit1).c_if(*self.condition2)
        circuit_expected.measure(self.left_qubit4, self.left_clbit0)
        circuit_expected.measure(self.left_qubit1, self.left_clbit1)

        self.assertEqual(circuit_composed, circuit_expected)

    def test_compose_classical(self):
        """Composing on classical bits.

                       ┌───┐                       ┌─────┐┌─┐
        lqr_1_0: |0>───┤ H ├───     rqr_0: |0>──■──┤ Tdg ├┤M├
                       ├───┤                  ┌─┴─┐└─┬─┬─┘└╥┘
        lqr_1_1: |0>───┤ X ├───     rqr_1: |0>┤ X ├──┤M├───╫─
                    ┌──┴───┴──┐               └───┘  └╥┘   ║
        lqr_1_2: |0>┤ U1(0.1) ├  +   rcr_0: 0 ════════╬════╩═  =
                    └─────────┘                       ║
        lqr_2_0: |0>─────■─────      rcr_1: 0 ════════╩══════
                       ┌─┴─┐
        lqr_2_1: |0>───┤ X ├───
                       └───┘
        lcr_0: 0 ══════════════

        lcr_1: 0 ══════════════

                       ┌───┐
        lqr_1_0: |0>───┤ H ├──────────────────
                       ├───┤        ┌─────┐┌─┐
        lqr_1_1: |0>───┤ X ├─────■──┤ Tdg ├┤M├
                    ┌──┴───┴──┐  │  └─────┘└╥┘
        lqr_1_2: |0>┤ U1(0.1) ├──┼──────────╫─
                    └─────────┘  │          ║
        lqr_2_0: |0>─────■───────┼──────────╫─
                       ┌─┴─┐   ┌─┴─┐  ┌─┐   ║
        lqr_2_1: |0>───┤ X ├───┤ X ├──┤M├───╫─
                       └───┘   └───┘  └╥┘   ║
           lcr_0: 0 ═══════════════════╩════╬═
                                            ║
           lcr_1: 0 ════════════════════════╩═
        """
        qreg = QuantumRegister(2, 'rqr')
        creg = ClassicalRegister(2, 'rcr')
        circuit_right = QuantumCircuit(qreg, creg)
        circuit_right.cx(qreg[0], qreg[1])
        circuit_right.tdg(qreg[0])
        circuit_right.measure(qreg, creg)

        dag_left = circuit_to_dag(self.circuit_left)
        dag_right = circuit_to_dag(circuit_right)

        # permuted subset of qubits and clbits
        dag_left.compose(dag_right, qubits=[self.left_qubit1, self.left_qubit4],
                         clbits=[self.left_clbit1, self.left_clbit0])
        circuit_composed = dag_to_circuit(dag_left)

        circuit_expected = self.circuit_left.copy()
        circuit_expected.cx(self.left_qubit1, self.left_qubit4)
        circuit_expected.tdg(self.left_qubit1)
        circuit_expected.measure(self.left_qubit4, self.left_clbit0)
        circuit_expected.measure(self.left_qubit1, self.left_clbit1)

        self.assertEqual(circuit_composed, circuit_expected)


if __name__ == '__main__':
    unittest.main()
