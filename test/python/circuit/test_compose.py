# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""Test QuantumCircuit.compose()."""

import unittest

import numpy as np

from qiskit import transpile
from qiskit.pulse import Schedule
from qiskit.circuit import (
    QuantumRegister,
    ClassicalRegister,
    Clbit,
    QuantumCircuit,
    Qubit,
    Parameter,
    Gate,
    Instruction,
    CASE_DEFAULT,
    SwitchCaseOp,
)
from qiskit.circuit.library import HGate, RZGate, CXGate, CCXGate, TwoLocal
from qiskit.circuit.classical import expr
from qiskit.test import QiskitTestCase


class TestCircuitCompose(QiskitTestCase):
    """Test composition of two circuits."""

    def setUp(self):
        super().setUp()
        qreg1 = QuantumRegister(3, "lqr_1")
        qreg2 = QuantumRegister(2, "lqr_2")
        creg = ClassicalRegister(2, "lcr")

        self.circuit_left = QuantumCircuit(qreg1, qreg2, creg)
        self.circuit_left.h(qreg1[0])
        self.circuit_left.x(qreg1[1])
        self.circuit_left.p(0.1, qreg1[2])
        self.circuit_left.cx(qreg2[0], qreg2[1])

        self.left_qubit0 = qreg1[0]
        self.left_qubit1 = qreg1[1]
        self.left_qubit2 = qreg1[2]
        self.left_qubit3 = qreg2[0]
        self.left_qubit4 = qreg2[1]
        self.left_clbit0 = creg[0]
        self.left_clbit1 = creg[1]
        self.condition = (creg, 3)

    def test_compose_inorder(self):
        """Composing two circuits of the same width, default order.

                      ┌───┐
        lqr_1_0: |0>──┤ H ├───     rqr_0: |0>──■───────
                      ├───┤                    │  ┌───┐
        lqr_1_1: |0>──┤ X ├───     rqr_1: |0>──┼──┤ X ├
                    ┌─┴───┴──┐                 │  ├───┤
        lqr_1_2: |0>┤ P(0.1) ├  +  rqr_2: |0>──┼──┤ Y ├  =
                    └────────┘               ┌─┴─┐└───┘
        lqr_2_0: |0>────■─────     rqr_3: |0>┤ X ├─────
                      ┌─┴─┐                  └───┘┌───┐
        lqr_2_1: |0>──┤ X ├───     rqr_4: |0>─────┤ Z ├
                      └───┘                       └───┘
        lcr_0: 0 ═══════════

        lcr_1: 0 ═══════════


                       ┌───┐
         lqr_1_0: |0>──┤ H ├─────■───────
                       ├───┤     │  ┌───┐
         lqr_1_1: |0>──┤ X ├─────┼──┤ X ├
                     ┌─┴───┴──┐  │  ├───┤
         lqr_1_2: |0>┤ P(0.1) ├──┼──┤ Y ├
                     └────────┘┌─┴─┐└───┘
         lqr_2_0: |0>────■─────┤ X ├─────
                       ┌─┴─┐   └───┘┌───┐
         lqr_2_1: |0>──┤ X ├────────┤ Z ├
                       └───┘        └───┘
         lcr_0: 0 ═══════════════════════

         lcr_1: 0 ═══════════════════════

        """
        qreg = QuantumRegister(5, "rqr")
        circuit_right = QuantumCircuit(qreg)
        circuit_right.cx(qreg[0], qreg[3])
        circuit_right.x(qreg[1])
        circuit_right.y(qreg[2])
        circuit_right.z(qreg[4])

        circuit_expected = self.circuit_left.copy()
        circuit_expected.cx(self.left_qubit0, self.left_qubit3)
        circuit_expected.x(self.left_qubit1)
        circuit_expected.y(self.left_qubit2)
        circuit_expected.z(self.left_qubit4)

        circuit_composed = self.circuit_left.compose(circuit_right, inplace=False)
        self.assertEqual(circuit_composed, circuit_expected)

    def test_compose_inorder_unusual_types(self):
        """Test that composition works in order, using Numpy integer types as well as regular
        integer types.  In general, it should be permissible to use any of the same `QubitSpecifier`
        types (or similar for `Clbit`) that `QuantumCircuit.append` uses."""
        qreg = QuantumRegister(5, "rqr")
        creg = ClassicalRegister(2, "rcr")
        circuit_right = QuantumCircuit(qreg, creg)
        circuit_right.cx(qreg[0], qreg[3])
        circuit_right.x(qreg[1])
        circuit_right.y(qreg[2])
        circuit_right.z(qreg[4])
        circuit_right.measure([0, 1], [0, 1])

        circuit_expected = self.circuit_left.copy()
        circuit_expected.cx(self.left_qubit0, self.left_qubit3)
        circuit_expected.x(self.left_qubit1)
        circuit_expected.y(self.left_qubit2)
        circuit_expected.z(self.left_qubit4)
        circuit_expected.measure(self.left_qubit0, self.left_clbit0)
        circuit_expected.measure(self.left_qubit1, self.left_clbit1)

        circuit_composed = self.circuit_left.compose(circuit_right, np.arange(5), slice(0, 2))
        self.assertEqual(circuit_composed, circuit_expected)

    def test_compose_inorder_inplace(self):
        """Composing two circuits of the same width, default order, inplace.

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
        qreg = QuantumRegister(5, "rqr")
        circuit_right = QuantumCircuit(qreg)
        circuit_right.cx(qreg[0], qreg[3])
        circuit_right.x(qreg[1])
        circuit_right.y(qreg[2])
        circuit_right.z(qreg[4])

        circuit_expected = self.circuit_left.copy()
        circuit_expected.cx(self.left_qubit0, self.left_qubit3)
        circuit_expected.x(self.left_qubit1)
        circuit_expected.y(self.left_qubit2)
        circuit_expected.z(self.left_qubit4)

        # inplace
        circuit_left = self.circuit_left.copy()
        circuit_left.compose(circuit_right, inplace=True)
        self.assertEqual(circuit_left, circuit_expected)

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
        qreg = QuantumRegister(2, "rqr")

        circuit_right = QuantumCircuit(qreg)
        circuit_right.cx(qreg[0], qreg[1])
        circuit_right.tdg(qreg[0])

        circuit_expected = self.circuit_left.copy()
        circuit_expected.cx(self.left_qubit0, self.left_qubit1)
        circuit_expected.tdg(self.left_qubit0)

        circuit_composed = self.circuit_left.compose(circuit_right)
        self.assertEqual(circuit_composed, circuit_expected)

    def test_compose_permuted(self):
        """Composing two dags of the same width, permuted wires.
                      ┌───┐
        lqr_1_0: |0>──┤ H ├───      rqr_0: |0>──■───────
                      ├───┤                     │  ┌───┐
        lqr_1_1: |0>──┤ X ├───      rqr_1: |0>──┼──┤ X ├
                    ┌─┴───┴──┐                  │  ├───┤
        lqr_1_2: |0>┤ P(0.1) ├      rqr_2: |0>──┼──┤ Y ├
                    └────────┘                ┌─┴─┐└───┘
        lqr_2_0: |0>────■─────  +   rqr_3: |0>┤ X ├─────   =
                      ┌─┴─┐                   └───┘┌───┐
        lqr_2_1: |0>──┤ X ├───      rqr_4: |0>─────┤ Z ├
                      └───┘                        └───┘
        lcr_0: 0 ══════════════

        lcr_1: 0 ══════════════

                      ┌───┐   ┌───┐
        lqr_1_0: |0>──┤ H ├───┤ Z ├
                      ├───┤   ├───┤
        lqr_1_1: |0>──┤ X ├───┤ X ├
                    ┌─┴───┴──┐├───┤
        lqr_1_2: |0>┤ P(0.1) ├┤ Y ├
                    └────────┘└───┘
        lqr_2_0: |0>────■───────■──
                      ┌─┴─┐   ┌─┴─┐
        lqr_2_1: |0>──┤ X ├───┤ X ├
                      └───┘   └───┘
        lcr_0: 0 ══════════════════

        lcr_1: 0 ══════════════════
        """
        qreg = QuantumRegister(5, "rqr")

        circuit_right = QuantumCircuit(qreg)
        circuit_right.cx(qreg[0], qreg[3])
        circuit_right.x(qreg[1])
        circuit_right.y(qreg[2])
        circuit_right.z(qreg[4])

        circuit_expected = self.circuit_left.copy()
        circuit_expected.z(self.left_qubit0)
        circuit_expected.x(self.left_qubit1)
        circuit_expected.y(self.left_qubit2)
        circuit_expected.cx(self.left_qubit3, self.left_qubit4)

        # permuted wiring
        circuit_composed = self.circuit_left.compose(
            circuit_right,
            qubits=[
                self.left_qubit3,
                self.left_qubit1,
                self.left_qubit2,
                self.left_qubit4,
                self.left_qubit0,
            ],
            inplace=False,
        )
        self.assertEqual(circuit_composed, circuit_expected)

    def test_compose_permuted_smaller(self):
        """Composing with a smaller RHS dag, and permuted wires.
        Compose using indices.

                      ┌───┐                       ┌─────┐
        lqr_1_0: |0>──┤ H ├───     rqr_0: |0>──■──┤ Tdg ├
                      ├───┤                  ┌─┴─┐└─────┘
        lqr_1_1: |0>──┤ X ├───     rqr_1: |0>┤ X ├───────
                    ┌─┴───┴──┐               └───┘
        lqr_1_2: |0>┤ P(0.1) ├  +                          =
                    └────────┘
        lqr_2_0: |0>────■─────
                      ┌─┴─┐
        lqr_2_1: |0>──┤ X ├───
                      └───┘
        lcr_0: 0 ═════════════

        lcr_1: 0 ═════════════

                       ┌───┐
         lqr_1_0: |0>──┤ H ├───────────────
                       ├───┤
         lqr_1_1: |0>──┤ X ├───────────────
                     ┌─┴───┴──┐┌───┐
         lqr_1_2: |0>┤ P(0.1) ├┤ X ├───────
                     └────────┘└─┬─┘┌─────┐
         lqr_2_0: |0>────■───────■──┤ Tdg ├
                       ┌─┴─┐        └─────┘
         lqr_2_1: |0>──┤ X ├───────────────
                       └───┘
         lcr_0: 0 ═════════════════════════

         lcr_1: 0 ═════════════════════════
        """
        qreg = QuantumRegister(2, "rqr")
        circuit_right = QuantumCircuit(qreg)
        circuit_right.cx(qreg[0], qreg[1])
        circuit_right.tdg(qreg[0])

        # permuted wiring of subset
        circuit_composed = self.circuit_left.compose(circuit_right, qubits=[3, 2])

        circuit_expected = self.circuit_left.copy()
        circuit_expected.cx(self.left_qubit3, self.left_qubit2)
        circuit_expected.tdg(self.left_qubit3)

        self.assertEqual(circuit_composed, circuit_expected)

    def test_compose_classical(self):
        """Composing on classical bits.

                      ┌───┐                       ┌─────┐┌─┐
        lqr_1_0: |0>──┤ H ├───     rqr_0: |0>──■──┤ Tdg ├┤M├
                      ├───┤                  ┌─┴─┐└─┬─┬─┘└╥┘
        lqr_1_1: |0>──┤ X ├───     rqr_1: |0>┤ X ├──┤M├───╫─
                    ┌─┴───┴──┐               └───┘  └╥┘   ║
        lqr_1_2: |0>┤ P(0.1) ├  +   rcr_0: 0 ════════╬════╩═  =
                    └────────┘                       ║
        lqr_2_0: |0>────■─────      rcr_1: 0 ════════╩══════
                      ┌─┴─┐
        lqr_2_1: |0>──┤ X ├───
                      └───┘
        lcr_0: 0 ══════════════

        lcr_1: 0 ══════════════

                      ┌───┐
        lqr_1_0: |0>──┤ H ├──────────────────
                      ├───┤        ┌─────┐┌─┐
        lqr_1_1: |0>──┤ X ├─────■──┤ Tdg ├┤M├
                    ┌─┴───┴──┐  │  └─────┘└╥┘
        lqr_1_2: |0>┤ P(0.1) ├──┼──────────╫─
                    └────────┘  │          ║
        lqr_2_0: |0>────■───────┼──────────╫─
                      ┌─┴─┐   ┌─┴─┐  ┌─┐   ║
        lqr_2_1: |0>──┤ X ├───┤ X ├──┤M├───╫─
                      └───┘   └───┘  └╥┘   ║
           lcr_0: 0 ══════════════════╩════╬═
                                           ║
           lcr_1: 0 ═══════════════════════╩═
        """
        qreg = QuantumRegister(2, "rqr")
        creg = ClassicalRegister(2, "rcr")

        circuit_right = QuantumCircuit(qreg, creg)
        circuit_right.cx(qreg[0], qreg[1])
        circuit_right.tdg(qreg[0])
        circuit_right.measure(qreg, creg)

        # permuted subset of qubits and clbits
        circuit_composed = self.circuit_left.compose(circuit_right, qubits=[1, 4], clbits=[1, 0])

        circuit_expected = self.circuit_left.copy()
        circuit_expected.cx(self.left_qubit1, self.left_qubit4)
        circuit_expected.tdg(self.left_qubit1)
        circuit_expected.measure(self.left_qubit4, self.left_clbit0)
        circuit_expected.measure(self.left_qubit1, self.left_clbit1)

        self.assertEqual(circuit_composed, circuit_expected)

    def test_compose_conditional(self):
        """Composing on classical bits.

                      ┌───┐                       ┌───┐ ┌─┐
        lqr_1_0: |0>──┤ H ├───     rqr_0: ────────┤ H ├─┤M├───
                      ├───┤                ┌───┐  └─┬─┘ └╥┘┌─┐
        lqr_1_1: |0>──┤ X ├───     rqr_1: ─┤ X ├────┼────╫─┤M├
                    ┌─┴───┴──┐             └─┬─┘    │    ║ └╥┘
        lqr_1_2: |0>┤ P(0.1) ├  +         ┌──┴──┐┌──┴──┐ ║  ║
                    └────────┘     rcr_0: ╡     ╞╡     ╞═╩══╬═
        lqr_2_0: |0>────■─────            │ = 3 ││ = 3 │    ║
                      ┌─┴─┐        rcr_1: ╡     ╞╡     ╞════╩═
        lqr_2_1: |0>──┤ X ├───            └─────┘└─────┘
                      └───┘
        lcr_0: 0 ══════════════

        lcr_1: 0 ══════════════

                   ┌───┐
        lqr_1_0: ──┤ H ├───────────────────────
                   ├───┤           ┌───┐    ┌─┐
        lqr_1_1: ──┤ X ├───────────┤ H ├────┤M├
                 ┌─┴───┴──┐        └─┬─┘    └╥┘
        lqr_1_2: ┤ P(0.1) ├──────────┼───────╫─
                 └────────┘          │       ║
        lqr_2_0: ────■───────────────┼───────╫─
                   ┌─┴─┐    ┌───┐    │   ┌─┐ ║
        lqr_2_1: ──┤ X ├────┤ X ├────┼───┤M├─╫─
                   └───┘    └─┬─┘    │   └╥┘ ║
                           ┌──┴──┐┌──┴──┐ ║  ║
        lcr_0: ════════════╡     ╞╡     ╞═╬══╩═
                           │ = 3 ││ = 3 │ ║
        lcr_1: ════════════╡     ╞╡     ╞═╩════
                           └─────┘└─────┘
        """
        qreg = QuantumRegister(2, "rqr")
        creg = ClassicalRegister(2, "rcr")

        circuit_right = QuantumCircuit(qreg, creg)
        circuit_right.x(qreg[1]).c_if(creg, 3)
        circuit_right.h(qreg[0]).c_if(creg, 3)
        circuit_right.measure(qreg, creg)

        # permuted subset of qubits and clbits
        circuit_composed = self.circuit_left.compose(circuit_right, qubits=[1, 4], clbits=[0, 1])

        circuit_expected = self.circuit_left.copy()
        circuit_expected.x(self.left_qubit4).c_if(*self.condition)
        circuit_expected.h(self.left_qubit1).c_if(*self.condition)
        circuit_expected.measure(self.left_qubit1, self.left_clbit0)
        circuit_expected.measure(self.left_qubit4, self.left_clbit1)

        self.assertEqual(circuit_composed, circuit_expected)

    def test_compose_conditional_no_match(self):
        """Test that compose correctly maps registers in conditions to the new circuit, even when
        there are no matching registers in the destination circuit.

        Regression test of gh-6583 and gh-6584."""
        right = QuantumCircuit(QuantumRegister(3), ClassicalRegister(1), ClassicalRegister(1))
        right.h(1)
        right.cx(1, 2)
        right.cx(0, 1)
        right.h(0)
        right.measure([0, 1], [0, 1])
        right.z(2).c_if(right.cregs[0], 1)
        right.x(2).c_if(right.cregs[1], 1)
        test = QuantumCircuit(3, 3).compose(right, range(3), range(2))
        z = next(ins.operation for ins in test.data[::-1] if ins.operation.name == "z")
        x = next(ins.operation for ins in test.data[::-1] if ins.operation.name == "x")
        # The registers should have been mapped, including the bits inside them.  Unlike the
        # previous test, there are no matching registers in the destination circuit, so the
        # composition needs to add new registers (bit groupings) over the existing mapped bits.
        self.assertIsNot(z.condition, None)
        self.assertIsInstance(z.condition[0], ClassicalRegister)
        self.assertEqual(len(z.condition[0]), len(right.cregs[0]))
        self.assertIs(z.condition[0][0], test.clbits[0])
        self.assertEqual(z.condition[1], 1)
        self.assertIsNot(x.condition, None)
        self.assertIsInstance(x.condition[0], ClassicalRegister)
        self.assertEqual(len(x.condition[0]), len(right.cregs[1]))
        self.assertEqual(z.condition[1], 1)
        self.assertIs(x.condition[0][0], test.clbits[1])

    def test_compose_switch_match(self):
        """Test that composition containing a `switch` with a register that matches proceeds
        correctly."""
        case_0 = QuantumCircuit(1, 2)
        case_0.x(0)
        case_1 = QuantumCircuit(1, 2)
        case_1.z(0)
        case_default = QuantumCircuit(1, 2)
        cr = ClassicalRegister(2, "target")
        right = QuantumCircuit(QuantumRegister(1), cr)
        right.switch(cr, [(0, case_0), (1, case_1), (CASE_DEFAULT, case_default)], [0], [0, 1])

        test = QuantumCircuit(QuantumRegister(3), cr, ClassicalRegister(2)).compose(
            right, [1], [0, 1]
        )

        expected = test.copy_empty_like()
        expected.switch(cr, [(0, case_0), (1, case_1), (CASE_DEFAULT, case_default)], [1], [0, 1])
        self.assertEqual(test, expected)

    def test_compose_switch_no_match(self):
        """Test that composition containing a `switch` with a register that matches proceeds
        correctly."""
        case_0 = QuantumCircuit(1, 2)
        case_0.x(0)
        case_1 = QuantumCircuit(1, 2)
        case_1.z(0)
        case_default = QuantumCircuit(1, 2)
        cr = ClassicalRegister(2, "target")
        right = QuantumCircuit(QuantumRegister(1), cr)
        right.switch(cr, [(0, case_0), (1, case_1), (CASE_DEFAULT, case_default)], [0], [0, 1])
        test = QuantumCircuit(3, 3).compose(right, [1], [0, 1])

        self.assertEqual(len(test.data), 1)
        self.assertIsInstance(test.data[0].operation, SwitchCaseOp)
        target = test.data[0].operation.target
        self.assertIn(target, test.cregs)
        self.assertEqual(list(target), test.clbits[0:2])

    def test_compose_gate(self):
        """Composing with a gate.

                   ┌───┐                               ┌───┐    ┌───┐
        lqr_1_0: ──┤ H ├───                 lqr_1_0: ──┤ H ├────┤ X ├
                   ├───┤                               ├───┤    └─┬─┘
        lqr_1_1: ──┤ X ├───                 lqr_1_1: ──┤ X ├──────┼───
                 ┌─┴───┴──┐     ───■───              ┌─┴───┴──┐   │
        lqr_1_2: ┤ P(0.1) ├  +   ┌─┴─┐   =  lqr_1_2: ┤ P(0.1) ├───┼───
                 └────────┘     ─┤ X ├─              └────────┘   │
        lqr_2_0: ────■─────      └───┘      lqr_2_0: ────■────────┼──
                   ┌─┴─┐                               ┌─┴─┐      │
        lqr_2_1: ──┤ X ├───                 lqr_2_1: ──┤ X ├──────■───
                   └───┘                               └───┘
        lcr_0: 0 ══════════                 lcr_0: 0 ═════════════════

        lcr_1: 0 ══════════                 lcr_1: 0 ═════════════════

        """
        circuit_composed = self.circuit_left.compose(CXGate(), qubits=[4, 0])

        circuit_expected = self.circuit_left.copy()
        circuit_expected.cx(self.left_qubit4, self.left_qubit0)

        self.assertEqual(circuit_composed, circuit_expected)

    def test_compose_calibrations(self):
        """Test that composing two circuits updates calibrations."""
        circ_left = QuantumCircuit(1)
        circ_left.add_calibration("h", [0], None)
        circ_right = QuantumCircuit(1)
        circ_right.add_calibration("rx", [0], None)
        circ = circ_left.compose(circ_right)
        self.assertEqual(len(circ.calibrations), 2)
        self.assertEqual(len(circ_left.calibrations), 1)

        circ_left = QuantumCircuit(1)
        circ_left.add_calibration("h", [0], None)
        circ_right = QuantumCircuit(1)
        circ_right.add_calibration("h", [1], None)
        circ = circ_left.compose(circ_right)
        self.assertEqual(len(circ.calibrations), 1)
        self.assertEqual(len(circ.calibrations["h"]), 2)
        self.assertEqual(len(circ_left.calibrations), 1)

        # Ensure that transpiled _calibration is defaultdict
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(0, 0)
        qc = transpile(qc, None, basis_gates=["h", "cx"], coupling_map=[[0, 1], [1, 0]])
        qc.add_calibration("cx", [0, 1], Schedule())

    def test_compose_one_liner(self):
        """Test building a circuit in one line, for fun."""
        circ = QuantumCircuit(3)
        h = HGate()
        rz = RZGate(0.1)
        cx = CXGate()
        ccx = CCXGate()
        circ = circ.compose(h, [0]).compose(cx, [0, 2]).compose(ccx, [2, 1, 0]).compose(rz, [1])

        expected = QuantumCircuit(3)
        expected.h(0)
        expected.cx(0, 2)
        expected.ccx(2, 1, 0)
        expected.rz(0.1, 1)

        self.assertEqual(circ, expected)

    def test_compose_global_phase(self):
        """Composing with global phase."""
        circ1 = QuantumCircuit(1, global_phase=1)
        circ1.rz(0.5, 0)
        circ2 = QuantumCircuit(1, global_phase=2)
        circ3 = QuantumCircuit(1, global_phase=3)
        circ4 = circ1.compose(circ2).compose(circ3)
        self.assertEqual(
            circ4.global_phase, circ1.global_phase + circ2.global_phase + circ3.global_phase
        )

    def test_compose_front_circuit(self):
        """Test composing a circuit at the front of a circuit."""

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        other = QuantumCircuit(2)
        other.cz(1, 0)
        other.z(1)

        output = qc.compose(other, front=True)

        expected = QuantumCircuit(2)
        expected.cz(1, 0)
        expected.z(1)
        expected.h(0)
        expected.cx(0, 1)

        self.assertEqual(output, expected)

    def test_compose_front_gate(self):
        """Test composing a gate at the front of a circuit."""

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        output = qc.compose(CXGate(), [1, 0], front=True)

        expected = QuantumCircuit(2)
        expected.cx(1, 0)
        expected.h(0)
        expected.cx(0, 1)

        self.assertEqual(output, expected)

    def test_compose_adds_parameters(self):
        """Test the composed circuit contains all parameters."""
        a, b = Parameter("a"), Parameter("b")

        qc_a = QuantumCircuit(1)
        qc_a.rx(a, 0)

        qc_b = QuantumCircuit(1)
        qc_b.rx(b, 0)

        with self.subTest("compose with other circuit out-of-place"):
            qc_1 = qc_a.compose(qc_b)
            self.assertEqual(qc_1.parameters, {a, b})

        with self.subTest("compose with other instruction out-of-place"):
            instr_b = qc_b.to_instruction()
            qc_2 = qc_a.compose(instr_b, [0])
            self.assertEqual(qc_2.parameters, {a, b})

        with self.subTest("compose with other circuit in-place"):
            qc_a.compose(qc_b, inplace=True)
            self.assertEqual(qc_a.parameters, {a, b})

    def test_wrapped_compose(self):
        """Test wrapping the circuit upon composition works."""
        qc_a = QuantumCircuit(1)
        qc_a.x(0)

        qc_b = QuantumCircuit(1, name="B")
        qc_b.h(0)

        qc_a.compose(qc_b, wrap=True, inplace=True)

        self.assertDictEqual(qc_a.count_ops(), {"B": 1, "x": 1})
        self.assertDictEqual(qc_a.decompose().count_ops(), {"h": 1, "u3": 1})

    def test_wrapping_unitary_circuit(self):
        """Test a unitary circuit will be wrapped as Gate, else as Instruction."""
        qc_init = QuantumCircuit(1)
        qc_init.x(0)

        qc_unitary = QuantumCircuit(1, name="a")
        qc_unitary.ry(0.23, 0)

        qc_nonunitary = QuantumCircuit(1)
        qc_nonunitary.reset(0)

        with self.subTest("wrapping a unitary circuit"):
            qc = qc_init.compose(qc_unitary, wrap=True)
            self.assertIsInstance(qc.data[1].operation, Gate)

        with self.subTest("wrapping a non-unitary circuit"):
            qc = qc_init.compose(qc_nonunitary, wrap=True)
            self.assertIsInstance(qc.data[1].operation, Instruction)

    def test_single_bit_condition(self):
        """Test that compose can correctly handle circuits that contain conditions on single
        bits.  This is a regression test of the bug that broke qiskit-experiments in gh-7653."""
        base = QuantumCircuit(1, 1)
        base.x(0).c_if(0, True)
        test = QuantumCircuit(1, 1).compose(base)
        self.assertIsNot(base.clbits[0], test.clbits[0])
        self.assertEqual(base, test)
        self.assertIs(test.data[0].operation.condition[0], test.clbits[0])

    def test_condition_mapping_ifelseop(self):
        """Test that the condition in an `IfElseOp` is correctly mapped to a new set of bits and
        registers."""
        base_loose = Clbit()
        base_creg = ClassicalRegister(2)
        base_qreg = QuantumRegister(1)
        base = QuantumCircuit(base_qreg, [base_loose], base_creg)
        with base.if_test((base_loose, True)):
            base.x(0)
        with base.if_test((base_creg, 3)):
            base.x(0)

        test_loose = Clbit()
        test_creg = ClassicalRegister(2)
        test_qreg = QuantumRegister(1)
        test = QuantumCircuit(test_qreg, [test_loose], test_creg).compose(base)

        bit_instruction = test.data[0].operation
        reg_instruction = test.data[1].operation
        self.assertIs(bit_instruction.condition[0], test_loose)
        self.assertEqual(bit_instruction.condition, (test_loose, True))
        self.assertIs(reg_instruction.condition[0], test_creg)
        self.assertEqual(reg_instruction.condition, (test_creg, 3))

    def test_condition_mapping_whileloopop(self):
        """Test that the condition in a `WhileLoopOp` is correctly mapped to a new set of bits and
        registers."""
        base_loose = Clbit()
        base_creg = ClassicalRegister(2)
        base_qreg = QuantumRegister(1)
        base = QuantumCircuit(base_qreg, [base_loose], base_creg)
        with base.while_loop((base_loose, True)):
            base.x(0)
        with base.while_loop((base_creg, 3)):
            base.x(0)

        test_loose = Clbit()
        test_creg = ClassicalRegister(2)
        test_qreg = QuantumRegister(1)
        test = QuantumCircuit(test_qreg, [test_loose], test_creg).compose(base)

        bit_instruction = test.data[0].operation
        reg_instruction = test.data[1].operation
        self.assertIs(bit_instruction.condition[0], test_loose)
        self.assertEqual(bit_instruction.condition, (test_loose, True))
        self.assertIs(reg_instruction.condition[0], test_creg)
        self.assertEqual(reg_instruction.condition, (test_creg, 3))

    def test_compose_no_clbits_in_one(self):
        """Test combining a circuit with cregs to one without"""
        ansatz = TwoLocal(2, rotation_blocks="ry", entanglement_blocks="cx")

        qc = QuantumCircuit(2)
        qc.measure_all()
        out = ansatz.compose(qc)
        self.assertEqual(out.clbits, qc.clbits)

    def test_compose_no_clbits_in_one_inplace(self):
        """Test combining a circuit with cregs to one without inplace"""
        ansatz = TwoLocal(2, rotation_blocks="ry", entanglement_blocks="cx")

        qc = QuantumCircuit(2)
        qc.measure_all()
        ansatz.compose(qc, inplace=True)
        self.assertEqual(ansatz.clbits, qc.clbits)

    def test_compose_no_clbits_in_one_multireg(self):
        """Test combining a circuit with cregs to one without, multi cregs"""
        ansatz = TwoLocal(2, rotation_blocks="ry", entanglement_blocks="cx")

        qa = QuantumRegister(2, "q")
        ca = ClassicalRegister(2, "a")
        cb = ClassicalRegister(2, "b")
        qc = QuantumCircuit(qa, ca, cb)
        qc.measure(0, cb[1])
        out = ansatz.compose(qc)
        self.assertEqual(out.clbits, qc.clbits)
        self.assertEqual(out.cregs, qc.cregs)

    def test_compose_noclbits_registerless(self):
        """Combining a circuit with cregs to one without, registerless case"""
        inner = QuantumCircuit([Qubit(), Qubit()], [Clbit(), Clbit()])
        inner.measure([0, 1], [0, 1])
        outer = QuantumCircuit(2)
        outer.compose(inner, inplace=True)
        self.assertEqual(outer.clbits, inner.clbits)
        self.assertEqual(outer.cregs, [])

    def test_expr_condition_is_mapped(self):
        """Test that an expression in a condition involving several registers is mapped correctly to
        the destination circuit."""
        inner = QuantumCircuit(1)
        inner.x(0)
        a_src = ClassicalRegister(2, "a_src")
        b_src = ClassicalRegister(2, "b_src")
        c_src = ClassicalRegister(name="c_src", bits=list(a_src) + list(b_src))
        source = QuantumCircuit(QuantumRegister(1), a_src, b_src, c_src)

        test_1 = lambda: expr.lift(a_src[0])
        test_2 = lambda: expr.logic_not(b_src[1])
        test_3 = lambda: expr.logic_and(expr.bit_and(b_src, 2), expr.less(c_src, 7))
        source.if_test(test_1(), inner.copy(), [0], [])
        source.if_else(test_2(), inner.copy(), inner.copy(), [0], [])
        source.while_loop(test_3(), inner.copy(), [0], [])

        a_dest = ClassicalRegister(2, "a_dest")
        b_dest = ClassicalRegister(2, "b_dest")
        dest = QuantumCircuit(QuantumRegister(1), a_dest, b_dest).compose(source)

        # Check that the input conditions weren't mutated.
        for in_condition, instruction in zip((test_1, test_2, test_3), source.data):
            self.assertEqual(in_condition(), instruction.operation.condition)

        # Should be `a_dest`, `b_dest` and an added one to account for `c_src`.
        self.assertEqual(len(dest.cregs), 3)
        mapped_reg = dest.cregs[-1]

        expected = QuantumCircuit(dest.qregs[0], a_dest, b_dest, mapped_reg)
        expected.if_test(expr.lift(a_dest[0]), inner.copy(), [0], [])
        expected.if_else(expr.logic_not(b_dest[1]), inner.copy(), inner.copy(), [0], [])
        expected.while_loop(
            expr.logic_and(expr.bit_and(b_dest, 2), expr.less(mapped_reg, 7)), inner.copy(), [0], []
        )
        self.assertEqual(dest, expected)

    def test_expr_target_is_mapped(self):
        """Test that an expression in a switch statement's target is mapping correctly to the
        destination circuit."""
        inner1 = QuantumCircuit(1)
        inner1.x(0)
        inner2 = QuantumCircuit(1)
        inner2.z(0)

        a_src = ClassicalRegister(2, "a_src")
        b_src = ClassicalRegister(2, "b_src")
        c_src = ClassicalRegister(name="c_src", bits=list(a_src) + list(b_src))
        source = QuantumCircuit(QuantumRegister(1), a_src, b_src, c_src)

        test_1 = lambda: expr.lift(a_src[0])
        test_2 = lambda: expr.logic_not(b_src[1])
        test_3 = lambda: expr.lift(b_src)
        test_4 = lambda: expr.bit_and(c_src, 7)
        source.switch(test_1(), [(False, inner1.copy()), (True, inner2.copy())], [0], [])
        source.switch(test_2(), [(False, inner1.copy()), (True, inner2.copy())], [0], [])
        source.switch(test_3(), [(0, inner1.copy()), (CASE_DEFAULT, inner2.copy())], [0], [])
        source.switch(test_4(), [(0, inner1.copy()), (CASE_DEFAULT, inner2.copy())], [0], [])

        a_dest = ClassicalRegister(2, "a_dest")
        b_dest = ClassicalRegister(2, "b_dest")
        dest = QuantumCircuit(QuantumRegister(1), a_dest, b_dest).compose(source)

        # Check that the input expressions weren't mutated.
        for in_target, instruction in zip((test_1, test_2, test_3, test_4), source.data):
            self.assertEqual(in_target(), instruction.operation.target)

        # Should be `a_dest`, `b_dest` and an added one to account for `c_src`.
        self.assertEqual(len(dest.cregs), 3)
        mapped_reg = dest.cregs[-1]

        expected = QuantumCircuit(dest.qregs[0], a_dest, b_dest, mapped_reg)
        expected.switch(
            expr.lift(a_dest[0]), [(False, inner1.copy()), (True, inner2.copy())], [0], []
        )
        expected.switch(
            expr.logic_not(b_dest[1]), [(False, inner1.copy()), (True, inner2.copy())], [0], []
        )
        expected.switch(
            expr.lift(b_dest), [(0, inner1.copy()), (CASE_DEFAULT, inner2.copy())], [0], []
        )
        expected.switch(
            expr.bit_and(mapped_reg, 7),
            [(0, inner1.copy()), (CASE_DEFAULT, inner2.copy())],
            [0],
            [],
        )

        self.assertEqual(dest, expected)


if __name__ == "__main__":
    unittest.main()
