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

from qiskit import transpile
from qiskit.pulse import Schedule
from qiskit.circuit import (
    QuantumRegister,
    ClassicalRegister,
    QuantumCircuit,
    Parameter,
    Gate,
    Instruction,
)
from qiskit.circuit.library import HGate, RZGate, CXGate, CCXGate
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
        lcr_0: ════════════╡     ╞╡     ╞═╩══╬═
                           │ = 3 ││ = 3 │    ║
        lcr_1: ════════════╡     ╞╡     ╞════╩═
                           └─────┘└─────┘
        """
        qreg = QuantumRegister(2, "rqr")
        creg = ClassicalRegister(2, "rcr")

        circuit_right = QuantumCircuit(qreg, creg)
        circuit_right.x(qreg[1]).c_if(creg, 3)
        circuit_right.h(qreg[0]).c_if(creg, 3)
        circuit_right.measure(qreg, creg)

        # permuted subset of qubits and clbits
        circuit_composed = self.circuit_left.compose(circuit_right, qubits=[1, 4], clbits=[1, 0])

        circuit_expected = self.circuit_left.copy()
        circuit_expected.x(self.left_qubit4).c_if(*self.condition)
        circuit_expected.h(self.left_qubit1).c_if(*self.condition)
        circuit_expected.measure(self.left_qubit4, self.left_clbit0)
        circuit_expected.measure(self.left_qubit1, self.left_clbit1)

        self.assertEqual(circuit_composed, circuit_expected)

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
            self.assertIsInstance(qc.data[1][0], Gate)

        with self.subTest("wrapping a non-unitary circuit"):
            qc = qc_init.compose(qc_nonunitary, wrap=True)
            self.assertIsInstance(qc.data[1][0], Instruction)


if __name__ == "__main__":
    unittest.main()
