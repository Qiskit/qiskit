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

"""Test Qiskit's inverse gate operation."""

import unittest
import numpy as np

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, pulse
from qiskit.circuit import Clbit
from qiskit.circuit.library import RXGate, RYGate
from qiskit.circuit.exceptions import CircuitError
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestCircuitProperties(QiskitTestCase):
    """QuantumCircuit properties tests."""

    def test_qarg_numpy_int(self):
        """Test castable to integer args for QuantumCircuit."""
        n = np.int64(12)
        qc1 = QuantumCircuit(n)
        self.assertEqual(qc1.num_qubits, 12)
        self.assertEqual(type(qc1), QuantumCircuit)

    def test_carg_numpy_int(self):
        """Test castable to integer cargs for QuantumCircuit."""
        n = np.int64(12)
        c1 = ClassicalRegister(n)
        qc1 = QuantumCircuit(c1)
        c_regs = qc1.cregs
        self.assertEqual(c_regs[0], c1)
        self.assertEqual(type(qc1), QuantumCircuit)

    def test_carg_numpy_int_2(self):
        """Test castable to integer cargs for QuantumCircuit."""
        qc1 = QuantumCircuit(12, np.int64(12))
        self.assertEqual(len(qc1.clbits), 12)
        self.assertTrue(all(isinstance(bit, Clbit) for bit in qc1.clbits))
        self.assertEqual(type(qc1), QuantumCircuit)

    def test_qarg_numpy_int_exception(self):
        """Test attempt to pass non-castable arg to QuantumCircuit."""
        self.assertRaises(CircuitError, QuantumCircuit, "string")

    def test_warning_on_noninteger_float(self):
        """Test warning when passing non-integer float to QuantumCircuit"""
        self.assertRaises(CircuitError, QuantumCircuit, 2.2)
        # but an integer float should pass
        qc = QuantumCircuit(2.0)
        self.assertEqual(qc.num_qubits, 2)

    def test_circuit_depth_empty(self):
        """Test depth of empty circuity"""
        q = QuantumRegister(5, "q")
        qc = QuantumCircuit(q)
        self.assertEqual(qc.depth(), 0)

    def test_circuit_depth_no_reg(self):
        """Test depth of no register circuits"""
        qc = QuantumCircuit()
        self.assertEqual(qc.depth(), 0)

    def test_circuit_depth_meas_only(self):
        """Test depth of measurement only"""
        q = QuantumRegister(1, "q")
        c = ClassicalRegister(1, "c")
        qc = QuantumCircuit(q, c)
        qc.measure(q, c)
        self.assertEqual(qc.depth(), 1)

    def test_circuit_depth_barrier(self):
        """Make sure barriers do not add to depth"""

        #         ┌───┐                     ░ ┌─┐
        #    q_0: ┤ H ├──■──────────────────░─┤M├────────────
        #         ├───┤┌─┴─┐                ░ └╥┘┌─┐
        #    q_1: ┤ H ├┤ X ├──■─────────────░──╫─┤M├─────────
        #         ├───┤└───┘  │  ┌───┐      ░  ║ └╥┘┌─┐
        #    q_2: ┤ H ├───────┼──┤ X ├──■───░──╫──╫─┤M├──────
        #         ├───┤       │  └─┬─┘┌─┴─┐ ░  ║  ║ └╥┘┌─┐
        #    q_3: ┤ H ├───────┼────┼──┤ X ├─░──╫──╫──╫─┤M├───
        #         ├───┤     ┌─┴─┐  │  └───┘ ░  ║  ║  ║ └╥┘┌─┐
        #    q_4: ┤ H ├─────┤ X ├──■────────░──╫──╫──╫──╫─┤M├
        #         └───┘     └───┘           ░  ║  ║  ║  ║ └╥┘
        #    c: 5/═════════════════════════════╩══╩══╩══╩══╩═
        #                                      0  1  2  3  4
        q = QuantumRegister(5, "q")
        c = ClassicalRegister(5, "c")
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.h(q[4])
        qc.cx(q[0], q[1])
        qc.cx(q[1], q[4])
        qc.cx(q[4], q[2])
        qc.cx(q[2], q[3])
        qc.barrier(q)
        qc.measure(q, c)
        self.assertEqual(qc.depth(), 6)

    def test_circuit_depth_simple(self):
        """Test depth for simple circuit"""
        #      ┌───┐
        # q_0: ┤ H ├──■────────────────────
        #      └───┘  │            ┌───┐┌─┐
        # q_1: ───────┼────────────┤ X ├┤M├
        #      ┌───┐  │  ┌───┐┌───┐└─┬─┘└╥┘
        # q_2: ┤ X ├──┼──┤ X ├┤ X ├──┼───╫─
        #      └───┘  │  └───┘└───┘  │   ║
        # q_3: ───────┼──────────────┼───╫─
        #           ┌─┴─┐┌───┐       │   ║
        # q_4: ─────┤ X ├┤ X ├───────■───╫─
        #           └───┘└───┘           ║
        # c: 1/══════════════════════════╩═
        #                                0
        q = QuantumRegister(5, "q")
        c = ClassicalRegister(1, "c")
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.cx(q[0], q[4])
        qc.x(q[2])
        qc.x(q[2])
        qc.x(q[2])
        qc.x(q[4])
        qc.cx(q[4], q[1])
        qc.measure(q[1], c[0])
        self.assertEqual(qc.depth(), 5)

    def test_circuit_depth_multi_reg(self):
        """Test depth for multiple registers"""

        #       ┌───┐
        # q1_0: ┤ H ├──■─────────────────
        #       ├───┤┌─┴─┐
        # q1_1: ┤ H ├┤ X ├──■────────────
        #       ├───┤└───┘  │  ┌───┐
        # q1_2: ┤ H ├───────┼──┤ X ├──■──
        #       ├───┤       │  └─┬─┘┌─┴─┐
        # q2_0: ┤ H ├───────┼────┼──┤ X ├
        #       ├───┤     ┌─┴─┐  │  └───┘
        # q2_1: ┤ H ├─────┤ X ├──■───────
        #       └───┘     └───┘
        q1 = QuantumRegister(3, "q1")
        q2 = QuantumRegister(2, "q2")
        c = ClassicalRegister(5, "c")
        qc = QuantumCircuit(q1, q2, c)
        qc.h(q1[0])
        qc.h(q1[1])
        qc.h(q1[2])
        qc.h(q2[0])
        qc.h(q2[1])
        qc.cx(q1[0], q1[1])
        qc.cx(q1[1], q2[1])
        qc.cx(q2[1], q1[2])
        qc.cx(q1[2], q2[0])
        self.assertEqual(qc.depth(), 5)

    def test_circuit_depth_3q_gate(self):
        """Test depth for 3q gate"""

        #       ┌───┐
        # q1_0: ┤ H ├──■────■─────────────────
        #       ├───┤  │  ┌─┴─┐
        # q1_1: ┤ H ├──┼──┤ X ├──■────────────
        #       ├───┤  │  └───┘  │  ┌───┐
        # q1_2: ┤ H ├──┼─────────┼──┤ X ├──■──
        #       ├───┤┌─┴─┐       │  └─┬─┘┌─┴─┐
        # q2_0: ┤ H ├┤ X ├───────┼────┼──┤ X ├
        #       ├───┤└─┬─┘     ┌─┴─┐  │  └───┘
        # q2_1: ┤ H ├──■───────┤ X ├──■───────
        #       └───┘          └───┘
        q1 = QuantumRegister(3, "q1")
        q2 = QuantumRegister(2, "q2")
        c = ClassicalRegister(5, "c")
        qc = QuantumCircuit(q1, q2, c)
        qc.h(q1[0])
        qc.h(q1[1])
        qc.h(q1[2])
        qc.h(q2[0])
        qc.h(q2[1])
        qc.ccx(q2[1], q1[0], q2[0])
        qc.cx(q1[0], q1[1])
        qc.cx(q1[1], q2[1])
        qc.cx(q2[1], q1[2])
        qc.cx(q1[2], q2[0])
        self.assertEqual(qc.depth(), 6)

    def test_circuit_depth_conditionals1(self):
        """Test circuit depth for conditional gates #1."""

        #      ┌───┐     ┌─┐
        # q_0: ┤ H ├──■──┤M├─────────────────
        #      ├───┤┌─┴─┐└╥┘┌─┐
        # q_1: ┤ H ├┤ X ├─╫─┤M├──────────────
        #      ├───┤└───┘ ║ └╥┘ ┌───┐
        # q_2: ┤ H ├──■───╫──╫──┤ H ├────────
        #      ├───┤┌─┴─┐ ║  ║  └─╥─┘  ┌───┐
        # q_3: ┤ H ├┤ X ├─╫──╫────╫────┤ H ├─
        #      └───┘└───┘ ║  ║    ║    └─╥─┘
        #                 ║  ║ ┌──╨──┐┌──╨──┐
        # c: 4/═══════════╩══╩═╡ 0x2 ╞╡ 0x4 ╞
        #                 0  1 └─────┘└─────┘
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)

        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.cx(q[0], q[1])
        qc.cx(q[2], q[3])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[1])
        qc.h(q[2]).c_if(c, 2)
        qc.h(q[3]).c_if(c, 4)
        self.assertEqual(qc.depth(), 5)

    def test_circuit_depth_conditionals2(self):
        """Test circuit depth for conditional gates #2."""

        #      ┌───┐     ┌─┐┌─┐
        # q_0: ┤ H ├──■──┤M├┤M├──────────────
        #      ├───┤┌─┴─┐└╥┘└╥┘
        # q_1: ┤ H ├┤ X ├─╫──╫───────────────
        #      ├───┤└───┘ ║  ║  ┌───┐
        # q_2: ┤ H ├──■───╫──╫──┤ H ├────────
        #      ├───┤┌─┴─┐ ║  ║  └─╥─┘  ┌───┐
        # q_3: ┤ H ├┤ X ├─╫──╫────╫────┤ H ├─
        #      └───┘└───┘ ║  ║    ║    └─╥─┘
        #                 ║  ║ ┌──╨──┐┌──╨──┐
        # c: 4/═══════════╩══╩═╡ 0x2 ╞╡ 0x4 ╞
        #                 0  0 └─────┘└─────┘
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)

        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.cx(q[0], q[1])
        qc.cx(q[2], q[3])
        qc.measure(q[0], c[0])
        qc.measure(q[0], c[0])
        qc.h(q[2]).c_if(c, 2)
        qc.h(q[3]).c_if(c, 4)
        self.assertEqual(qc.depth(), 6)

    def test_circuit_depth_conditionals3(self):
        """Test circuit depth for conditional gates #3."""

        #      ┌───┐┌─┐
        # q_0: ┤ H ├┤M├───■────────────
        #      ├───┤└╥┘   │   ┌─┐
        # q_1: ┤ H ├─╫────┼───┤M├──────
        #      ├───┤ ║    │   └╥┘┌─┐
        # q_2: ┤ H ├─╫────┼────╫─┤M├───
        #      ├───┤ ║  ┌─┴─┐  ║ └╥┘┌─┐
        # q_3: ┤ H ├─╫──┤ X ├──╫──╫─┤M├
        #      └───┘ ║  └─╥─┘  ║  ║ └╥┘
        #            ║ ┌──╨──┐ ║  ║  ║
        # c: 4/══════╩═╡ 0x2 ╞═╩══╩══╩═
        #            0 └─────┘ 1  2  3
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)

        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.measure(q[0], c[0])
        qc.cx(q[0], q[3]).c_if(c, 2)

        qc.measure(q[1], c[1])
        qc.measure(q[2], c[2])
        qc.measure(q[3], c[3])
        self.assertEqual(qc.depth(), 4)

    def test_circuit_depth_bit_conditionals1(self):
        """Test circuit depth for single bit conditional gates #1."""

        #      ┌───┐┌─┐
        # q_0: ┤ H ├┤M├─────────────────────────
        #      ├───┤└╥┘      ┌───┐
        # q_1: ┤ H ├─╫───────┤ H ├──────────────
        #      ├───┤ ║ ┌─┐   └─╥─┘
        # q_2: ┤ H ├─╫─┤M├─────╫────────────────
        #      ├───┤ ║ └╥┘     ║        ┌───┐
        # q_3: ┤ H ├─╫──╫──────╫────────┤ H ├───
        #      └───┘ ║  ║      ║        └─╥─┘
        #            ║  ║ ┌────╨────┐┌────╨────┐
        # c: 4/══════╩══╩═╡ c_0=0x1 ╞╡ c_2=0x0 ╞
        #            0  2 └─────────┘└─────────┘
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)

        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.measure(q[0], c[0])
        qc.measure(q[2], c[2])
        qc.h(q[1]).c_if(c[0], True)
        qc.h(q[3]).c_if(c[2], False)
        self.assertEqual(qc.depth(), 3)

    def test_circuit_depth_bit_conditionals2(self):
        """Test circuit depth for single bit conditional gates #2."""

        #      ┌───┐┌─┐                                                          »
        # q_0: ┤ H ├┤M├──────────────────────────────■─────────────────────■─────»
        #      ├───┤└╥┘      ┌───┐                 ┌─┴─┐                   │     »
        # q_1: ┤ H ├─╫───────┤ H ├─────────────────┤ X ├───────────────────┼─────»
        #      ├───┤ ║ ┌─┐   └─╥─┘                 └─╥─┘                 ┌─┴─┐   »
        # q_2: ┤ H ├─╫─┤M├─────╫─────────────────────╫──────────■────────┤ H ├───»
        #      ├───┤ ║ └╥┘     ║        ┌───┐        ║        ┌─┴─┐      └─╥─┘   »
        # q_3: ┤ H ├─╫──╫──────╫────────┤ H ├────────╫────────┤ X ├────────╫─────»
        #      └───┘ ║  ║      ║        └─╥─┘        ║        └─╥─┘        ║     »
        #            ║  ║ ┌────╨────┐┌────╨────┐┌────╨────┐┌────╨────┐┌────╨────┐»
        # c: 4/══════╩══╩═╡ c_1=0x1 ╞╡ c_3=0x1 ╞╡ c_0=0x0 ╞╡ c_2=0x0 ╞╡ c_1=0x1 ╞»
        #            0  2 └─────────┘└─────────┘└─────────┘└─────────┘└─────────┘»
        # «
        # «q_0: ───────────
        # «
        # «q_1: ─────■─────
        # «          │
        # «q_2: ─────┼─────
        # «        ┌─┴─┐
        # «q_3: ───┤ H ├───
        # «        └─╥─┘
        # «     ┌────╨────┐
        # «c: 4/╡ c_3=0x1 ╞
        # «     └─────────┘
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)

        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.measure(q[0], c[0])
        qc.measure(q[2], c[2])
        qc.h(q[1]).c_if(c[1], True)
        qc.h(q[3]).c_if(c[3], True)
        qc.cx(0, 1).c_if(c[0], False)
        qc.cx(2, 3).c_if(c[2], False)
        qc.ch(0, 2).c_if(c[1], True)
        qc.ch(1, 3).c_if(c[3], True)
        self.assertEqual(qc.depth(), 4)

    def test_circuit_depth_bit_conditionals3(self):
        """Test circuit depth for single bit conditional gates #3."""

        #      ┌───┐┌─┐
        # q_0: ┤ H ├┤M├──────────────────────────────────────
        #      ├───┤└╥┘   ┌───┐                     ┌─┐
        # q_1: ┤ H ├─╫────┤ H ├─────────────────────┤M├──────
        #      ├───┤ ║    └─╥─┘    ┌───┐            └╥┘┌─┐
        # q_2: ┤ H ├─╫──────╫──────┤ H ├─────────────╫─┤M├───
        #      ├───┤ ║      ║      └─╥─┘    ┌───┐    ║ └╥┘┌─┐
        # q_3: ┤ H ├─╫──────╫────────╫──────┤ H ├────╫──╫─┤M├
        #      └───┘ ║      ║        ║      └─╥─┘    ║  ║ └╥┘
        #            ║ ┌────╨────┐┌──╨──┐┌────╨────┐ ║  ║  ║
        # c: 4/══════╩═╡ c_0=0x1 ╞╡ 0x2 ╞╡ c_3=0x1 ╞═╩══╩══╩═
        #            0 └─────────┘└─────┘└─────────┘ 1  2  3
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)

        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.measure(q[0], c[0])
        qc.h(1).c_if(c[0], True)
        qc.h(q[2]).c_if(c, 2)
        qc.h(3).c_if(c[3], True)
        qc.measure(q[1], c[1])
        qc.measure(q[2], c[2])
        qc.measure(q[3], c[3])
        self.assertEqual(qc.depth(), 6)

    def test_circuit_depth_measurements1(self):
        """Test circuit depth for measurements #1."""

        #      ┌───┐┌─┐
        # q_0: ┤ H ├┤M├─────────
        #      ├───┤└╥┘┌─┐
        # q_1: ┤ H ├─╫─┤M├──────
        #      ├───┤ ║ └╥┘┌─┐
        # q_2: ┤ H ├─╫──╫─┤M├───
        #      ├───┤ ║  ║ └╥┘┌─┐
        # q_3: ┤ H ├─╫──╫──╫─┤M├
        #      └───┘ ║  ║  ║ └╥┘
        # c: 4/══════╩══╩══╩══╩═
        #            0  1  2  3
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)

        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[1])
        qc.measure(q[2], c[2])
        qc.measure(q[3], c[3])
        self.assertEqual(qc.depth(), 2)

    def test_circuit_depth_measurements2(self):
        """Test circuit depth for measurements #2."""

        #      ┌───┐┌─┐┌─┐┌─┐┌─┐
        # q_0: ┤ H ├┤M├┤M├┤M├┤M├
        #      ├───┤└╥┘└╥┘└╥┘└╥┘
        # q_1: ┤ H ├─╫──╫──╫──╫─
        #      ├───┤ ║  ║  ║  ║
        # q_2: ┤ H ├─╫──╫──╫──╫─
        #      ├───┤ ║  ║  ║  ║
        # q_3: ┤ H ├─╫──╫──╫──╫─
        #      └───┘ ║  ║  ║  ║
        # c: 4/══════╩══╩══╩══╩═
        #            0  1  2  3
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)

        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.measure(q[0], c[0])
        qc.measure(q[0], c[1])
        qc.measure(q[0], c[2])
        qc.measure(q[0], c[3])
        self.assertEqual(qc.depth(), 5)

    def test_circuit_depth_measurements3(self):
        """Test circuit depth for measurements #3."""

        #      ┌───┐┌─┐
        # q_0: ┤ H ├┤M├─────────
        #      ├───┤└╥┘┌─┐
        # q_1: ┤ H ├─╫─┤M├──────
        #      ├───┤ ║ └╥┘┌─┐
        # q_2: ┤ H ├─╫──╫─┤M├───
        #      ├───┤ ║  ║ └╥┘┌─┐
        # q_3: ┤ H ├─╫──╫──╫─┤M├
        #      └───┘ ║  ║  ║ └╥┘
        # c: 4/══════╩══╩══╩══╩═
        #            0  0  0  0
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)

        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[0])
        qc.measure(q[2], c[0])
        qc.measure(q[3], c[0])
        self.assertEqual(qc.depth(), 5)

    def test_circuit_depth_barriers1(self):
        """Test circuit depth for barriers #1."""

        #      ┌───┐      ░
        # q_0: ┤ H ├──■───░───────────
        #      └───┘┌─┴─┐ ░
        # q_1: ─────┤ X ├─░───────────
        #           └───┘ ░ ┌───┐
        # q_2: ───────────░─┤ H ├──■──
        #                 ░ └───┘┌─┴─┐
        # q_3: ───────────░──────┤ X ├
        #                 ░      └───┘
        q = QuantumRegister(4, "q")
        c = ClassicalRegister(4, "c")
        circ = QuantumCircuit(q, c)
        circ.h(0)
        circ.cx(0, 1)
        circ.barrier(q)
        circ.h(2)
        circ.cx(2, 3)
        self.assertEqual(circ.depth(), 4)

    def test_circuit_depth_barriers2(self):
        """Test circuit depth for barriers #2."""

        #      ┌───┐ ░       ░       ░
        # q_0: ┤ H ├─░───■───░───────░──────
        #      └───┘ ░ ┌─┴─┐ ░       ░
        # q_1: ──────░─┤ X ├─░───────░──────
        #            ░ └───┘ ░ ┌───┐ ░
        # q_2: ──────░───────░─┤ H ├─░───■──
        #            ░       ░ └───┘ ░ ┌─┴─┐
        # q_3: ──────░───────░───────░─┤ X ├
        #            ░       ░       ░ └───┘
        q = QuantumRegister(4, "q")
        c = ClassicalRegister(4, "c")
        circ = QuantumCircuit(q, c)
        circ.h(0)
        circ.barrier(q)
        circ.cx(0, 1)
        circ.barrier(q)
        circ.h(2)
        circ.barrier(q)
        circ.cx(2, 3)
        self.assertEqual(circ.depth(), 4)

    def test_circuit_depth_barriers3(self):
        """Test circuit depth for barriers #3."""

        #      ┌───┐ ░       ░  ░  ░       ░
        # q_0: ┤ H ├─░───■───░──░──░───────░──────
        #      └───┘ ░ ┌─┴─┐ ░  ░  ░       ░
        # q_1: ──────░─┤ X ├─░──░──░───────░──────
        #            ░ └───┘ ░  ░  ░ ┌───┐ ░
        # q_2: ──────░───────░──░──░─┤ H ├─░───■──
        #            ░       ░  ░  ░ └───┘ ░ ┌─┴─┐
        # q_3: ──────░───────░──░──░───────░─┤ X ├
        #            ░       ░  ░  ░       ░ └───┘
        q = QuantumRegister(4, "q")
        c = ClassicalRegister(4, "c")
        circ = QuantumCircuit(q, c)
        circ.h(0)
        circ.barrier(q)
        circ.cx(0, 1)
        circ.barrier(q)
        circ.barrier(q)
        circ.barrier(q)
        circ.h(2)
        circ.barrier(q)
        circ.cx(2, 3)
        self.assertEqual(circ.depth(), 4)

    def test_circuit_depth_2qubit(self):
        """Test finding depth of two-qubit gates only."""

        #      ┌───┐
        # q_0: ┤ H ├──■───────────────────
        #      └───┘┌─┴─┐┌─────────┐   ┌─┐
        # q_1: ─────┤ X ├┤ Rz(0.1) ├─■─┤M├
        #      ┌───┐└───┘└─────────┘ │ └╥┘
        # q_2: ┤ H ├──■──────────────┼──╫─
        #      └───┘┌─┴─┐            │  ║
        # q_3: ─────┤ X ├────────────■──╫─
        #           └───┘               ║
        # c: 1/═════════════════════════╩═
        #                               0
        circ = QuantumCircuit(4, 1)
        circ.h(0)
        circ.cx(0, 1)
        circ.h(2)
        circ.cx(2, 3)
        circ.rz(0.1, 1)
        circ.cz(1, 3)
        circ.measure(1, 0)
        self.assertEqual(circ.depth(lambda x: x.operation.num_qubits == 2), 2)

    def test_circuit_depth_multiqubit_or_conditional(self):
        """Test finding depth of multi-qubit or conditional gates."""

        #      ┌───┐                              ┌───┐
        # q_0: ┤ H ├──■───────────────────────────┤ X ├───
        #      └───┘  │  ┌─────────┐        ┌─┐   └─╥─┘
        # q_1: ───────■──┤ Rz(0.1) ├──────■─┤M├─────╫─────
        #           ┌─┴─┐└──┬───┬──┘      │ └╥┘     ║
        # q_2: ─────┤ X ├───┤ H ├─────■───┼──╫──────╫─────
        #           └───┘   └───┘   ┌─┴─┐ │  ║      ║
        # q_3: ─────────────────────┤ X ├─■──╫──────╫─────
        #                           └───┘    ║ ┌────╨────┐
        # c: 1/══════════════════════════════╩═╡ c_0 = T ╞
        #                                    0 └─────────┘
        circ = QuantumCircuit(4, 1)
        circ.h(0)
        circ.ccx(0, 1, 2)
        circ.h(2)
        circ.cx(2, 3)
        circ.rz(0.1, 1)
        circ.cz(1, 3)
        circ.measure(1, 0)
        circ.x(0).c_if(0, 1)
        self.assertEqual(
            circ.depth(lambda x: x.operation.num_qubits >= 2 or x.operation.condition is not None),
            4,
        )

    def test_circuit_depth_first_qubit(self):
        """Test finding depth of gates touching q0 only."""

        #      ┌───┐        ┌───┐
        # q_0: ┤ H ├──■─────┤ T ├─────────
        #      └───┘┌─┴─┐┌──┴───┴──┐   ┌─┐
        # q_1: ─────┤ X ├┤ Rz(0.1) ├─■─┤M├
        #      ┌───┐└───┘└─────────┘ │ └╥┘
        # q_2: ┤ H ├──■──────────────┼──╫─
        #      └───┘┌─┴─┐            │  ║
        # q_3: ─────┤ X ├────────────■──╫─
        #           └───┘               ║
        # c: 1/═════════════════════════╩═
        #                               0
        circ = QuantumCircuit(4, 1)
        circ.h(0)
        circ.cx(0, 1)
        circ.t(0)
        circ.h(2)
        circ.cx(2, 3)
        circ.rz(0.1, 1)
        circ.cz(1, 3)
        circ.measure(1, 0)
        self.assertEqual(circ.depth(lambda x: circ.qubits[0] in x.qubits), 3)

    def test_circuit_size_empty(self):
        """Circuit.size should return 0 for an empty circuit."""
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)

        self.assertEqual(qc.size(), 0)

    def test_circuit_size_single_qubit_gates(self):
        """Circuit.size should increment for each added single qubit gate."""
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)

        qc.h(q[0])
        self.assertEqual(qc.size(), 1)
        qc.h(q[1])
        self.assertEqual(qc.size(), 2)

    def test_circuit_size_2qubit(self):
        """Circuit.size of only 2-qubit gates."""
        size = 3
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)

        qc.cx(q[0], q[1])
        qc.rz(0.1, q[1])
        qc.rzz(0.1, q[1], q[2])
        self.assertEqual(qc.size(lambda x: x.operation.num_qubits == 2), 2)

    def test_circuit_count_ops(self):
        """Test circuit count ops."""
        q = QuantumRegister(6, "q")
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.x(q[1])
        qc.y(q[2:4])
        qc.z(q[3:])
        result = qc.count_ops()

        expected = {"h": 6, "z": 3, "y": 2, "x": 1}

        self.assertIsInstance(result, dict)
        self.assertEqual(expected, result)

    def test_circuit_count_ops_on_qubits(self):
        """Test circuit count ops on specified qubits."""
        q = QuantumRegister(5, "q")
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.x(q[1])
        qc.y(q[2:4])
        qc.z(q[3:])
        qc.cx(q[0], q[1])
        qc.ccx(q[2], q[3], q[4])
        result = qc.count_ops(qubits=[1, 2])
        expected = {"x": 1, 'cx': 1, 'h': 2, 'y': 1, 'ccx': 1}
        self.assertIsInstance(result, dict)
        self.assertEqual(expected, result)

    def test_circuit_nonlocal_gates(self):
        """Test num_nonlocal_gates."""

        #      ┌───┐                   ┌────────┐
        # q_0: ┤ H ├───────────────────┤0       ├
        #      ├───┤   ┌───┐           │        │
        # q_1: ┤ H ├───┤ X ├─────────■─┤        ├
        #      ├───┤   └───┘         │ │        │
        # q_2: ┤ H ├─────■───────────X─┤  Iswap ├
        #      ├───┤     │     ┌───┐ │ │        │
        # q_3: ┤ H ├─────┼─────┤ Z ├─X─┤        ├
        #      ├───┤┌────┴────┐├───┤   │        │
        # q_4: ┤ H ├┤ Ry(0.1) ├┤ Z ├───┤1       ├
        #      ├───┤└──┬───┬──┘└───┘   └───╥────┘
        # q_5: ┤ H ├───┤ Z ├───────────────╫─────
        #      └───┘   └───┘            ┌──╨──┐
        # c: 2/═════════════════════════╡ 0x2 ╞══
        #                               └─────┘
        q = QuantumRegister(6, "q")
        c = ClassicalRegister(2, "c")
        qc = QuantumCircuit(q, c)
        qc.h(q)
        qc.x(q[1])
        qc.cry(0.1, q[2], q[4])
        qc.z(q[3:])
        qc.cswap(q[1], q[2], q[3])
        qc.iswap(q[0], q[4]).c_if(c, 2)
        result = qc.num_nonlocal_gates()
        expected = 3
        self.assertEqual(expected, result)

    def test_circuit_nonlocal_gates_no_instruction(self):
        """Verify num_nunlocal_gates does not include barriers."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/4500
        n = 3
        qc = QuantumCircuit(n)
        qc.h(range(n))

        qc.barrier()

        self.assertEqual(qc.num_nonlocal_gates(), 0)

    def test_circuit_connected_components_empty(self):
        """Verify num_connected_components is width for empty"""
        q = QuantumRegister(7, "q")
        qc = QuantumCircuit(q)
        self.assertEqual(7, qc.num_connected_components())

    def test_circuit_connected_components_multi_reg(self):
        """Test tensor factors works over multi registers"""

        #       ┌───┐
        # q1_0: ┤ H ├──■─────────────────
        #       ├───┤┌─┴─┐
        # q1_1: ┤ H ├┤ X ├──■────────────
        #       ├───┤└───┘  │  ┌───┐
        # q1_2: ┤ H ├───────┼──┤ X ├──■──
        #       ├───┤       │  └─┬─┘┌─┴─┐
        # q2_0: ┤ H ├───────┼────┼──┤ X ├
        #       ├───┤     ┌─┴─┐  │  └───┘
        # q2_1: ┤ H ├─────┤ X ├──■───────
        #       └───┘     └───┘
        q1 = QuantumRegister(3, "q1")
        q2 = QuantumRegister(2, "q2")
        qc = QuantumCircuit(q1, q2)
        qc.h(q1[0])
        qc.h(q1[1])
        qc.h(q1[2])
        qc.h(q2[0])
        qc.h(q2[1])
        qc.cx(q1[0], q1[1])
        qc.cx(q1[1], q2[1])
        qc.cx(q2[1], q1[2])
        qc.cx(q1[2], q2[0])
        self.assertEqual(qc.num_connected_components(), 1)

    def test_circuit_connected_components_multi_reg2(self):
        """Test tensor factors works over multi registers #2."""

        # q1_0: ──■────────────
        #         │
        # q1_1: ──┼─────────■──
        #         │  ┌───┐  │
        # q1_2: ──┼──┤ X ├──┼──
        #         │  └─┬─┘┌─┴─┐
        # q2_0: ──┼────■──┤ X ├
        #       ┌─┴─┐     └───┘
        # q2_1: ┤ X ├──────────
        #       └───┘
        q1 = QuantumRegister(3, "q1")
        q2 = QuantumRegister(2, "q2")
        qc = QuantumCircuit(q1, q2)
        qc.cx(q1[0], q2[1])
        qc.cx(q2[0], q1[2])
        qc.cx(q1[1], q2[0])
        self.assertEqual(qc.num_connected_components(), 2)

    def test_circuit_connected_components_disconnected(self):
        """Test tensor factors works with 2q subspaces."""

        # q1_0: ──■──────────────────────
        #         │
        # q1_1: ──┼────■─────────────────
        #         │    │
        # q1_2: ──┼────┼────■────────────
        #         │    │    │
        # q1_3: ──┼────┼────┼────■───────
        #         │    │    │    │
        # q1_4: ──┼────┼────┼────┼────■──
        #         │    │    │    │  ┌─┴─┐
        # q2_0: ──┼────┼────┼────┼──┤ X ├
        #         │    │    │  ┌─┴─┐└───┘
        # q2_1: ──┼────┼────┼──┤ X ├─────
        #         │    │  ┌─┴─┐└───┘
        # q2_2: ──┼────┼──┤ X ├──────────
        #         │  ┌─┴─┐└───┘
        # q2_3: ──┼──┤ X ├───────────────
        #       ┌─┴─┐└───┘
        # q2_4: ┤ X ├────────────────────
        #       └───┘
        q1 = QuantumRegister(5, "q1")
        q2 = QuantumRegister(5, "q2")
        qc = QuantumCircuit(q1, q2)
        qc.cx(q1[0], q2[4])
        qc.cx(q1[1], q2[3])
        qc.cx(q1[2], q2[2])
        qc.cx(q1[3], q2[1])
        qc.cx(q1[4], q2[0])
        self.assertEqual(qc.num_connected_components(), 5)

    def test_circuit_connected_components_with_clbits(self):
        """Test tensor components with classical register."""

        #      ┌───┐┌─┐
        # q_0: ┤ H ├┤M├─────────
        #      ├───┤└╥┘┌─┐
        # q_1: ┤ H ├─╫─┤M├──────
        #      ├───┤ ║ └╥┘┌─┐
        # q_2: ┤ H ├─╫──╫─┤M├───
        #      ├───┤ ║  ║ └╥┘┌─┐
        # q_3: ┤ H ├─╫──╫──╫─┤M├
        #      └───┘ ║  ║  ║ └╥┘
        # c: 4/══════╩══╩══╩══╩═
        #           0  1  2  3
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[1])
        qc.measure(q[2], c[2])
        qc.measure(q[3], c[3])
        self.assertEqual(qc.num_connected_components(), 4)

    def test_circuit_connected_components_with_cond(self):
        """Test tensor components with one conditional gate."""

        #      ┌───┐┌─┐
        # q_0: ┤ H ├┤M├───■────────────
        #      ├───┤└╥┘   │   ┌─┐
        # q_1: ┤ H ├─╫────┼───┤M├──────
        #      ├───┤ ║    │   └╥┘┌─┐
        # q_2: ┤ H ├─╫────┼────╫─┤M├───
        #      ├───┤ ║  ┌─┴─┐  ║ └╥┘┌─┐
        # q_3: ┤ H ├─╫──┤ X ├──╫──╫─┤M├
        #      └───┘ ║  └─╥─┘  ║  ║ └╥┘
        #            ║ ┌──╨──┐ ║  ║  ║
        # c: 4/══════╩═╡ 0x2 ╞═╩══╩══╩═
        #            0 └─────┘ 1  2  3
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.measure(q[0], c[0])
        qc.cx(q[0], q[3]).c_if(c, 2)
        qc.measure(q[1], c[1])
        qc.measure(q[2], c[2])
        qc.measure(q[3], c[3])
        self.assertEqual(qc.num_connected_components(), 1)

    def test_circuit_connected_components_with_cond2(self):
        """Test tensor components with two conditional gates."""

        #      ┌───┐ ┌───┐
        # q_0: ┤ H ├─┤ H ├────────
        #      ├───┤ └─╥─┘
        # q_1: ┤ H ├───╫──────■───
        #      ├───┤   ║    ┌─┴─┐
        # q_2: ┤ H ├───╫────┤ X ├─
        #      ├───┤   ║    └─╥─┘
        # q_3: ┤ H ├───╫──────╫───
        #      └───┘┌──╨──┐┌──╨──┐
        # c: 8/═════╡ 0x0 ╞╡ 0x4 ╞
        #           └─────┘└─────┘
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(2 * size, "c")
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.h(0).c_if(c, 0)
        qc.cx(1, 2).c_if(c, 4)
        self.assertEqual(qc.num_connected_components(), 2)

    def test_circuit_connected_components_with_cond3(self):
        """Test tensor components with three conditional gates and measurements."""

        #       ┌───┐┌─┐ ┌───┐
        # q0_0: ┤ H ├┤M├─┤ H ├──────────────────
        #       ├───┤└╥┘ └─╥─┘
        # q0_1: ┤ H ├─╫────╫──────■─────────────
        #       ├───┤ ║    ║    ┌─┴─┐ ┌─┐
        # q0_2: ┤ H ├─╫────╫────┤ X ├─┤M├───────
        #       ├───┤ ║    ║    └─╥─┘ └╥┘ ┌───┐
        # q0_3: ┤ H ├─╫────╫──────╫────╫──┤ X ├─
        #       └───┘ ║    ║      ║    ║  └─╥─┘
        #             ║ ┌──╨──┐┌──╨──┐ ║ ┌──╨──┐
        # c0: 4/══════╩═╡ 0x0 ╞╡ 0x1 ╞═╩═╡ 0x2 ╞
        #             0 └─────┘└─────┘ 2 └─────┘
        size = 4
        q = QuantumRegister(size)
        c = ClassicalRegister(size)
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.measure(q[0], c[0])
        qc.h(q[0]).c_if(c, 0)
        qc.cx(q[1], q[2]).c_if(c, 1)
        qc.measure(q[2], c[2])
        qc.x(q[3]).c_if(c, 2)
        self.assertEqual(qc.num_connected_components(), 1)

    def test_circuit_connected_components_with_bit_cond(self):
        """Test tensor components with one single bit conditional gate."""

        #      ┌───┐┌─┐
        # q_0: ┤ H ├┤M├───────────■────────
        #      ├───┤└╥┘┌─┐        │
        # q_1: ┤ H ├─╫─┤M├────────┼────────
        #      ├───┤ ║ └╥┘┌─┐     │
        # q_2: ┤ H ├─╫──╫─┤M├─────┼────────
        #      ├───┤ ║  ║ └╥┘   ┌─┴─┐   ┌─┐
        # q_3: ┤ H ├─╫──╫──╫────┤ X ├───┤M├
        #      └───┘ ║  ║  ║    └─╥─┘   └╥┘
        #            ║  ║  ║ ┌────╨────┐ ║
        # c: 4/══════╩══╩══╩═╡ c_0=0x1 ╞═╩═
        #            0  1  2 └─────────┘ 3
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.measure(q[0], c[0])
        qc.cx(q[0], q[3]).c_if(c[0], True)
        qc.measure(q[1], c[1])
        qc.measure(q[2], c[2])
        qc.measure(q[3], c[3])
        self.assertEqual(qc.num_connected_components(), 3)

    def test_circuit_connected_components_with_bit_cond2(self):
        """Test tensor components with two bit conditional gates."""

        #      ┌───┐   ┌───┐                 ┌───┐
        # q_0: ┤ H ├───┤ H ├─────────────────┤ X ├───
        #      ├───┤   └─╥─┘                 └─┬─┘
        # q_1: ┤ H ├─────╫─────────────────────■─────
        #      ├───┤     ║                     ║
        # q_2: ┤ H ├─────╫──────────■──────────╫─────
        #      ├───┤     ║          │          ║
        # q_3: ┤ H ├─────╫──────────■──────────╫─────
        #      └───┘┌────╨────┐┌────╨────┐┌────╨────┐
        # c: 6/═════╡ c_1=0x1 ╞╡ c_0=0x1 ╞╡ c_4=0x0 ╞
        #           └─────────┘└─────────┘└─────────┘
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size + 2, "c")
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.h(0).c_if(c[1], True)
        qc.cx(1, 0).c_if(c[4], False)
        qc.cz(2, 3).c_if(c[0], True)
        self.assertEqual(qc.num_connected_components(), 5)

    def test_circuit_connected_components_with_bit_cond3(self):
        """Test tensor components with register and bit conditional gates."""

        #       ┌───┐   ┌───┐
        # q0_0: ┤ H ├───┤ H ├───────────────────────
        #       ├───┤   └─╥─┘
        # q0_1: ┤ H ├─────╫─────────■───────────────
        #       ├───┤     ║       ┌─┴─┐
        # q0_2: ┤ H ├─────╫───────┤ X ├─────────────
        #       ├───┤     ║       └─╥─┘    ┌───┐
        # q0_3: ┤ H ├─────╫─────────╫──────┤ X ├────
        #       └───┘     ║         ║      └─╥─┘
        #            ┌────╨─────┐┌──╨──┐┌────╨─────┐
        # c0: 4/═════╡ c0_0=0x1 ╞╡ 0x1 ╞╡ c0_2=0x1 ╞
        #            └──────────┘└─────┘└──────────┘
        size = 4
        q = QuantumRegister(size)
        c = ClassicalRegister(size)
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.h(q[0]).c_if(c[0], True)
        qc.cx(q[1], q[2]).c_if(c, 1)
        qc.x(q[3]).c_if(c[2], True)
        self.assertEqual(qc.num_connected_components(), 1)

    def test_circuit_unitary_factors1(self):
        """Test unitary factors empty circuit."""
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)
        self.assertEqual(qc.num_unitary_factors(), 4)

    def test_circuit_unitary_factors2(self):
        """Test unitary factors multi qregs"""
        q1 = QuantumRegister(2, "q1")
        q2 = QuantumRegister(2, "q2")
        c = ClassicalRegister(4, "c")
        qc = QuantumCircuit(q1, q2, c)
        self.assertEqual(qc.num_unitary_factors(), 4)

    def test_circuit_unitary_factors3(self):
        """Test unitary factors measurements and conditionals."""

        #      ┌───┐                                      ┌─┐
        # q_0: ┤ H ├────────■──────────■────■──────────■──┤M├───
        #      ├───┤        │          │    │  ┌─┐     │  └╥┘
        # q_1: ┤ H ├──■─────┼─────■────┼────┼──┤M├─────┼───╫────
        #      ├───┤┌─┴─┐   │   ┌─┴─┐  │    │  └╥┘┌─┐  │   ║
        # q_2: ┤ H ├┤ X ├───┼───┤ X ├──┼────┼───╫─┤M├──┼───╫────
        #      ├───┤└───┘ ┌─┴─┐ └───┘┌─┴─┐┌─┴─┐ ║ └╥┘┌─┴─┐ ║ ┌─┐
        # q_3: ┤ H ├──────┤ X ├──────┤ X ├┤ X ├─╫──╫─┤ X ├─╫─┤M├
        #      └───┘      └─╥─┘      └───┘└───┘ ║  ║ └───┘ ║ └╥┘
        #                ┌──╨──┐                ║  ║       ║  ║
        # c: 4/══════════╡ 0x2 ╞════════════════╩══╩═══════╩══╩═
        #                └─────┘                1  2       0  3
        size = 4
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.cx(q[1], q[2])
        qc.cx(q[1], q[2])
        qc.cx(q[0], q[3]).c_if(c, 2)
        qc.cx(q[0], q[3])
        qc.cx(q[0], q[3])
        qc.cx(q[0], q[3])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[1])
        qc.measure(q[2], c[2])
        qc.measure(q[3], c[3])
        self.assertEqual(qc.num_unitary_factors(), 2)

    def test_circuit_unitary_factors4(self):
        """Test unitary factors measurements go to same cbit."""

        #      ┌───┐┌─┐
        # q_0: ┤ H ├┤M├─────────
        #      ├───┤└╥┘┌─┐
        # q_1: ┤ H ├─╫─┤M├──────
        #      ├───┤ ║ └╥┘┌─┐
        # q_2: ┤ H ├─╫──╫─┤M├───
        #      ├───┤ ║  ║ └╥┘┌─┐
        # q_3: ┤ H ├─╫──╫──╫─┤M├
        #      └───┘ ║  ║  ║ └╥┘
        # q_4: ──────╫──╫──╫──╫─
        #            ║  ║  ║  ║
        # c: 5/══════╩══╩══╩══╩═
        #            0  0  0  0
        size = 5
        q = QuantumRegister(size, "q")
        c = ClassicalRegister(size, "c")
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.h(q[1])
        qc.h(q[2])
        qc.h(q[3])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[0])
        qc.measure(q[2], c[0])
        qc.measure(q[3], c[0])
        self.assertEqual(qc.num_unitary_factors(), 5)

    def test_num_qubits_qubitless_circuit(self):
        """Check output in absence of qubits."""
        c_reg = ClassicalRegister(3)
        circ = QuantumCircuit(c_reg)
        self.assertEqual(circ.num_qubits, 0)

    def test_num_qubits_qubitfull_circuit(self):
        """Check output in presence of qubits"""
        q_reg = QuantumRegister(4)
        c_reg = ClassicalRegister(3)
        circ = QuantumCircuit(q_reg, c_reg)
        self.assertEqual(circ.num_qubits, 4)

    def test_num_qubits_registerless_circuit(self):
        """Check output for circuits with direct argument for qubits."""
        circ = QuantumCircuit(5)
        self.assertEqual(circ.num_qubits, 5)

    def test_num_qubits_multiple_register_circuit(self):
        """Check output for circuits with multiple quantum registers."""
        q_reg1 = QuantumRegister(5)
        q_reg2 = QuantumRegister(6)
        q_reg3 = QuantumRegister(7)
        circ = QuantumCircuit(q_reg1, q_reg2, q_reg3)
        self.assertEqual(circ.num_qubits, 18)

    def test_calibrations_basis_gates(self):
        """Check if the calibrations for basis gates provided are added correctly."""
        circ = QuantumCircuit(2)

        with pulse.build() as q0_x180:
            pulse.play(pulse.library.Gaussian(20, 1.0, 3.0), pulse.DriveChannel(0))
        with pulse.build() as q1_y90:
            pulse.play(pulse.library.Gaussian(20, -1.0, 3.0), pulse.DriveChannel(1))

        # Add calibration
        circ.add_calibration(RXGate(3.14), [0], q0_x180)
        circ.add_calibration(RYGate(1.57), [1], q1_y90)

        self.assertEqual(set(circ.calibrations.keys()), {"rx", "ry"})
        self.assertEqual(set(circ.calibrations["rx"].keys()), {((0,), (3.14,))})
        self.assertEqual(set(circ.calibrations["ry"].keys()), {((1,), (1.57,))})
        self.assertEqual(
            circ.calibrations["rx"][((0,), (3.14,))].instructions, q0_x180.instructions
        )
        self.assertEqual(circ.calibrations["ry"][((1,), (1.57,))].instructions, q1_y90.instructions)

    def test_calibrations_custom_gates(self):
        """Check if the calibrations for custom gates with params provided are added correctly."""
        circ = QuantumCircuit(3)

        with pulse.build() as q0_x180:
            pulse.play(pulse.library.Gaussian(20, 1.0, 3.0), pulse.DriveChannel(0))

        # Add calibrations with a custom gate 'rxt'
        circ.add_calibration("rxt", [0], q0_x180, params=[1.57, 3.14, 4.71])

        self.assertEqual(set(circ.calibrations.keys()), {"rxt"})
        self.assertEqual(set(circ.calibrations["rxt"].keys()), {((0,), (1.57, 3.14, 4.71))})
        self.assertEqual(
            circ.calibrations["rxt"][((0,), (1.57, 3.14, 4.71))].instructions, q0_x180.instructions
        )

    def test_calibrations_no_params(self):
        """Check calibrations if the no params is provided with just gate name."""
        circ = QuantumCircuit(3)

        with pulse.build() as q0_x180:
            pulse.play(pulse.library.Gaussian(20, 1.0, 3.0), pulse.DriveChannel(0))

        circ.add_calibration("h", [0], q0_x180)

        self.assertEqual(set(circ.calibrations.keys()), {"h"})
        self.assertEqual(set(circ.calibrations["h"].keys()), {((0,), ())})
        self.assertEqual(circ.calibrations["h"][((0,), ())].instructions, q0_x180.instructions)

    def test_has_calibration_for(self):
        """Test that `has_calibration_for` returns a correct answer."""
        qc = QuantumCircuit(3)

        with pulse.build() as q0_x180:
            pulse.play(pulse.library.Gaussian(20, 1.0, 3.0), pulse.DriveChannel(0))
        qc.add_calibration("h", [0], q0_x180)

        qc.h(0)
        qc.h(1)

        self.assertTrue(qc.has_calibration_for(qc.data[0]))
        self.assertFalse(qc.has_calibration_for(qc.data[1]))

    def test_has_calibration_for_legacy(self):
        """Test that `has_calibration_for` returns a correct answer when presented with a legacy 3
        tuple."""
        qc = QuantumCircuit(3)

        with pulse.build() as q0_x180:
            pulse.play(pulse.library.Gaussian(20, 1.0, 3.0), pulse.DriveChannel(0))
        qc.add_calibration("h", [0], q0_x180)

        qc.h(0)
        qc.h(1)

        self.assertTrue(
            qc.has_calibration_for(
                (qc.data[0].operation, list(qc.data[0].qubits), list(qc.data[0].clbits))
            )
        )
        self.assertFalse(
            qc.has_calibration_for(
                (qc.data[1].operation, list(qc.data[1].qubits), list(qc.data[1].clbits))
            )
        )

    def test_metadata_copy_does_not_share_state(self):
        """Verify mutating the metadata of a circuit copy does not impact original."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/6057

        qc1 = QuantumCircuit(1)
        qc1.metadata = {"a": 0}

        qc2 = qc1.copy()
        qc2.metadata["a"] = 1000

        self.assertEqual(qc1.metadata["a"], 0)

    def test_metadata_is_dict(self):
        """Verify setting metadata to None in the constructor results in an empty dict."""
        qc = QuantumCircuit(1)
        metadata1 = qc.metadata
        self.assertEqual(metadata1, {})

    def test_metadata_raises(self):
        """Test that we must set metadata to a dict."""
        qc = QuantumCircuit(1)
        with self.assertRaises(TypeError):
            qc.metadata = 1

    def test_scheduling(self):
        """Test cannot return schedule information without scheduling."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        with self.assertRaises(AttributeError):
            # pylint: disable=pointless-statement
            qc.op_start_times


if __name__ == "__main__":
    unittest.main()
