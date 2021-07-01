# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test dynamical decoupling insertion pass."""

import unittest
from numpy import pi

from qiskit.circuit import QuantumCircuit, Delay
from qiskit.circuit.library import XGate, YGate, RXGate, UGate
from qiskit.quantum_info import Operator
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.passes import ASAPSchedule, ALAPSchedule, DynamicalDecoupling
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.exceptions import TranspilerError

from qiskit.test import QiskitTestCase


class TestDynamicalDecoupling(QiskitTestCase):
    """Tests DynamicalDecoupling pass."""

    def setUp(self):
        super().setUp()
        self.ghz4 = QuantumCircuit(4)
        self.ghz4.h(0)
        self.ghz4.cx(0, 1)
        self.ghz4.cx(1, 2)
        self.ghz4.cx(2, 3)

        self.durations = InstructionDurations(
            [
                ("h", 0, 50),
                ("cx", [0, 1], 700),
                ("cx", [1, 2], 200),
                ("cx", [2, 3], 300),
                ("x", None, 50),
                ("y", None, 50),
                ("u", None, 100),
                ("rx", None, 100),
                ("measure", None, 1000),
            ]
        )

        self.midmeas = QuantumCircuit(3, 1)
        self.midmeas.cx(0, 1)
        self.midmeas.cx(1, 2)
        self.midmeas.u(pi, 0, pi, 0)
        self.midmeas.measure(2, 0)
        self.midmeas.cx(1, 2)
        self.midmeas.cx(0, 1)

    def test_insert_dd_ghz(self):
        """Test DD gates are inserted in correct spots."""
        dd_sequence = [XGate(), XGate()]
        pm = PassManager(
            [ALAPSchedule(self.durations), DynamicalDecoupling(self.durations, dd_sequence)]
        )

        ghz4_dd = pm.run(self.ghz4.measure_all(inplace=False))

        expected = self.ghz4.copy()
        expected = expected.compose(Delay(50), [1], front=True)
        expected = expected.compose(Delay(750), [2], front=True)
        expected = expected.compose(Delay(950), [3], front=True)

        expected = expected.compose(Delay(100), [0])
        expected = expected.compose(XGate(), [0])
        expected = expected.compose(Delay(200), [0])
        expected = expected.compose(XGate(), [0])
        expected = expected.compose(Delay(100), [0])

        expected = expected.compose(Delay(50), [1])
        expected = expected.compose(XGate(), [1])
        expected = expected.compose(Delay(100), [1])
        expected = expected.compose(XGate(), [1])
        expected = expected.compose(Delay(50), [1])

        expected.measure_all()

        self.assertEqual(ghz4_dd, expected)

    def test_insert_dd_ghz_one_qubit(self):
        """Test DD gates are inserted on only one qubit."""
        dd_sequence = [XGate(), XGate()]
        pm = PassManager(
            [
                ALAPSchedule(self.durations),
                DynamicalDecoupling(self.durations, dd_sequence, qubits=[0]),
            ]
        )

        ghz4_dd = pm.run(self.ghz4.measure_all(inplace=False))

        expected = self.ghz4.copy()
        expected = expected.compose(Delay(50), [1], front=True)
        expected = expected.compose(Delay(750), [2], front=True)
        expected = expected.compose(Delay(950), [3], front=True)

        expected = expected.compose(Delay(100), [0])
        expected = expected.compose(XGate(), [0])
        expected = expected.compose(Delay(200), [0])
        expected = expected.compose(XGate(), [0])
        expected = expected.compose(Delay(100), [0])

        expected = expected.compose(Delay(300), [1])

        expected.measure_all()

        self.assertEqual(ghz4_dd, expected)

    def test_insert_dd_ghz_everywhere(self):
        """Test DD gates even on initial idle spots."""
        dd_sequence = [YGate(), YGate()]
        pm = PassManager(
            [
                ALAPSchedule(self.durations),
                DynamicalDecoupling(self.durations, dd_sequence, skip_reset_qubits=False),
            ]
        )

        ghz4_dd = pm.run(self.ghz4.measure_all(inplace=False))

        expected = self.ghz4.copy()
        expected = expected.compose(Delay(50), [1], front=True)

        expected = expected.compose(Delay(162), [2], front=True)
        expected = expected.compose(YGate(), [2], front=True)
        expected = expected.compose(Delay(326), [2], front=True)
        expected = expected.compose(YGate(), [2], front=True)
        expected = expected.compose(Delay(162), [2], front=True)

        expected = expected.compose(Delay(212), [3], front=True)
        expected = expected.compose(YGate(), [3], front=True)
        expected = expected.compose(Delay(426), [3], front=True)
        expected = expected.compose(YGate(), [3], front=True)
        expected = expected.compose(Delay(212), [3], front=True)

        expected = expected.compose(Delay(100), [0])
        expected = expected.compose(YGate(), [0])
        expected = expected.compose(Delay(200), [0])
        expected = expected.compose(YGate(), [0])
        expected = expected.compose(Delay(100), [0])

        expected = expected.compose(Delay(50), [1])
        expected = expected.compose(YGate(), [1])
        expected = expected.compose(Delay(100), [1])
        expected = expected.compose(YGate(), [1])
        expected = expected.compose(Delay(50), [1])

        expected.measure_all()

        self.assertEqual(ghz4_dd, expected)

    def test_insert_dd_ghz_xy4(self):
        """Test XY4 sequence of DD gates."""
        dd_sequence = [XGate(), YGate(), XGate(), YGate()]
        pm = PassManager(
            [ALAPSchedule(self.durations), DynamicalDecoupling(self.durations, dd_sequence)]
        )

        ghz4_dd = pm.run(self.ghz4.measure_all(inplace=False))

        expected = self.ghz4.copy()
        expected = expected.compose(Delay(50), [1], front=True)
        expected = expected.compose(Delay(750), [2], front=True)
        expected = expected.compose(Delay(950), [3], front=True)

        expected = expected.compose(Delay(37), [0])
        expected = expected.compose(XGate(), [0])
        expected = expected.compose(Delay(75), [0])
        expected = expected.compose(YGate(), [0])
        expected = expected.compose(Delay(76), [0])
        expected = expected.compose(XGate(), [0])
        expected = expected.compose(Delay(75), [0])
        expected = expected.compose(YGate(), [0])
        expected = expected.compose(Delay(37), [0])

        expected = expected.compose(Delay(12), [1])
        expected = expected.compose(XGate(), [1])
        expected = expected.compose(Delay(25), [1])
        expected = expected.compose(YGate(), [1])
        expected = expected.compose(Delay(26), [1])
        expected = expected.compose(XGate(), [1])
        expected = expected.compose(Delay(25), [1])
        expected = expected.compose(YGate(), [1])
        expected = expected.compose(Delay(12), [1])

        expected.measure_all()

        self.assertEqual(ghz4_dd, expected)

    def test_insert_midmeas_hahn_alap(self):
        """Test a single X gate as Hahn echo can absorb in the downstream circuit."""
        dd_sequence = [XGate()]
        pm = PassManager(
            [ALAPSchedule(self.durations), DynamicalDecoupling(self.durations, dd_sequence)]
        )

        midmeas_dd = pm.run(self.midmeas)

        combined_u = UGate(0, pi / 2, -pi / 2)

        expected = QuantumCircuit(3, 1)
        expected.cx(0, 1)
        expected.delay(625, 0)
        expected.x(0)
        expected.delay(625, 0)
        expected.compose(combined_u, [0], inplace=True)
        expected.delay(700, 2)
        expected.cx(1, 2)
        expected.delay(1000, 1)
        expected.measure(2, 0)
        expected.cx(1, 2)
        expected.cx(0, 1)
        expected.delay(700, 2)
        expected.global_phase = 4.71238898038469

        self.assertEqual(midmeas_dd, expected)
        # check the absorption into U was done correctly
        self.assertEqual(Operator(combined_u), Operator(XGate()) & Operator(XGate()))

    def test_insert_midmeas_hahn_asap(self):
        """Test a single X gate as Hahn echo can absorb in the upstream circuit."""
        dd_sequence = [RXGate(pi / 4)]
        pm = PassManager(
            [ASAPSchedule(self.durations), DynamicalDecoupling(self.durations, dd_sequence)]
        )

        midmeas_dd = pm.run(self.midmeas)

        combined_u = UGate(3 * pi / 4, -pi / 2, pi / 2)

        expected = QuantumCircuit(3, 1)
        expected.cx(0, 1)
        expected.compose(combined_u, [0], inplace=True)
        expected.delay(600, 0)
        expected.rx(pi / 4, 0)
        expected.delay(600, 0)
        expected.delay(700, 2)
        expected.cx(1, 2)
        expected.delay(1000, 1)
        expected.measure(2, 0)
        expected.cx(1, 2)
        expected.cx(0, 1)
        expected.delay(700, 2)

        self.assertEqual(midmeas_dd, expected)
        # check the absorption into U was done correctly
        self.assertTrue(
            Operator(XGate()).equiv(
                Operator(UGate(3 * pi / 4, -pi / 2, pi / 2)) & Operator(RXGate(pi / 4))
            )
        )

    def test_insert_dd_bad_sequence(self):
        """Test DD raises when non-identity sequence is inserted."""
        dd_sequence = [XGate(), YGate()]
        pm = PassManager(
            [
                ALAPSchedule(self.durations),
                DynamicalDecoupling(self.durations, dd_sequence),
            ]
        )

        with self.assertRaises(TranspilerError):
            pm.run(self.ghz4)


if __name__ == "__main__":
    unittest.main()
