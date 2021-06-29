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

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import XGate
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.passes import ALAPSchedule, InsertDD
from qiskit.transpiler.passmanager import PassManager

from qiskit.test import QiskitTestCase


class TestInsertDD(QiskitTestCase):
    """Tests InsertDD pass."""

    def setUp(self):
        super().setUp()
        self.ghz4 = QuantumCircuit(4)
        self.ghz4.h(0)
        self.ghz4.cx(0, 1)
        self.ghz4.cx(1, 2)
        self.ghz4.cx(2, 3)
        self.ghz4.measure_all()

        self.durations = InstructionDurations(
            [("h", 0, 50), ("cx", [0, 1], 700),
             ("cx", [1, 2], 200), ("cx", [2, 3], 300),
             ("x", None, 50), ("measure", None, 1000)]
        )


    def test_insert_dd_ghz(self):
        """Test DD gates are inserted in correct spots."""
        dd_sequence = [XGate(), XGate()]
        pm = PassManager([ALAPSchedule(self.durations),
                          InsertDD(self.durations, dd_sequence)])

        ghz4_dd = pm.run(self.ghz4)

        expected = QuantumCircuit(4)
        expected.h(0)
        expected.delay(50, 1)
        expected.delay(750, 2)
        expected.delay(950, 3)

        expected.cx(0, 1)
        expected.cx(1, 2)
        expected.cx(2, 3)

        expected.delay(100, 0)
        expected.x(0)
        expected.delay(200, 0)
        expected.x(0)
        expected.delay(100, 0)

        expected.delay(50, 1)
        expected.x(1)
        expected.delay(100, 1)
        expected.x(1)
        expected.delay(50, 1)

        expected.measure_all()

        self.assertEqual(ghz4_dd, expected)


if __name__ == "__main__":
    unittest.main()
