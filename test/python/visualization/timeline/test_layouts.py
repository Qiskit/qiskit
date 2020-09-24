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

"""Tests for layouts of timeline drawer."""

import qiskit
from qiskit.test import QiskitTestCase
from qiskit.visualization.timeline import layouts


class TestBitArrange(QiskitTestCase):
    """Tests for layout.bit_arrange."""

    def setUp(self) -> None:
        """Setup."""
        super().setUp()

        qregs = qiskit.QuantumRegister(3)
        cregs = qiskit.ClassicalRegister(3)

        self.regs = list(qregs) + list(cregs)

    def test_qreg_creg_ascending(self):
        """Test qreg_creg_ascending layout function."""
        sorted_regs = layouts.qreg_creg_ascending(self.regs)
        ref_regs = [self.regs[0], self.regs[1], self.regs[2],
                    self.regs[3], self.regs[4], self.regs[5]]

        self.assertListEqual(sorted_regs, ref_regs)

    def test_qreg_creg_descending(self):
        """Test qreg_creg_descending layout function."""
        sorted_regs = layouts.qreg_creg_descending(self.regs)
        ref_regs = [self.regs[2], self.regs[1], self.regs[0],
                    self.regs[5], self.regs[4], self.regs[3]]

        self.assertListEqual(sorted_regs, ref_regs)


class TestAxisMap(QiskitTestCase):
    """Tests for layout.time_axis_map."""



