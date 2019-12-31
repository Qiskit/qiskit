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
"""Quick program to test the quantum operators  modules."""

import unittest
import numpy as np

from qiskit.quantum_info import Operator, Choi
from qiskit.quantum_info import process_fidelity
from qiskit.quantum_info import average_gate_fidelity
from qiskit.quantum_info import gate_error
from qiskit.test import QiskitTestCase


class TestOperators(QiskitTestCase):
    """Tests for qi.py"""
    def test_operator_process_fidelity(self):
        """Test the process_fidelity function for operator inputs"""
        # Orthogonal operator
        op = Operator.from_label('X')
        f_pro = process_fidelity(op, require_cp=True, require_tp=True)
        self.assertAlmostEqual(f_pro, 0.0, places=7)

        # Global phase operator
        op1 = Operator.from_label('X')
        op2 = -1j * op1
        f_pro = process_fidelity(op1, op2, require_cp=True, require_tp=True)
        self.assertAlmostEqual(f_pro, 1.0, places=7)

    def test_channel_process_fidelity(self):
        """Test the process_fidelity function for channel inputs"""
        depol = Choi(np.eye(4) / 2)
        iden = Choi(Operator.from_label('I'))

        # Completely depolarizing channel
        f_pro = process_fidelity(depol, require_cp=True, require_tp=True)
        self.assertAlmostEqual(f_pro, 0.25, places=7)

        # Identity
        f_pro = process_fidelity(iden, require_cp=True, require_tp=True)
        self.assertAlmostEqual(f_pro, 1.0, places=7)

        # Depolarizing channel
        p = 0.3
        chan = p * depol + (1 - p) * iden
        f_pro = process_fidelity(chan, require_cp=True, require_tp=True)
        f_target = p * 0.25 + (1 - p)
        self.assertAlmostEqual(f_pro, f_target, places=7)

        # Depolarizing channel
        p = 0.5
        op = Operator.from_label('Y')
        chan = (p * depol + (1 - p) * iden) @ op
        f_pro = process_fidelity(chan, op, require_cp=True, require_tp=True)
        target = p * 0.25 + (1 - p)
        self.assertAlmostEqual(f_pro, target, places=7)

    def test_operator_average_gate_fidelity(self):
        """Test the average_gate_fidelity function for operator inputs"""
        # Orthogonal operator
        op = Operator.from_label('Z')
        f_ave = average_gate_fidelity(op, require_cp=True, require_tp=True)
        self.assertAlmostEqual(f_ave, 1 / 3, places=7)

        # Global phase operator
        op1 = Operator.from_label('Y')
        op2 = -1j * op1
        f_ave = average_gate_fidelity(op1,
                                      op2,
                                      require_cp=True,
                                      require_tp=True)
        self.assertAlmostEqual(f_ave, 1.0, places=7)

    def test_channel_average_gate_fidelity(self):
        """Test the average_gate_fidelity function for channel inputs"""
        depol = Choi(np.eye(4) / 2)
        iden = Choi(Operator.from_label('I'))

        # Completely depolarizing channel
        f_ave = average_gate_fidelity(depol, require_cp=True, require_tp=True)
        self.assertAlmostEqual(f_ave, 0.5, places=7)

        # Identity
        f_ave = average_gate_fidelity(iden, require_cp=True, require_tp=True)
        self.assertAlmostEqual(f_ave, 1.0, places=7)

        # Depolarizing channel
        p = 0.11
        chan = p * depol + (1 - p) * iden
        f_ave = average_gate_fidelity(chan, require_cp=True, require_tp=True)
        f_target = (2 * (p * 0.25 + (1 - p)) + 1) / 3
        self.assertAlmostEqual(f_ave, f_target, places=7)

        # Depolarizing channel
        p = 0.5
        op = Operator.from_label('Y')
        chan = (p * depol + (1 - p) * iden) @ op
        f_ave = average_gate_fidelity(chan,
                                      op,
                                      require_cp=True,
                                      require_tp=True)
        target = (2 * (p * 0.25 + (1 - p)) + 1) / 3
        self.assertAlmostEqual(f_ave, target, places=7)

    def test_operator_gate_error(self):
        """Test the gate_error function for operator inputs"""
        # Orthogonal operator
        op = Operator.from_label('Z')
        err = gate_error(op, require_cp=True, require_tp=True)
        self.assertAlmostEqual(err, 2 / 3, places=7)

        # Global phase operator
        op1 = Operator.from_label('Y')
        op2 = -1j * op1
        err = gate_error(op1, op2, require_cp=True, require_tp=True)
        self.assertAlmostEqual(err, 0, places=7)

    def test_channel_gate_error(self):
        """Test the gate_error function for channel inputs"""
        depol = Choi(np.eye(4) / 2)
        iden = Choi(Operator.from_label('I'))

        # Depolarizing channel
        p = 0.11
        chan = p * depol + (1 - p) * iden
        err = gate_error(chan, require_cp=True, require_tp=True)
        target = 1 - average_gate_fidelity(chan)
        self.assertAlmostEqual(err, target, places=7)

        # Depolarizing channel
        p = 0.5
        op = Operator.from_label('Y')
        chan = (p * depol + (1 - p) * iden) @ op
        err = gate_error(chan, op, require_cp=True, require_tp=True)
        target = 1 - average_gate_fidelity(chan, op)
        self.assertAlmostEqual(err, target, places=7)


if __name__ == '__main__':
    unittest.main()
