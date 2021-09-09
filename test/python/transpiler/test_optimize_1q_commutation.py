# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the Optimize1qGatesSimpleCommutation pass"""

from collections import Counter

import unittest

import ddt
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes.optimization.optimize_1q_commutation import (
    Optimize1qGatesSimpleCommutation,
)
from qiskit.test import QiskitTestCase


@ddt.ddt
class TestOptimize1qSimpleCommutation(QiskitTestCase):
    """Test for 1q gate optimizations."""

    def test_successor_commutation(self):
        """
        Check that Optimize1qGatesSimpleCommutation correctly moves 1Q gates later.
        """
        qc = QuantumCircuit(2)
        qc.sx(1)
        qc.cx(0, 1)
        qc.p(-np.pi, 1)
        qc.sx(1)
        qc.p(np.pi, 1)

        optimize_pass = Optimize1qGatesSimpleCommutation(basis=["sx", "p"], run_to_completion=True)
        result = optimize_pass(qc)

        expected = QuantumCircuit(2, global_phase=np.pi / 2)
        expected.cx(0, 1)
        msg = f"expected:\n{expected}\nresult:\n{result}"
        self.assertEqual(expected, result, msg=msg)

    def test_predecessor_commutation(self):
        """
        Check that Optimize1qGatesSimpleCommutation correctly moves 1Q gates earlier.
        """
        qc = QuantumCircuit(2)
        qc.p(-np.pi, 1)
        qc.sx(1)
        qc.p(np.pi, 1)
        qc.cx(0, 1)
        qc.sx(1)

        optimize_pass = Optimize1qGatesSimpleCommutation(basis=["sx", "p"], run_to_completion=True)
        result = optimize_pass(qc)

        expected = QuantumCircuit(2, global_phase=np.pi / 2)
        expected.cx(0, 1)
        msg = f"expected:\n{expected}\nresult:\n{result}"
        self.assertEqual(expected, result, msg=msg)

    def test_elaborate_commutation(self):
        """
        Check that Optimize1qGatesSimpleCommutation can perform several steps without fumbling.
        """
        qc = QuantumCircuit(2)

        qc.p(np.pi / 8, 0)
        qc.sx(0)
        qc.p(np.pi / 7, 0)

        qc.p(np.pi / 4, 1)
        qc.sx(1)

        qc.cx(0, 1)

        qc.p(-np.pi, 1)
        qc.sx(1)
        qc.p(-np.pi, 1)

        qc.p(np.pi / 7, 0)
        qc.sx(0)
        qc.p(np.pi / 8, 0)

        optimize_pass = Optimize1qGatesSimpleCommutation(basis=["sx", "p"], run_to_completion=True)
        result = optimize_pass(qc)

        expected = QuantumCircuit(2, global_phase=np.pi / 2)
        expected.p(np.pi / 8, 0)
        expected.sx(0)

        expected.p(np.pi / 4, 1)

        expected.cx(0, 1)

        expected.p(2 * np.pi / 7, 0)
        expected.sx(0)
        expected.p(np.pi / 8, 0)

        msg = f"expected:\n{expected}\nresult:\n{result}"
        self.assertEqual(expected, result, msg=msg)

    def test_midcircuit_double_commutation(self):
        """
        Check that Optimize1qGatesSimpleCommutation can push gates forward and backward out of a run
        in the middle of a circuit.
        """
        qc = QuantumCircuit(2)

        qc.rz(2.15, 0)  # this block will get modified by resynthesis
        qc.sx(0)
        qc.rz(-2.75, 0)
        qc.sx(0)
        qc.rz(0.255, 0)

        qc.rz(0.138, 1)
        qc.sx(1)
        qc.rz(-2.87, 1)
        qc.sx(1)
        qc.rz(-2.1, 1)

        qc.cx(1, 0)

        qc.sx(0)  # this will get moved
        qc.rz(1.03, 0)
        qc.sx(0)  # this will get moved

        qc.sx(1)
        qc.rz(1.45, 1)
        qc.sx(1)
        qc.rz(1.33, 1)  # this will get moved

        qc.cx(1, 0)

        qc.rz(2.01, 0)  # this block will get modified by resynthesis
        qc.sx(0)
        qc.rz(-1.62, 0)
        qc.sx(0)
        qc.rz(-1.16, 0)

        qc.rz(-0.732, 1)  # this one gate will get modified by resynthesis
        qc.sx(1)
        qc.rz(-2.65, 1)
        qc.sx(1)
        qc.rz(2.17, 1)

        optimize_pass = Optimize1qGatesSimpleCommutation(basis=["sx", "rz"], run_to_completion=True)
        result = optimize_pass(qc)
        runs = circuit_to_dag(result).collect_1q_runs()
        oneq_counts = Counter([len(run) for run in runs])

        self.assertEqual(oneq_counts, Counter([5, 5, 3, 1, 5, 5]))


if __name__ == "__main__":
    unittest.main()
