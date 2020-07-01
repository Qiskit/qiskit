# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-function-docstring

"""Test DelayInDt pass."""

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes.scheduling.delayindt import DelayInDt, delay_in_dt
from qiskit.test import QiskitTestCase


class TestDelayInDtPass(QiskitTestCase):
    """ Tests for DelayInDt pass. """

    def test_single_delay(self):
        qc = QuantumCircuit(1)
        qc.delay(500, 0, unit='ns')
        actual = DelayInDt(1e-10).run(circuit_to_dag(qc))
        qc = QuantumCircuit(1)
        qc.delay(5000, 0)
        expected = circuit_to_dag(qc)
        self.assertEqual(actual, expected)

    def test_rounding(self):
        qc = QuantumCircuit(1)
        qc.delay(50, 0, unit='s')
        with self.assertWarns(UserWarning):
            actual = DelayInDt(0.333).run(circuit_to_dag(qc))
        qc = QuantumCircuit(1)
        qc.delay(150, 0)  # 50 / 0.333 = 150.15015
        expected = circuit_to_dag(qc)
        self.assertEqual(actual, expected)

    def test_do_nothing_for_circuit_with_delay_in_dt(self):
        qc = QuantumCircuit(1)
        qc.delay(500, 0)
        actual = DelayInDt(0.77).run(circuit_to_dag(qc))
        qc = QuantumCircuit(1)
        qc.delay(500, 0)
        expected = circuit_to_dag(qc)
        self.assertEqual(actual, expected)

    def test_do_nothing_for_circuit_without_delays(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        actual = DelayInDt(0.77).run(circuit_to_dag(qc))
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        expected = circuit_to_dag(qc)
        self.assertEqual(actual, expected)

    def test_delay_in_dt(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.delay(500, 0, unit='ns')
        qc.h(0)
        actual = delay_in_dt(qc, dt_in_sec=1e-10)
        expected = QuantumCircuit(1)
        expected.h(0)
        expected.delay(5000, 0)
        expected.h(0)
        self.assertEqual(actual, expected)
