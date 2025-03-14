# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the optimize-1q-gate pass"""

import itertools
import unittest
import numpy as np
from ddt import ddt, idata, unpack

from qiskit import QuantumCircuit
from qiskit.circuit import Delay
from qiskit.circuit.library import XGate
from qiskit.transpiler import InstructionDurations, InstructionProperties, PassManager, Target
from qiskit.transpiler.passes import TimeUnitConversion
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.target import Target
from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
class TestTimeUnitConversion(QiskitTestCase):
    """Test for time unit conversion pass."""

    # test w inst_durations, target, none
    # test all si pass or convert
    # test all dt pass
    # test mixed convert
    # test mixed error

    def setUp(self):
        super().setUp()

        self.dt = 1e-9
        self.x_dt = 10
        self.delay_dt = 5

        self.target = {}
        self.target["dt"] = Target(dt=self.dt)
        self.target["dt"].add_instruction(
            XGate(),
            properties={(0,): InstructionProperties(duration=self.x_dt * self.dt)},
        )
        self.target["nodt"] = Target()
        self.target["nodt"].add_instruction(XGate())
        self.target["none"] = None

        self.durations = {
            "si": InstructionDurations([("x", None, self.x_dt * self.dt, None, "s")]),
            "dt": InstructionDurations([("x", None, self.x_dt, None, "dt")]),
            "none": None,
        }

    @idata(itertools.product(("s", "dt"), ("dt", "si", "none"), ("dt", "nodt", "none")))
    @unpack
    def test_gate_and_delay(self, delay_unit, inst_durations_unit, target_type):
        """Test delays are converted, passed through or flagged as appropriate"""
        if delay_unit == "s":
            delay_val = self.delay_dt * self.dt
        else:
            delay_val = self.delay_dt

        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.delay(delay_val, 0, unit=delay_unit)
        dag = circuit_to_dag(circuit)

        pass_ = TimeUnitConversion(self.durations[inst_durations_unit], self.target[target_type])

        if target_type == "none" and (delay_unit, inst_durations_unit) in (
            ("s", "dt"),
            ("dt", "si"),
        ):
            # Error case when delay and instruction units do not match
            with self.assertRaises(TranspilerError):
                after = pass_.run(dag)
        elif delay_unit == "s" and (
            (inst_durations_unit != "dt" and target_type == "none") or target_type == "nodt"
        ):
            # SI units on delay not overridden by instruction durations
            # Note that a target without a dt overrides passed instruction durations
            after = pass_.run(dag)
            node = after.op_nodes(op=Delay).pop()
            self.assertEqual(node.op.params[0], self.delay_dt * self.dt)
        else:
            # Delay passed as dt or converted using target dt
            after = pass_.run(dag)
            node = after.op_nodes(op=Delay).pop()
            self.assertEqual(node.op.params[0], self.delay_dt)
