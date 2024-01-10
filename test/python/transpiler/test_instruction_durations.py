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

# pylint: disable=missing-function-docstring

"""Test InstructionDurations class."""
from copy import deepcopy
from qiskit.circuit import Delay, Parameter
from qiskit.providers.fake_provider import FakeParis, FakePerth
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations

from qiskit.test.base import QiskitTestCase


class TestInstructionDurations(QiskitTestCase):
    """Test InstructionDurations class."""

    def test_empty(self):
        durations = InstructionDurations()
        self.assertEqual(durations.dt, None)
        with self.assertRaises(TranspilerError):
            durations.get("cx", [0, 1], "dt")

    def test_fail_if_invalid_dict_is_supplied_when_construction(self):
        invalid_dic = [("cx", [0, 1])]  # no duration
        with self.assertRaises(TranspilerError):
            InstructionDurations(invalid_dic)

    def test_update_with_parameters(self):
        durations = InstructionDurations(
            [("rzx", (0, 1), 150, (0.5,)), ("rzx", (0, 1), 300, (1.0,))]
        )
        self.assertEqual(durations.get("rzx", [0, 1], parameters=[0.5]), 150)
        self.assertEqual(durations.get("rzx", [0, 1], parameters=[1.0]), 300)

    def test_can_get_unbounded_duration_without_unit_conversion(self):
        param = Parameter("t")
        parameterized_delay = Delay(param, "dt")
        actual = InstructionDurations().get(parameterized_delay, 0)
        self.assertEqual(actual, param)

    def test_can_get_unbounded_duration_with_unit_conversion_when_dt_is_provided(self):
        param = Parameter("t")
        parameterized_delay = Delay(param, "s")
        actual = InstructionDurations(dt=100).get(parameterized_delay, 0)
        self.assertEqual(actual, param / 100)

    def test_fail_if_get_unbounded_duration_with_unit_conversion_when_dt_is_not_provided(self):
        param = Parameter("t")
        parameterized_delay = Delay(param, "s")
        with self.assertRaises(TranspilerError):
            InstructionDurations().get(parameterized_delay, 0)


class TestInstrctionDurationsFromBackendV1(QiskitTestCase):
    """Test :meth:`~.from_backend` of :class:`.InstructionDurations` with
    :class:`.BackendV1`"""

    def setUp(self):
        super().setUp()

        self.backend = FakeParis()
        self.backend_config = self.backend.configuration()
        self.backend_props = self.backend.properties()
        self.example_qubit = (0,)
        self.example_gate = "x"

        # Setting dt for the copy of backend to be None
        self.backend_cpy = deepcopy(self.backend)
        self.backend_cpy.configuration().dt = None

    def test_backend_dt_equals_inst_dur_dt(self):
        durations = InstructionDurations.from_backend(self.backend)
        self.assertEqual(durations.dt, self.backend_config.dt)

    def test_backend_gate_length_equals_inst_dur(self):
        durations = InstructionDurations.from_backend(self.backend)
        inst_dur_duration = durations.get(self.example_gate, self.example_qubit[0], "s")
        backend_inst_dur = self.backend_props.gate_length(
            gate=self.example_gate, qubits=self.example_qubit
        )
        self.assertEqual(inst_dur_duration, backend_inst_dur)

    def test_backend_without_dt_sets_inst_dur_None(self):
        durations = InstructionDurations.from_backend(self.backend_cpy)
        self.assertIsNone(durations.dt)

    def test_get_dur_s_with_dt_None(self):
        durations = InstructionDurations.from_backend(self.backend_cpy)
        self.assertEqual(
            durations.get(self.example_gate, self.example_qubit[0], "s"), 3.5555555555555554e-08
        )

    def test_raise_dur_get_dt_with_backend_dt_None(self):
        durations = InstructionDurations.from_backend(self.backend_cpy)
        with self.assertRaises(TranspilerError):
            durations.get(self.example_gate, self.example_qubit[0])

    def test_works_unequal_dt_dtm(self):
        self.backend_cpy.configuration().dt = 1.0

        # This is expcted to fail
        InstructionDurations.from_backend(self.backend_cpy)

        self.backend_cpy.configuration().dt = None  # Resetting to None
        # Check if dt and dtm were indeed unequal
        self.assertNotEqual(self.backend_cpy.configuration().dtm, 1.0)


class TestInstrctionDurationsFromBackendV2(QiskitTestCase):
    """Test :meth:`~.from_backend` of :class:`.InstructionDurations` with
    :class:`.BackendV2`"""

    def setUp(self):
        super().setUp()

        self.backend = FakePerth()
        self.example_gate = "x"
        self.example_qubit = (0,)

        # Setting dt for the copy  for BackendV2 to None
        self.backend_cpy = deepcopy(self.backend)
        self.backend_cpy.target.dt = None

    def test_backend_dt_equals_inst_dur_dt(self):
        durations = InstructionDurations.from_backend(self.backend)
        self.assertEqual(durations.dt, self.backend.dt)

    def test_backend_gate_length_equals_inst_dur(self):
        durations = InstructionDurations.from_backend(self.backend)
        inst_dur_duration = durations.get(
            inst=self.example_gate, qubits=self.example_qubit[0], unit="s"
        )
        backend_inst_dur = self.backend.target._gate_map[self.example_gate][
            self.example_qubit
        ].duration
        self.assertEqual(inst_dur_duration, backend_inst_dur)

    def test_backend_without_dt_sets_inst_dur_None(self):
        durations = InstructionDurations.from_backend(self.backend_cpy)
        self.assertIsNone(durations.dt)

    def test_get_dur_s_with_dt_None(self):
        durations = InstructionDurations.from_backend(self.backend_cpy)
        self.assertEqual(
            durations.get(self.example_gate, self.example_qubit[0], "s"), 3.5555555555555554e-08
        )

    def test_raise_dur_get_dt_with_backend_dt_None(self):
        durations = InstructionDurations.from_backend(self.backend_cpy)
        with self.assertRaises(TranspilerError):
            durations.get(self.example_gate, self.example_qubit[0])

    def test_works_unequal_dt_dtm(self):
        self.backend_cpy.target.dt = 1.0

        # This is expcted to fail
        InstructionDurations.from_backend(self.backend_cpy)

        self.backend_cpy.target.dt = None  # Resetting to None
        # Check if dt and dtm were indeed unequal
        self.assertNotEqual(self.backend_cpy.dtm, 1.0)
