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

from qiskit.circuit import Delay, Parameter
from qiskit.test.mock.backends import FakeParis, FakeTokyo
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations

from qiskit.test.base import QiskitTestCase


class TestInstructionDurationsClass(QiskitTestCase):
    """Test Test InstructionDurations class."""

    def test_empty(self):
        durations = InstructionDurations()
        self.assertEqual(durations.dt, None)
        with self.assertRaises(TranspilerError):
            durations.get("cx", [0, 1], "dt")

    def test_fail_if_invalid_dict_is_supplied_when_construction(self):
        invalid_dic = [("cx", [0, 1])]  # no duration
        with self.assertRaises(TranspilerError):
            InstructionDurations(invalid_dic)

    def test_from_backend_for_backend_with_dt(self):
        backend = FakeParis()
        gate = self._find_gate_with_length(backend)
        durations = InstructionDurations.from_backend(backend)
        self.assertGreater(durations.dt, 0)
        self.assertGreater(durations.get(gate, 0), 0)

    def test_from_backend_for_backend_without_dt(self):
        backend = FakeTokyo()
        gate = self._find_gate_with_length(backend)
        durations = InstructionDurations.from_backend(backend)
        self.assertIsNone(durations.dt)
        self.assertGreater(durations.get(gate, 0, "s"), 0)
        with self.assertRaises(TranspilerError):
            durations.get(gate, 0)

    def test_durations_by_name_param_qubits(self):
        dt = 2.222222e-10
        chunk_size = 16
        durations = InstructionDurations(
            [
                # duration_by_name
                ('rz', None, None, 0, 'dt'),
                ('sx', None, None, 6, 'dt'),

                # duration_by_name_params
                ('rx', pi, None, 12, 'dt'),
                ('rx', pi/4, None, 3, 'dt'),

                # duration_by_name_qubits
                ('y', None, 0, 8, 'dt'),
                ('y', None, 1, 4, 'dt'),

                # duration_by_name_params_qubits
                ('cx', None, (0, 1), 101, 'dt'),
                ('cx', None, (2, 1), 70, 'dt'),
                ('rzx', pi/2, (0, 1), 101, 'dt'),
                ('rzx', pi/2, (2, 1), 70, 'dt'),
                ('rzx', pi/4, (0, 1), 52, 'dt'),
                ('rzx', pi/4, (2, 1), 37, 'dt'),
                ('rzx', pi/6, (0, 1), 37, 'dt'),
                ('rzx', pi/6, (2, 1), 24, 'dt')
            ],
            dt=dt
        )
        self.assertEqual(durations.get('sx', unit='dt'), 6)

    def _find_gate_with_length(self, backend):
        """Find a gate that has gate length."""
        props = backend.properties()
        for gate in props.gates:
            try:
                if props.gate_length(gate.gate, 0):
                    return gate.gate
            except Exception:  # pylint: disable=broad-except
                pass
        raise ValueError("Unable to find a gate with gate length.")

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
