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

from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations

from qiskit.test.base import QiskitTestCase
from qiskit.test.mock.backends import FakeParis, FakeVigo


class TestInstructionDurationsClass(QiskitTestCase):
    """Test Test InstructionDurations class."""

    def test_empty(self):
        durations = InstructionDurations()
        self.assertEqual(durations.dt, None)
        with self.assertRaises(TranspilerError):
            durations.get('cx', [0, 1], 'dt')

    def test_fail_if_invalid_dict_is_supplied_when_construction(self):
        invalid_dic = [('cx', [0, 1])]  # no duration
        with self.assertRaises(TranspilerError):
            InstructionDurations(invalid_dic)

    def test_from_backend_for_backend_with_dt(self):
        durations = InstructionDurations.from_backend(FakeParis())
        self.assertGreater(durations.dt, 0)
        self.assertGreater(durations.get('u2', 0), 0)

    def test_from_backend_for_backend_without_dt(self):
        durations = InstructionDurations.from_backend(FakeVigo())
        self.assertIsNone(durations.dt)
        self.assertGreater(durations.get('u2', 0, 's'), 0)
        with self.assertRaises(TranspilerError):
            durations.get('u2', 0)
