# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for calibration entries."""

import numpy as np

from qiskit.circuit.parameter import Parameter
from qiskit.pulse import (
    Schedule,
    ScheduleBlock,
    Play,
    Constant,
    DriveChannel,
)
from qiskit.pulse.calibration_entries import (
    ScheduleDef,
    CallableDef,
)
from qiskit.pulse.exceptions import PulseError
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from qiskit.utils.deprecate_pulse import decorate_test_methods, ignore_pulse_deprecation_warnings


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestSchedule(QiskitTestCase):
    """Test case for the ScheduleDef."""

    def test_add_schedule(self):
        """Basic test pulse Schedule format."""
        program = Schedule()
        program.insert(
            0,
            Play(Constant(duration=10, amp=0.1, angle=0.0), DriveChannel(0)),
            inplace=True,
        )

        entry = ScheduleDef()
        entry.define(program)

        signature_to_test = list(entry.get_signature().parameters.keys())
        signature_ref = []
        self.assertListEqual(signature_to_test, signature_ref)

        schedule_to_test = entry.get_schedule()
        schedule_ref = program
        self.assertEqual(schedule_to_test, schedule_ref)

    def test_add_block(self):
        """Basic test pulse Schedule format."""
        program = ScheduleBlock()
        program.append(
            Play(Constant(duration=10, amp=0.1, angle=0.0), DriveChannel(0)),
            inplace=True,
        )

        entry = ScheduleDef()
        entry.define(program)

        signature_to_test = list(entry.get_signature().parameters.keys())
        signature_ref = []
        self.assertListEqual(signature_to_test, signature_ref)

        schedule_to_test = entry.get_schedule()
        schedule_ref = program
        self.assertEqual(schedule_to_test, schedule_ref)

    def test_parameterized_schedule(self):
        """Test adding and managing parameterized schedule."""
        param1 = Parameter("P1")
        param2 = Parameter("P2")

        program = ScheduleBlock()
        program.append(
            Play(Constant(duration=param1, amp=param2, angle=0.0), DriveChannel(0)),
            inplace=True,
        )

        entry = ScheduleDef()
        entry.define(program)

        signature_to_test = list(entry.get_signature().parameters.keys())
        signature_ref = ["P1", "P2"]
        self.assertListEqual(signature_to_test, signature_ref)

        schedule_to_test = entry.get_schedule(P1=10, P2=0.1)
        schedule_ref = program.assign_parameters({param1: 10, param2: 0.1}, inplace=False)
        self.assertEqual(schedule_to_test, schedule_ref)

    def test_parameterized_schedule_with_user_args(self):
        """Test adding schedule with user signature.

        Bind parameters to a pulse schedule but expecting non-lexicographical order.
        """
        theta = Parameter("theta")
        lam = Parameter("lam")
        phi = Parameter("phi")

        program = ScheduleBlock()
        program.append(
            Play(Constant(duration=10, amp=phi, angle=0.0), DriveChannel(0)),
            inplace=True,
        )
        program.append(
            Play(Constant(duration=10, amp=theta, angle=0.0), DriveChannel(0)),
            inplace=True,
        )
        program.append(
            Play(Constant(duration=10, amp=lam, angle=0.0), DriveChannel(0)),
            inplace=True,
        )

        entry = ScheduleDef(arguments=["theta", "lam", "phi"])
        entry.define(program)

        signature_to_test = list(entry.get_signature().parameters.keys())
        signature_ref = ["theta", "lam", "phi"]
        self.assertListEqual(signature_to_test, signature_ref)

        # Do not specify kwargs. This is order sensitive.
        schedule_to_test = entry.get_schedule(0.1, 0.2, 0.3)
        schedule_ref = program.assign_parameters(
            {theta: 0.1, lam: 0.2, phi: 0.3},
            inplace=False,
        )
        self.assertEqual(schedule_to_test, schedule_ref)

    def test_parameterized_schedule_with_wrong_signature(self):
        """Test raising PulseError when signature doesn't match."""
        param1 = Parameter("P1")

        program = ScheduleBlock()
        program.append(
            Play(Constant(duration=10, amp=param1, angle=0.0), DriveChannel(0)),
            inplace=True,
        )

        entry = ScheduleDef(arguments=["This_is_wrong_param_name"])

        with self.assertRaises(PulseError):
            entry.define(program)

    def test_equality(self):
        """Test equality evaluation between the schedule entries."""
        program1 = Schedule()
        program1.insert(
            0,
            Play(Constant(duration=10, amp=0.1, angle=0.0), DriveChannel(0)),
            inplace=True,
        )

        program2 = Schedule()
        program2.insert(
            0,
            Play(Constant(duration=10, amp=0.2, angle=0.0), DriveChannel(0)),
            inplace=True,
        )

        entry1 = ScheduleDef()
        entry1.define(program1)

        entry2 = ScheduleDef()
        entry2.define(program2)

        entry3 = ScheduleDef()
        entry3.define(program1)

        self.assertEqual(entry1, entry3)
        self.assertNotEqual(entry1, entry2)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestCallable(QiskitTestCase):
    """Test case for the CallableDef."""

    def test_add_callable(self):
        """Basic test callable format."""
        program = Schedule()
        program.insert(
            0,
            Play(Constant(duration=10, amp=0.1, angle=0.0), DriveChannel(0)),
            inplace=True,
        )

        def factory():
            return program

        entry = CallableDef()
        entry.define(factory)

        signature_to_test = list(entry.get_signature().parameters.keys())
        signature_ref = []
        self.assertListEqual(signature_to_test, signature_ref)

        schedule_to_test = entry.get_schedule()
        schedule_ref = program
        self.assertEqual(schedule_to_test, schedule_ref)

    def test_add_callable_with_argument(self):
        """Basic test callable format."""

        def factory(var1, var2):
            program = Schedule()
            if var1 > 0:
                program.insert(
                    0,
                    Play(Constant(duration=var2, amp=var1, angle=0.0), DriveChannel(0)),
                    inplace=True,
                )
            else:
                program.insert(
                    0,
                    Play(Constant(duration=var2, amp=np.abs(var1), angle=np.pi), DriveChannel(0)),
                    inplace=True,
                )
            return program

        entry = CallableDef()
        entry.define(factory)

        signature_to_test = list(entry.get_signature().parameters.keys())
        signature_ref = ["var1", "var2"]
        self.assertListEqual(signature_to_test, signature_ref)

        schedule_to_test = entry.get_schedule(0.1, 10)
        schedule_ref = Schedule()
        schedule_ref.insert(
            0,
            Play(Constant(duration=10, amp=0.1, angle=0.0), DriveChannel(0)),
            inplace=True,
        )
        self.assertEqual(schedule_to_test, schedule_ref)

        schedule_to_test = entry.get_schedule(-0.1, 10)
        schedule_ref = Schedule()
        schedule_ref.insert(
            0,
            Play(Constant(duration=10, amp=0.1, angle=np.pi), DriveChannel(0)),
            inplace=True,
        )
        self.assertEqual(schedule_to_test, schedule_ref)

    def test_equality(self):
        """Test equality evaluation between the callable entries.

        This does NOT compare the code. Just object equality.
        """

        def factory1():
            return Schedule()

        def factory2():
            return Schedule()

        entry1 = CallableDef()
        entry1.define(factory1)

        entry2 = CallableDef()
        entry2.define(factory2)

        entry3 = CallableDef()
        entry3.define(factory1)

        self.assertEqual(entry1, entry3)
        self.assertNotEqual(entry1, entry2)
