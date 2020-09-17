# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test cases for parameters used in Schedules."""
from qiskit.test import QiskitTestCase

from qiskit import pulse, assemble
from qiskit.circuit import Parameter
from qiskit.pulse.channels import DriveChannel, AcquireChannel, MemorySlot
from qiskit.test.mock import FakeAlmaden


GHz = 1e9
"Conversion factor."


# TODO: remove assemble, too slow

class TestPulseParameters(QiskitTestCase):
    """Tests usage of Parameters in qiskit.pulse; specifically in Schedules,
    Instructions, and Pulses.
    """

    def setUp(self):
        """Just some useful, reusable Parameters and constants."""
        super().setUp()
        self.alpha = Parameter('⍺')
        self.beta = Parameter('beta')
        self.gamma = Parameter('γ')
        self.phi = Parameter('ϕ')
        self.theta = Parameter('ϑ')
        self.amp = Parameter('amp')
        self.sigma = Parameter('sigma')
        self.qubit = Parameter('q')

        self.FREQ = 4.5e9
        self.SHIFT = 0.2e9
        self.PHASE = 3.1415 / 4

        self.backend = FakeAlmaden()

    def test_straight_schedule_bind(self):
        """"""
        schedule = pulse.Schedule()
        schedule += pulse.SetFrequency(self.alpha, DriveChannel(0))
        schedule += pulse.ShiftFrequency(self.gamma, DriveChannel(0))
        schedule += pulse.SetPhase(self.phi, DriveChannel(1))
        schedule += pulse.ShiftPhase(self.theta, DriveChannel(1))

        schedule.assign_parameters({self.alpha: self.FREQ, self.gamma: self.SHIFT,
                                    self.phi: self.PHASE, self.theta: -self.PHASE})

        insts = assemble(schedule, self.backend).experiments[0].instructions
        self.assertEqual(float(insts[0].frequency*GHz), self.FREQ)
        self.assertEqual(float(insts[1].frequency*GHz), self.SHIFT)
        self.assertEqual(float(insts[2].phase), self.PHASE)
        self.assertEqual(float(insts[3].phase), -self.PHASE)

    def test_multiple_parameters(self):
        """
        """
        schedule = pulse.Schedule()
        schedule += pulse.SetFrequency(self.alpha + self.beta, DriveChannel(0))
        schedule += pulse.ShiftFrequency(self.gamma + self.beta, DriveChannel(0))
        schedule += pulse.SetPhase(self.phi, DriveChannel(1))

        # Partial bind
        DELTA = 1e9
        schedule.assign_parameters({self.alpha: self.FREQ - DELTA})
        schedule.assign_parameters({self.beta: DELTA})
        schedule.assign_parameters({self.gamma: self.SHIFT - DELTA})
        schedule.assign_parameters({self.phi: self.PHASE})

        insts = assemble(schedule, self.backend).experiments[0].instructions
        self.assertEqual(float(insts[0].frequency*GHz), self.FREQ)
        self.assertEqual(float(insts[1].frequency*GHz), self.SHIFT)
        self.assertEqual(float(insts[2].phase), self.PHASE)

    def test_with_function(self):
        """
        """
        def get_frequency(variable):
            return 2*variable

        def get_shift(variable):
            return variable - 1

        schedule = pulse.Schedule()
        schedule += pulse.SetFrequency(get_frequency(self.alpha), DriveChannel(0))
        schedule += pulse.ShiftFrequency(get_shift(self.gamma), DriveChannel(0))

        schedule.assign_parameters({self.alpha: self.FREQ / 2, self.gamma: self.SHIFT + 1})

        insts = assemble(schedule, self.backend).experiments[0].instructions
        self.assertEqual(float(insts[0].frequency*GHz), self.FREQ)
        self.assertEqual(float(insts[1].frequency*GHz), self.SHIFT)

    def test_substitution(self):
        """
        """
        schedule = pulse.Schedule()
        schedule += pulse.SetFrequency(self.alpha, DriveChannel(0))

        schedule.assign_parameters({self.alpha: 2*self.beta})
        schedule.assign_parameters({self.beta: self.FREQ / 2})

        insts = assemble(schedule, self.backend).experiments[0].instructions
        self.assertEqual(float(insts[0].frequency*GHz), self.FREQ)

    def test_channels(self):
        """
        """
        schedule = pulse.Schedule()
        schedule += pulse.ShiftPhase(self.PHASE, DriveChannel(2*self.qubit))

        schedule.assign_parameters({self.qubit: 4})

        insts = assemble(schedule, self.backend).experiments[0].instructions
        self.assertEqual(insts[0].ch, 'd8')

    def test_acquire_channels(self):
        """
        """
        schedule = pulse.Schedule()
        schedule += pulse.Acquire(16000, AcquireChannel(self.qubit), MemorySlot(self.qubit))
        schedule.assign_parameters({self.qubit: 1})

        insts = assemble(schedule, self.backend, meas_map=[[1]]).experiments[0].instructions
        self.assertEqual(int(insts[0].qubits[0]), 1)

    def test_with_pulses(self):
        """
        """
        waveform = pulse.library.Gaussian(duration=128, sigma=self.sigma, amp=self.amp)

        schedule = pulse.Schedule()
        schedule += pulse.Play(waveform, DriveChannel(10))
        schedule.assign_parameters({self.amp: 0.2, self.sigma: 4})

        self.backend.configuration().parametric_pulses = ['gaussian', 'drag']
        insts = assemble(schedule, self.backend).experiments[0].instructions
        self.assertEqual(complex(insts[0].parameters['amp']), 0.2)
        self.assertEqual(float(insts[0].parameters['sigma']), 4.)

        waveform = pulse.library.GaussianSquare(duration=1280, sigma=self.sigma, amp=self.amp, width=1000)
        waveform = pulse.library.Constant(duration=1280, amp=self.amp)
        waveform = pulse.library.Drag(duration=1280, sigma=self.sigma, amp=self.amp, beta=2)

        schedule = pulse.Schedule()
        schedule += pulse.Play(waveform, DriveChannel(0))
        schedule.assign_parameters({self.amp: 0.2, self.sigma: 4})
