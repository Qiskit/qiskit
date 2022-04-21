# -*- coding: utf-8 -*

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

# pylint: disable=missing-docstring,invalid-name,no-member
# pylint: disable=attribute-defined-outside-init

from qiskit import schedule, QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.pulse import Schedule, Gaussian, DriveChannel, Play
from qiskit.providers.fake_provider import FakeOpenPulse2Q


def build_parametric_pulse_schedule(number_of_unique_pulses,
                                    number_of_channels):
    sched = Schedule()
    for _ in range(number_of_unique_pulses):
        for channel in range(number_of_channels):
            sched.append(
                Play(
                    Gaussian(duration=25, sigma=4, amp=0.5j),
                    DriveChannel(channel),
                ),
                inplace=True,
            )
    return sched


class ScheduleToInstructionBench:
    params = ([1, 2, 5], [8, 128, 2048])
    param_names = ['number_of_unique_pulses', 'number_of_channels']
    timeout = 600

    def setup(self, unique_pulses, channels):
        self.parametric_sched = build_parametric_pulse_schedule(unique_pulses,
                                                                channels)
        qr = QuantumRegister(1)
        self.qc = QuantumCircuit(qr)
        self.qc.append(Gate('my_pulse', 1, []), qargs=[qr[0]])
        self.backend = FakeOpenPulse2Q()
        self.inst_map = self.backend.defaults().instruction_schedule_map
        self.add_inst_map = self.inst_map
        self.add_inst_map.add('my_pulse', [0], self.parametric_sched)

    def time_build_instruction(self, _, __):
        self.inst_map.add('my_pulse', [0], self.parametric_sched)

    def time_instruction_to_schedule(self, _, __):
        schedule(self.qc, self.backend, inst_map=self.add_inst_map)
