# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Compile function for converting a list of schedules to the qobj.
"""
import uuid
import warnings
import numpy as np

from qiskit.pulse.schedule import PulseSchedule
from qiskit.qobj import Qobj, QobjConfig, QobjExperiment, QobjInstruction, QobjHeader
from qiskit.qobj import QobjExperimentConfig, QobjExperimentHeader, QobjConditional
from qiskit.qobj.run_config import RunConfig
from qiskit.qobj._utils import QobjType
from qiskit.pulse.commands import *
from qiskit.exceptions import QiskitError


def schedules_to_qobj(schedules, user_qobj_header=None, run_config=None,
                     qobj_id=None):
    """Convert a list of schedules into a qobj.

    Args:
        circuits (list[PulseSchedule] or PulseSchedule): Schedules to compile.
        user_qobj_header (QobjHeader): Header to pass to the results.
        run_config (RunConfig): RunConfig object.
        qobj_id (int): Identifier for the generated qobj.

    Returns:
        Qobj: the Qobj to be run on the backends.
    Raises:
        QiskitError: when invalid command is given.
    """
    user_qobj_header = user_qobj_header or QobjHeader()
    run_config = run_config or RunConfig()

    if isinstance(schedules, PulseSchedule):
        schedules = [schedules]

    userconfig = QobjConfig(**run_config.to_dict())
    experiments = []
    for schedule in schedules:
        # user defined lo frequency
        user_lo_freqs = {}
        if any(schedule.qubit_lo_freq):
            user_lo_freqs['qubit_lo_freq'] = np.where(schedule.qubit_lo_freq,
                                                      schedule.qubit_lo_freq,
                                                      userconfig.qubit_lo_freq)
        if any(schedule.meas_lo_freq):
            user_lo_freqs['meas_lo_freq'] = np.where(schedule.meas_lo_freq,
                                                     schedule.meas_lo_freq,
                                                     userconfig.meas_lo_freq)

        # generate experimental configuration
        experimentconfig = QobjExperimentConfig(**user_lo_freqs)

        # generate experimental header
        experimentheader = QobjExperimentHeader(name=schedule.name)

        #TODO: support conditional gate
        instructions = []
        for pulse in schedule.flat_pulse_sequence():
            current_instruction = QobjInstruction(name=pulse.command.name,
                                                  t0=pulse.start_time())
            if isinstance(pulse.command, (SamplePulse, FunctionalPulse)):
                current_instruction.ch = pulse.channel.name()
            elif isinstance(pulse.command, FrameChange):
                current_instruction.ch = pulse.channel.name()
                current_instruction.phase = pulse.command.phase
            elif isinstance(pulse.command, PersistentValue):
                current_instruction.ch = pulse.channel.name()
                current_instruction.value = pulse.command.value
            elif isinstance(pulse.command, Acquire):
                current_instruction.duration = pulse.command.duration
                #TODO: qubit-register mapping
                current_instruction.qubits = []
                current_instruction.memory_slot = []
                current_instruction.register_slot = []
                #TODO: backend defaults
                current_instruction.kernels = pulse.command.kernel or None
                current_instruction.discriminators = pulse.command.discriminator or None
            elif isinstance(pulse.command, Snapshot):
                current_instruction.label = pulse.command.label
                current_instruction.type = pulse.command.type
            else:
                raise QiskitError('Invalid command is given, %s' % pulse.command.name)
            instructions.append(current_instruction)

        experiments.append(QobjExperiment(instructions=instructions, header=experimentheader,
                                          config=experimentconfig))

    return Qobj(qobj_id=qobj_id or str(uuid.uuid4()), config=userconfig,
                experiments=experiments, header=user_qobj_header,
                type=QobjType.PULSE.value)
