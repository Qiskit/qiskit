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

"""Assemble function for converting a list of circuits into a qobj"""
import logging

from qiskit.exceptions import QiskitError
from qiskit.pulse.commands import PulseInstruction
from qiskit.qobj import (PulseQobj, QobjExperimentHeader, QasmQobjConfig,
                         PulseQobjInstruction, PulseQobjExperimentConfig,
                         PulseQobjExperiment, PulseQobjConfig, QobjPulseLibrary)
from qiskit.qobj.converters import PulseQobjConverter, LoConfigConverter

logger = logging.getLogger(__name__)


def assemble_schedules(schedules, qobj_id=None, qobj_header=None, run_config=None):
    """Assembles a list of schedules into a qobj which can be run on the backend.
    Args:
        schedules (list[Schedule]): schedules to assemble
        qobj_id (int): identifier for the generated qobj
        qobj_header (QobjHeader): header to pass to the results
        run_config (RunConfig): configuration of the runtime environment
    Returns:
        PulseQobj: the Qobj to be run on the backends
    Raises:
        QiskitError: when invalid schedules or configs are provided
    """
    qobj_config = QasmQobjConfig()
    if run_config:
        qobj_config = QasmQobjConfig(**run_config.to_dict())

    # Get appropriate convertors
    instruction_converter = PulseQobjConverter
    instruction_converter = instruction_converter(PulseQobjInstruction, **run_config.to_dict())
    lo_converter = LoConfigConverter(PulseQobjExperimentConfig, run_config.qubit_lo_freq,
                                     run_config.meas_lo_freq, **run_config.to_dict())

    # Pack everything into the Qobj
    qobj_schedules = []
    user_pulselib = set()
    for idx, schedule in enumerate(schedules):
        # instructions
        qobj_instructions = []
        # Instructions are returned as tuple of shifted time and instruction
        for shift, instruction in schedule.instructions:
            # TODO: support conditional gate
            qobj_instructions.append(instruction_converter(shift, instruction))
            if isinstance(instruction, PulseInstruction):
                # add samples to pulse library
                user_pulselib.add(instruction.command)
        # experiment header
        qobj_experiment_header = QobjExperimentHeader(
            name=schedule.name or 'Experiment-%d' % idx
        )

        qobj_schedules.append({
            'header': qobj_experiment_header,
            'instructions': qobj_instructions
        })

    # setup pulse_library
    run_config.pulse_library = [QobjPulseLibrary(name=pulse.name, samples=pulse.samples)
                                for pulse in user_pulselib]

    # create qob experiment field
    experiments = []
    if len(run_config.schedule_los) == 1:
        lo_dict = run_config.schedule_los.pop()
        # update global config
        q_los = lo_converter.get_qubit_los(lo_dict)
        if q_los:
            run_config.qubit_lo_freq = q_los
        m_los = lo_converter.get_meas_los(lo_dict)
        if m_los:
            run_config.meas_lo_freq = m_los

    if run_config.schedule_los:
        # multiple frequency setups
        if len(qobj_schedules) == 1:
            # frequency sweep
            for lo_dict in run_config.schedule_los:
                experiments.append(PulseQobjExperiment(
                    instructions=qobj_schedules[0]['instructions'],
                    experimentheader=qobj_schedules[0]['header'],
                    experimentconfig=lo_converter(lo_dict)
                ))
        elif len(qobj_schedules) == len(run_config.schedule_los):
            # n:n setup
            for lo_dict, schedule in zip(run_config.schedule_los, qobj_schedules):
                experiments.append(PulseQobjExperiment(
                    instructions=schedule['instructions'],
                    experimentheader=schedule['header'],
                    experimentconfig=lo_converter(lo_dict)
                ))
        else:
            raise QiskitError('Invalid LO setting is specified. '
                              'The LO should be configured for each schedule, or '
                              'single setup for all schedules (unique), or '
                              'multiple setups for a single schedule (frequency sweep),'
                              'or no LO configured at all.')
    else:
        # unique frequency setup
        for schedule in qobj_schedules:
            experiments.append(PulseQobjExperiment(
                instructions=schedule['instructions'],
                experimentheader=schedule['header'],
            ))

    qobj_config = PulseQobjConfig(**run_config.to_dict())

    return PulseQobj(qobj_id=qobj_id,
                     config=qobj_config,
                     experiments=experiments,
                     header=qobj_header)
