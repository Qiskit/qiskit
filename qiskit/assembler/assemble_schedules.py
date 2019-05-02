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
from qiskit.pulse.commands import PulseInstruction, AcquireInstruction
from qiskit.qobj import (PulseQobj, QobjExperimentHeader,
                         PulseQobjInstruction, PulseQobjExperimentConfig,
                         PulseQobjExperiment, PulseQobjConfig, PulseLibraryItem)
from qiskit.qobj.converters import InstructionToQobjConverter, LoConfigConverter

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
    if hasattr(run_config, 'instruction_converter'):
        instruction_converter = run_config.instruction_converter
    else:
        instruction_converter = InstructionToQobjConverter

    qobj_config = run_config.to_dict()
    qubit_lo_range = qobj_config.pop('qubit_lo_range')
    meas_lo_range = qobj_config.pop('meas_lo_range')
    meas_map = qobj_config.pop('meas_map', None)
    instruction_converter = instruction_converter(PulseQobjInstruction, **qobj_config)

    lo_converter = LoConfigConverter(PulseQobjExperimentConfig, qubit_lo_range=qubit_lo_range,
                                     meas_lo_range=meas_lo_range, **qobj_config)

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
            if isinstance(instruction, AcquireInstruction):
                if meas_map:
                    # verify all acquires satisfy meas_map
                    _validate_meas_map(instruction, meas_map)

        # experiment header
        qobj_experiment_header = QobjExperimentHeader(
            name=schedule.name or 'Experiment-%d' % idx
        )

        qobj_schedules.append({
            'header': qobj_experiment_header,
            'instructions': qobj_instructions
        })

    # setup pulse_library
    qobj_config['pulse_library'] = [PulseLibraryItem(name=pulse.name, samples=pulse.samples)
                                    for pulse in user_pulselib]

    # create qob experiment field
    experiments = []
    schedule_los = qobj_config.pop('schedule_los', [])

    if len(schedule_los) == 1:
        lo_dict = schedule_los[0]
        # update global config
        q_los = lo_converter.get_qubit_los(lo_dict)
        if q_los:
            qobj_config['qubit_lo_freq'] = q_los
        m_los = lo_converter.get_meas_los(lo_dict)
        if m_los:
            qobj_config['meas_lo_freq'] = m_los

    if schedule_los:
        # multiple frequency setups
        if len(qobj_schedules) == 1:
            # frequency sweep
            for lo_dict in schedule_los:
                experiments.append(PulseQobjExperiment(
                    instructions=qobj_schedules[0]['instructions'],
                    header=qobj_schedules[0]['header'],
                    config=lo_converter(lo_dict)
                ))
        elif len(qobj_schedules) == len(schedule_los):
            # n:n setup
            for lo_dict, schedule in zip(schedule_los, qobj_schedules):
                experiments.append(PulseQobjExperiment(
                    instructions=schedule['instructions'],
                    header=schedule['header'],
                    config=lo_converter(lo_dict)
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
                header=schedule['header'],
            ))

    qobj_config = PulseQobjConfig(**qobj_config)

    return PulseQobj(qobj_id=qobj_id,
                     config=qobj_config,
                     experiments=experiments,
                     header=qobj_header)


def _validate_meas_map(acquire, meas_map):
    """Validate all qubits tied in meas_map are to be acquired."""
    meas_map_set = [set(m) for m in meas_map]
    # Verify that each qubit is listed once in measurement map
    measured_qubits = set(acq_ch.index for acq_ch in acquire.acquires)
    tied_qubits = set()
    for meas_qubit in measured_qubits:
        for map_inst in meas_map_set:
            if meas_qubit in map_inst:
                tied_qubits |= map_inst

    if measured_qubits != tied_qubits:
        raise QiskitError('Qubits to be acquired: {0} do not satisfy required qubits '
                          'in measurement map: {1}'.format(measured_qubits, tied_qubits))
    return True
