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
    """
    user_qobj_header = user_qobj_header or QobjHeader()
    run_config = run_config or RunConfig()

    if isinstance(schedules, PulseSchedule):
        schedules = [schedules]

    userconfig = QobjConfig(**run_config.to_dict())
    experiments = []
    max_n_qubits = 0
    max_memory_slots = 0
    for schedule in schedules:
        # user defined qubit frequency
        if any(schedule.qubit_lo_freq):
            qubit_lo_freq = np.where(schedule.qubit_lo_freq,
                                     schedule.qubit_lo_freq,
                                     userconfig.qubit_lo_freq)
        else:
            qubit_lo_freq = []

        # user defined meas frequency
        if any(schedule.meas_lo_freq):
            meas_lo_freq = np.where(schedule.meas_lo_freq,
                                    schedule.meas_lo_freq,
                                    userconfig.meas_lo_freq)
        else:
            meas_lo_freq = []

        # generate experimental configuration
        if any(qubit_lo_freq) and any(meas_lo_freq):
            experimentconfig = QobjExperimentConfig(qubit_lo_freq=qubit_lo_freq,
                                                    meas_lo_freq=meas_lo_freq)
        elif any(qubit_lo_freq):
            experimentconfig = QobjExperimentConfig(qubit_lo_freq=qubit_lo_freq)
        elif any(meas_lo_freq):
            experimentconfig = QobjExperimentConfig(meas_lo_freq=meas_lo_freq)
        else:
            experimentconfig = QobjExperimentConfig()

        # generate experimental header
        experimentheader = QobjExperimentHeader(name=schedule.name)

        instructions = []
        for pulse in schedule.flat_pulse_sequence():
            current_instruction = QobjInstruction()


        experiments.append(QobjExperiment(instructions=instructions, header=experimentheader,
                                          config=experimentconfig))

    return Qobj(qobj_id=qobj_id or str(uuid.uuid4()), config=userconfig,
                experiments=experiments, header=user_qobj_header,
                type=QobjType.PULSE.value)
