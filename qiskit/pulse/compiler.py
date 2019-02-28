# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Helper module for simplified Qiskit usage.
"""

import warnings
import logging
import numpy as np

from qiskit.converters import schedules_to_qobj
from qiskit.pulse.schedule import PulseSchedule
from qiskit.qobj import RunConfig
from qiskit.qobj import QobjHeader
from qiskit.exceptions import QiskitError


logger = logging.getLogger(__name__)


# pylint: disable=redefined-builtin
def compile(schedules, backend, config=None, shots=1024, max_credits=10,
            seed=None, meas_level=1, memory_slot_size=100,
            meas_return="avg", rep_time=None, qobj_id=None):
    """Compile a list of pulses into a qobj.

    Args:
        schedules (PulseSchedule or list[PulseSchedule]): pulses to execute.
        backend (BaseBackend): a backend to execute the circuits on.
        config (dict): dictionary of parameters (e.g. noise) used by runner
        shots (int): number of repetitions of each circuit, for sampling
        max_credits (int): maximum credits to use
        seed (int): random seed for simulators
        meas_level (int): set the appropriate level of the measurement output.
        memory_slot_size (int): size of each memory slot if the output is Level 0.
        meas_return (str): indicates the level of measurement information to return.
            "single" returns information from every shot of the experiment.
            "avg" returns the average measurement output (averaged over the number of shots).
            If the meas level is 2 then this is fixed to single.
        rep_time (int): repetition time of the experiment in μs.
            The delay between experiments will be rep_time.
            Must be from the list provided by the device.
        qobj_id (int): identifier for the generated qobj

    Returns:
        Qobj: the qobj to be run on the backends.
    """

    if config:
        warnings.warn('The `config` argument is deprecated and '
                      'does not do anything', DeprecationWarning)

    backend_config = backend.configuration()
    backend_defaults = backend_config.defaults

    run_config = RunConfig()

    if seed:
        run_config.seed = seed
    if shots:
        run_config.shots = shots
    if max_credits:
        run_config.max_credits = max_credits

    run_config.meas_level = meas_level
    # merge all pulse commands in the library to reduce data size
    pulse_library = backend_defaults.get('pulse_library', [])
    for schedule in schedules:
        cmds = schedule.command_library()
        for cmd in cmds:
            _name = cmd.name
            _samples = list(map(lambda x: [np.real(x), np.imag(x)], cmd.samples))
            pulse = {'name': _name, 'samples': _samples}
            if pulse not in pulse_library:
                pulse_library.append(pulse)
    run_config.pulse_library = pulse_library
    run_config.memory_slot_size = memory_slot_size
    run_config.meas_return = meas_return
    run_config.qubit_lo_freq = backend_defaults['qubit_freq_est']
    run_config.meas_lo_freq = backend_defaults['meas_freq_est']
    if rep_time:
        # check if rep_time is supported by the backend
        if rep_time in backend_config.rep_times:
            run_config.rep_time = rep_time
        else:
            raise QiskitError('Invalid rep_time is specified.')
    else:
        run_config.rep_time = config.rep_times[0]

    qobj = schedules_to_qobj(schedules, user_qobj_header=QobjHeader(), run_config=run_config,
                             qobj_id=qobj_id)

    return qobj
    
def execute(schedules, backend, config=None, shots=1024, max_credits=10,
            seed=None, meas_level=1, memory_slot_size=100,
            meas_return="avg", rep_time=None, qobj_id=None):
    """Executes a set of pulses.

    Args:
        schedules (PulseSchedule or list[PulseSchedule]): pulses to execute.
        backend (BaseBackend): a backend to execute the circuits on.
        config (dict): dictionary of parameters (e.g. noise) used by runner
        shots (int): number of repetitions of each circuit, for sampling
        max_credits (int): maximum credits to use
        seed (int): random seed for simulators
        meas_level (int): set the appropriate level of the measurement output.
        memory_slot_size (int): size of each memory slot if the output is Level 0.
        meas_return (str): indicates the level of measurement information to return.
            "single" returns information from every shot of the experiment.
            "avg" returns the average measurement output (averaged over the number of shots).
            If the meas level is 2 then this is fixed to single.
        rep_time (int): repetition time of the experiment in μs.
            The delay between experiments will be rep_time.
            Must be from the list provided by the device.
        qobj_id (int): identifier for the generated qobj

    Returns:
        BaseJob: returns job instance derived from BaseJob
    """

    qobj = compile(schedules, backend,
                   config, shots, max_credits, seed,
                   meas_level, memory_slot_size, meas_return,
                   rep_time, qobj_id)

    return backend.run(qobj)
