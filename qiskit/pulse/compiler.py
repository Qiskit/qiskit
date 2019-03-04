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
from qiskit.pulse.channels import DriveChannel, MeasureChannel
from qiskit.pulse.commands import Discriminator, Kernel
from qiskit.qobj import RunConfig
from qiskit.qobj import QobjHeader, QobjConfig
from qiskit.exceptions import QiskitError
from qiskit.pulse.commands import *


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

    run_config = RunConfig()

    if isinstance(schedules, PulseSchedule):
        schedules = [schedules]

    if seed:
        run_config.seed = seed
    if shots:
        run_config.shots = shots
    if max_credits:
        run_config.max_credits = max_credits

    # additional config for OpenPulse
    run_config = embed_pulse_config(schedules, run_config, backend,
                                    meas_level, memory_slot_size,
                                    meas_return, rep_time)

    # add backend information to schedules
    for schedule in schedules:
        embed_backend_frequency(schedule, backend)
        embed_backend_defaults(schedule, backend)

    qobj = schedules_to_qobj(schedules, user_qobj_header=QobjHeader(), run_config=run_config,
                             qobj_id=qobj_id)

    return qobj


def embed_pulse_config(schedules, run_config, backend,
                       meas_level, memory_slot_size,
                       meas_return, rep_time):
    """ Add OpenPulse configurations to Qobj configuration.

    Args:
        schedules (List[PulseSchedule]): a list of PulseSchedule.
        run_config (RunConfig): RunConfig object.
        backend (BaseBackend): a backend to execute the circuits on.
        meas_level (int): set the appropriate level of the measurement output.
        memory_slot_size (int): size of each memory slot if the output is Level 0.
        meas_return (str): indicates the level of measurement information to return.
        rep_time (int): repetition time of the experiment in μs.
    Returns:
        RunConfig: the returned default configuration.
    """
    userconfig = run_config
    config = backend.configuration()

    # add OpenPulse configuration
    userconfig.meas_level = meas_level
    userconfig.memory_slot_size = memory_slot_size
    userconfig.meas_return = meas_return
    userconfig.qubit_lo_freq = config.defaults['qubit_freq_est']
    userconfig.meas_lo_freq = config.defaults['meas_freq_est']

    # check if rep_time is supported by backend
    if rep_time:
        if rep_time in config.rep_times:
            userconfig.rep_time = rep_time
        else:
            raise QiskitError('Invalid rep_time is specified.')
    else:
        userconfig.rep_time = config.rep_times[0]

    # merge all pulse commands in the library to reduce data size
    pulse_library = config.defaults.get('pulse_library', [])
    for schedle in schedules:
        cmds = schedle.command_library()
        for cmd in cmds:
            _name = cmd.name
            _samples = list(map(lambda x: [np.real(x), np.imag(x)], cmd.samples))
            pulse = {'name': _name, 'samples': _samples}
            if pulse not in pulse_library:
                pulse_library.append(pulse)
    userconfig.pulse_library = pulse_library

    return userconfig


def embed_backend_frequency(schedule, backend):
    """Add default LO frequencies to PulseSchedules.

    Args:
        schedule (PulseSchedule): a PulseSchedule.
        backend (BaseBackend): a backend to execute the circuits on.
    """
    config = backend.configuration()

    for chs in schedule.channel_list:
        # qubit_lo_freq
        if all(isinstance(ch, DriveChannel) for ch in chs):
            for ii, ch in enumerate(chs):
                ch.lo_frequency = ch.lo_frequency or config.defaults['qubit_freq_est'][ii]
        # meas_los_freq
        if all(isinstance(ch, MeasureChannel) for ch in chs):
            for ii, ch in enumerate(chs):
                ch.lo_frequency = ch.lo_frequency or config.defaults['meas_freq_est'][ii]


def embed_backend_defaults(schedule, backend):
    """Add default backend settings to PulseCommands.

    Args:
        schedule (PulseSchedule): a PulseSchedule.
        backend (BaseBackend): a backend to execute the circuits on.
    """
    config = backend.configuration()

    for pulse in schedule.flat_pulse_sequence():
        if isinstance(pulse.command, Acquire):
            # fill measurement kernel and discriminator
            _k = config.defaults['meas_kernel']
            _d = config.defaults['discriminator']

            if not pulse.command.kernel.name:
                pulse.command.kernel = Kernel(_k['name'], **_k['params'])
            if not pulse.command.discriminator.name:
                pulse.command.discriminator = Discriminator(_d['name'], **_d['params'])


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
