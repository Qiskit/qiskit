# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Helper module for simplified Qiskit usage.

This module includes
    execute_circuits: compile and run a list of quantum circuits.
    execute: simplified usage of either execute_circuits or execute_schedules

In general we recommend using the SDK functions directly. However, to get something
running quickly we have provider this wrapper module.
"""

import logging
import warnings

from qiskit.compiler import RunConfig, TranspileConfig
from qiskit.compiler import assemble_circuits, assemble_schedules, transpile
from qiskit.qobj import QobjHeader
from qiskit.validation.exceptions import ModelValidationError

logger = logging.getLogger(__name__)


def execute(circuits, backend, qobj_header=None, config=None, basis_gates=None,
            coupling_map=None, initial_layout=None, shots=1024, max_credits=10,
            seed=None, qobj_id=None, seed_mapper=None, pass_manager=None,
            memory=False, **kwargs):
    """Executes a set of circuits.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]): circuits to execute
        backend (BaseBackend): a backend to execute the circuits on
        qobj_header (QobjHeader or dict): user input to go into the header
        config (dict): dictionary of parameters (e.g. noise) used by runner
        basis_gates (list[str]): list of basis gate names supported by the
            target. Default: ['u1','u2','u3','cx','id']
        coupling_map (list): coupling map (perhaps custom) to target in mapping
        initial_layout (list): initial layout of qubits in mapping
        shots (int): number of repetitions of each circuit, for sampling
        max_credits (int): maximum credits to use
        seed (int): random seed for simulators
        seed_mapper (int): random seed for swapper mapper
        qobj_id (int): identifier for the generated qobj
        pass_manager (PassManager): a pass manger for the transpiler pipeline
        memory (bool): if True, per-shot measurement bitstrings are returned as well.
        kwargs: extra arguments used by AER for running configurable backends.
                Refer to the backend documentation for details on these arguments

    Returns:
        BaseJob: returns job instance derived from BaseJob
    """

    transpile_config = TranspileConfig()
    run_config = RunConfig()

    if config:
        warnings.warn('config is deprecated in terra 0.8', DeprecationWarning)
    if qobj_id:
        warnings.warn('qobj_id is deprecated in terra 0.8', DeprecationWarning)
    if basis_gates:
        transpile_config.basis_gate = basis_gates
    if coupling_map:
        transpile_config.coupling_map = coupling_map
    if initial_layout:
        transpile_config.initial_layout = initial_layout
    if seed_mapper:
        transpile_config.seed_mapper = seed_mapper
    if shots:
        run_config.shots = shots
    if max_credits:
        run_config.max_credits = max_credits
    if seed is not None:
        run_config.seed = seed
    if memory:
        run_config.memory = memory
    if pass_manager:
        warnings.warn('pass_manager in the execute function is deprecated in terra 0.8.',
                      DeprecationWarning)

    job = execute_circuits(circuits, backend, qobj_header=qobj_header,
                           run_config=run_config,
                           transpile_config=transpile_config, **kwargs)

    return job


def execute_circuits(circuits, backend, qobj_header=None,
                     transpile_config=None, run_config=None, **kwargs):
    """Executes a list of circuits.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]): circuits to execute
        backend (BaseBackend): a backend to execute the circuits on
        qobj_header (QobjHeader or dict): User input to go in the header
        transpile_config (TranspileConfig): Configurations for the transpiler
        run_config (RunConfig): Run Configuration
        kwargs: extra arguments used by AER for running configurable backends.
                Refer to the backend documentation for details on these arguments

    Returns:
        BaseJob: returns job instance derived from BaseJob
    """

    # TODO: a hack, remove when backend is not needed in transpile
    # ------
    transpile_config = transpile_config or TranspileConfig()
    transpile_config.backend = backend
    # ------

    # filling in the header with the backend name the qobj was run on
    qobj_header = qobj_header or QobjHeader()
    if isinstance(qobj_header, dict):
        qobj_header = QobjHeader(**qobj_header)
    qobj_header.backend_name = backend.name()

    # default values
    if not run_config:
        # TODO remove max_credits from the default when it is not
        # required by by the backend.
        run_config = RunConfig(shots=1024, max_credits=10, memory=False)

    # transpiling the circuits using the transpiler_config
    new_circuits = transpile(circuits, transpile_config=transpile_config)

    # assembling the circuits into a qobj to be run on the backend
    qobj = assemble_circuits(new_circuits, qobj_header=qobj_header,
                             run_config=run_config)

    # executing the circuits on the backend and returning the job
    return backend.run(qobj, **kwargs)


def execute_schedules(schedules, backend, schedule_los=None, shots=1024,
                      meas_level=2, meas_return='avg', memory=None,
                      memory_slots=None, memory_slot_size=100,
                      rep_time=None, max_credits=10, seed=None,
                      qobj_header=None, run_config=None,
                      **kwargs):
    """Executes a list of schedules.

    Args:
        schedules (Schedule or List[Schedule]): schedules to execute
        backend (BaseBackend): a backend to execute the schedules on
        schedule_los(None or list[Union[Dict[OutputChannel, float], LoConfig]] or
                        Union[Dict[OutputChannel, float], LoConfig]): Experiment LO configurations
        shots (int): number of repetitions of each circuit, for sampling
        meas_level (int): set the appropriate level of the measurement output.
        meas_return (str): indicates the level of measurement data for the backend to return
            for `meas_level` 0 and 1:
                "single" returns information from every shot of the experiment.
                "avg" returns the average measurement output (averaged over the number of shots).
        memory (bool or None): For `meas_level` 2, return the individual shot results.
        memory_slots (int): number of classical memory slots used in this job.
        memory_slot_size (int): size of each memory slot if the output is Level 0.
        rep_time (int): repetition time of the experiment in μs.
            The delay between experiments will be rep_time.
            Must be from the list provided by the device.
        max_credits (int): maximum credits to use
        seed (int): random seed for simulators
        qobj_header (QobjHeader or dict): user input to go into the header
        run_config (dict): Additional run time configuration arguments to be inserted
                    in the qobj configuration.
        kwargs: extra arguments to configure backend

    Returns:
        BaseJob: returns job instance derived from BaseJob
    """
    backend_config = backend.configuration()

    # TODO : Remove usage of config.defaults when backend.defaults() is updated.
    try:
        backend_default = backend.defaults()
    except ModelValidationError:
        from collections import namedtuple
        BackendDefault = namedtuple('BackendDefault', ('qubit_freq_est', 'meas_freq_est'))

        backend_default = BackendDefault(
            qubit_freq_est=backend_config.defaults['qubit_freq_est'],
            meas_freq_est=backend_config.defaults['meas_freq_est']
        )

    memory_slots = memory_slots if memory_slots else backend_config.n_qubits
    rep_time = rep_time if rep_time else backend_config.rep_times[-1]

    # filling in the header with the backend name the qobj was run on
    qobj_header = qobj_header or QobjHeader()
    if isinstance(qobj_header, dict):
        qobj_header = QobjHeader(**qobj_header)

    qobj_header.backend_name = backend.name()
    qobj_header.backend_version = backend_config.backend_version

    if run_config is None:
        run_config = {}

    qobj = assemble_schedules(schedules,
                              backend_default.qubit_freq_est,
                              backend_default.meas_freq_est,
                              schedule_los=schedule_los, shots=shots,
                              meas_level=meas_level, meas_return=meas_return,
                              memory=memory, memory_slots=memory_slots,
                              memory_slot_size=memory_slot_size,
                              rep_time=rep_time, max_credits=max_credits,
                              seed=seed, qobj_header=qobj_header,
                              **run_config)

    return backend.run(qobj, **kwargs)
