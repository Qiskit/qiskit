# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Helper module for simplified Qiskit usage.

This module includes
    execute_circuits: compile and run a list of quantum circuits.
    execute_schedules: compile and run a list of pulse schedules.
    execute: legacy wrapper that automatically selects
        execute_circuits or execute_schedules based on input

In general we recommend using the SDK modules directly. However, to get something
running quickly we have provided this wrapper module.
"""

import logging

from qiskit.circuit import QuantumCircuit
from qiskit.pulse import Schedule
from qiskit.compiler import assemble_circuits, assemble_schedules, transpile
from qiskit.qobj import QobjHeader
from qiskit.validation.exceptions import ModelValidationError
from qiskit.exceptions import QiskitError

logger = logging.getLogger(__name__)


def execute_circuits(circuits, backend,
                     basis_gates=None, coupling_map=None,  # transpile options
                     backend_properties=None, initial_layout=None,
                     seed_transpiler=None, optimization_level=None, transpile_config=None,
                     qobj_id=None, qobj_header=None, shots=1024,  # run options
                     memory=False, max_credits=10, seed_simulator=None, run_config=None,
                     seed=None, seed_mapper=None,  # deprecated
                     config=None, pass_manager=None,
                     **kwargs):
    """Executes a list of circuits on a backend.

    The execution is asynchronous, and a handle to a job instance is returned.

    This is a wrapper function around the following 3 stages:
    1. new_circuits = compiler.transpile(circuits)
    2. qobj = compiler.assemble_circuits(new_circuits)
    3. job = backend.run(qobj)

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]):
            Circuit(s) to execute

        backend (BaseBackend):
            Backend to execute circuits on.
            Transpiler options are automatically grabbed from
            backend.configuration() and backend.properties().
            If any other option is explicitly set (e.g. coupling_map), it
            will override the backend's.
            If any other options is set in the transpile_config, it will
            also override the backend's.

        basis_gates (list[str]):
            List of basis gate names to unroll to.
            e.g:
                ['u1', 'u2', 'u3', 'cx']
            If None, do not unroll.

        coupling_map (CouplingMap or list):
            Coupling map (perhaps custom) to target in mapping.
            Multiple formats are supported:
            a. CouplingMap instance

            b. list
                Must be given as an adjacency matrix, where each entry
                specifies all two-qubit interactions supported by backend
                e.g:
                    [[0, 1], [0, 3], [1, 2], [1, 5], [2, 5], [4, 1], [5, 3]]

        backend_properties (BackendProperties):
            properties returned by a backend, including information on gate
            errors, readout errors, qubit coherence times, etc. For a backend
            that provides this information, it can be obtained with:
            ``backend.properties()``

        initial_layout (Layout or dict or list):
            Initial position of virtual qubits on physical qubits.
            If this layout makes the circuit compatible with the coupling_map
            constraints, it will be used.
            The final layout is not guaranteed to be the same, as the transpiler
            may permute qubits through swaps or other means.

            Multiple formats are supported:
            a. Layout instance

            b. dict
                virtual to physical:
                    {qr[0]: 0,
                     qr[1]: 3,
                     qr[2]: 5}

                physical to virtual:
                    {0: qr[0],
                     3: qr[1],
                     5: qr[2]}

            c. list
                virtual to physical:
                    [0, 3, 5]  # virtual qubits are ordered (in addition to named)

                physical to virtual:
                    [qr[0], None, None, qr[1], None, qr[2]]

        seed_transpiler (int):
            sets random seed for the stochastic parts of the transpiler

        optimization_level (int):
            How much optimization to perform on the circuits.
            Higher levels generate more optimized circuits,
            at the expense of longer transpilation time.
                0: no optimization
                1: light optimization
                2: heavy optimization

        transpile_config (TranspileConfig):
            Transpiler configuration, containing some or all of the above options.
            If any other option is explicitly set (e.g. coupling_map), it
            will override the transpile_config's.

        qobj_id (str):
            String identifier to annotate the Qobj

        qobj_header (QobjHeader or dict):
            User input that will be inserted in Qobj header, and will also be
            copied to the corresponding Result header. Headers do not affect the run.

        shots (int):
            number of repetitions of each circuit, for sampling. Default: 2014

        memory (bool):
            if True, per-shot measurement bitstrings are returned as well
            (provided the backend supports it). Default: False

        max_credits (int):
            maximum credits to spend on job. Default: 10

        seed_simulator (int):
            random seed to control sampling, for when backend is a simulator

        run_config (RunConfig):
            Qobj runtime configuration, containing some or all of the above options.
            If any other option is explicitly set (e.g. shots), it
            will override the run_config's.

        seed (int):
            DEPRECATED in 0.8: use ``seed_simulator`` kwarg instead

        seed_mapper (int):
            DEPRECATED in 0.8: use ``seed_transpiler`` kwarg instead

        config (dict):
            DEPRECATED in 0.8: use run_config instead

        pass_manager (PassManager):
            DEPRECATED in 0.8: use pass_manager.run() to transpile the circuit,
            then assemble and run it.

        kwargs: extra arguments used by Aer for running configurable backends.
                Refer to the backend documentation for details on these arguments

    Returns:
        BaseJob: returns job instance derived from BaseJob
    """
    # transpiling the circuits using given transpile options
    new_circuits = transpile(circuits,
                             basis_gates=basis_gates,
                             coupling_map=coupling_map,
                             backend_properties=backend_properties,
                             initial_layout=initial_layout,
                             seed_transpiler=seed_transpiler,
                             optimization_level=optimization_level,
                             transpile_config=transpile_config,
                             backend=backend,
                             seed_mapper=seed_mapper,  # deprecated
                             pass_manager=pass_manager  # deprecated
                             )

    # filling in the header with the backend name the qobj was run on
    backend_config = backend.configuration()
    qobj_header = qobj_header or QobjHeader()
    if isinstance(qobj_header, dict):
        qobj_header = QobjHeader(**qobj_header)
    qobj_header.backend_name = backend_config.backend_name
    qobj_header.backend_version = backend_config.backend_version

    # assembling the circuits into a qobj to be run on the backend
    qobj = assemble_circuits(new_circuits,
                             qobj_id=qobj_id,
                             qobj_header=qobj_header,
                             shots=shots,
                             memory=memory,
                             max_credits=max_credits,
                             seed_simulator=seed_simulator,
                             run_config=run_config,
                             config=config,  # deprecated
                             seed=seed  # deprecated
                             )

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

    # executing the schedules on the backend and returning the job
    return backend.run(qobj, **kwargs)


def execute(circuits, backend,
            basis_gates=None, coupling_map=None,  # transpile options
            backend_properties=None, initial_layout=None,
            seed_transpiler=None, optimization_level=None, transpile_config=None,
            qobj_id=None, qobj_header=None, shots=1024,  # run options
            memory=False, max_credits=10, seed_simulator=None, run_config=None,
            seed=None, seed_mapper=None,  # deprecated
            config=None, pass_manager=None,
            **kwargs):
    """Execute a list of circuits or pulse schedules on a backend.

    This is a wrapper around either ``execute_circuits()`` or ``execute_schedules()``.
    Refer to those functions for more details.

    The execution is asynchronous, and a handle to a job instance is returned.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]):
            Circuit(s) to execute

        backend (BaseBackend):
            Backend to execute circuits on.
            Transpiler options are automatically grabbed from
            backend.configuration() and backend.properties().
            If any other option is explicitly set (e.g. coupling_map), it
            will override the backend's.
            If any other options is set in the transpile_config, it will
            also override the backend's.

        basis_gates (list[str]):
            List of basis gate names to unroll to.
            e.g:
                ['u1', 'u2', 'u3', 'cx']
            If None, do not unroll.

        coupling_map (CouplingMap or list):
            Coupling map (perhaps custom) to target in mapping.
            Multiple formats are supported:
            a. CouplingMap instance

            b. list
                Must be given as an adjacency matrix, where each entry
                specifies all two-qubit interactions supported by backend
                e.g:
                    [[0, 1], [0, 3], [1, 2], [1, 5], [2, 5], [4, 1], [5, 3]]

        backend_properties (BackendProperties):
            properties returned by a backend, including information on gate
            errors, readout errors, qubit coherence times, etc. For a backend
            that provides this information, it can be obtained with:
            ``backend.properties()``

        initial_layout (Layout or dict or list):
            Initial position of virtual qubits on physical qubits.
            If this layout makes the circuit compatible with the coupling_map
            constraints, it will be used.
            The final layout is not guaranteed to be the same, as the transpiler
            may permute qubits through swaps or other means.

            Multiple formats are supported:
            a. Layout instance

            b. dict
                virtual to physical:
                    {qr[0]: 0,
                     qr[1]: 3,
                     qr[2]: 5}

                physical to virtual:
                    {0: qr[0],
                     3: qr[1],
                     5: qr[2]}

            c. list
                virtual to physical:
                    [0, 3, 5]  # virtual qubits are ordered (in addition to named)

                physical to virtual:
                    [qr[0], None, None, qr[1], None, qr[2]]

        seed_transpiler (int):
            sets random seed for the stochastic parts of the transpiler

        optimization_level (int):
            How much optimization to perform on the circuits.
            Higher levels generate more optimized circuits,
            at the expense of longer transpilation time.
                0: no optimization
                1: light optimization
                2: heavy optimization

        transpile_config (TranspileConfig):
            Transpiler configuration, containing some or all of the above options.
            If any other option is explicitly set (e.g. coupling_map), it
            will override the transpile_config's.

        qobj_id (str):
            String identifier to annotate the Qobj

        qobj_header (QobjHeader):
            User input that will be inserted in Qobj header, and will also be
            copied to the corresponding Result header. Headers do not affect the run.

        shots (int):
            number of repetitions of each circuit, for sampling. Default: 2014

        memory (bool):
            if True, per-shot measurement bitstrings are returned as well
            (provided the backend supports it). Default: False

        max_credits (int):
            maximum credits to spend on job. Default: 10

        seed_simulator (int):
            random seed to control sampling, for when backend is a simulator

        run_config (RunConfig):
            Qobj runtime configuration, containing some or all of the above options.
            If any other option is explicitly set (e.g. shots), it
            will override the run_config's.

        seed (int):
            DEPRECATED in 0.8: use ``seed_simulator`` kwarg instead

        seed_mapper (int):
            DEPRECATED in 0.8: use ``seed_transpiler`` kwarg instead

        config (dict):
            DEPRECATED in 0.8: use run_config instead

        pass_manager (PassManager):
            DEPRECATED in 0.8: use pass_manager.run() to transpile the circuit,
            then assemble and run it.

        kwargs: extra arguments used by Aer for running configurable backends.
                Refer to the backend documentation for details on these arguments

    Returns:
        BaseJob: returns job instance derived from BaseJob

    Raises:
        QiskitError: if the execution cannot be interpreted as either circuits or pulses
    """
    if (isinstance(circuits, QuantumCircuit) or
            (isinstance(circuits, list) and
             all(isinstance(c, QuantumCircuit) for c in circuits))):
        return execute_circuits(circuits=circuits,
                                backend=backend,
                                basis_gates=basis_gates,  # transpile options
                                coupling_map=coupling_map,
                                backend_properties=backend_properties,
                                initial_layout=initial_layout,
                                seed_transpiler=seed_transpiler,
                                optimization_level=optimization_level,
                                transpile_config=transpile_config,
                                qobj_id=qobj_id,  # run options
                                qobj_header=qobj_header,
                                shots=shots,
                                memory=memory,
                                max_credits=max_credits,
                                seed_simulator=seed_simulator,
                                run_config=run_config,
                                seed=seed,  # deprecated
                                seed_mapper=seed_mapper,
                                config=config,
                                pass_manager=pass_manager,
                                **kwargs)

    elif (isinstance(circuits, Schedule) or
          (isinstance(circuits, list) and
           all(isinstance(c, Schedule) for c in circuits))):
        return execute_schedules()

    else:
        raise QiskitError("bad input to execute function; must be either circuits or schedules")
