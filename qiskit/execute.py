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

"""
=============================================
Executing Experiments (:mod:`qiskit.execute`)
=============================================

.. currentmodule:: qiskit.execute

.. autofunction:: execute
"""
import logging
from time import time
from qiskit.compiler import transpile, assemble, schedule
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit.pulse import Schedule
from qiskit.exceptions import QiskitError

logger = logging.getLogger(__name__)


def _log_submission_time(start_time, end_time):
    log_msg = ("Total Job Submission Time - %.5f (ms)"
               % ((end_time - start_time) * 1000))
    logger.info(log_msg)


def execute(experiments, backend,
            basis_gates=None, coupling_map=None,  # circuit transpile options
            backend_properties=None, initial_layout=None,
            seed_transpiler=None, optimization_level=None, pass_manager=None,
            qobj_id=None, qobj_header=None, shots=1024,  # common run options
            memory=False, max_credits=10, seed_simulator=None,
            default_qubit_los=None, default_meas_los=None,  # schedule run options
            schedule_los=None, meas_level=MeasLevel.CLASSIFIED,
            meas_return=MeasReturnType.AVERAGE,
            memory_slots=None, memory_slot_size=100, rep_time=None, rep_delay=None,
            parameter_binds=None, schedule_circuit=False, inst_map=None, meas_map=None,
            scheduling_method=None, init_qubits=None,
            **run_config):
    """Execute a list of :class:`qiskit.circuit.QuantumCircuit` or
    :class:`qiskit.pulse.Schedule` on a backend.

    The execution is asynchronous, and a handle to a job instance is returned.

    Args:
        experiments (QuantumCircuit or list[QuantumCircuit] or Schedule or list[Schedule]):
            Circuit(s) or pulse schedule(s) to execute

        backend (BaseBackend):
            Backend to execute circuits on.
            Transpiler options are automatically grabbed from
            backend.configuration() and backend.properties().
            If any other option is explicitly set (e.g. coupling_map), it
            will override the backend's.

        basis_gates (list[str]):
            List of basis gate names to unroll to.
            e.g: ``['u1', 'u2', 'u3', 'cx']``
            If ``None``, do not unroll.

        coupling_map (CouplingMap or list): Coupling map (perhaps custom) to
            target in mapping. Multiple formats are supported:

            #. CouplingMap instance
            #. list
               Must be given as an adjacency matrix, where each entry
               specifies all two-qubit interactions supported by backend
               e.g:
               ``[[0, 1], [0, 3], [1, 2], [1, 5], [2, 5], [4, 1], [5, 3]]``

        backend_properties (BackendProperties):
            Properties returned by a backend, including information on gate
            errors, readout errors, qubit coherence times, etc. Find a backend
            that provides this information with:
            ``backend.properties()``

        initial_layout (Layout or dict or list):
            Initial position of virtual qubits on physical qubits.
            If this layout makes the circuit compatible with the coupling_map
            constraints, it will be used.
            The final layout is not guaranteed to be the same, as the transpiler
            may permute qubits through swaps or other means.

            Multiple formats are supported:

            #. :class:`qiskit.transpiler.Layout` instance
            #. ``dict``:
               virtual to physical::

                    {qr[0]: 0,
                     qr[1]: 3,
                     qr[2]: 5}

               physical to virtual::
                    {0: qr[0],
                     3: qr[1],
                     5: qr[2]}

            #. ``list``
               virtual to physical::

                    [0, 3, 5]  # virtual qubits are ordered (in addition to named)

               physical to virtual::

                    [qr[0], None, None, qr[1], None, qr[2]]

        seed_transpiler (int): Sets random seed for the stochastic parts of the transpiler

        optimization_level (int): How much optimization to perform on the circuits.
            Higher levels generate more optimized circuits,
            at the expense of longer transpilation time.
            #. No optimization
            #. Light optimization
            #. Heavy optimization
            #. Highest optimization
            If None, level 1 will be chosen as default.

        pass_manager (PassManager): The pass manager to use during transpilation. If this
            arg is present, auto-selection of pass manager based on the transpile options
            will be turned off and this pass manager will be used directly.

        qobj_id (str): String identifier to annotate the Qobj

        qobj_header (QobjHeader or dict): User input that will be inserted in Qobj header,
            and will also be copied to the corresponding :class:`qiskit.result.Result`
            header. Headers do not affect the run.

        shots (int): Number of repetitions of each circuit, for sampling. Default: 1024

        memory (bool): If True, per-shot measurement bitstrings are returned as well
            (provided the backend supports it). For OpenPulse jobs, only
            measurement level 2 supports this option. Default: False

        max_credits (int): Maximum credits to spend on job. Default: 10

        seed_simulator (int): Random seed to control sampling, for when backend is a simulator

        default_qubit_los (list): List of default qubit LO frequencies in Hz

        default_meas_los (list): List of default meas LO frequencies in Hz

        schedule_los (None or list or dict or LoConfig): Experiment LO
            configurations, if specified the list is in the format::

                list[Union[Dict[PulseChannel, float], LoConfig]] or
                     Union[Dict[PulseChannel, float], LoConfig]

        meas_level (int or MeasLevel): Set the appropriate level of the
            measurement output for pulse experiments.

        meas_return (str or MeasReturn): Level of measurement data for the
            backend to return For ``meas_level`` 0 and 1:
            ``"single"`` returns information from every shot.
            ``"avg"`` returns average measurement output (averaged over number
            of shots).

        memory_slots (int): Number of classical memory slots used in this job.

        memory_slot_size (int): Size of each memory slot if the output is Level 0.

        rep_time (int): Time per program execution in seconds. Must be from the list provided
            by the backend (``backend.configuration().rep_times``). Defaults to the first entry.

        rep_delay (float): Delay between programs in seconds. Only supported on certain
            backends (``backend.configuration().dynamic_reprate_enabled`` ). If supported,
            ``rep_delay`` will be used instead of ``rep_time`` and must be from the range supplied
            by the backend (``backend.configuration().rep_delay_range``). Default is given by
            ``backend.configuration().default_rep_delay``.

        parameter_binds (list[dict]): List of Parameter bindings over which the set of
            experiments will be executed. Each list element (bind) should be of the form
            ``{Parameter1: value1, Parameter2: value2, ...}``. All binds will be
            executed across all experiments, e.g. if parameter_binds is a
            length-n list, and there are m experiments, a total of :math:`m x n`
            experiments will be run (one for each experiment/bind pair).

        schedule_circuit (bool): If ``True``, ``experiments`` will be converted to
            :class:`qiskit.pulse.Schedule` objects prior to execution.

        inst_map (InstructionScheduleMap):
            Mapping of circuit operations to pulse schedules. If None, defaults to the
            ``instruction_schedule_map`` of ``backend``.

        meas_map (list(list(int))):
            List of sets of qubits that must be measured together. If None, defaults to
            the ``meas_map`` of ``backend``.

        scheduling_method (str or list(str)):
            Optionally specify a particular scheduling method.

        init_qubits (bool): Whether to reset the qubits to the ground state for each shot.
                            Default: ``True``.

        run_config (dict):
            Extra arguments used to configure the run (e.g. for Aer configurable backends).
            Refer to the backend documentation for details on these arguments.
            Note: for now, these keyword arguments will both be copied to the
            Qobj config, and passed to backend.run()

    Returns:
        BaseJob: returns job instance derived from BaseJob

    Raises:
        QiskitError: if the execution cannot be interpreted as either circuits or schedules

    Example:
        Construct a 5-qubit GHZ circuit and execute 4321 shots on a backend.

        .. jupyter-execute::

            from qiskit import QuantumCircuit, execute, BasicAer

            backend = BasicAer.get_backend('qasm_simulator')

            qc = QuantumCircuit(5, 5)
            qc.h(0)
            qc.cx(0, range(1, 5))
            qc.measure_all()

            job = execute(qc, backend, shots=4321)
    """
    if isinstance(experiments, Schedule) or (isinstance(experiments, list) and
                                             isinstance(experiments[0], Schedule)):
        # do not transpile a schedule circuit
        if schedule_circuit:
            raise QiskitError("Must supply QuantumCircuit to schedule circuit.")
    elif pass_manager is not None:
        # transpiling using pass_manager
        _check_conflicting_argument(optimization_level=optimization_level,
                                    basis_gates=basis_gates,
                                    coupling_map=coupling_map,
                                    seed_transpiler=seed_transpiler,
                                    backend_properties=backend_properties,
                                    initial_layout=initial_layout,
                                    backend=backend)
        experiments = pass_manager.run(experiments)
    else:
        # transpiling the circuits using given transpile options
        experiments = transpile(experiments,
                                basis_gates=basis_gates,
                                coupling_map=coupling_map,
                                backend_properties=backend_properties,
                                initial_layout=initial_layout,
                                seed_transpiler=seed_transpiler,
                                optimization_level=optimization_level,
                                backend=backend)

    if schedule_circuit:
        experiments = schedule(circuits=experiments,
                               backend=backend,
                               inst_map=inst_map,
                               meas_map=meas_map,
                               method=scheduling_method)

    # assembling the circuits into a qobj to be run on the backend
    qobj = assemble(experiments,
                    qobj_id=qobj_id,
                    qobj_header=qobj_header,
                    shots=shots,
                    memory=memory,
                    max_credits=max_credits,
                    seed_simulator=seed_simulator,
                    default_qubit_los=default_qubit_los,
                    default_meas_los=default_meas_los,
                    schedule_los=schedule_los,
                    meas_level=meas_level,
                    meas_return=meas_return,
                    memory_slots=memory_slots,
                    memory_slot_size=memory_slot_size,
                    rep_time=rep_time,
                    rep_delay=rep_delay,
                    parameter_binds=parameter_binds,
                    backend=backend,
                    init_qubits=init_qubits,
                    **run_config)

    # executing the circuits on the backend and returning the job
    start_time = time()
    job = backend.run(qobj, **run_config)
    end_time = time()
    _log_submission_time(start_time, end_time)
    return job


def _check_conflicting_argument(**kargs):
    conflicting_args = [arg for arg, value in kargs.items() if value]
    if conflicting_args:
        raise QiskitError("The parameters pass_manager conflicts with the following "
                          "parameter(s): {}.".format(', '.join(conflicting_args)))
