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
======================================================
Executing Experiments (:mod:`qiskit.execute_function`)
======================================================

.. currentmodule:: qiskit.execute_function

.. autofunction:: execute
"""
import logging
from time import time
import warnings
from qiskit.compiler import transpile, schedule
from qiskit.providers.backend import Backend
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.exceptions import QiskitError

logger = logging.getLogger(__name__)


def _log_submission_time(start_time, end_time):
    log_msg = "Total Job Submission Time - %.5f (ms)" % ((end_time - start_time) * 1000)
    logger.info(log_msg)


def execute(
    experiments,
    backend,
    basis_gates=None,
    coupling_map=None,  # circuit transpile options
    backend_properties=None,
    initial_layout=None,
    seed_transpiler=None,
    optimization_level=None,
    pass_manager=None,
    qobj_id=None,
    qobj_header=None,
    shots=None,  # common run options
    memory=None,
    max_credits=None,
    seed_simulator=None,
    default_qubit_los=None,
    default_meas_los=None,  # schedule run options
    qubit_lo_range=None,
    meas_lo_range=None,
    schedule_los=None,
    meas_level=None,
    meas_return=None,
    memory_slots=None,
    memory_slot_size=None,
    rep_time=None,
    rep_delay=None,
    parameter_binds=None,
    schedule_circuit=False,
    inst_map=None,
    meas_map=None,
    scheduling_method=None,
    init_qubits=None,
    **run_config,
):
    """Execute a list of :class:`qiskit.circuit.QuantumCircuit` or
    :class:`qiskit.pulse.Schedule` on a backend.

    This function provides a higher level abstraction for applications where the details
    of how the circuit is compiled and run on the backend don't matter and all that is
    required is that you want to execute a :class:`~.QuantumCircuit` on a given backend.
    The options on this function are minimal by design, if you desire more control over
    the compilation of the circuit prior to execution you should instead use the
    :class:`~.transpile` function with the :class:`.BackendV2.run` method to execute the
    circuit. For example::

        from qiskit.providers.fake_provider import FakeKolkata
        from qiskit.circuit import QuantumCircuit
        from qiskit.compiler import transpile

        backend = FakeKolkata()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        backend.run(transpile(qc, backend, optimization_level=3))


    Args:
        experiments (QuantumCircuit or list[QuantumCircuit] or Schedule or list[Schedule]):
            Circuit(s) or pulse schedule(s) to execute

        backend (Backend):
            Backend to execute circuits on.
            Transpiler options are automatically grabbed from
            backend.configuration() and backend.properties().
            If any other option is explicitly set (e.g. coupling_map), it
            will override the backend's.

        basis_gates (list[str]):
            DEPRECATED:
            List of basis gate names to unroll to.
            e.g: ``['u1', 'u2', 'u3', 'cx']``
            If ``None``, do not unroll.

        coupling_map (CouplingMap or list): DEPRECATED: Coupling map (perhaps custom) to
            target in mapping. Multiple formats are supported:

            #. CouplingMap instance
            #. list
               Must be given as an adjacency matrix, where each entry
               specifies all two-qubit interactions supported by backend
               e.g:
               ``[[0, 1], [0, 3], [1, 2], [1, 5], [2, 5], [4, 1], [5, 3]]``

        backend_properties (BackendProperties):
            DEPRECATED: Properties returned by a backend, including information on gate
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

        pass_manager (PassManager): DEPRECATED the pass manager to use during transpilation. If this
            arg is present, auto-selection of pass manager based on the transpile options
            will be turned off and this pass manager will be used directly. If you require compilation
            with a custom passmanager you should use the :meth:`.PassManager.run` method directly to
            compile the circuit and ``backend.run()`` to execute it.

        qobj_id (str): DEPRECATED: String identifier to annotate the Qobj.  This has no effect
            and the :attr:`~.QuantumCircuit.name` attribute of the input circuit(s) should be used
            instead.

        qobj_header (QobjHeader or dict): DEPRECATED: User input that will be inserted in Qobj
            header, and will also be copied to the corresponding :class:`qiskit.result.Result`
            header. Headers do not affect the run. Headers do not affect the run. This kwarg
            has no effect anymore and the :attr:`~.QuantumCircuit.metadata` attribute of the
            input circuit(s) should be used instead.

        shots (int): DEPRECATED: Number of repetitions of each circuit, for sampling. Default: 1024

        memory (bool): DEPRECATED If True, per-shot measurement bitstrings are returned as well
            (provided the backend supports it). For OpenPulse jobs, only
            measurement level 2 supports this option. Default: False. This is a per backend option
            and should be part of run_config for backends that support it.

        max_credits (int): DEPRECATED This parameter is deprecated as of Qiskit Terra 0.20.0
            and will be removed in a future release. This parameter has no effect on modern
            IBM Quantum systems, no alternative is necessary.

        seed_simulator (int): DEPRECATED Random seed to control sampling, for when backend is a
            simulator.

        default_qubit_los (Optional[List[float]]): DEPRECATED: List of job level qubit drive LO
            frequencies in Hz. Overridden by ``schedule_los`` if specified. Must have length
            ``n_qubits``. This is deprecated if control of the LO frequencies is needed that should
            be done at the backend level

        default_meas_los (Optional[List[float]]): DEPRECATED: List of job level measurement LO
            frequencies in Hz. Overridden by ``schedule_los`` if specified. Must have
            length ``n_qubits``.

        qubit_lo_range (Optional[List[List[float]]]): DEPRECATED: List of job level drive LO
            anges each of form ``[range_min, range_max]`` in Hz. Used to validate
            ``qubit_lo_freq``. Must have length ``n_qubits``.

        meas_lo_range (Optional[List[List[float]]]): DEPRECATED: List of job level measurement
            LO ranges each of form ``[range_min, range_max]`` in Hz. Used to
            validate ``meas_lo_freq``. Must have length ``n_qubits``.

        schedule_los (list):
            DEPRECATED: Experiment level (ie circuit or schedule) LO frequency configurations
            for qubit drive and measurement channels. These values override the job level
            values from ``default_qubit_los`` and ``default_meas_los``. Frequencies
            are in Hz. Settable for qasm and pulse jobs.

            If a single LO config or dict is used, the values are set at job level. If a list is
            used, the list must be the size of the number of experiments in the job, except in the
            case of a single experiment. In this case, a frequency sweep will be assumed and one
            experiment will be created for every list entry.

            Not every channel is required to be specified. If not specified, the backend default
            value will be used.

        meas_level (int or MeasLevel): DEPRECATED: Set the appropriate level of the
            measurement output for pulse experiments.

        meas_return (str or MeasReturn): DEPRECATED: Level of measurement data for the
            backend to return For ``meas_level`` 0 and 1:
            ``"single"`` returns information from every shot.
            ``"avg"`` returns average measurement output (averaged over number
            of shots).

        memory_slots (int): DEPRECATED: Number of classical memory slots used in this job.

        memory_slot_size (int): DEPRECATED: Size of each memory slot if the output is Level 0.

        rep_time (int): DEPRECATED: Time per program execution in seconds. Must be from the list provided
            by the backend (``backend.configuration().rep_times``). Defaults to the first entry.

        rep_delay (float): DEPRECATED: Delay between programs in seconds. Only supported on certain
            backends (``backend.configuration().dynamic_reprate_enabled`` ). If supported,
            ``rep_delay`` will be used instead of ``rep_time`` and must be from the range supplied
            by the backend (``backend.configuration().rep_delay_range``). Default is given by
            ``backend.configuration().default_rep_delay``.

        parameter_binds (list[dict]): List of Parameter bindings over which the set of
            experiments will be executed. Each list element (bind) should be of the form
            ``{Parameter1: value1, Parameter2: value2, ...}``. All binds will be
            executed across all experiments, e.g. if parameter_binds is a
            length-:math:`n` list, and there are :math:`m` experiments, a total of :math:`m \\times n`
            experiments will be run (one for each experiment/bind pair).

        schedule_circuit (bool): DEPRECATED: If ``True``, ``experiments`` will be converted to
            :class:`qiskit.pulse.Schedule` objects prior to execution.

        inst_map (InstructionScheduleMap):
            DEPRECATED: Mapping of circuit operations to pulse schedules. If None, defaults to the
            ``instruction_schedule_map`` of ``backend``.

        meas_map (list(list(int))):
            DEPRECATED: List of sets of qubits that must be measured together. If None, defaults to
            the ``meas_map`` of ``backend``.

        scheduling_method (str or list(str)):
            DEPRECATED: Optionally specify a particular scheduling method. If a custom scheduling method
            is required :func:`~.transpile` and :func:`~.schedule` should be used and pass the output
            to the ``backend.run()`` directly.

        init_qubits (bool): DEPRECATED: Whether to reset the qubits to the ground state for each shot.
            Default: ``True``. This is a backend specific option and should be set as part of
            run_config on a specific backend

        run_config (dict):
            Extra arguments used to configure the run. Refer to the backend documentation for
            details on these arguments.

    Returns:
        Job: returns job instance derived from Job

    Raises:
        QiskitError: if the execution cannot be interpreted as either circuits or schedules

    Example:
        Construct a 5-qubit GHZ circuit and execute 4321 shots on a backend.

        .. code-block::

            from qiskit import QuantumCircuit, execute, BasicAer

            backend = BasicAer.get_backend('qasm_simulator')

            qc = QuantumCircuit(5, 5)
            qc.h(0)
            qc.cx(0, range(1, 5))
            qc.measure_all()

            job = execute(qc, backend, shots=4321)
    """
    if basis_gates is not None:
        warnings.warn(
            "The basis_gates argument is deprecated as of Qiskit Terra 0.21.0, "
            "and will be removed in a future release. If you require compiling to a "
            "custom set of basis gates you should use the transpile() function instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    if coupling_map is not None:
        warnings.warn(
            "The coupling_map argument is deprecated as of Qiskit Terra 0.21.0, "
            "and will be removed in a future release. If you require compiling to a "
            "custom coupling graph you should use the transpile() function instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    if backend_properties is not None:
        warnings.warn(
            "The backend_properties argument is deprecated as of Qiskit Terra 0.21.0, "
            "and will be removed in a future release. If you require compiling to a "
            "custom BackendProperties object you should use the transpile() function instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    if pass_manager is not None:
        warnings.warn(
            "The pass_manager argument is deprecated as of Qiskit Terra 0.21.0, "
            "and will be removed in a future release. If you require compiling with a "
            "custom PassManager object you should use the run() method on that custom "
            "Passmanager directly instead of using execute.",
            DeprecationWarning,
            stacklevel=2,
        )
    if shots is not None:
        if not hasattr(backend.options, "shots"):
            warnings.warn(
                "The shots argument is deprecated as of Qiskit Terra 0.21.0, "
                "and will be removed in a future release. The backend you are running on does "
                "not support setting the number of shots so this option will have no effect.",
                DeprecationWarning,
                stacklevel=2,
            )
    if memory is not None:
        if not hasattr(backend.options, "memory"):
            warnings.warn(
                "The memory argument is deprecated as of Qiskit Terra 0.21.0, "
                "and will be removed in a future release. The backend you are running on does "
                "not support setting this as an option so the argument will have no effect.",
                DeprecationWarning,
                stacklevel=2,
            )
    if seed_simulator is not None:
        if not hasattr(backend.options, "seed_simulator"):
            warnings.warn(
                "The seed_simulator argument is deprecated as of Qiskit Terra 0.21.0, "
                "and will be removed in a future release. The backend you are running on does "
                "not support setting this as an option so this argument will have no effect.",
                DeprecationWarning,
                stacklevel=2,
            )
    if default_qubit_los is not None:
        warnings.warn(
            "The default_qubit_los argument is deprecated as of Qiskit Terra 0.21.0, "
            "and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
    if default_meas_los is not None:
        warnings.warn(
            "The default_meas_los argument is deprecated as of Qiskit Terra 0.21.0, "
            "and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
    if qubit_lo_range is not None:
        warnings.warn(
            "The qubit_lo_range argument is deprecated as of Qiskit Terra 0.21.0, "
            "and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
    if meas_lo_range is not None:
        warnings.warn(
            "The meas_lo_range argument is deprecated as of Qiskit Terra 0.21.0, "
            "and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
    if schedule_los is not None:
        warnings.warn(
            "The schedule_los argument is deprecated as of Qiskit Terra 0.21.0, "
            "and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
    if meas_return is not None:
        warnings.warn(
            "The meas_return argument is deprecated as of Qiskit Terra 0.21.0, "
            "and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
    if memory_slots is not None:
        warnings.warn(
            "The memory_slots argument is deprecated as of Qiskit Terra 0.21.0, "
            "and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
    if memory_slot_size is not None:
        warnings.warn(
            "The memory_slot_size argument is deprecated as of Qiskit Terra 0.21.0, "
            "and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
    if rep_time is not None:
        warnings.warn(
            "The rep_time argument is deprecated as of Qiskit Terra 0.21.0, "
            "and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
    if rep_delay is not None:
        warnings.warn(
            "The rep_delay argument is deprecated as of Qiskit Terra 0.21.0, "
            "and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
    if not schedule_circuit:
        warnings.warn(
            "The schedule_circuit argument is deprecated as of Qiskit Terra 0.21.0, "
            "and will be removed in a future release. If you require scheduling your circuit "
            "prior to execution you should use the :func:`~.schedule` function explicitly with "
            "prior to running the circuit.",
            DeprecationWarning,
            stacklevel=2,
        )
    if inst_map is not None:
        warnings.warn(
            "The inst_map argument is deprecated as of Qiskit Terra 0.21.0, "
            "and will be removed in a future release. If you require compiling to a "
            "custom :class:`~.InstructionScheduleMap` you should use the transpile() function "
            "instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    if meas_map is not None:
        warnings.warn(
            "The meas_map argument is deprecated as of Qiskit Terra 0.21.0, "
            "and will be removed in a future release. If you require scheduling your circuit "
            "prior to execution you should use the :func:`~.schedule` function explicitly with "
            "prior to running the circuit.",
            DeprecationWarning,
            stacklevel=2,
        )
    if scheduling_method is not None:
        warnings.warn(
            "The scheduling_method argument is deprecated as of Qiskit Terra 0.21.0, "
            "and will be removed in a future release. If you require scheduling your circuit "
            "prior to execution you should use the :func:`~.schedule` function explicitly with "
            "prior to running the circuit.",
            DeprecationWarning,
            stacklevel=2,
        )
    if init_qubits is not None:
        warnings.warn(
            "The init_qubits argument is deprecated as of Qiskit Terra 0.21.0, "
            "and will be removed in a future release. If you require scheduling your circuit "
            "prior to execution you should use the :func:`~.schedule` function explicitly with "
            "prior to running the circuit.",
            DeprecationWarning,
            stacklevel=2,
        )

    if isinstance(experiments, (Schedule, ScheduleBlock)) or (
        isinstance(experiments, list) and isinstance(experiments[0], (Schedule, ScheduleBlock))
    ):
        # do not transpile a schedule circuit
        if schedule_circuit:
            raise QiskitError("Must supply QuantumCircuit to schedule circuit.")
    elif pass_manager is not None:
        # transpiling using pass_manager
        _check_conflicting_argument(
            optimization_level=optimization_level,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            seed_transpiler=seed_transpiler,
            backend_properties=backend_properties,
            initial_layout=initial_layout,
        )
        experiments = pass_manager.run(experiments)
    else:
        # transpiling the circuits using given transpile options
        experiments = transpile(
            experiments,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            backend_properties=backend_properties,
            initial_layout=initial_layout,
            seed_transpiler=seed_transpiler,
            optimization_level=optimization_level,
            backend=backend,
        )

    if schedule_circuit:
        experiments = schedule(
            circuits=experiments,
            backend=backend,
            inst_map=inst_map,
            meas_map=meas_map,
            method=scheduling_method,
        )
    if max_credits is not None:
        warnings.warn(
            "The `max_credits` parameter is deprecated as of Qiskit Terra 0.20.0, "
            "and will be removed in a future release. This parameter has no effect on "
            "modern IBM Quantum systems, and no alternative is necessary.",
            DeprecationWarning,
            stacklevel=2,
        )

    if qobj_id is not None:
        warnings.warn(
            "The qobj_id argument is deprecated as of the Qiskit Terra 0.21.0, "
            "and will be remvoed in a future release. This argument has no effect and "
            "is not used by any backends."
        )

    if qobj_header is not None:
        warnings.warn(
            "The qobj_header argument is deprecated as of the Qiskit Terra 0.21.0, "
            "and will be remvoed in a future release. This argument has no effect and "
            "is not used by any backends."
        )

    if isinstance(backend, Backend):
        start_time = time()
        run_kwargs = {
            "shots": shots,
            "memory": memory,
            "seed_simulator": seed_simulator,
            "qubit_lo_freq": default_qubit_los,
            "meas_lo_freq": default_meas_los,
            "qubit_lo_range": qubit_lo_range,
            "meas_lo_range": meas_lo_range,
            "schedule_los": schedule_los,
            "meas_level": meas_level,
            "meas_return": meas_return,
            "memory_slots": memory_slots,
            "memory_slot_size": memory_slot_size,
            "rep_time": rep_time,
            "rep_delay": rep_delay,
            "init_qubits": init_qubits,
        }
        for key in list(run_kwargs.keys()):
            if not hasattr(backend.options, key):
                if run_kwargs[key] is not None:
                    logger.info(
                        "%s backend doesn't support option %s so not passing that kwarg to run()",
                        backend.name,
                        key,
                    )
                del run_kwargs[key]
            elif run_kwargs[key] is None:
                del run_kwargs[key]

        if parameter_binds:
            run_kwargs["parameter_binds"] = parameter_binds
        run_kwargs.update(run_config)
        job = backend.run(experiments, **run_kwargs)
        end_time = time()
        _log_submission_time(start_time, end_time)
    else:
        raise QiskitError("Invalid backend type %s" % type(backend))
    return job


def _check_conflicting_argument(**kargs):
    conflicting_args = [arg for arg, value in kargs.items() if value]
    if conflicting_args:
        raise QiskitError(
            "The parameters pass_manager conflicts with the following "
            "parameter(s): {}.".format(", ".join(conflicting_args))
        )
