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
import uuid
import copy
import logging
import warnings
from time import time
from typing import Union, List, Dict, Optional
from qiskit.circuit import QuantumCircuit, Qubit, Parameter
from qiskit.exceptions import QiskitError
from qiskit.pulse import ScheduleComponent, LoConfig
from qiskit.assembler.run_config import RunConfig
from qiskit.assembler import assemble_circuits, assemble_schedules
from qiskit.qobj import QobjHeader, Qobj
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit.validation.jsonschema import SchemaValidationError
from qiskit.providers import BaseBackend
from qiskit.pulse.channels import PulseChannel
from qiskit.pulse import Schedule

LOG = logging.getLogger(__name__)


def _log_assembly_time(start_time, end_time):
    log_msg = "Total Assembly Time - %.5f (ms)" % ((end_time - start_time) * 1000)
    LOG.info(log_msg)


# TODO: parallelize over the experiments (serialize each separately, then add global header/config)
def assemble(experiments: Union[QuantumCircuit, List[QuantumCircuit], Schedule, List[Schedule]],
             backend: Optional[BaseBackend] = None,
             qobj_id: Optional[str] = None,
             qobj_header: Optional[Union[QobjHeader, Dict]] = None,
             shots: Optional[int] = None, memory: Optional[bool] = False,
             max_credits: Optional[int] = None,
             seed_simulator: Optional[int] = None,
             qubit_lo_freq: Optional[List[int]] = None,
             meas_lo_freq: Optional[List[int]] = None,
             qubit_lo_range: Optional[List[int]] = None,
             meas_lo_range: Optional[List[int]] = None,
             schedule_los: Optional[Union[List[Union[Dict[PulseChannel, float], LoConfig]],
                                          Union[Dict[PulseChannel, float], LoConfig]]] = None,
             meas_level: Union[int, MeasLevel] = MeasLevel.CLASSIFIED,
             meas_return: Union[str, MeasReturnType] = MeasReturnType.AVERAGE,
             meas_map: Optional[List[List[Qubit]]] = None,
             memory_slot_size: int = 100,
             rep_time: Optional[int] = None,
             rep_delay: Optional[float] = None,
             parameter_binds: Optional[List[Dict[Parameter, float]]] = None,
             parametric_pulses: Optional[List[str]] = None,
             init_qubits: bool = True,
             **run_config: Dict) -> Qobj:
    """Assemble a list of circuits or pulse schedules into a ``Qobj``.

    This function serializes the payloads, which could be either circuits or schedules,
    to create ``Qobj`` "experiments". It further annotates the experiment payload with
    header and configurations.

    Args:
        experiments: Circuit(s) or pulse schedule(s) to execute
        backend: If set, some runtime options are automatically grabbed from
            ``backend.configuration()`` and ``backend.defaults()``.
            If any other option is explicitly set (e.g., ``rep_time``), it
            will override the backend's.
            If any other options is set in the run_config, it will
            also override the backend's.
        qobj_id: String identifier to annotate the ``Qobj``
        qobj_header: User input that will be inserted in ``Qobj`` header, and will also be
            copied to the corresponding Result header. Headers do not affect the run.
        shots: Number of repetitions of each circuit, for sampling. Default: 1024
            or ``max_shots`` from the backend configuration, whichever is smaller
        memory: If ``True``, per-shot measurement bitstrings are returned as well
            (provided the backend supports it). For OpenPulse jobs, only
            measurement level 2 supports this option.
        max_credits: Maximum credits to spend on job. Default: 10
        seed_simulator: Random seed to control sampling, for when backend is a simulator
        qubit_lo_freq: List of default qubit LO frequencies in Hz. Will be overridden by
            ``schedule_los`` if set.
        meas_lo_freq: List of default measurement LO frequencies in Hz. Will be overridden
            by ``schedule_los`` if set.
        qubit_lo_range: List of drive LO ranges each of form ``[range_min, range_max]`` in Hz.
            Used to validate the supplied qubit frequencies.
        meas_lo_range: List of measurement LO ranges each of form ``[range_min, range_max]`` in Hz.
            Used to validate the supplied qubit frequencies.
        schedule_los: Experiment LO configurations, frequencies are given in Hz.
        meas_level: Set the appropriate level of the measurement output for pulse experiments.
        meas_return: Level of measurement data for the backend to return.

            For ``meas_level`` 0 and 1:
                * ``single`` returns information from every shot.
                * ``avg`` returns average measurement output (averaged over number of shots).
        meas_map: List of lists, containing qubits that must be measured together.
        memory_slot_size: Size of each memory slot if the output is Level 0.
        rep_time: Time per program execution in sec. Must be from the list provided
            by the backend (``backend.configuration().rep_times``).
        rep_delay: Delay between programs in sec. Only supported on certain
            backends (``backend.configuration().dynamic_reprate_enabled`` ).
            If supported, ``rep_delay`` will be used instead of ``rep_time``. Must be from the list
            provided by the backend (``backend.configuration().rep_delays``).
        parameter_binds: List of Parameter bindings over which the set of experiments will be
            executed. Each list element (bind) should be of the form
            {Parameter1: value1, Parameter2: value2, ...}. All binds will be
            executed across all experiments; e.g., if parameter_binds is a
            length-n list, and there are m experiments, a total of m x n
            experiments will be run (one for each experiment/bind pair).
        parametric_pulses: A list of pulse shapes which are supported internally on the backend.
            Example::

            ['gaussian', 'constant']
        init_qubits: Whether to reset the qubits to the ground state for each shot.
                     Default: ``True``.
        **run_config: Extra arguments used to configure the run (e.g., for Aer configurable
            backends). Refer to the backend documentation for details on these
            arguments.

    Returns:
            A ``Qobj`` that can be run on a backend. Depending on the type of input,
            this will be either a ``QasmQobj`` or a ``PulseQobj``.

    Raises:
        QiskitError: if the input cannot be interpreted as either circuits or schedules
    """
    start_time = time()
    experiments = experiments if isinstance(experiments, list) else [experiments]
    qobj_id, qobj_header, run_config_common_dict = _parse_common_args(backend, qobj_id, qobj_header,
                                                                      shots, memory, max_credits,
                                                                      seed_simulator, init_qubits,
                                                                      **run_config)

    # assemble either circuits or schedules
    if all(isinstance(exp, QuantumCircuit) for exp in experiments):
        run_config = _parse_circuit_args(parameter_binds, **run_config_common_dict)

        # If circuits are parameterized, bind parameters and remove from run_config
        bound_experiments, run_config = _expand_parameters(circuits=experiments,
                                                           run_config=run_config)
        end_time = time()
        _log_assembly_time(start_time, end_time)
        return assemble_circuits(circuits=bound_experiments, qobj_id=qobj_id,
                                 qobj_header=qobj_header, run_config=run_config)

    elif all(isinstance(exp, ScheduleComponent) for exp in experiments):
        run_config = _parse_pulse_args(backend, qubit_lo_freq, meas_lo_freq,
                                       qubit_lo_range, meas_lo_range,
                                       schedule_los, meas_level, meas_return,
                                       meas_map, memory_slot_size,
                                       rep_time, rep_delay,
                                       parametric_pulses,
                                       **run_config_common_dict)

        end_time = time()
        _log_assembly_time(start_time, end_time)
        return assemble_schedules(schedules=experiments, qobj_id=qobj_id,
                                  qobj_header=qobj_header, run_config=run_config)

    else:
        raise QiskitError("bad input to assemble() function; "
                          "must be either circuits or schedules")


# TODO: rework to return a list of RunConfigs (one for each experiments), and a global one
def _parse_common_args(backend, qobj_id, qobj_header, shots,
                       memory, max_credits, seed_simulator, init_qubits,
                       **run_config):
    """Resolve the various types of args allowed to the assemble() function through
    duck typing, overriding args, etc. Refer to the assemble() docstring for details on
    what types of inputs are allowed.

    Here the args are resolved by converting them to standard instances, and prioritizing
    them in case a run option is passed through multiple args (explicitly setting an arg
    has more priority than the arg set by backend)

    Returns:
        RunConfig: a run config, which is a standardized object that configures the qobj
            and determines the runtime environment.

    Raises:
        QiskitError: if the memory arg is True and the backend does not support
        memory. Also if shots exceeds max_shots for the configured backend.
    """
    # grab relevant info from backend if it exists
    backend_config = None
    if backend:
        backend_config = backend.configuration()
        # check for memory flag applied to backend that does not support memory
        if memory and not backend_config.memory:
            raise QiskitError("memory not supported by backend {}"
                              .format(backend_config.backend_name))

    # an identifier for the Qobj
    qobj_id = qobj_id or str(uuid.uuid4())

    # The header that goes at the top of the Qobj (and later Result)
    # we process it as dict, then write entries that are not None to a QobjHeader object
    qobj_header = qobj_header or {}
    if isinstance(qobj_header, QobjHeader):
        qobj_header = qobj_header.to_dict()
    backend_name = getattr(backend_config, 'backend_name', None)
    backend_version = getattr(backend_config, 'backend_version', None)
    qobj_header = {**dict(backend_name=backend_name, backend_version=backend_version),
                   **qobj_header}
    qobj_header = QobjHeader(**{k: v for k, v in qobj_header.items() if v is not None})

    max_shots = getattr(backend_config, 'max_shots', None)
    if shots is None:
        if max_shots:
            shots = min(1024, max_shots)
        else:
            shots = 1024
    elif max_shots and max_shots < shots:
        raise QiskitError(
            'Number of shots specified: %s exceeds max_shots property of the '
            'backend: %s.' % (shots, max_shots))

    # create run configuration and populate
    run_config_dict = dict(shots=shots,
                           memory=memory,
                           max_credits=max_credits,
                           seed_simulator=seed_simulator,
                           init_qubits=init_qubits,
                           **run_config)

    return qobj_id, qobj_header, run_config_dict


def _parse_pulse_args(backend, qubit_lo_freq, meas_lo_freq, qubit_lo_range,
                      meas_lo_range, schedule_los, meas_level,
                      meas_return, meas_map,
                      memory_slot_size,
                      rep_time, rep_delay,
                      parametric_pulses,
                      **run_config):
    """Build a pulse RunConfig replacing unset arguments with defaults derived from the `backend`.
    See `assemble` for more information on the required arguments.

    Returns:
        RunConfig: a run config, which is a standardized object that configures the qobj
            and determines the runtime environment.
    Raises:
        SchemaValidationError: if the given meas_level is not allowed for the given `backend`.
    """
    # grab relevant info from backend if it exists
    backend_config = None
    backend_default = None
    if backend:
        backend_default = backend.defaults()
        backend_config = backend.configuration()

        if meas_level not in getattr(backend_config, 'meas_levels', [MeasLevel.CLASSIFIED]):
            raise SchemaValidationError(
                ('meas_level = {} not supported for backend {}, only {} is supported'
                 ).format(meas_level, backend_config.backend_name, backend_config.meas_levels)
            )

    meas_map = meas_map or getattr(backend_config, 'meas_map', None)

    schedule_los = schedule_los or []
    if isinstance(schedule_los, (LoConfig, dict)):
        schedule_los = [schedule_los]

    # Convert to LoConfig if LO configuration supplied as dictionary
    schedule_los = [lo_config if isinstance(lo_config, LoConfig) else LoConfig(lo_config)
                    for lo_config in schedule_los]

    if not qubit_lo_freq and hasattr(backend_default, 'qubit_freq_est'):
        qubit_lo_freq = backend_default.qubit_freq_est
    if not meas_lo_freq and hasattr(backend_default, 'meas_freq_est'):
        meas_lo_freq = backend_default.meas_freq_est

    qubit_lo_range = qubit_lo_range or getattr(backend_config, 'qubit_lo_range', None)
    meas_lo_range = meas_lo_range or getattr(backend_config, 'meas_lo_range', None)

    dynamic_reprate_enabled = getattr(backend_config, 'dynamic_reprate_enabled', False)

    rep_time = rep_time or getattr(backend_config, 'rep_times', None)
    if rep_time:
        if dynamic_reprate_enabled:
            warnings.warn("Dynamic rep rates are supported on this backend. 'rep_delay' will be "
                          "used instead, if specified.", RuntimeWarning)
        if isinstance(rep_time, list):
            rep_time = rep_time[0]
        rep_time = int(rep_time * 1e6)  # convert sec to μs

    rep_delay = rep_delay or getattr(backend_config, 'rep_delays', None)
    if rep_delay:
        if not dynamic_reprate_enabled:
            warnings.warn("Dynamic rep rates not supported on this backend. 'rep_time' will be "
                          "used instead.", RuntimeWarning)

        if isinstance(rep_delay, list):
            rep_delay = rep_delay[0]
        rep_delay = rep_delay * 1e6  # convert sec to μs

    parametric_pulses = parametric_pulses or getattr(backend_config, 'parametric_pulses', [])

    # create run configuration and populate
    run_config_dict = dict(qubit_lo_freq=qubit_lo_freq,
                           meas_lo_freq=meas_lo_freq,
                           qubit_lo_range=qubit_lo_range,
                           meas_lo_range=meas_lo_range,
                           schedule_los=schedule_los,
                           meas_level=meas_level,
                           meas_return=meas_return,
                           meas_map=meas_map,
                           memory_slot_size=memory_slot_size,
                           rep_time=rep_time,
                           rep_delay=rep_delay,
                           parametric_pulses=parametric_pulses,
                           **run_config)
    run_config = RunConfig(**{k: v for k, v in run_config_dict.items() if v is not None})

    return run_config


def _parse_circuit_args(parameter_binds, **run_config):
    """Build a circuit RunConfig replacing unset arguments with defaults derived from the `backend`.
    See `assemble` for more information on the required arguments.

    Returns:
        RunConfig: a run config, which is a standardized object that configures the qobj
            and determines the runtime environment.
    """
    parameter_binds = parameter_binds or []

    # create run configuration and populate
    run_config_dict = dict(parameter_binds=parameter_binds, **run_config)
    run_config = RunConfig(**{k: v for k, v in run_config_dict.items() if v is not None})

    return run_config


def _expand_parameters(circuits, run_config):
    """Verifies that there is a single common set of parameters shared between
    all circuits and all parameter binds in the run_config. Returns an expanded
    list of circuits (if parameterized) with all parameters bound, and a copy of
    the run_config with parameter_binds cleared.

    If neither the circuits nor the run_config specify parameters, the two are
    returned unmodified.

    Raises:
        QiskitError: if run_config parameters are not compatible with circuit parameters

    Returns:
        Tuple(List[QuantumCircuit], RunConfig):
          - List of input circuits expanded and with parameters bound
          - RunConfig with parameter_binds removed
    """

    parameter_binds = run_config.parameter_binds
    if parameter_binds or \
       any(circuit.parameters for circuit in circuits):

        all_bind_parameters = [bind.keys()
                               for bind in parameter_binds]
        all_circuit_parameters = [circuit.parameters for circuit in circuits]

        # Collect set of all unique parameters across all circuits and binds
        unique_parameters = {param
                             for param_list in all_bind_parameters + all_circuit_parameters
                             for param in param_list}

        # Check that all parameters are common to all circuits and binds
        if not all_bind_parameters \
           or not all_circuit_parameters \
           or any(unique_parameters != bind_params for bind_params in all_bind_parameters) \
           or any(unique_parameters != parameters for parameters in all_circuit_parameters):
            raise QiskitError(
                ('Mismatch between run_config.parameter_binds and all circuit parameters. ' +
                 'Parameter binds: {} ' +
                 'Circuit parameters: {}').format(all_bind_parameters, all_circuit_parameters))

        circuits = [circuit.bind_parameters(binds)
                    for circuit in circuits
                    for binds in parameter_binds]

        # All parameters have been expanded and bound, so remove from run_config
        run_config = copy.deepcopy(run_config)
        run_config.parameter_binds = []

    return circuits, run_config
