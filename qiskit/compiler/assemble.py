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
import logging
import copy

from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.pulse import ScheduleComponent, LoConfig
from qiskit.assembler.run_config import RunConfig
from qiskit.assembler import assemble_circuits, assemble_schedules
from qiskit.qobj import QobjHeader
from qiskit.validation.exceptions import ModelValidationError

logger = logging.getLogger(__name__)


# TODO: parallelize over the experiments (serialize each separately, then add global header/config)
def assemble(experiments,
             backend=None,
             qobj_id=None, qobj_header=None,  # common run options
             shots=1024, memory=False, max_credits=None, seed_simulator=None,
             qubit_lo_freq=None, meas_lo_freq=None,  # schedule run options
             qubit_lo_range=None, meas_lo_range=None,
             schedule_los=None, meas_level=2, meas_return='avg', meas_map=None,
             memory_slots=None, memory_slot_size=100, rep_time=None, parameter_binds=None,
             **run_config):
    """Assemble a list of circuits or pulse schedules into a Qobj.

    This function serializes the payloads, which could be either circuits or schedules,
    to create Qobj "experiments". It further annotates the experiment payload with
    header and configurations.

    Args:
        experiments (QuantumCircuit or list[QuantumCircuit] or Schedule or list[Schedule]):
            Circuit(s) or pulse schedule(s) to execute

        backend (BaseBackend):
            If set, some runtime options are automatically grabbed from
            backend.configuration() and backend.defaults().
            If any other option is explicitly set (e.g. rep_rate), it
            will override the backend's.
            If any other options is set in the run_config, it will
            also override the backend's.

        qobj_id (str):
            String identifier to annotate the Qobj

        qobj_header (QobjHeader or dict):
            User input that will be inserted in Qobj header, and will also be
            copied to the corresponding Result header. Headers do not affect the run.

        shots (int):
            Number of repetitions of each circuit, for sampling. Default: 2014

        memory (bool):
            If True, per-shot measurement bitstrings are returned as well
            (provided the backend supports it). For OpenPulse jobs, only
            measurement level 2 supports this option. Default: False

        max_credits (int):
            Maximum credits to spend on job. Default: 10

        seed_simulator (int):
            Random seed to control sampling, for when backend is a simulator

        qubit_lo_freq (list):
            List of default qubit lo frequencies

        meas_lo_freq (list):
            List of default meas lo frequencies

        qubit_lo_range (list):
            List of drive lo ranges

        meas_lo_range (list):
            List of meas lo ranges

        schedule_los (None or list[Union[Dict[PulseChannel, float], LoConfig]] or
                      Union[Dict[PulseChannel, float], LoConfig]):
            Experiment LO configurations

        meas_level (int):
            Set the appropriate level of the measurement output for pulse experiments.

        meas_return (str):
            Level of measurement data for the backend to return
            For `meas_level` 0 and 1:
                "single" returns information from every shot.
                "avg" returns average measurement output (averaged over number of shots).

        meas_map (list):
            List of lists, containing qubits that must be measured together.

        memory_slots (int):
            Number of classical memory slots used in this job.

        memory_slot_size (int):
            Size of each memory slot if the output is Level 0.

        rep_time (int): repetition time of the experiment in μs.
            The delay between experiments will be rep_time.
            Must be from the list provided by the device.

        parameter_binds (list[dict{Parameter: Value}]):
            List of Parameter bindings over which the set of experiments will be
            executed. Each list element (bind) should be of the form
            {Parameter1: value1, Parameter2: value2, ...}. All binds will be
            executed across all experiments, e.g. if parameter_binds is a
            length-n list, and there are m experiments, a total of m x n
            experiments will be run (one for each experiment/bind pair).

        run_config (dict):
            extra arguments used to configure the run (e.g. for Aer configurable backends)
            Refer to the backend documentation for details on these arguments

    Returns:
        Qobj: a qobj which can be run on a backend. Depending on the type of input,
            this will be either a QasmQobj or a PulseQobj.

    Raises:
        QiskitError: if the input cannot be interpreted as either circuits or schedules
    """
    # Get RunConfig(s) that will be inserted in Qobj to configure the run
    experiments = experiments if isinstance(experiments, list) else [experiments]
    qobj_id, qobj_header, run_config = _parse_run_args(backend, qobj_id, qobj_header,
                                                       shots, memory, max_credits, seed_simulator,
                                                       qubit_lo_freq, meas_lo_freq,
                                                       qubit_lo_range, meas_lo_range,
                                                       schedule_los, meas_level, meas_return,
                                                       meas_map, memory_slots,
                                                       memory_slot_size, rep_time,
                                                       parameter_binds, **run_config)

    # assemble either circuits or schedules
    if all(isinstance(exp, QuantumCircuit) for exp in experiments):
        # If circuits are parameterized, bind parameters and remove from run_config
        bound_experiments, run_config = _expand_parameters(circuits=experiments,
                                                           run_config=run_config)
        return assemble_circuits(circuits=bound_experiments, qobj_id=qobj_id,
                                 qobj_header=qobj_header, run_config=run_config)

    elif all(isinstance(exp, ScheduleComponent) for exp in experiments):
        return assemble_schedules(schedules=experiments, qobj_id=qobj_id,
                                  qobj_header=qobj_header, run_config=run_config)

    else:
        raise QiskitError("bad input to assemble() function; "
                          "must be either circuits or schedules")


# TODO: rework to return a list of RunConfigs (one for each experiments), and a global one
def _parse_run_args(backend, qobj_id, qobj_header,
                    shots, memory, max_credits, seed_simulator,
                    qubit_lo_freq, meas_lo_freq,
                    qubit_lo_range, meas_lo_range,
                    schedule_los, meas_level, meas_return,
                    meas_map, memory_slots,
                    memory_slot_size, rep_time,
                    parameter_binds, **run_config):
    """Resolve the various types of args allowed to the assemble() function through
    duck typing, overriding args, etc. Refer to the assemble() docstring for details on
    what types of inputs are allowed.

    Here the args are resolved by converting them to standard instances, and prioritizing
    them in case a run option is passed through multiple args (explicitly setting an arg
    has more priority than the arg set by backend)

    Returns:
        RunConfig: a run config, which is a standardized object that configures the qobj
            and determines the runtime environment.
    """
    # grab relevant info from backend if it exists
    backend_config = None
    backend_default = None
    if backend:
        backend_config = backend.configuration()
        # TODO : Remove usage of config.defaults when backend.defaults() is updated.
        try:
            backend_default = backend.defaults()
        except (ModelValidationError, AttributeError):
            from collections import namedtuple
            backend_config_defaults = getattr(backend_config, 'defaults', {})
            BackendDefault = namedtuple('BackendDefault', ('qubit_freq_est', 'meas_freq_est'))
            backend_default = BackendDefault(
                qubit_freq_est=backend_config_defaults.get('qubit_freq_est'),
                meas_freq_est=backend_config_defaults.get('meas_freq_est')
            )

    meas_map = meas_map or getattr(backend_config, 'meas_map', None)
    memory_slots = memory_slots or getattr(backend_config, 'memory_slots', None)
    rep_time = rep_time or getattr(backend_config, 'rep_times', None)
    if isinstance(rep_time, list):
        rep_time = rep_time[-1]

    parameter_binds = parameter_binds or []

    # add default empty lo config
    schedule_los = schedule_los or []
    if isinstance(schedule_los, (LoConfig, dict)):
        schedule_los = [schedule_los]

    # Convert to LoConfig if lo configuration supplied as dictionary
    schedule_los = [lo_config if isinstance(lo_config, LoConfig) else LoConfig(lo_config)
                    for lo_config in schedule_los]

    qubit_lo_freq = qubit_lo_freq or getattr(backend_default, 'qubit_freq_est', [])
    meas_lo_freq = meas_lo_freq or getattr(backend_default, 'meas_freq_est', [])

    qubit_lo_range = qubit_lo_range or getattr(backend_config, 'qubit_lo_range', [])
    meas_lo_range = meas_lo_range or getattr(backend_config, 'meas_lo_range', [])
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

    # create run configuration and populate
    run_config_dict = dict(shots=shots,
                           memory=memory,
                           max_credits=max_credits,
                           seed_simulator=seed_simulator,
                           qubit_lo_freq=qubit_lo_freq,
                           meas_lo_freq=meas_lo_freq,
                           qubit_lo_range=qubit_lo_range,
                           meas_lo_range=meas_lo_range,
                           schedule_los=schedule_los,
                           meas_level=meas_level,
                           meas_return=meas_return,
                           meas_map=meas_map,
                           memory_slots=memory_slots,
                           memory_slot_size=memory_slot_size,
                           rep_time=rep_time,
                           parameter_binds=parameter_binds,
                           **run_config)
    run_config = RunConfig(**{k: v for k, v in run_config_dict.items() if v is not None})

    return qobj_id, qobj_header, run_config


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
        unique_parameters = set(param
                                for param_list in all_bind_parameters + all_circuit_parameters
                                for param in param_list)

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
