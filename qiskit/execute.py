# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper module for simplified Qiskit usage.

    This module includes
        execute_circuits: runs a list of quantum circuits.

    In general we recommend using the SDK functions directly. However, to get something
    running quickly we have provider this wrapper module.
"""

import logging
import warnings

from qiskit.compiler import assemble_circuits, transpile
from qiskit.compiler import RunConfig, TranspileConfig
from qiskit.qobj import QobjHeader

logger = logging.getLogger(__name__)


def execute(circuits, backend, config=None, basis_gates=None, coupling_map=None,
            initial_layout=None, shots=1024, max_credits=10, seed=None,
            qobj_id=None, seed_mapper=None, pass_manager=None,
            memory=False, **kwargs):
    """Executes a set of circuits.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]): circuits to execute
        backend (BaseBackend): a backend to execute the circuits on
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
    if seed:
        run_config.seed = seed
    if memory:
        run_config.memory = memory
    if pass_manager:
        warnings.warn('pass_manager in the execute function is deprecated in terra 0.8.',
                      DeprecationWarning)

    job = execute_circuits(circuits, backend, qobj_header=None,
                           run_config=run_config,
                           transpile_config=transpile_config, **kwargs)

    return job


def execute_circuits(circuits, backend, qobj_header=None, run_config=None,
                     transpile_config=None, **kwargs):
    """Executes a list of circuits.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]): circuits to execute
        backend (BaseBackend): a backend to execute the circuits on
        qobj_header (QobjHeader): User input to go in the header
        run_config (RunConfig): Run Configuration
        transpile_config (TranspileConfig): Configurations for the transpiler
        kwargs: extra arguments used by AER for running configurable backends.
                Refer to the backend documentation for details on these arguments

    Returns:
        BaseJob: returns job instance derived from BaseJob
    """

    # HACK TO BE REMOVED when backend is not needed in transpile
    # ------
    transpile_config = transpile_config or TranspileConfig()
    transpile_config.backend = backend
    # ------

    # filling in the header with the backend name the qob was rune on
    qobj_header = qobj_header or QobjHeader()
    qobj_header.backend_name = backend.name()

    # default values
    if not run_config:
        # TODO remove max_credits from the default when it is not required by
        # by the backend.
        run_config = RunConfig(shots=1024, max_credits=10, memory=False)

    # synthesizing the circuits using the transpiler_config
    new_circuits = transpile(circuits, transpile_config=transpile_config)

    # assembling the circuits into a qobj to be run on the backend
    qobj = assemble_circuits(new_circuits, qobj_header=qobj_header,
                             run_config=run_config)

    # executing the circuits on the backend and returning the job
    return backend.run(qobj, **kwargs)
