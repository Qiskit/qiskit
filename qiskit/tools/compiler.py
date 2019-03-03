# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper module for simplified Qiskit usage. THIS WILL BE REMOVED IN AFTER 0.8."""
import warnings
import logging

from qiskit.compiler import assemble_qobj, RunConfig, TranspileConfig
from qiskit.execute_wrapper import execute_circuits
from qiskit import transpiler
from qiskit.mapper import Layout

logger = logging.getLogger(__name__)


# pylint: disable=redefined-builtin
def compile(circuits, backend,
            config=None, basis_gates=None, coupling_map=None, initial_layout=None,
            shots=1024, max_credits=10, seed=None, qobj_id=None, seed_mapper=None,
            pass_manager=None, memory=False):
    """Compile a list of circuits into a qobj.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]): circuits to compile
        backend (BaseBackend): a backend to compile for
        config (dict): dictionary of parameters (e.g. noise) used by runner
        basis_gates (list[str]): list of basis gates names supported by the
            target. Default: ['u1','u2','u3','cx','id']
        coupling_map (list): coupling map (perhaps custom) to target in mapping
        initial_layout (list): initial layout of qubits in mapping
        shots (int): number of repetitions of each circuit, for sampling
        max_credits (int): maximum credits to use
        seed (int): random seed for simulators
        seed_mapper (int): random seed for swapper mapper
        qobj_id (int): identifier for the generated qobj
        pass_manager (PassManager): a pass manger for the transpiler pipeline
        memory (bool): if True, per-shot measurement bitstrings are returned as well

    Returns:
        Qobj: the qobj to be run on the backends

    Raises:
        QiskitError: if the desired options are not supported by backend
    """
    warnings.warn('compile is replace with in 0.8 with synthesis and assemble',
                  DeprecationWarning)

    run_config = RunConfig()

    if config:
        warnings.warn('config is not used anymore. Set all configs in '
                      'run_config.', DeprecationWarning)
    if shots:
        run_config.shots = shots
    if max_credits:
        run_config.max_credits = max_credits
    if seed:
        run_config.seed = seed
    if memory:
        run_config.memory = memory

    if initial_layout is not None and not isinstance(initial_layout, Layout):
        initial_layout = Layout(initial_layout)

    new_circuits = transpiler.transpile(circuits, backend, basis_gates, coupling_map,
                                        initial_layout, seed_mapper, pass_manager)

    qobj = assemble_qobj(new_circuits, user_qobj_header=None, run_config=run_config,
                         qobj_id=qobj_id)

    return qobj


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

    warnings.warn('execute is replace with run_circuits in 0.8', DeprecationWarning)

    transpile_config = TranspileConfig()
    run_config = RunConfig()

    if config:
        warnings.warn('config is not used anymore. Set all configs in '
                      'run_config.', DeprecationWarning)
    if qobj_id:
        warnings.warn('qobj_id is not used anymore', DeprecationWarning)
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
        warnings.warn('pass_manager if using pass_manager please dont use execute',
                      DeprecationWarning)

    job = execute_circuits(circuits, backend, user_qobj_header=None,
                           run_config=run_config,
                           transpile_config=transpile_config, **kwargs)

    return job
