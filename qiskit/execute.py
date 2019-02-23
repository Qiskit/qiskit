# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper module for simplified Qiskit usage."""
import warnings
import logging

from qiskit import transpiler
from qiskit.converters import circuits_to_qobj
from qiskit.compiler import RunConfig
from qiskit.qobj import QobjHeader
from qiskit.mapper import Layout
from qiskit.tools.compiler import compile


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

    qobj = compile(circuits, backend,
                   config, basis_gates, coupling_map, initial_layout,
                   shots, max_credits, seed, qobj_id, seed_mapper,
                   pass_manager, memory)

    return backend.run(qobj, **kwargs)
