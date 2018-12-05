# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper module for simplified Qiskit usage."""
import warnings
import logging

from qiskit import transpiler
from qiskit.transpiler._passmanager import PassManager
from qiskit.converters import circuits_to_qobj

logger = logging.getLogger(__name__)


# pylint: disable=redefined-builtin
def compile(circuits, backend,
            config=None, basis_gates=None, coupling_map=None, initial_layout=None,
            shots=1024, max_credits=10, seed=None, qobj_id=None,
            skip_transpiler=False, seed_mapper=None, pass_manager=None):
    """Compile a list of circuits into a qobj.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]): circuits to compile
        backend (BaseBackend): a backend to compile for
        config (dict): dictionary of parameters (e.g. noise) used by runner
        basis_gates (str): comma-separated basis gate set to compile to
        coupling_map (list): coupling map (perhaps custom) to target in mapping
        initial_layout (list): initial layout of qubits in mapping
        shots (int): number of repetitions of each circuit, for sampling
        max_credits (int): maximum credits to use
        seed (int): random seed for simulators
        seed_mapper (int): random seed for swapper mapper
        qobj_id (int): identifier for the generated qobj
        pass_manager (PassManager): a pass manger for the transpiler pipeline
        skip_transpiler (bool): DEPRECATED skip transpiler and create qobj directly

    Returns:
        Qobj: the qobj to be run on the backends
    """
    if skip_transpiler:  # empty pass manager which does nothing
        pass_manager = PassManager()
        warnings.warn('The skip_transpiler option has been deprecated. '
                      'Please pass an empty PassManager() instance instead',
                      DeprecationWarning)

    circuits = transpiler.transpile(circuits, backend, basis_gates, coupling_map, initial_layout,
                                    seed_mapper, pass_manager)

    # step 4: Making a qobj
    qobj = circuits_to_qobj(circuits, backend_name=backend.name(),
                            config=config, shots=shots, max_credits=max_credits,
                            qobj_id=qobj_id, basis_gates=basis_gates,
                            coupling_map=coupling_map, seed=seed)

    return qobj


def execute(circuits, backend, config=None, basis_gates=None, coupling_map=None,
            initial_layout=None, shots=1024, max_credits=10, seed=None,
            qobj_id=None, skip_transpiler=False, seed_mapper=None, pass_manager=None,
            **kwargs):
    """Executes a set of circuits.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]): circuits to execute
        backend (BaseBackend): a backend to execute the circuits on
        config (dict): dictionary of parameters (e.g. noise) used by runner
        basis_gates (str): comma-separated basis gate set to compile to
        coupling_map (list): coupling map (perhaps custom) to target in mapping
        initial_layout (list): initial layout of qubits in mapping
        shots (int): number of repetitions of each circuit, for sampling
        max_credits (int): maximum credits to use
        seed (int): random seed for simulators
        seed_mapper (int): random seed for swapper mapper
        qobj_id (int): identifier for the generated qobj
        pass_manager (PassManager): a pass manger for the transpiler pipeline
        skip_transpiler (bool): DEPRECATED skip transpiler and create qobj directly
        kwargs: extra arguments used by AER for running configurable backends.
                Refer to the backend documentation for details on these arguments

    Returns:
        BaseJob: returns job instance derived from BaseJob
    """
    if skip_transpiler:  # empty pass manager which does nothing
        pass_manager = PassManager()
        warnings.warn('The skip_transpiler option has been deprecated. '
                      'Please pass an empty PassManager() instance instead',
                      DeprecationWarning)

    qobj = compile(circuits, backend,
                   config, basis_gates, coupling_map, initial_layout,
                   shots, max_credits, seed, qobj_id,
                   skip_transpiler, seed_mapper, pass_manager)

    return backend.run(qobj, **kwargs)
