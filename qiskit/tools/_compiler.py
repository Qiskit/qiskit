# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper module for simplified Qiskit usage."""
from copy import deepcopy
import warnings
import uuid
import logging

from qiskit import transpiler
from qiskit.transpiler._passmanager import PassManager
from qiskit.qobj import Qobj, QobjConfig, QobjExperiment, QobjItem, QobjHeader
from qiskit.unroll import DagUnroller, JsonBackend
from qiskit.dagcircuit import DAGCircuit
from qiskit._quantumcircuit import QuantumCircuit

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


def circuits_to_qobj(circuits, backend_name, config=None, shots=1024,
                     max_credits=10, qobj_id=None, basis_gates=None, coupling_map=None,
                     seed=None):
    """Convert a list of circuits into a qobj.

    Args:
        circuits (list[QuantumCircuits] or QuantumCircuit): circuits to compile
        backend_name (str): name of runner backend
        config (dict): dictionary of parameters (e.g. noise) used by runner
        shots (int): number of repetitions of each circuit, for sampling
        max_credits (int): maximum credits to use
        qobj_id (int): identifier for the generated qobj
        basis_gates (list[str])): basis gates for the experiment
        coupling_map (list): coupling map (perhaps custom) to target in mapping
        seed (int): random seed for simulators

    Returns:
        Qobj: the Qobj to be run on the backends
    """
    # TODO: the following will be removed from qobj and thus removed here:
    # `basis_gates`, `coupling_map`

    # Step 1: create the Qobj, with empty experiments.
    # Copy the configuration: the values in `config` have preference
    qobj_config = deepcopy(config or {})
    qobj_config.update({'shots': shots,
                        'max_credits': max_credits,
                        'memory_slots': 0})

    qobj = Qobj(qobj_id=qobj_id or str(uuid.uuid4()),
                config=QobjConfig(**qobj_config),
                experiments=[],
                header=QobjHeader(backend_name=backend_name))
    if seed:
        qobj.config.seed = seed

    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]

    for circuit in circuits:
        qobj.experiments.append(_circuit_to_experiment(circuit,
                                                       config,
                                                       basis_gates,
                                                       coupling_map))

    # Update the global `memory_slots` and `n_qubits` values.
    qobj.config.memory_slots = max(experiment.config.memory_slots for
                                   experiment in qobj.experiments)

    qobj.config.n_qubits = max(experiment.config.n_qubits for
                               experiment in qobj.experiments)

    return qobj


def _circuit_to_experiment(circuit, config=None, basis_gates=None,
                           coupling_map=None):
    """Helper function for dags to qobj in parallel (if available).

    Args:
        circuit (QuantumCircuit): QuantumCircuit to convert into qobj experiment
        config (dict): dictionary of parameters (e.g. noise) used by runner
        basis_gates (list[str])): basis gates for the experiment
        coupling_map (list): coupling map (perhaps custom) to target in mapping

    Returns:
        Qobj: Qobj to be run on the backends
    """
    # pylint: disable=unused-argument
    #  TODO: if arguments are really unused, consider changing the signature

    dag = DAGCircuit.fromQuantumCircuit(circuit)
    json_circuit = DagUnroller(dag, JsonBackend(dag.basis)).execute()
    # Step 3a: create the Experiment based on json_circuit
    experiment = QobjExperiment.from_dict(json_circuit)
    # Step 3b: populate the Experiment configuration and header
    experiment.header.name = circuit.name
    experiment_config = deepcopy(config or {})
    experiment_config.update({
        'memory_slots': sum([creg.size for creg in dag.cregs.values()]),
        'n_qubits': sum([qreg.size for qreg in dag.qregs.values()])
        })
    experiment.config = QobjItem(**experiment_config)

    # set eval_symbols=True to evaluate each symbolic expression
    # TODO: after transition to qobj, we can drop this
    experiment.header.compiled_circuit_qasm = circuit.qasm()
    # Step 3c: add the Experiment to the Qobj
    return experiment


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
