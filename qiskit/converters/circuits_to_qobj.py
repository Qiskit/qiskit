# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper function for converting a list of circuits to a qobj"""
from copy import deepcopy
import uuid

from qiskit.qobj import Qobj, QobjConfig, QobjExperiment, QobjItem, QobjHeader
from qiskit.circuit.quantumcircuit import QuantumCircuit


def circuits_to_qobj(circuits, backend_name, config=None, shots=1024,
                     max_credits=10, qobj_id=None, basis_gates=None, coupling_map=None,
                     seed=None, memory=False):
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
        memory (bool): if True, per-shot measurement bitstrings are returned as well

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
                        'memory_slots': 0,
                        'memory': memory})

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
    # TODO: removed the DAG from this function
    from qiskit.converters import circuit_to_dag
    from qiskit.unroll import DagUnroller, JsonBackend
    dag = circuit_to_dag(circuit)
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
