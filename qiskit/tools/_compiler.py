# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper module for simplified Qiskit usage."""
from copy import deepcopy
import uuid
import logging

from qiskit import transpiler
from qiskit.transpiler._passmanager import PassManager
from qiskit.qobj import Qobj, QobjConfig, QobjExperiment, QobjItem, QobjHeader
from qiskit.unroll import DagUnroller, JsonBackend
from qiskit.transpiler._parallel import parallel_map

logger = logging.getLogger(__name__)


# pylint: disable=redefined-builtin
def compile(circuits, backend,
            config=None, basis_gates=None, coupling_map=None, initial_layout=None,
            shots=1024, max_credits=10, seed=None, qobj_id=None, hpc=None,
            skip_transpiler=False, seed_mapper=None):
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
        hpc (dict): HPC simulator parameters
        skip_transpiler (bool): skip most of the compile steps and produce qobj directly

    Returns:
        Qobj: the qobj to be run on the backends

    Raises:
        TranspilerError: in case of bad compile options, e.g. the hpc options.

    """

    pass_manager = None  # default pass manager which executes predetermined passes
    if skip_transpiler:  # empty pass manager which does nothing
        pass_manager = PassManager()

    dags = transpiler.transpile(circuits, backend, basis_gates, coupling_map, initial_layout,
                                seed_mapper, hpc, pass_manager)

    # step 3: Making a qobj
    qobj_standard = dags_2_qobj(dags, backend_name=backend.name(),
                                config=config, shots=shots, max_credits=max_credits,
                                qobj_id=qobj_id, basis_gates=basis_gates,
                                coupling_map=coupling_map, seed=seed)

    return qobj_standard


def dags_2_qobj(dags, backend_name, config=None, shots=None,
                max_credits=None, qobj_id=None, basis_gates=None, coupling_map=None,
                seed=None):
    """Convert a list of dags into a qobj.

    Args:
        dags (list[DAGCircuit]): dags to compile
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
    # TODO: "memory_slots" is required by the qobj schema in the top-level
    # qobj.config, and is user-defined. At the moment is set to the maximum
    # number of *register* slots for the circuits, in order to have `measure`
    # behave properly until the transition is over; and each circuit stores
    # its memory_slots in its configuration.
    qobj_config.update({'shots': shots,
                        'max_credits': max_credits,
                        'memory_slots': 0})

    qobj = Qobj(qobj_id=qobj_id or str(uuid.uuid4()),
                config=QobjConfig(**qobj_config),
                experiments=[],
                header=QobjHeader(backend_name=backend_name))
    if seed:
        qobj.config.seed = seed

    qobj.experiments = parallel_map(_dags_2_qobj_parallel, dags,
                                    task_kwargs={'basis_gates': basis_gates,
                                                 'config': config,
                                                 'coupling_map': coupling_map})

    # Update the `memory_slots` value.
    # TODO: remove when `memory_slots` can be provided by the user.
    qobj.config.memory_slots = max(experiment.config.memory_slots for
                                   experiment in qobj.experiments)

    # Update the `n_qubits` global value.
    # TODO: num_qubits is not part of the qobj specification, but needed
    # for the simulator.
    qobj.config.n_qubits = max(experiment.config.n_qubits for
                               experiment in qobj.experiments)

    return qobj


def _dags_2_qobj_parallel(dag, config=None, basis_gates=None, coupling_map=None):
    """Helper function for dags to qobj in parallel (if available).

    Args:
        dag (DAGCircuit): DAG to compile
        config (dict): dictionary of parameters (e.g. noise) used by runner
        basis_gates (list[str])): basis gates for the experiment
        coupling_map (list): coupling map (perhaps custom) to target in mapping

    Returns:
        Qobj: Qobj to be run on the backends
    """
    json_circuit = DagUnroller(dag, JsonBackend(dag.basis)).execute()
    # Step 3a: create the Experiment based on json_circuit
    experiment = QobjExperiment.from_dict(json_circuit)
    # Step 3b: populate the Experiment configuration and header
    experiment.header.name = dag.name
    # TODO: place in header or config?
    experiment_config = deepcopy(config or {})
    experiment_config.update({
        'coupling_map': coupling_map,
        'basis_gates': basis_gates,
        'layout': [[[i[0][0].name, i[0][1]], [i[1][0].name, i[1][1]]]
                   for i in dag.layout] if dag.layout else [],
        'memory_slots': sum([creg.size for creg in dag.cregs.values()]),
        # TODO: `n_qubits` is not part of the qobj spec, but needed for the simulator.
        'n_qubits': sum([qreg.size for qreg in dag.qregs.values()])
        })
    experiment.config = QobjItem(**experiment_config)

    # set eval_symbols=True to evaluate each symbolic expression
    # TODO: after transition to qobj, we can drop this
    experiment.header.compiled_circuit_qasm = dag.qasm(
        qeflag=True, eval_symbols=True)
    # Step 3c: add the Experiment to the Qobj
    return experiment


def execute(circuits, backend,
            config=None, basis_gates=None, coupling_map=None, initial_layout=None,
            shots=1024, max_credits=10, seed=None, qobj_id=None, hpc=None,
            skip_transpiler=False, seed_mapper=None, **kwargs):
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
        hpc (dict): HPC simulator parameters
        skip_transpiler (bool): skip most of the compile steps and produce qobj directly
        kwargs: extra arguments used by AER for runing configurable backends. Refer to the
        backend documentation for details on these arguments

    Returns:
        BaseJob: returns job instance derived from BaseJob
    """
    qobj = compile(circuits, backend,
                   config, basis_gates, coupling_map, initial_layout,
                   shots, max_credits, seed, qobj_id, hpc,
                   skip_transpiler, seed_mapper)
    return backend.run(qobj, **kwargs)
