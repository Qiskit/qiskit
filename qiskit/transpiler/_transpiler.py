# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tools for compiling a batch of quantum circuits."""
from copy import deepcopy
import logging
import uuid

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs

from qiskit.transpiler._transpilererror import TranspilerError
from qiskit._qiskiterror import QISKitError
from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.unroll import DagUnroller, DAGBackend, JsonBackend
from qiskit.mapper import (Coupling, optimize_1q_gates, coupling_list2dict, swap_mapper,
                           cx_cancellation, direction_mapper,
                           remove_last_measurements, return_last_measurements)
from qiskit.qobj import Qobj, QobjConfig, QobjExperiment, QobjItem, QobjHeader

logger = logging.getLogger(__name__)


# pylint: disable=redefined-builtin
def compile(circuits, backend,
            config=None, basis_gates=None, coupling_map=None, initial_layout=None,
            shots=1024, max_credits=10, seed=None, qobj_id=None, hpc=None,
            pass_manager=None):
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
        qobj_id (int): identifier for the generated qobj
        hpc (dict): HPC simulator parameters
        pass_manager (PassManager): a pass_manager for the transpiler stage

    Returns:
        Qobj: the Qobj to be run on the backends

    Raises:
        TranspilerError: in case of bad compile options, e.g. the hpc options.
    """
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]

    # FIXME: THIS NEEDS TO BE CLEANED UP -- some things to decide for list of circuits:
    # We have bugs if 3 is not true
    # 1. do all circuits have same coupling map?
    # 2. do all circuit have the same basis set?
    # 3. do they all have same registers etc?
    backend_conf = backend.configuration
    backend_name = backend_conf['name']
    # Check for valid parameters for the experiments.
    if hpc is not None and \
            not all(key in hpc for key in ('multi_shot_optimization', 'omp_num_threads')):
        raise TranspilerError('Unknown HPC parameter format!')
    basis_gates = basis_gates or backend_conf['basis_gates']
    coupling_map = coupling_map or backend_conf['coupling_map']
    num_qubits_first = sum((len(qreg) for qreg in circuits[0].get_qregs().values()))
    # FIXME: THIS IS A BUG if the second circuit has more than 1 qubit
    if num_qubits_first == 1 or coupling_map == "all-to-all":
        coupling_map = None

    # step 1: Making the list of dag circuits
    dags = _circuits_2_dags(circuits)

    # step 2: Transpile all the dags
    # Work-around for compiling multiple circuits with different qreg names.
    # Should later make it so that the initial_layout can be a list of layouts.
    list_layout = []    
    _initial_layout = initial_layout.copy() if initial_layout is not None else None
    for i, dag in enumerate(dags):
        # pick a good initial layout if coupling_map is not already satisfied
        # otherwise keep it as q[i]->q[i]. TODO: move this inside mapper pass.
        if (initial_layout is None and not backend.configuration['simulator']
                and not _matches_coupling_map(dag, coupling_map)):
            _initial_layout = _pick_best_layout(dag, backend)

        dags[i], final_layout = transpile(
            dag,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            initial_layout=_initial_layout,
            get_layout=True,
            seed=seed,
            pass_manager=pass_manager)
        list_layout.append([[k, v] for k, v in final_layout.items()] if final_layout else None)

    # step 3: Making a qobj
    # FIXME: Things to go: circuits (needs additions to dag)
    # TODO: we are not keeping many in qobj in future so we remove then
    qobj = _dags_2_qobj(dags, circuits, backend_name=backend_name, list_layout=list_layout,
                        config=config, shots=shots, max_credits=max_credits,
                        qobj_id=qobj_id, basis_gates=basis_gates,
                        coupling_map=coupling_map, seed=seed)

    return qobj


def _circuits_2_dags(circuits):
    """Convert a list of circuits into a list of dags.

    Args:
        circuits (list[QuantumCircuit]): circuit to compile

    Returns:
        list[DAGCircuit]: the dag representation of the circuits
        to be used in the transpiler
    """
    dags = []
    for circuit in circuits:
        dag = DAGCircuit.fromQuantumCircuit(circuit)
        dags.append(dag)
    return dags


def _dags_2_qobj(dags, circuits, backend_name, list_layout=None, config=None, shots=None,
                 max_credits=None, qobj_id=None, basis_gates=None, coupling_map=None,
                 seed=None):
    """Convert a list of dags into a qobj.

    Args:
        dags (list[DAGCircuit]): dags to compile

    Returns:
        Qobj: the qobj to run on the backend
    """
    # Step 1: create the Qobj, with empty experiments.
    # Copy the configuration: the values in `config` have prefernce
    qobj_config = deepcopy(config or {})
    # TODO: "memory_slots" is required by the qobj schema in the top-level
    # qobj.config, and is user-defined. At the moment is set to the maximum
    # number of *register* slots for the circuits, in order to have `measure`
    # behave properly until the transition is over; and each circuit stores
    # its memory_slots in its configuration.
    qobj_config.update({'shots': shots,
                        'max_credits': max_credits,
                        'memory_slots': 0})

    qobj = Qobj(id=qobj_id or str(uuid.uuid4()),
                config=QobjConfig(**qobj_config),
                experiments=[],
                header=QobjHeader(backend_name=backend_name))
    if seed:
        qobj.config.seed = seed

    # change to standard for if dags gets circuit field
    for i, dag in enumerate(dags):

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
            'layout': list_layout[i],
            'memory_slots': sum(register.size for register
                                in circuits[i].get_cregs().values())})
        experiment.config = QobjItem(**experiment_config)

        # set eval_symbols=True to evaluate each symbolic expression
        # TODO: after transition to qobj, we can drop this
        experiment.header.compiled_circuit_qasm = dag.qasm(qeflag=True, eval_symbols=True)
        # Step 3c: add the Experiment to the Qobj
        qobj.experiments.append(experiment)

    # Update the `memory_slots` value.
    # TODO: remove when `memory_slots` can be provided by the user.
    qobj.config.memory_slots = max(experiment.config.memory_slots for
                                   experiment in qobj.experiments)

    return qobj


def transpile(dag, basis_gates='u1,u2,u3,cx,id', coupling_map=None,
              initial_layout=None, get_layout=False,
              format='dag', seed=None, pass_manager=None):
    """Transform a dag circuit into another dag circuit (transpile), through
    consecutive passes on the dag.

    Args:
        dag (DAGCircuit): dag circuit to transform via transpilation
        basis_gates (str): a comma seperated string for the target basis gates
        coupling_map (list): A graph of coupling::

            [
             [control0(int), target0(int)],
             [control1(int), target1(int)],
            ]

            eg. [[0, 2], [1, 2], [1, 3], [3, 4]}

        initial_layout (dict): A mapping of qubit to qubit::

                              {
                                ("q", start(int)): ("q", final(int)),
                                ...
                              }
                              eg.
                              {
                                ("q", 0): ("q", 0),
                                ("q", 1): ("q", 1),
                                ("q", 2): ("q", 2),
                                ("q", 3): ("q", 3)
                              }
        get_layout (bool): flag for returning the final layout after mapping
        format (str): The target format of the compilation:
            {'dag', 'json', 'qasm'}
        seed (int): random seed for the swap mapper
        pass_manager (PassManager): pass manager instance for the tranpilation process
            If None, a default set of passes are run.
            Otherwise, the passes defined in it will run.
            If contains no passes in it, no dag transformations occur.

    Returns:
        object: If get_layout == False, the compiled circuit in the specified
            format. If get_layout == True, a tuple is returned, with the
            second element being the final layout.

    Raises:
        TranspilerError: if the format is not valid.
    """
    # TODO: `basis_gates` will be removed after we have the unroller pass.
    # TODO: `coupling_map`, `initial_layout`, `get_layout`, `seed` removed after mapper pass.
    final_layout = None

    if pass_manager:
        # run the passes specified by the pass manager
        for pass_ in pass_manager.passes():
            pass_.run(dag)
    else:
        # default set of passes
        # TODO: move each step here to a pass, and use a default passmanager below
        basis = basis_gates.split(',') if basis_gates else []
        dag_unroller = DagUnroller(dag, DAGBackend(basis))
        dag = dag_unroller.expand_gates()
        # if a coupling map is given compile to the map
        if coupling_map:
            logger.info("pre-mapping properties: %s",
                        dag.property_summary())
            # Insert swap gates
            coupling = Coupling(coupling_list2dict(coupling_map))
            removed_meas = remove_last_measurements(dag)
            logger.info("measurements moved: %s", removed_meas)
            logger.info("initial layout: %s", initial_layout)
            dag, final_layout, last_layout = swap_mapper(
                    dag, coupling, initial_layout, trials=20, seed=seed)
            logger.info("final layout: %s", final_layout)
            # Expand swaps
            dag_unroller = DagUnroller(dag, DAGBackend(basis))
            dag = dag_unroller.expand_gates()
            # Change cx directions
            dag = direction_mapper(dag, coupling)
            # Simplify cx gates
            cx_cancellation(dag)
            # Simplify single qubit gates
            dag = optimize_1q_gates(dag)
            return_last_measurements(dag, removed_meas,
                                     last_layout)
            logger.info("post-mapping properties: %s",
                        dag.property_summary())

    # choose output format
    # TODO: do we need all of these formats, or just the dag?
    if format == 'dag':
        compiled_circuit = dag
    elif format == 'json':
        # FIXME: JsonBackend is wrongly taking an ordered dict as basis, not list
        dag_unroller = DagUnroller(dag, JsonBackend(dag.basis))
        compiled_circuit = dag_unroller.execute()
    elif format == 'qasm':
        compiled_circuit = dag.qasm()
    else:
        raise TranspilerError('unrecognized circuit format')

    if get_layout:
        return compiled_circuit, final_layout
    return compiled_circuit


def _best_subset(backend, n_qubits):
    """Computes the qubit mapping with the best
    connectivity.

    Parameters:
        backend (Qiskit.BaseBackend): A QISKit backend instance.
        n_qubits (int): Number of subset qubits to consider.

    Returns:
        ndarray: Array of qubits to use for best
                connectivity mapping.

    Raises:
        QISKitError: Wrong number of qubits given.
    """
    if n_qubits == 1:
        return np.array([0])
    elif n_qubits <= 0:
        raise QISKitError('Number of qubits <= 0.')

    device_qubits = backend.configuration['n_qubits']
    if n_qubits > device_qubits:
        raise QISKitError('Number of qubits greater than device.')

    cmap = np.asarray(backend.configuration['coupling_map'])
    data = np.ones_like(cmap[:, 0])
    sp_cmap = sp.coo_matrix((data, (cmap[:, 0], cmap[:, 1])),
                            shape=(device_qubits, device_qubits)).tocsr()
    best = 0
    best_map = None
    # do bfs with each node as starting point
    for k in range(sp_cmap.shape[0]):
        bfs = cs.breadth_first_order(sp_cmap, i_start=k, directed=False,
                                     return_predecessors=False)

        connection_count = 0
        for i in range(n_qubits):
            node_idx = bfs[i]
            for j in range(sp_cmap.indptr[node_idx],
                           sp_cmap.indptr[node_idx + 1]):
                node = sp_cmap.indices[j]
                for counter in range(n_qubits):
                    if node == bfs[counter]:
                        connection_count += 1
                        break

        if connection_count > best:
            best = connection_count
            best_map = bfs[0:n_qubits]
    return best_map


def _matches_coupling_map(dag, coupling_map):
    """Iterate over circuit gates to check if all multi-qubit couplings
    match the qubit coupling graph in the backend.

    Parameters:
            dag (DAGCircuit): DAG representation of circuit.
            coupling_map (list): Backend coupling map, represented as an adjacency list.

    Returns:
            bool: True if all gates readily fit the backend coupling graph.
                  False if there's at least one gate that uses multiple qubits
                  which does not match the backend couplings.
    """
    match = True
    for _, data in dag.multi_graph.nodes(data=True):
        if data['type'] == 'op':
            gate_map = [qr[1] for qr in data['qargs']]
            if gate_map not in coupling_map:
                match = False
                break
    return match


def _pick_best_layout(dag, backend):
    """Pick a convenient layout depending on the best matching qubit connectivity

    Parameters:
        dag (DAGCircuit): DAG representation of circuit.
        backend (BaseBackend) : The backend with the coupling_map for searching

    Returns:
        dict: A special ordered initial_layout

    """
    num_qubits = sum(dag.qregs.values())            
    best_sub = _best_subset(backend, num_qubits)
    layout = {}
    map_iter = 0
    for key, value in qregs.items():
        for i in range(value):
            layout[(key, i)] = ('q', best_sub[map_iter])
            map_iter += 1
    return layout
