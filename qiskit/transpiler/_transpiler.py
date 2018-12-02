# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tools for compiling a batch of quantum circuits."""
import logging
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs

from qiskit.transpiler._transpilererror import TranspilerError
from qiskit._qiskiterror import QiskitError
from qiskit.dagcircuit import DAGCircuit
from qiskit import _quantumcircuit, _quantumregister
from qiskit.unrollers import _dagunroller
from qiskit.unrollers import _dagbackend
from qiskit.mapper import (Coupling, optimize_1q_gates, coupling_list2dict, swap_mapper,
                           cx_cancellation, direction_mapper,
                           remove_last_measurements, return_last_measurements)
from qiskit._pubsub import Publisher, Subscriber
from ._parallel import parallel_map


logger = logging.getLogger(__name__)


def transpile(circuits, backend, basis_gates=None, coupling_map=None, initial_layout=None,
              seed_mapper=None, hpc=None, pass_manager=None):
    """transpile a list of circuits into a dags.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]): circuits to compile
        backend (BaseBackend): a backend to compile for
        basis_gates (str): comma-separated basis gate set to compile to
        coupling_map (list): coupling map (perhaps custom) to target in mapping
        initial_layout (list): initial layout of qubits in mapping
        seed_mapper (int): random seed for the swap_mapper
        hpc (dict): HPC simulator parameters
        pass_manager (PassManager): a pass_manager for the transpiler stage

    Returns:
        dags: a list of dags.

    Raises:
        TranspilerError: in case of bad compile options, e.g. the hpc options.
    """
    if isinstance(circuits, _quantumcircuit.QuantumCircuit):
        circuits = [circuits]

    # FIXME: THIS NEEDS TO BE CLEANED UP -- some things to decide for list of circuits:
    # 1. do all circuits have same coupling map?
    # 2. do all circuit have the same basis set?
    # 3. do they all have same registers etc?
    # Check for valid parameters for the experiments.
    if hpc is not None and \
            not all(key in hpc for key in ('multi_shot_optimization', 'omp_num_threads')):
        raise TranspilerError('Unknown HPC parameter format!')
    basis_gates = basis_gates or ','.join(backend.configuration().basis_gates)
    coupling_map = coupling_map or getattr(backend.configuration(),
                                           'coupling_map', None)

    # step 1: Making the list of dag circuits
    dags = _circuits_2_dags(circuits)

    # step 2: Transpile all the dags

    # FIXME: Work-around for transpiling multiple circuits with different qreg names.
    # Make compile take a list of initial_layouts.
    _initial_layout = initial_layout

    # Pick a good initial layout if coupling_map is not already satisfied
    # otherwise keep it as q[i]->q[i].
    # TODO: move this inside mapper pass.
    initial_layouts = []
    for dag in dags:
        if (initial_layout is None and not backend.configuration().simulator
                and not _matches_coupling_map(dag, coupling_map)):
            _initial_layout = _pick_best_layout(dag, backend)
        initial_layouts.append(_initial_layout)

    dags = _dags_2_dags(dags, basis_gates=basis_gates, coupling_map=coupling_map,
                        initial_layouts=initial_layouts, seed_mapper=seed_mapper,
                        pass_manager=pass_manager)

    # TODO: change it to circuits
    # TODO: make it parallel
    return dags


def _circuits_2_dags(circuits):
    """Convert a list of circuits into a list of dags.

    Args:
        circuits (list[QuantumCircuit]): circuit to compile

    Returns:
        list[DAGCircuit]: the dag representation of the circuits
        to be used in the transpiler
    """
    dags = parallel_map(DAGCircuit.fromQuantumCircuit, circuits)
    return dags


def _dags_2_dags(dags, basis_gates='u1,u2,u3,cx,id', coupling_map=None,
                 initial_layouts=None, seed_mapper=None, pass_manager=None):
    """Transform multiple dags through a sequence of passes.

    Args:
        dags (list[DAGCircuit]): dag circuits to transform
        basis_gates (str): a comma separated string for the target basis gates
        coupling_map (list): A graph of coupling
        initial_layouts (list[dict]): A mapping of qubit to qubit for each dag
        seed_mapper (int): random seed_mapper for the swap mapper
        pass_manager (PassManager): pass manager instance for the transpilation process
            If None, a default set of passes are run.
            Otherwise, the passes defined in it will run.
            If contains no passes in it, no dag transformations occur.

    Returns:
        list[DAGCircuit]: the dag circuits after going through transpilation

    Events:
        terra.transpiler.transpile_dag.start: When the transpilation of the dags is about to start
        terra.transpiler.transpile_dag.done: When one of the dags has finished it's transpilation
        terra.transpiler.transpile_dag.finish: When all the dags have finished transpiling
    """

    def _emit_start(num_dags):
        """ Emit a dag transpilation start event
        Arg:
            num_dags: Number of dags to be transpiled"""
        Publisher().publish("terra.transpiler.transpile_dag.start", num_dags)
    Subscriber().subscribe("terra.transpiler.parallel.start", _emit_start)

    def _emit_done(progress):
        """ Emit a dag transpilation done event
        Arg:
            progress: The dag number that just has finshed transpile"""
        Publisher().publish("terra.transpiler.transpile_dag.done", progress)
    Subscriber().subscribe("terra.transpiler.parallel.done", _emit_done)

    def _emit_finish():
        """ Emit a dag transpilation finish event
        Arg:
            progress: The dag number that just has finshed transpile"""
        Publisher().publish("terra.transpiler.transpile_dag.finish")
    Subscriber().subscribe("terra.transpiler.parallel.finish", _emit_finish)

    dags_layouts = list(zip(dags, initial_layouts))
    final_dags = parallel_map(_transpile_dags_parallel, dags_layouts,
                              task_kwargs={'basis_gates': basis_gates,
                                           'coupling_map': coupling_map,
                                           'seed_mapper': seed_mapper,
                                           'pass_manager': pass_manager})
    return final_dags


def _transpile_dags_parallel(dag_layout_tuple, basis_gates='u1,u2,u3,cx,id',
                             coupling_map=None, seed_mapper=None, pass_manager=None):
    """Helper function for transpiling in parallel (if available).

    Args:
        dag_layout_tuple (tuple): Tuples of dags and their initial_layouts
        basis_gates (str): a comma separated string for the target basis gates
        coupling_map (list): A graph of coupling
        seed_mapper (int): random seed_mapper for the swap mapper
        pass_manager (PassManager): pass manager instance for the transpilation process
            If None, a default set of passes are run.
            Otherwise, the passes defined in it will run.
            If contains no passes in it, no dag transformations occur.
    Returns:
        DAGCircuit: DAG circuit after going through transpilation.
    """
    final_dag, final_layout = transpile_dag(
        dag_layout_tuple[0],
        basis_gates=basis_gates,
        coupling_map=coupling_map,
        initial_layout=dag_layout_tuple[1],
        get_layout=True,
        seed_mapper=seed_mapper,
        pass_manager=pass_manager)
    final_dag.layout = [[k, v]
                        for k, v in final_layout.items()] if final_layout else None
    return final_dag


# pylint: disable=redefined-builtin
def transpile_dag(dag, basis_gates='u1,u2,u3,cx,id', coupling_map=None,
                  initial_layout=None, get_layout=False,
                  format='dag', seed_mapper=None, pass_manager=None):
    """Transform a dag circuit into another dag circuit (transpile), through
    consecutive passes on the dag.

    Args:
        dag (DAGCircuit): dag circuit to transform via transpilation
        basis_gates (str): a comma separated string for the target basis gates
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
        format (str): DEPRECATED The target format of the compilation: {'dag', 'json', 'qasm'}
        seed_mapper (int): random seed_mapper for the swap mapper
        pass_manager (PassManager): pass manager instance for the transpilation process
            If None, a default set of passes are run.
            Otherwise, the passes defined in it will run.
            If contains no passes in it, no dag transformations occur.

    Returns:
        DAGCircuit: transformed dag
        DAGCircuit, dict: transformed dag along with the final layout on backend qubits

    Raises:
        TranspilerError: if the format is not valid.
    """
    # TODO: `basis_gates` will be removed after we have the unroller pass.
    # TODO: `coupling_map`, `initial_layout`, `get_layout`, `seed_mapper` removed after mapper pass.

    # TODO: move this to the mapper pass
    num_qubits = sum([qreg.size for qreg in dag.qregs.values()])
    if num_qubits == 1 or coupling_map == "all-to-all":
        coupling_map = None

    final_layout = None

    if pass_manager:
        # run the passes specified by the pass manager
        # TODO return the property set too. See #1086
        dag = pass_manager.run_passes(dag)
    else:
        # default set of passes
        # TODO: move each step here to a pass, and use a default passmanager below
        basis = basis_gates.split(',') if basis_gates else []
        dag_unroller = _dagunroller.DagUnroller(
            dag, _dagbackend.DAGBackend(basis))
        dag = dag_unroller.expand_gates()
        # if a coupling map is given compile to the map
        if coupling_map:
            logger.info("pre-mapping properties: %s",
                        dag.properties())
            # Insert swap gates
            coupling = Coupling(coupling_list2dict(coupling_map))
            removed_meas = remove_last_measurements(dag)
            logger.info("measurements moved: %s", removed_meas)
            logger.info("initial layout: %s", initial_layout)
            dag, final_layout, last_layout = swap_mapper(
                dag, coupling, initial_layout, trials=20, seed=seed_mapper)
            logger.info("final layout: %s", final_layout)
            # Expand swaps
            dag_unroller = _dagunroller.DagUnroller(
                dag, _dagbackend.DAGBackend(basis))
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
                        dag.properties())

    if format != 'dag':
        warnings.warn("transpiler no longer supports different formats. "
                      "only dag to dag transformations are supported.",
                      DeprecationWarning)

    if get_layout:
        return dag, final_layout
    return dag


def _best_subset(backend, n_qubits):
    """Computes the qubit mapping with the best
    connectivity.

    Parameters:
        backend (BaseBackend): A Qiskit backend instance.
        n_qubits (int): Number of subset qubits to consider.

    Returns:
        ndarray: Array of qubits to use for best
                connectivity mapping.

    Raises:
        QiskitError: Wrong number of qubits given.
    """
    if n_qubits == 1:
        return np.array([0])
    elif n_qubits <= 0:
        raise QiskitError('Number of qubits <= 0.')

    device_qubits = backend.configuration().n_qubits
    if n_qubits > device_qubits:
        raise QiskitError('Number of qubits greater than device.')

    cmap = np.asarray(getattr(backend.configuration(), 'coupling_map', None))
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
            if len(gate_map) > 1:
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
    num_qubits = sum([qreg.size for qreg in dag.qregs.values()])
    best_sub = _best_subset(backend, num_qubits)
    layout = {}
    map_iter = 0
    device_qubits = backend.configuration().n_qubits
    q = _quantumregister.QuantumRegister(device_qubits, 'q')
    for qreg in dag.qregs.values():
        for i in range(qreg.size):
            layout[(qreg.name, i)] = (q, int(best_sub[map_iter]))
            map_iter += 1
    return layout
