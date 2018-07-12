# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tools for compiling a batch of quantum circuits."""
import logging
import uuid

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs

from qiskit.qobj._qobj import QobjHeader
from qiskit.transpiler._transpilererror import TranspilerError
from qiskit._qiskiterror import QISKitError
from qiskit._quantumcircuit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.unroll import DagUnroller, DAGBackend, JsonBackend
from qiskit.mapper import (Coupling, optimize_1q_gates, coupling_list2dict, swap_mapper,
                           cx_cancellation, direction_mapper)
from qiskit._gate import Gate
from qiskit.qobj import Qobj, QobjConfig, QobjExperiment, QobjItem

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

    backend_conf = backend.configuration
    backend_name = backend_conf['name']

    # TODO: solve "config" and "shots" arguments
    # TODO: calculate "register_slots"

    # Step 1: create the Qobj, with empty experiments.
    qobj = Qobj(id=qobj_id or str(uuid.uuid4()),
                config=QobjConfig(shots=shots,
                                  max_credits=max_credits,
                                  register_slots=1234),
                experiments=[],
                header=QobjHeader(backend_name=backend_name))
    if seed:
        qobj.config.seed = seed

    # Check for valid parameters for the experiments.
    if hpc is not None and \
            not all(key in hpc for key in ('multi_shot_optimization', 'omp_num_threads')):
        raise TranspilerError('Unknown HPC parameter format!')
    basis_gates = basis_gates or backend_conf['basis_gates']
    coupling_map = coupling_map or backend_conf['coupling_map']

    # Step 2 and 3: transpile and populate the circuits
    for circuit in circuits:
        # TODO: A better solution is to have options to enable/disable optimizations
        num_qubits = sum((len(qreg) for qreg in circuit.get_qregs().values()))
        if num_qubits == 1 or coupling_map == "all-to-all":
            coupling_map = None
        # Step 2a: circuit -> dag
        dag_circuit = DAGCircuit.fromQuantumCircuit(circuit)

        # TODO: move this inside the mapper pass
        # pick a good initial layout if coupling_map is not already satisfied
        # otherwise keep it as q[i]->q[i]
        if (initial_layout is None and
                not backend_conf['simulator'] and
                not _matches_coupling_map(circuit.data, coupling_map)):
            initial_layout = _pick_best_layout(backend, num_qubits, circuit.get_qregs())

        # Step 2b: transpile (dag -> dag)
        dag_circuit, final_layout = transpile(
            dag_circuit,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            initial_layout=initial_layout,
            get_layout=True,
            seed=seed,
            pass_manager=pass_manager)

        # Step 2c: dag -> json
        # the compiled circuit to be run saved as a dag
        # we assume that transpile() has already expanded gates
        # to the target basis, so we just need to generate json
        list_layout = [[k, v] for k, v in final_layout.items()] if final_layout else None

        json_circuit = DagUnroller(dag_circuit, JsonBackend(dag_circuit.basis)).execute()

        # Step 3a: create the Experiment based on json_circuit
        experiment = QobjExperiment.from_dict(json_circuit)
        # Step 3b: populate the Experiment configuration and header
        experiment.header.name = circuit.name
        # TODO: place in header or config?
        experiment.config = QobjItem(coupling_map=coupling_map,
                                     basis_gates=basis_gates,
                                     layout=list_layout)

        # set eval_symbols=True to evaluate each symbolic expression
        # TODO after transition to qobj, we can drop this
        experiment.header.compiled_circuit_qasm = dag_circuit.qasm(
            qeflag=True, eval_symbols=True)

        # Step 3c: add the Experiment to the Qobj
        qobj.experiments.append(experiment)

    return qobj


# pylint: disable=redefined-builtin
def transpile(dag_circuit, basis_gates='u1,u2,u3,cx,id', coupling_map=None,
              initial_layout=None, get_layout=False,
              format='dag', seed=None, pass_manager=None):
    """Transform a dag circuit into another dag circuit (transpile), through
    consecutive passes on the dag.

    Args:
        dag_circuit (DAGCircuit): dag circuit to transform via transpilation
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
        get_layout (bool): flag for returning the layout
        format (str): The target format of the compilation:
            {'dag', 'json', 'qasm'}
        seed (int): random seed for simulators
        pass_manager (PassManager): pass manager instance for the tranpilation process
            If None, a default set of passes are run.
            Otherwise, the passes defined in it will run.
            If contains no passes in it, no dag transformations occur.

    Returns:
        object: If get_layout == False, the compiled circuit in the specified
            format. If get_layout == True, a tuple is returned, with the
            second element being the layout.

    Raises:
        TranspilerError: if the format is not valid.
    """
    final_layout = None

    if pass_manager:
        # run the passes specified by the pass manager
        for pass_ in pass_manager.passes():
            pass_.run(dag_circuit)
    else:
        # default set of passes
        # TODO: move each step here to a pass, and use a default passmanager below
        basis = basis_gates.split(',') if basis_gates else []
        dag_unroller = DagUnroller(dag_circuit, DAGBackend(basis))
        dag_circuit = dag_unroller.expand_gates()
        # if a coupling map is given compile to the map
        if coupling_map:
            logger.info("pre-mapping properties: %s",
                        dag_circuit.property_summary())
            # Insert swap gates
            coupling = Coupling(coupling_list2dict(coupling_map))
            logger.info("initial layout: %s", initial_layout)
            dag_circuit, final_layout = swap_mapper(
                dag_circuit, coupling, initial_layout, trials=20, seed=seed)
            logger.info("final layout: %s", final_layout)
            # Expand swaps
            dag_unroller = DagUnroller(dag_circuit, DAGBackend(basis))
            dag_circuit = dag_unroller.expand_gates()
            # Change cx directions
            dag_circuit = direction_mapper(dag_circuit, coupling)
            # Simplify cx gates
            cx_cancellation(dag_circuit)
            # Simplify single qubit gates
            dag_circuit = optimize_1q_gates(dag_circuit)
            logger.info("post-mapping properties: %s",
                        dag_circuit.property_summary())

    # choose output format
    # TODO: do we need all of these formats, or just the dag?
    if format == 'dag':
        compiled_circuit = dag_circuit
    elif format == 'json':
        # FIXME: JsonBackend is wrongly taking an ordered dict as basis, not list
        dag_unroller = DagUnroller(dag_circuit, JsonBackend(dag_circuit.basis))
        compiled_circuit = dag_unroller.execute()
    elif format == 'qasm':
        compiled_circuit = dag_circuit.qasm()
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


def _matches_coupling_map(instructions, coupling_map):
    """Iterate over circuit instructions to check if all multi-qubit couplings
    match the qubit coupling graph in the backend.

    Parameters:
            instructions (list): List of circuit instructions.
            coupling_map (list): Backend coupling map, represented as an adjacency list.

    Returns:
            True: If all instructions readily fit the backend coupling graph.

            False: If there's at least one instruction that uses multiple qubits
                   which does not match the backend couplings.
    """
    for instruction in instructions:
        if isinstance(instruction, Gate) and instruction.is_multi_qubit():
            if instruction.get_qubit_coupling() not in coupling_map:
                return False
    return True


def _pick_best_layout(backend, num_qubits, qregs):
    """ Pick a convenient layout depending on the best matching qubit connectivity

    Parameters:
        backend (BaseBackend) : The backend with the coupling_map for searching
        num_qubits (int): Number of qubits
        qregs (list): The list of quantum registers

    Returns:
        initial_layout: A special ordered layout

    """
    best_sub = _best_subset(backend, num_qubits)
    layout = {}
    map_iter = 0
    for key, value in qregs.items():
        for i in range(value.size):
            layout[(key, i)] = ('q', best_sub[map_iter])
            map_iter += 1
    return layout
