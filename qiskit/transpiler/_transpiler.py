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

from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.mapper import CouplingMap, swap_mapper
from qiskit.tools.parallel import parallel_map
from qiskit.converters import circuit_to_dag
from qiskit.converters import dag_to_circuit
from qiskit.extensions.standard import SwapGate
from .passes import (Unroller, CXDirection, CXCancellation,
                     Decompose, Optimize1qGates, BarrierBeforeFinalMeasurements)
from ._transpilererror import TranspilerError

logger = logging.getLogger(__name__)


def transpile(circuits, backend=None, basis_gates=None, coupling_map=None,
              initial_layout=None, seed_mapper=None, pass_manager=None):
    """transpile one or more circuits.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]): circuits to compile
        backend (BaseBackend): a backend to compile for
        basis_gates (str): comma-separated basis gate set to compile to
        coupling_map (list): coupling map (perhaps custom) to target in mapping
        initial_layout (list): initial layout of qubits in mapping
        seed_mapper (int): random seed for the swap_mapper
        pass_manager (PassManager): a pass_manager for the transpiler stages

    Returns:
        QuantumCircuit or list[QuantumCircuit]: transpiled circuit(s).

    Raises:
        TranspilerError: if args are not complete for the transpiler to function
    """
    return_form_is_single = False
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]
        return_form_is_single = True

    # Check for valid parameters for the experiments.
    basis_gates = basis_gates or ','.join(backend.configuration().basis_gates)
    coupling_map = coupling_map or getattr(backend.configuration(),
                                           'coupling_map', None)

    if not basis_gates:
        raise TranspilerError('no basis_gates or backend to compile to')

    circuits = parallel_map(_transpilation, circuits,
                            task_kwargs={'backend': backend,
                                         'basis_gates': basis_gates,
                                         'coupling_map': coupling_map,
                                         'initial_layout': initial_layout,
                                         'seed_mapper': seed_mapper,
                                         'pass_manager': pass_manager})
    if return_form_is_single:
        return circuits[0]
    return circuits


def _transpilation(circuit, backend=None, basis_gates=None, coupling_map=None,
                   initial_layout=None, seed_mapper=None,
                   pass_manager=None):
    """Perform transpilation of a single circuit.

    Args:
        circuit (QuantumCircuit): A circuit to transpile.
        backend (BaseBackend): a backend to compile for
        basis_gates (str): comma-separated basis gate set to compile to
        coupling_map (list): coupling map (perhaps custom) to target in mapping
        initial_layout (list): initial layout of qubits in mapping
        seed_mapper (int): random seed for the swap_mapper
        pass_manager (PassManager): a pass_manager for the transpiler stage

    Returns:
        QuantumCircuit: A transpiled circuit.

    Raises:
        TranspilerError: if args are not complete for transpiler to function.
    """
    if pass_manager and not pass_manager.working_list:
        return circuit

    dag = circuit_to_dag(circuit)
    if not backend and not initial_layout:
        raise TranspilerError('initial layout not supplied, and cannot '
                              'be inferred from backend.')
    if (initial_layout is None and not backend.configuration().simulator
            and not _matches_coupling_map(dag, coupling_map)):
        initial_layout = _pick_best_layout(dag, backend)

    final_dag = transpile_dag(dag, basis_gates=basis_gates,
                              coupling_map=coupling_map,
                              initial_layout=initial_layout,
                              format='dag',
                              seed_mapper=seed_mapper,
                              pass_manager=pass_manager)

    out_circuit = dag_to_circuit(final_dag)

    return out_circuit


# pylint: disable=redefined-builtin
def transpile_dag(dag, basis_gates='u1,u2,u3,cx,id', coupling_map=None,
                  initial_layout=None, format='dag', seed_mapper=None,
                  pass_manager=None):
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
        format (str): DEPRECATED The target format of the compilation: {'dag', 'json', 'qasm'}
        seed_mapper (int): random seed_mapper for the swap mapper
        pass_manager (PassManager): pass manager instance for the transpilation process
            If None, a default set of passes are run.
            Otherwise, the passes defined in it will run.
            If contains no passes in it, no dag transformations occur.

    Returns:
        DAGCircuit: transformed dag
        DAGCircuit, dict: transformed dag along with the final layout on backend qubits
    """
    # TODO: `basis_gates` will be removed after we have the unroller pass.
    # TODO: `coupling_map`, `initial_layout`, `seed_mapper` removed after mapper pass.

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
        dag = Unroller(basis).run(dag)
        name = dag.name
        # if a coupling map is given compile to the map
        if coupling_map:
            logger.info("pre-mapping properties: %s",
                        dag.properties())
            # Insert swap gates
            coupling = CouplingMap(couplinglist=coupling_map)
            logger.info("initial layout: %s", initial_layout)
            dag = BarrierBeforeFinalMeasurements().run(dag)
            dag, final_layout = swap_mapper(
                dag, coupling, initial_layout, trials=20, seed=seed_mapper)
            logger.info("final layout: %s", final_layout)
            # Expand swaps
            dag = Decompose(SwapGate).run(dag)
            # Change cx directions
            dag = CXDirection(coupling).run(dag)
            # Simplify cx gates
            dag = CXCancellation().run(dag)
            # Unroll to the basis
            dag = Unroller(['u1', 'u2', 'u3', 'id', 'cx']).run(dag)
            # Simplify single qubit gates
            dag = Optimize1qGates().run(dag)
            logger.info("post-mapping properties: %s",
                        dag.properties())
        dag.name = name

    if format != 'dag':
        warnings.warn("transpiler no longer supports different formats. "
                      "only dag to dag transformations are supported.",
                      DeprecationWarning)

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
        TranspilerError: Wrong number of qubits given.
    """
    if n_qubits == 1:
        return np.array([0])
    elif n_qubits <= 0:
        raise TranspilerError('Number of qubits <= 0.')

    device_qubits = backend.configuration().n_qubits
    if n_qubits > device_qubits:
        raise TranspilerError('Number of qubits greater than device.')

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
    q = QuantumRegister(device_qubits, 'q')
    for qreg in dag.qregs.values():
        for i in range(qreg.size):
            layout[(qreg.name, i)] = (q, int(best_sub[map_iter]))
            map_iter += 1
    return layout
