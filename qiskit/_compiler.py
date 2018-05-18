# -*- coding: utf-8 -*-
# pylint: disable=redefined-builtin

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Tools for compiling a batch of quantum circuits."""
import logging

import random
import string
import copy
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs

# Stable Modules
from ._qiskiterror import QISKitError
from ._quantumcircuit import QuantumCircuit
from .qasm import Qasm

# Beta Modules
from .dagcircuit import DAGCircuit
from .unroll import DagUnroller, DAGBackend, JsonBackend, Unroller, CircuitBackend
from .mapper import (Coupling, optimize_1q_gates, coupling_list2dict, swap_mapper,
                     cx_cancellation, direction_mapper)

logger = logging.getLogger(__name__)


def compile(circuits, backend,
            config=None, basis_gates=None, coupling_map=None, initial_layout=None,
            shots=1024, max_credits=10, seed=None, qobj_id=None, hpc=None,
            skip_translation=False):
    """Compile a list of circuits into a qobj.

    FIXME THIS FUNCTION WILL BE REWRITTEN IN VERSION 0.6. It will be a thin wrapper
    of circuit->dag, transpiler (dag -> dag) and dags-> qobj

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
        skip_translation (bool): If True, bypass most of the compilation process and
            creates a qobj with minimal check nor translation

    Returns:
        obj: the qobj to be run on the backends

    Raises:
        QISKitError: if any of the circuit names cannot be found on the
            Quantum Program.
    """
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]

    backend_conf = backend.configuration
    backend_name = backend_conf['name']

    qobj = {}
    if not qobj_id:
        qobj_id = "".join([random.choice(string.ascii_letters + string.digits)
                           for n in range(30)])
    qobj['id'] = qobj_id
    qobj['config'] = {'max_credits': max_credits,
                      'shots': shots,
                      'backend_name': backend_name}

    if hpc is not None and \
            not all(key in hpc for key in ('multi_shot_optimization', 'omp_num_threads')):
        raise QISKitError('Unknown HPC parameter format!')

    qobj['circuits'] = []
    if not basis_gates:
        if 'basis_gates' in backend_conf:
            basis_gates = backend_conf['basis_gates']
    if len(basis_gates.split(',')) < 2:
        # catches deprecated basis specification like 'SU2+CNOT'
        logger.warning('encountered deprecated basis specification: '
                       '"%s" substituting u1,u2,u3,cx,id', str(basis_gates))
        basis_gates = 'u1,u2,u3,cx,id'
    if not coupling_map:
        coupling_map = backend_conf['coupling_map']

    for circuit in circuits:
        num_qubits = sum((len(qreg) for qreg in circuit.get_qregs().values()))
        # TODO: A better solution is to have options to enable/disable optimizations
        if num_qubits == 1:
            coupling_map = None
        if coupling_map == 'all-to-all':
            coupling_map = None

        # making the job to be added to qobj
        job = {}
        job["name"] = circuit.name
        # config parameters used by the runner
        if config is None:
            config = {}  # default to empty config dict
        job["config"] = copy.deepcopy(config)
        job["config"]["coupling_map"] = coupling_map
        # TODO: Jay: make config options optional for different backends
        job["config"]["basis_gates"] = basis_gates
        if seed is None:
            job["config"]["seed"] = None
        else:
            job["config"]["seed"] = seed

        if skip_translation:  # Just return the qobj, without any transformation or analysis
            job["config"]["layout"] = None
            job["compiled_circuit_qasm"] = circuit.qasm()
            job["compiled_circuit"] = DagUnroller(
                DAGCircuit.fromQuantumCircuit(circuit),
                JsonBackend(job['config']['basis_gates'].split(','))).execute()
        else:
            # Pick good initial layout if None is given and not simulator
            if initial_layout is None and not backend.configuration['simulator']:
                best_sub = best_subset(backend, num_qubits)
                qreg_list = []
                for key, value in circuit.get_qregs().items():
                    qreg_list += [key]*len(value)

                initial_layout = {(rr, kk): ('q', best_sub[kk])
                                  for rr in qreg_list
                                  for kk in range(len(qreg_list))}
            dag_circuit, final_layout = compile_circuit(
                circuit,
                basis_gates=basis_gates,
                coupling_map=coupling_map,
                initial_layout=initial_layout,
                get_layout=True)
            # Map the layout to a format that can be json encoded
            list_layout = None
            if final_layout:
                list_layout = [[k, v] for k, v in final_layout.items()]
            job["config"]["layout"] = list_layout

            # the compiled circuit to be run saved as a dag
            # we assume that compile_circuit has already expanded gates
            # to the target basis, so we just need to generate json
            json_circuit = DagUnroller(dag_circuit, JsonBackend(dag_circuit.basis)).execute()
            job["compiled_circuit"] = json_circuit
            # set eval_symbols=True to evaluate each symbolic expression
            # TODO after transition to qobj, we can drop this
            job["compiled_circuit_qasm"] = dag_circuit.qasm(qeflag=True,
                                                            eval_symbols=True)
        # add job to the qobj
        qobj["circuits"].append(job)
    return qobj


def compile_circuit(quantum_circuit, basis_gates='u1,u2,u3,cx,id', coupling_map=None,
                    initial_layout=None, get_layout=False, format='dag'):
    """Compile the circuit.

    This builds the internal "to execute" list which is list of quantum
    circuits to run on different backends.

    Args:
        quantum_circuit (QuantumCircuit): circuit to compile
        basis_gates (str): a comma seperated string and are the base gates,
                           which by default are: u1,u2,u3,cx,id
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
        get_layout (bool): flag for returning the layout.
        format (str): The target format of the compilation:
            {'dag', 'json', 'qasm'}

    Returns:
        object: If get_layout == False, the compiled circuit in the specified
            format. If get_layout == True, a tuple is returned, with the
            second element being the layout.

    Raises:
        QISKitCompilerError: if the format is not valid.
    """
    compiled_dag_circuit = DAGCircuit.fromQuantumCircuit(quantum_circuit)
    basis = basis_gates.split(',') if basis_gates else []

    dag_unroller = DagUnroller(compiled_dag_circuit, DAGBackend(basis))
    compiled_dag_circuit = dag_unroller.expand_gates()
    final_layout = None
    # if a coupling map is given compile to the map
    if coupling_map:
        logger.info("pre-mapping properties: %s",
                    compiled_dag_circuit.property_summary())
        # Insert swap gates
        coupling = Coupling(coupling_list2dict(coupling_map))
        logger.info("initial layout: %s", initial_layout)
        compiled_dag_circuit, final_layout = swap_mapper(
            compiled_dag_circuit, coupling, initial_layout, trials=20, seed=13)
        logger.info("final layout: %s", final_layout)
        # Expand swaps
        dag_unroller = DagUnroller(compiled_dag_circuit, DAGBackend(basis))
        compiled_dag_circuit = dag_unroller.expand_gates()
        # Change cx directions
        compiled_dag_circuit = direction_mapper(compiled_dag_circuit, coupling)
        # Simplify cx gates
        cx_cancellation(compiled_dag_circuit)
        # Simplify single qubit gates
        compiled_dag_circuit = optimize_1q_gates(compiled_dag_circuit)
        logger.info("post-mapping properties: %s",
                    compiled_dag_circuit.property_summary())
    # choose output format
    if format == 'dag':
        compiled_circuit = compiled_dag_circuit
    elif format == 'json':
        dag_unroller = DagUnroller(compiled_dag_circuit,
                                   JsonBackend(list(compiled_dag_circuit.basis.keys())))
        compiled_circuit = dag_unroller.execute()
    elif format == 'qasm':
        compiled_circuit = compiled_dag_circuit.qasm()
    else:
        raise QISKitCompilerError('unrecognized circuit format')

    if get_layout:
        return compiled_circuit, final_layout
    return compiled_circuit


def load_unroll_qasm_file(filename, basis_gates='u1,u2,u3,cx,id'):
    """Load qasm file and return unrolled circuit

    XXX HOW IS THIS FUNCTION USED. COPIED FROM OPENQUANTUMCOMPILER

    Args:
        filename (str): a string for the filename including its location.
        basis_gates (str): basis to unroll circuit to.
    Returns:
        object: Returns a unrolled QuantumCircuit object
    """
    # create Program object Node (AST)
    node_circuit = Qasm(filename=filename).parse()
    node_unroller = Unroller(node_circuit, CircuitBackend(basis_gates.split(",")))
    circuit_unrolled = node_unroller.execute()
    return circuit_unrolled


def best_subset(backend, n_qubits):
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


class QISKitCompilerError(QISKitError):
    """Exceptions raised during compilation"""
    pass
