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

from .dagcircuit import DAGCircuit
from .unroll import DagUnroller, DAGBackend, JsonBackend

from . import backends
from ._qiskiterror import QISKitError
from ._measure import Measure
from ._gate import Gate
from ._quantumcircuit import QuantumCircuit
from .unroll import Unroller, CircuitBackend
from .extensions.standard.barrier import Barrier
from .mapper import (Coupling, optimize_1q_gates, coupling_list2dict, swap_mapper, 
                     cx_cancellation, direction_mapper)
from ._quantumjob import QuantumJob
from .qasm import Qasm


logger = logging.getLogger(__name__)

COMPILE_CONFIG_DEFAULT = {
    'backend': "local_qasm_simulator",
    'config': None,
    'basis_gates': None,
    'coupling_map': None,
    'initial_layout': None,
    'shots': 1024,
    'max_credits': 10,
    'seed': 1,
    'qobj_id': None,
    'hpc': None
}


def compile(list_of_circuits=None, compile_config=None):
    """Compile a list of circuits into a qobj.

    XXX THIS FUNCTION WILL BE REWRITTEN IN VERSION 0.6

    Args:
        list_of_circuits (list[QuantumCircuits]): list of circuits
        compile_config (dict or None): a dictionary of compile configurations.
            If `None`, the default compile configuration will be used.

    Returns:
        obj: the qobj to be run on the backends

    Raises:
        QISKitError: if any of the circuit names cannot be found on the
            Quantum Program.
    """
    if isinstance(list_of_circuits, QuantumCircuit):
        list_of_circuits = [list_of_circuits]

    compile_config = compile_config or COMPILE_CONFIG_DEFAULT
    backend = compile_config['backend']
    config = compile_config['config']
    basis_gates = compile_config['basis_gates']
    coupling_map = compile_config['coupling_map']
    initial_layout = compile_config['initial_layout']
    shots = compile_config['shots']
    max_credits = compile_config['max_credits']
    seed = compile_config['seed']
    qobj_id = compile_config['qobj_id']
    hpc = compile_config['hpc']

    qobj = {}
    if not qobj_id:
        qobj_id = "".join([random.choice(string.ascii_letters + string.digits)
                           for n in range(30)])
    qobj['id'] = qobj_id
    qobj["config"] = {"max_credits": max_credits, 'backend': backend,
                      "shots": shots}

    # TODO This backend needs HPC parameters to be passed in order to work
    if backend == 'ibmqx_hpc_qasm_simulator':
        if hpc is None:
            logger.info('ibmqx_hpc_qasm_simulator backend needs HPC '
                        'parameter. Setting defaults to hpc.multi_shot_optimization '
                        '= true and hpc.omp_num_threads = 16')
            hpc = {'multi_shot_optimization': True, 'omp_num_threads': 16}

        if not all(key in hpc for key in
                   ('multi_shot_optimization', 'omp_num_threads')):
            raise QISKitError('Unknown HPC parameter format!')

        qobj['config']['hpc'] = hpc
    elif hpc is not None:
        logger.info('HPC parameter is only available for '
                    'ibmqx_hpc_qasm_simulator. You are passing an HPC parameter '
                    'but you are not using ibmqx_hpc_qasm_simulator, so we will '
                    'ignore it.')
        hpc = None

    qobj['circuits'] = []
    backend_conf = backends.configuration(backend)
    if not basis_gates:
        if 'basis_gates' in backend_conf:
            basis_gates = backend_conf['basis_gates']
    elif len(basis_gates.split(',')) < 2:
        # catches deprecated basis specification like 'SU2+CNOT'
        logger.warning('encountered deprecated basis specification: '
                       '"%s" substituting u1,u2,u3,cx,id', str(basis_gates))
        basis_gates = 'u1,u2,u3,cx,id'
    if not coupling_map:
        coupling_map = backend_conf['coupling_map']
    for circuit in list_of_circuits:
        num_qubits = sum((len(qreg) for qreg in circuit.get_qregs().values()))
        # TODO: A better solution is to have options to enable/disable optimizations
        if num_qubits == 1:
            coupling_map = None
        if coupling_map == 'all-to-all':
            coupling_map = None
        # if the backend is a real chip, insert barrier before measurements
        if not backend_conf['simulator']:
            measured_qubits = []
            qasm_idx = []
            for i, instruction in enumerate(circuit.data):
                if isinstance(instruction, Measure):
                    measured_qubits.append(instruction.arg[0])
                    qasm_idx.append(i)
                elif isinstance(instruction, Gate) and bool(set(instruction.arg) &
                                                            set(measured_qubits)):
                    raise QISKitError('backend "{0}" rejects gate after '
                                      'measurement in circuit "{1}"'.format(backend, circuit.name))
            for i, qubit in zip(qasm_idx, measured_qubits):
                circuit.data.insert(i, Barrier([qubit], circuit))
        dag_circuit, final_layout = compile_circuit(
            circuit,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            initial_layout=initial_layout,
            get_layout=True)
        # making the job to be added to qobj
        job = {}
        job["name"] = circuit.name
        # config parameters used by the runner
        if config is None:
            config = {}  # default to empty config dict
        job["config"] = copy.deepcopy(config)
        job["config"]["coupling_map"] = coupling_map
        # TODO: Jay: make config options optional for different backends
        # Map the layout to a format that can be json encoded
        list_layout = None
        if final_layout:
            list_layout = [[k, v] for k, v in final_layout.items()]
        job["config"]["layout"] = list_layout
        job["config"]["basis_gates"] = basis_gates
        if seed is None:
            job["config"]["seed"] = None
        else:
            job["config"]["seed"] = seed
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


def execute(list_of_circuits, compile_config=None, wait=5, timeout=60):
    """Executes a set of circuits.

    Args:
        list_of_circuits (list[QuantumCircuits]): list of circuits

        wait (int): XXX -- I DONT THINK WE NEED TO KEEP THIS
        timeout (int): XXX -- I DONT THINK WE NEED TO KEEP THIS
        compile_config (dict or None): a dictionary of compile configurations.

    Returns:
        obj: The results object
    """
    compile_config = compile_config or COMPILE_CONFIG_DEFAULT

    backend = compile_config['backend']
    my_backend = backends.get_backend_instance(backend)
    qobj = compile(list_of_circuits, compile_config)

    q_job = QuantumJob(qobj, preformatted=True, resources={
        'max_credits': qobj['config']['max_credits'], 'wait': wait, 'timeout': timeout})
    result = my_backend.run(q_job)
    return result

def load_unroll_qasm_file(filename, basis_gates='u1,u2,u3,cx,id'):
    """Load qasm file and return unrolled circuit

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

class QISKitCompilerError(QISKitError):
    """Exceptions raised during compilation"""
    pass
