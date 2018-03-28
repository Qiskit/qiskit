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

"""Tools for compiling Quantum Programs."""
import logging

import qiskit.qasm as qasm
import qiskit.mapper as mapper
from qiskit._qiskiterror import QISKitError
from qiskit.dagcircuit import DAGCircuit
from qiskit.unroll import Unroller, CircuitBackend, DagUnroller, DAGBackend, JsonBackend

logger = logging.getLogger(__name__)


def compile(quantum_circuit, basis_gates='u1,u2,u3,cx,id', coupling_map=None,
            initial_layout=None, get_layout=False, format='dag'):
    """Compile the circuit.

    This builds the internal "to execute" list which is list of quantum
    circuits to run on different backends.

    Args:
        quantum_circuit (QuantumCircuit): circuit to compile
        basis_gates (str): a comma seperated string and are the base gates,
                           which by default are: u1,u2,u3,cx,id
        coupling_map (dict): A directed graph of coupling::

            {
             control(int):
                 [
                     target1(int),
                     target2(int),
                     , ...
                 ],
                 ...
            }

            eg. {0: [2], 1: [2], 3: [2]}

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
        coupling = mapper.Coupling(coupling_map)
        logger.info("initial layout: %s", initial_layout)
        compiled_dag_circuit, final_layout = mapper.swap_mapper(
            compiled_dag_circuit, coupling, initial_layout, trials=20, seed=13)
        logger.info("final layout: %s", final_layout)
        # Expand swaps
        dag_unroller = DagUnroller(compiled_dag_circuit, DAGBackend(basis))
        compiled_dag_circuit = dag_unroller.expand_gates()
        # Change cx directions
        compiled_dag_circuit = mapper.direction_mapper(compiled_dag_circuit, coupling)
        # Simplify cx gates
        mapper.cx_cancellation(compiled_dag_circuit)
        # Simplify single qubit gates
        compiled_dag_circuit = mapper.optimize_1q_gates(compiled_dag_circuit)
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

    Args:
        filename (str): a string for the filename including its location.
        basis_gates (str): basis to unroll circuit to.
    Returns:
        object: Returns a unrolled QuantumCircuit object
    """
    # create Program object Node (AST)
    node_circuit = qasm.Qasm(filename=filename).parse()
    node_unroller = Unroller(node_circuit, CircuitBackend(basis_gates.split(",")))
    circuit_unrolled = node_unroller.execute()
    return circuit_unrolled


class QISKitCompilerError(QISKitError):
    """Exceptions raised during compilation"""
    pass
