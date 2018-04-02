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
import warnings

import qiskit.qasm as qasm
from qiskit._qiskiterror import QISKitError
from qiskit._compiler import compile_circuit
from qiskit.unroll import Unroller, CircuitBackend

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
    warnings.warn(
        "openquantumcompiler will be deprecated in upcoming versions (>0.5.0). "
        "Using qiskit.compile instead is recommended.", DeprecationWarning)

    compiled_circuit = compile_circuit(quantum_circuit, basis_gates, coupling_map,
                                       initial_layout, get_layout, format)
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
