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
from qiskit.unroll import DagUnroller, JsonBackend


# TODO: This is here for backward compatibility with QISKit Developer Challenge
# Once the challenge is finished, we have to remove this entire module.
def dag2json(dag_circuit, basis_gates='u1,u2,u3,cx,id'):
    """Make a Json representation of the circuit.
    Takes a circuit dag and returns json circuit obj. This is an internal
    function.
    Args:
        dag_circuit (QuantumCircuit): a dag representation of the circuit.
        basis_gates (str): a comma seperated string and are the base gates,
                               which by default are: u1,u2,u3,cx,id
    Returns:
        json: the json version of the dag
    """
    return DagUnroller(dag_circuit, JsonBackend(basis_gates.split(","))).execute()
