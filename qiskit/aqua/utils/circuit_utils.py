# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
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

from qiskit import transpiler, BasicAer, QuantumRegister
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller


def convert_to_basis_gates(circuit):
    # unroll the circuit using the basis u1, u2, u3, cx, and id gates
    unroller = Unroller(basis=['u1', 'u2', 'u3', 'cx', 'id'])
    pm = PassManager(passes=[unroller])
    qc = transpiler.transpile(circuit, BasicAer.get_backend('qasm_simulator'), pass_manager=pm)
    return qc


def is_qubit(qb):
    # check if the input is a qubit, which is in the form (QuantumRegister, int)
    return isinstance(qb, tuple) and isinstance(qb[0], QuantumRegister) and isinstance(qb[1], int)
